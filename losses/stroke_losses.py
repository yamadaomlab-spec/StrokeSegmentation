"""
stroke_losses.py – StrokeSegNet の各ヘッドに対応する損失関数

損失の構成 (設計書 Section 6):
    L = λ_fg * L_fg + λ_emb * L_emb + λ_ori * L_ori
          + λ_end * L_end + λ_junc * L_junc

各損失:
    L_fg   : BCE + Dice (前景セグメンテーション)
    L_emb  : Discriminative loss (埋め込み空間でのストローク分離)
    L_ori  : Cosine similarity 損失 (接線方向場, 方向の曖昧性に対応)
    L_end  : Focal loss (端点ヒートマップ)
    L_junc : Focal loss (交差点ヒートマップ)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ #
#  Foreground: BCE + Dice                                            #
# ------------------------------------------------------------------ #

def fg_loss(pred_logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Args:
        pred_logit : (B, 1, H, W) – シグモイド適用前
        target     : (B, H, W) float32 – 0/1 の前景マスク
    """
    t = target.unsqueeze(1)                      # (B, 1, H, W)
    bce = F.binary_cross_entropy_with_logits(pred_logit, t)

    p = torch.sigmoid(pred_logit)
    intersection = (p * t).sum(dim=(1, 2, 3))
    dice = 1.0 - (2.0 * intersection + 1.0) / (p.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3)) + 1.0)
    return bce + dice.mean()


# ------------------------------------------------------------------ #
#  Embedding: Discriminative Loss                                     #
# ------------------------------------------------------------------ #

def discriminative_loss(
    embedding: torch.Tensor,      # (B, D, H, W)
    stroke_id_map: torch.Tensor,  # (B, H, W) int64, 0=bg, 1〜N=stroke
    fg_mask: torch.Tensor,        # (B, H, W) float32
    delta_v: float = 0.5,
    delta_d: float = 1.5,
    lambda_reg: float = 1e-3,
) -> torch.Tensor:
    """
    De Brabandere et al. "Semantic Instance Segmentation with a Discriminative Loss Function"

    L_var : 同一ストローク内の埋め込みを重心へ引き寄せる
    L_dist: 異なるストロークの重心を互いに遠ざける
    L_reg : 重心を原点付近に保つ
    """
    B, D, H, W = embedding.shape
    device = embedding.device
    total_loss = torch.zeros(1, device=device)
    valid_batch = 0

    for b in range(B):
        fg_b = fg_mask[b].bool()               # (H, W)
        if not fg_b.any():
            continue

        emb_b = embedding[b, :, fg_b].T        # (N_fg, D)
        ids_b = stroke_id_map[b][fg_b]         # (N_fg,)  1〜K
        unique_ids = torch.unique(ids_b)
        unique_ids = unique_ids[unique_ids > 0]
        K = len(unique_ids)
        if K == 0:
            continue

        means = []
        L_var = torch.zeros(1, device=device)
        for k_id in unique_ids:
            mask_k = (ids_b == k_id)
            e_k = emb_b[mask_k]                # (n_k, D)
            mu_k = e_k.mean(0)                  # (D,)
            means.append(mu_k)
            dist_k = torch.norm(mu_k.detach() - e_k, dim=1)   # (n_k,)
            L_var = L_var + torch.clamp(dist_k - delta_v, min=0).pow(2).mean()
        L_var = L_var / K

        L_dist = torch.zeros(1, device=device)
        L_reg  = torch.zeros(1, device=device)
        means_t = torch.stack(means, dim=0)     # (K, D)
        for i in range(K):
            L_reg = L_reg + torch.norm(means_t[i])
            for j in range(i + 1, K):
                d_ij = torch.norm(means_t[i] - means_t[j])
                L_dist = L_dist + torch.clamp(2.0 * delta_d - d_ij, min=0).pow(2)
        L_reg  = L_reg / K
        L_dist = L_dist / max(K * (K - 1) / 2, 1)

        total_loss = total_loss + L_var + L_dist + lambda_reg * L_reg
        valid_batch += 1

    if valid_batch == 0:
        return total_loss
    return total_loss / valid_batch


# ------------------------------------------------------------------ #
#  Orientation: Cosine Similarity Loss (方向の正負を不問にする)        #
# ------------------------------------------------------------------ #

def orientation_loss(
    pred_ori: torch.Tensor,    # (B, 2, H, W)
    target_ori: torch.Tensor,  # (B, 2, H, W)
    fg_mask: torch.Tensor,     # (B, H, W) float32
) -> torch.Tensor:
    """
    前景画素のみで計算する方向場損失。
    L = 1 - |cos(θ_pred - θ_gt)| = 1 - |<pred_unit, gt_unit>|

    接線の正負 (ストローク進行方向) の曖昧性に対応するため絶対値を取る。
    """
    # 単位ベクトル化
    pred_n = F.normalize(pred_ori, dim=1, eps=1e-8)   # (B, 2, H, W)
    gt_n   = F.normalize(target_ori, dim=1, eps=1e-8) # (B, 2, H, W)

    cosine = (pred_n * gt_n).sum(dim=1)               # (B, H, W)
    loss_map = 1.0 - cosine.abs()                      # 0=完全一致

    mask = fg_mask.bool()
    if not mask.any():
        return loss_map.mean() * 0.0
    return loss_map[mask].mean()


# ------------------------------------------------------------------ #
#  Endpoint / Junction: Focal Loss                                    #
# ------------------------------------------------------------------ #

def focal_loss(
    pred_logit: torch.Tensor,  # (B, 1, H, W)
    target: torch.Tensor,      # (B, H, W) float32
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Binary focal loss: FL = -α_t (1 - p_t)^γ log(p_t)
    端点・交差点のように正例が少ない場合に有効。
    """
    t = target.unsqueeze(1).clamp(0, 1)   # (B, 1, H, W)
    bce = F.binary_cross_entropy_with_logits(pred_logit, t, reduction="none")
    p_t = torch.exp(-bce)
    alpha_t = alpha * t + (1.0 - alpha) * (1.0 - t)
    fl = alpha_t * (1.0 - p_t).pow(gamma) * bce
    return fl.mean()


# ------------------------------------------------------------------ #
#  Combined Loss                                                      #
# ------------------------------------------------------------------ #

class StrokeLoss(nn.Module):
    """
    5つのヘッド損失を重み付き合計する。
    weights: dict  {fg, emb, ori, end, junc}
    """

    def __init__(
        self,
        lambda_fg:   float = 1.0,
        lambda_emb:  float = 1.0,
        lambda_ori:  float = 0.5,
        lambda_end:  float = 0.5,
        lambda_junc: float = 0.5,
        delta_v:     float = 0.5,
        delta_d:     float = 1.5,
        lambda_reg:  float = 1e-3,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.lam = dict(fg=lambda_fg, emb=lambda_emb, ori=lambda_ori,
                        end=lambda_end, junc=lambda_junc)
        self.disc_kw  = dict(delta_v=delta_v, delta_d=delta_d, lambda_reg=lambda_reg)
        self.focal_kw = dict(alpha=focal_alpha, gamma=focal_gamma)

    def forward(self, preds: dict, targets: dict) -> tuple:
        """
        Args:
            preds   : model forward の出力 dict
            targets : DataLoader の batch dict (デバイス上のテンソル)

        Returns:
            total_loss, loss_dict (各損失の値)
        """
        fg_mask       = targets["fg_mask"]
        stroke_id_map = targets["stroke_id_map"]

        L_fg   = fg_loss(preds["fg"], fg_mask)

        L_emb  = discriminative_loss(
            preds["embedding"], stroke_id_map, fg_mask, **self.disc_kw
        )

        L_ori  = orientation_loss(
            preds["orientation"], targets["orientation"], fg_mask
        )

        L_end  = focal_loss(preds["endpoint"],  targets["endpoint_heatmap"], **self.focal_kw)
        L_junc = focal_loss(preds["junction"],  targets["junction_heatmap"], **self.focal_kw)

        total = (
            self.lam["fg"]   * L_fg
            + self.lam["emb"]  * L_emb
            + self.lam["ori"]  * L_ori
            + self.lam["end"]  * L_end
            + self.lam["junc"] * L_junc
        )

        loss_dict = {
            "total": total.item(),
            "fg":    L_fg.item(),
            "emb":   L_emb.item(),
            "ori":   L_ori.item(),
            "end":   L_end.item(),
            "junc":  L_junc.item(),
        }
        return total, loss_dict
