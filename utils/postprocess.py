"""
postprocess.py – 推論時のストローク分離処理

設計書 Section 10 の Step 1〜5 を実装:
    Step 1: foreground 閾値処理
    Step 2: 前景画素の embedding 抽出
    Step 3: DBSCAN クラスタリング
    Step 4: 各クラスタのバイナリマスク生成
    Step 5: カラー可視化
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from sklearn.cluster import DBSCAN


def predict(model, image_tensor: torch.Tensor, device: torch.device) -> dict:
    """
    モデルを推論モードで実行し、シグモイドを適用した出力を返す。

    Args:
        model        : StrokeSegNet
        image_tensor : (1, 1, H, W) float32 – 正規化済み入力
        device       : 実行デバイス

    Returns:
        dict: fg, embedding, orientation, endpoint, junction (numpy 配列)
    """
    model.eval()
    with torch.no_grad():
        out = model(image_tensor.to(device))

    return {
        "fg":          torch.sigmoid(out["fg"]).squeeze().cpu().numpy(),        # (H, W)
        "embedding":   out["embedding"].squeeze().cpu().numpy(),                # (D, H, W)
        "orientation": out["orientation"].squeeze().cpu().numpy(),              # (2, H, W)
        "endpoint":    torch.sigmoid(out["endpoint"]).squeeze().cpu().numpy(),  # (H, W)
        "junction":    torch.sigmoid(out["junction"]).squeeze().cpu().numpy(),  # (H, W)
    }


def separate_strokes(
    preds: dict,
    fg_threshold:       float = 0.5,
    dbscan_eps:         float = 0.3,
    dbscan_min_samples: int   = 5,
) -> list[np.ndarray]:
    """
    予測結果からストロークごとのバイナリマスクを生成する。

    Args:
        preds             : predict() の出力
        fg_threshold      : 前景確率の閾値
        dbscan_eps        : DBSCAN の近傍半径 (埋め込み空間)
        dbscan_min_samples: DBSCAN の最小点数

    Returns:
        list of (H, W) bool ndarray – 各ストロークのバイナリマスク
        (ストローク数はデータによって異なる)
    """
    fg = preds["fg"]
    H, W = fg.shape
    fg_mask = fg >= fg_threshold

    if not fg_mask.any():
        return []

    # 前景画素の embedding を抽出
    emb = preds["embedding"]          # (D, H, W)
    ys, xs = np.where(fg_mask)
    fg_emb = emb[:, ys, xs].T         # (N_fg, D)

    # DBSCAN でクラスタリング
    clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric="euclidean")
    labels = clustering.fit_predict(fg_emb)  # -1 = ノイズ

    # クラスタごとにマスクを生成
    unique_labels = sorted(set(labels.tolist()) - {-1})
    masks = []
    for lbl in unique_labels:
        mask = np.zeros((H, W), dtype=bool)
        pts_idx = np.where(labels == lbl)[0]
        mask[ys[pts_idx], xs[pts_idx]] = True
        masks.append(mask)

    return masks


def visualize_strokes(image: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
    """
    入力画像の上に各ストロークを色付きで重ねた可視化画像を返す。

    Args:
        image : (H, W) float32 または uint8 – グレースケール入力
        masks : separate_strokes() の出力

    Returns:
        (H, W, 3) uint8 RGB 画像
    """
    if image.dtype != np.uint8:
        img_u8 = np.clip(image * 255, 0, 255).astype(np.uint8)
    else:
        img_u8 = image

    vis = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)

    # 各ストロークに色を割り当て (HSV 等間隔)
    N = max(len(masks), 1)
    colors = [
        tuple(
            int(c)
            for c in cv2.cvtColor(
                np.array([[[int(i * 180 / N), 200, 220]]], dtype=np.uint8),
                cv2.COLOR_HSV2RGB,
            )[0, 0]
        )
        for i in range(N)
    ]

    overlay = vis.copy()
    for mask, color in zip(masks, colors):
        overlay[mask] = color

    # 半透明合成
    result = cv2.addWeighted(vis, 0.4, overlay, 0.6, 0)
    return result


def visualize_targets(targets: dict) -> dict:
    """
    訓練ターゲットを可視化用 uint8 画像に変換して返す。
    (デバッグ・検証用)

    Returns:
        dict of (H, W, 3) or (H, W) uint8 numpy arrays
    """
    def _to_u8(arr):
        a = arr.squeeze()
        a = (a - a.min()) / (a.max() - a.min() + 1e-8)
        return (a * 255).astype(np.uint8)

    result = {}

    # 入力画像
    img = targets["image"].squeeze().numpy()
    result["image"] = _to_u8(img)

    # 前景マスク
    result["fg_mask"] = _to_u8(targets["fg_mask"].numpy())

    # ストロークID マップ (色分け)
    sid = targets["stroke_id_map"].numpy()
    N = int(sid.max())
    sid_rgb = np.zeros((*sid.shape, 3), dtype=np.uint8)
    for i in range(1, N + 1):
        hue = int((i - 1) * 180 / max(N, 1))
        color = cv2.cvtColor(
            np.array([[[hue, 200, 220]]], dtype=np.uint8), cv2.COLOR_HSV2RGB
        )[0, 0]
        sid_rgb[sid == i] = color
    result["stroke_id_map"] = sid_rgb

    # 接線方向場 (角度を色で表現)
    ori = targets["orientation"].numpy()  # (2, H, W)
    angle = np.arctan2(ori[1], ori[0])    # -π〜π
    hue = ((angle + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
    fg = (targets["fg_mask"].numpy() * 255).astype(np.uint8)
    hsv = np.stack([hue, np.full_like(hue, 200), fg], axis=-1)
    result["orientation"] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # 端点・交差点ヒートマップ
    result["endpoint_heatmap"] = _to_u8(targets["endpoint_heatmap"].numpy())
    result["junction_heatmap"] = _to_u8(targets["junction_heatmap"].numpy())

    return result
