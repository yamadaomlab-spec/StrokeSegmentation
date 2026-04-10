"""
evaluate.py – 学習済みモデルのストローク分離精度を評価する

評価指標 (設計書 Section 12):
    前景セグメンテーション: mIoU, Dice
    ストローク分離 (最大 IoU マッチング):
        stroke precision, recall, F1
        over-segmentation rate, under-segmentation rate

使い方:
    python evaluate.py --checkpoint outputs/ckpt_best.pth [--config configs/config.yaml]
    python evaluate.py --checkpoint outputs/ckpt_best.pth --visualize --vis_dir ./vis_output
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
import cv2

from datasets import LMDBStrokeDataset
from models import StrokeSegNet
from utils import predict, separate_strokes, visualize_strokes


# ------------------------------------------------------------------ #
#  評価指標                                                           #
# ------------------------------------------------------------------ #

def binary_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(inter) / (float(union) + 1e-8)


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = (pred & gt).sum()
    return 2.0 * float(inter) / (float(pred.sum()) + float(gt.sum()) + 1e-8)


def match_strokes(pred_masks, gt_masks, iou_threshold: float = 0.5):
    """
    predicted masks と GT masks を最大 IoU で貪欲マッチングし、
    (TP, FP, FN, over_seg_count, under_seg_count) を返す。

    over_seg  : 1 GT に複数 pred がマッチ
    under_seg : 1 pred に複数 GT がマッチ
    """
    if not gt_masks:
        return 0, len(pred_masks), 0, 0, 0
    if not pred_masks:
        return 0, 0, len(gt_masks), 0, 0

    matched_gt = set()
    matched_pred = set()
    tp = 0
    over_seg = 0
    under_seg = 0

    # GT ごとに最もよくマッチする pred を探す
    for gi, gm in enumerate(gt_masks):
        best_iou, best_pi = 0.0, -1
        for pi, pm in enumerate(pred_masks):
            iou = binary_iou(pm, gm)
            if iou > best_iou:
                best_iou, best_pi = iou, pi
        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(gi)
            matched_pred.add(best_pi)

    # 1 GT に複数 pred がマッチしているか確認 (over-segmentation)
    for pi, pm in enumerate(pred_masks):
        matching_gts = sum(
            1 for gi, gm in enumerate(gt_masks) if binary_iou(pm, gm) >= iou_threshold
        )
        if matching_gts >= 2:
            over_seg += 1

    # 1 pred に複数 GT がマッチしているか (under-segmentation)
    for gi, gm in enumerate(gt_masks):
        matching_preds = sum(
            1 for pi, pm in enumerate(pred_masks) if binary_iou(pm, gm) >= iou_threshold
        )
        if matching_preds >= 2:
            under_seg += 1

    fp = len(pred_masks) - len(matched_pred)
    fn = len(gt_masks) - len(matched_gt)
    return tp, fp, fn, over_seg, under_seg


def gt_masks_from_batch(stroke_id_map: np.ndarray) -> list[np.ndarray]:
    """stroke_id_map から GT バイナリマスクのリストを生成。"""
    ids = np.unique(stroke_id_map)
    ids = ids[ids > 0]
    return [(stroke_id_map == i) for i in ids]


# ------------------------------------------------------------------ #
#  評価メインループ                                                   #
# ------------------------------------------------------------------ #

@torch.no_grad()
def evaluate(
    model, loader, device,
    fg_threshold: float = 0.5,
    dbscan_eps: float = 0.3,
    dbscan_min_samples: int = 5,
    visualize: bool = False,
    vis_dir: str = "./vis_output",
    max_vis: int = 50,
):
    model.eval()
    vis_count = 0
    if visualize:
        Path(vis_dir).mkdir(parents=True, exist_ok=True)

    # 累積値
    fg_iou_sum, fg_dice_sum = 0.0, 0.0
    total_tp, total_fp, total_fn = 0, 0, 0
    total_over, total_under = 0, 0
    n_samples = 0

    for batch_idx, batch in enumerate(loader):
        B = batch["image"].shape[0]
        for i in range(B):
            image_t     = batch["image"][i : i + 1]       # (1,1,H,W)
            fg_gt       = batch["fg_mask"][i].numpy()     # (H,W)
            sid_map     = batch["stroke_id_map"][i].numpy()
            tag_char    = batch["tag_char"][i]

            # 推論
            preds = predict(model, image_t, device)

            # 前景評価
            fg_pred = (preds["fg"] >= fg_threshold)
            fg_iou_sum  += binary_iou(fg_pred, fg_gt.astype(bool))
            fg_dice_sum += dice_score(fg_pred, fg_gt.astype(bool))

            # ストローク分離評価
            pred_masks = separate_strokes(
                preds,
                fg_threshold=fg_threshold,
                dbscan_eps=dbscan_eps,
                dbscan_min_samples=dbscan_min_samples,
            )
            gt_masks = gt_masks_from_batch(sid_map)

            tp, fp, fn, over, under = match_strokes(pred_masks, gt_masks)
            total_tp    += tp
            total_fp    += fp
            total_fn    += fn
            total_over  += over
            total_under += under
            n_samples   += 1

            # 可視化
            if visualize and vis_count < max_vis:
                img_np = image_t.squeeze().numpy()
                vis = visualize_strokes(img_np, pred_masks)
                # GT overlayも並べて保存
                gt_vis = visualize_strokes(img_np, gt_masks)
                combined = np.concatenate([vis, gt_vis], axis=1)
                fname = Path(vis_dir) / f"{batch_idx:04d}_{i:02d}_{tag_char}.png"
                cv2.imwrite(str(fname), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
                vis_count += 1

    # 集計
    fg_iou  = fg_iou_sum  / max(n_samples, 1)
    fg_dice = fg_dice_sum / max(n_samples, 1)

    prec  = total_tp / max(total_tp + total_fp, 1)
    rec   = total_tp / max(total_tp + total_fn, 1)
    f1    = 2 * prec * rec / max(prec + rec, 1e-8)
    over_rate  = total_over  / max(n_samples, 1)
    under_rate = total_under / max(n_samples, 1)

    results = {
        "fg_iou":            fg_iou,
        "fg_dice":           fg_dice,
        "stroke_precision":  prec,
        "stroke_recall":     rec,
        "stroke_f1":         f1,
        "over_seg_rate":     over_rate,
        "under_seg_rate":    under_rate,
        "n_samples":         n_samples,
    }
    return results


# ------------------------------------------------------------------ #
#  エントリポイント                                                   #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--visualize",  action="store_true")
    parser.add_argument("--vis_dir",    default="./vis_output")
    parser.add_argument("--max_vis",    type=int, default=50)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Dataset ---
    val_ds = LMDBStrokeDataset(
        lmdb_path       = cfg["DATA"]["lmdb_test"],
        img_size        = cfg["DATA"]["img_size"],
        margin          = cfg["DATA"]["margin"],
        thickness_range = (cfg["DATA"]["thickness_min"], cfg["DATA"]["thickness_max"]),
        thickness_fixed = cfg["DATA"]["thickness_test"],
        endpoint_sigma  = cfg["DATA"]["endpoint_sigma"],
        junction_sigma  = cfg["DATA"]["junction_sigma"],
        augment         = False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg["TRAIN"]["batch_size"],
        shuffle     = False,
        num_workers = cfg["DATA"]["num_workers"],
        pin_memory  = True,
    )

    # --- Model ---
    model = StrokeSegNet(emb_dim=cfg["MODEL"]["emb_dim"]).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded: {args.checkpoint}  (epoch {ckpt.get('epoch', '?')})")

    # --- Evaluate ---
    infer_cfg = cfg["INFER"]
    results = evaluate(
        model, val_loader, device,
        fg_threshold       = infer_cfg["fg_threshold"],
        dbscan_eps         = infer_cfg["dbscan_eps"],
        dbscan_min_samples = infer_cfg["dbscan_min_samples"],
        visualize          = args.visualize,
        vis_dir            = args.vis_dir,
        max_vis            = args.max_vis,
    )

    print("\n===== Evaluation Results =====")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:<22} : {v:.4f}")
        else:
            print(f"  {k:<22} : {v}")


if __name__ == "__main__":
    main()
