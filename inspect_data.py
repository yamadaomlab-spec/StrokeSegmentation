"""
inspect_data.py – データ・ターゲット生成の動作確認スクリプト

使い方:
    python inspect_data.py [--n 8] [--out_dir ./inspect_out]

指定した枚数のサンプルについて:
    - 入力画像
    - 前景マスク
    - ストロークID マップ (色分け)
    - 接線方向場
    - 端点・交差点ヒートマップ
をまとめた確認画像を out_dir に保存する。
"""

import argparse
import pickle
import random
from pathlib import Path

import lmdb
import numpy as np
import cv2
import yaml

from datasets.lmdb_stroke_dataset import _parse_coords
from datasets.target_generator import TargetGenerator
from utils.postprocess import visualize_targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="configs/config.yaml")
    parser.add_argument("--n",       type=int, default=8, help="出力サンプル数")
    parser.add_argument("--out_dir", default="./inspect_out")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lmdb_path = cfg["DATA"]["lmdb_train"]
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    with env.begin() as txn:
        num = int(txn.get(b"num_sample"))

    gen = TargetGenerator(
        img_size       = cfg["DATA"]["img_size"],
        margin         = cfg["DATA"]["margin"],
        endpoint_sigma = cfg["DATA"]["endpoint_sigma"],
        junction_sigma = cfg["DATA"]["junction_sigma"],
    )

    print(f"LMDB: {lmdb_path}  ({num} samples)")
    indices = random.sample(range(num), min(args.n, num))

    for idx in indices:
        with env.begin() as txn:
            data = pickle.loads(txn.get(str(idx).encode()))

        tag_char = data["tag_char"]
        strokes  = _parse_coords(data["coordinates"])
        if not strokes:
            continue

        tgt = gen.generate(strokes, thickness=2, augment=False)
        if tgt is None:
            continue

        import torch
        tgt_tensors = {k: torch.from_numpy(v) for k, v in tgt.items()}
        vis_dict = visualize_targets(tgt_tensors)

        # 6パネルを横に並べる
        panels = [
            vis_dict["image"],
            vis_dict["fg_mask"],
            vis_dict["stroke_id_map"],
            vis_dict["orientation"],
            vis_dict["endpoint_heatmap"],
            vis_dict["junction_heatmap"],
        ]
        labels = ["image", "fg_mask", "stroke_id", "orientation", "endpoint", "junction"]

        rows = []
        for panel, label in zip(panels, labels):
            if panel.ndim == 2:
                panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2RGB)
            H, W = panel.shape[:2]
            cv2.putText(panel, label, (4, 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 0), 1, cv2.LINE_AA)
            rows.append(panel)

        combined = np.concatenate(rows, axis=1)
        fname = out_dir / f"{idx:06d}_{tag_char}.png"
        cv2.imwrite(str(fname), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        print(f"  saved: {fname}  (char={tag_char!r}, strokes={len(strokes)})")

    print(f"\nDone. Output in: {out_dir}")


if __name__ == "__main__":
    main()
