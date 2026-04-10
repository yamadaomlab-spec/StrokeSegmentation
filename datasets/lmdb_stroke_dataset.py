"""
lmdb_stroke_dataset.py – PEN-Net の data_lmdb を読み込む PyTorch Dataset

data_lmdb の各レコード形式:
    key  : b'0', b'1', ...  (連番)
    value: pickle({'tag_char': str, 'coordinates': [[x0,y0,x1,y1,...], ...]})

coordinates の各ストロークは [x0, y0, x1, y1, ...] のフラットリストで
(x_i, y_i) は POT 生座標（整数）。
"""

import pickle
import random
import lmdb
import torch
from torch.utils.data import Dataset

from .target_generator import TargetGenerator


def _parse_coords(raw_coords):
    """
    フラットリスト [[x0,y0,x1,y1,...], ...] を [(x,y), ...] のリストのリストに変換。
    点数が 1 以下のストロークは除外。
    """
    strokes = []
    for s in raw_coords:
        pts = [(s[i], s[i + 1]) for i in range(0, len(s) - 1, 2)]
        if len(pts) >= 2:
            strokes.append(pts)
    return strokes


class LMDBStrokeDataset(Dataset):
    """
    PEN-Net の data_lmdb から手書き文字のストロークデータを読み込み、
    StrokeSegNet の訓練に必要なターゲットを生成して返す。

    返却する dict:
        image            Tensor (1, H, W)  – 入力文字画像
        fg_mask          Tensor (H, W)     – 前景マスク
        stroke_id_map    Tensor (H, W)     – 画素ごとストロークID (int64)
        orientation      Tensor (2, H, W)  – (cos θ, sin θ) 接線方向場
        endpoint_heatmap Tensor (H, W)     – 端点ガウスヒートマップ
        junction_heatmap Tensor (H, W)     – 交差点ガウスヒートマップ
        num_strokes      int
        tag_char         str
    """

    def __init__(
        self,
        lmdb_path: str,
        img_size: int = 256,
        margin: int = 12,
        thickness_range: tuple = (1, 3),
        thickness_fixed: int = 2,      # テスト時に使う固定太さ
        endpoint_sigma: float = 3.0,
        junction_sigma: float = 5.0,
        augment: bool = True,
        max_retries: int = 10,
    ):
        self.thickness_range = thickness_range
        self.thickness_fixed = thickness_fixed
        self.augment = augment
        self.max_retries = max_retries

        self.gen = TargetGenerator(
            img_size=img_size,
            margin=margin,
            endpoint_sigma=endpoint_sigma,
            junction_sigma=junction_sigma,
        )

        self.env = lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.env.begin() as txn:
            self.num_samples = int(txn.get(b"num_sample"))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        for attempt in range(self.max_retries):
            try:
                sample = self._load_one((idx + attempt) % self.num_samples)
                if sample is not None:
                    return sample
            except Exception:
                pass
        # フォールバック: ランダムサンプル
        return self._load_one(random.randint(0, self.num_samples - 1))

    def _load_one(self, idx):
        with self.env.begin() as txn:
            raw = txn.get(str(idx).encode())
        if raw is None:
            return None
        data = pickle.loads(raw)

        strokes = _parse_coords(data["coordinates"])
        if not strokes:
            return None

        thickness = (
            random.randint(*self.thickness_range)
            if self.augment
            else self.thickness_fixed
        )

        targets = self.gen.generate(strokes, thickness=thickness, augment=self.augment)
        if targets is None:
            return None

        return {
            "image":             torch.from_numpy(targets["image"]),
            "fg_mask":           torch.from_numpy(targets["fg_mask"]),
            "stroke_id_map":     torch.from_numpy(targets["stroke_id_map"]),
            "orientation":       torch.from_numpy(targets["orientation"]),
            "endpoint_heatmap":  torch.from_numpy(targets["endpoint_heatmap"]),
            "junction_heatmap":  torch.from_numpy(targets["junction_heatmap"]),
            "num_strokes":       len(strokes),
            "tag_char":          data["tag_char"],
        }
