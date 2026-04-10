"""
target_generator.py – ストローク座標から訓練ターゲットを生成する

入力: 各ストロークの座標列 (list of list of (x, y))
出力:
  image            (1, H, W) float32  – 全ストロークを描画した入力画像 (1=筆画, 0=背景)
  fg_mask          (H, W)    float32  – 前景(筆画)マスク
  stroke_id_map    (H, W)    int64    – 画素ごとのストロークID (0=背景, 1〜N=各ストローク)
  orientation      (2, H, W) float32  – (cos θ, sin θ) 接線方向場
  endpoint_heatmap (H, W)    float32  – ストローク端点ガウスヒートマップ
  junction_heatmap (H, W)    float32  – 交差点ガウスヒートマップ
"""

import random
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree


class TargetGenerator:
    def __init__(
        self,
        img_size: int = 256,
        margin: int = 12,
        endpoint_sigma: float = 3.0,
        junction_sigma: float = 5.0,
    ):
        self.img_size = img_size
        self.margin = margin
        self.endpoint_sigma = endpoint_sigma
        self.junction_sigma = junction_sigma

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def generate(self, strokes, thickness: int = 2, augment: bool = False):
        """
        Args:
            strokes  : list of strokes. 各ストロークは [(x, y), ...] のリスト
            thickness: ポリライン描画の太さ
            augment  : True のとき回転・水平反転を適用

        Returns:
            dict (image, fg_mask, stroke_id_map, orientation,
                  endpoint_heatmap, junction_heatmap) or None
        """
        strokes = [s for s in strokes if len(s) >= 2]
        if not strokes:
            return None

        H = W = self.img_size

        norm = self._normalize(strokes)
        if norm is None:
            return None

        if augment:
            norm = self._augment(norm, H, W)

        # --- 各ストロークのバイナリマスク ---
        stroke_masks = [self._render_stroke(pts, H, W, thickness) for pts in norm]

        # --- 入力画像: 全ストロークを白背景・黒線で描画, 正規化 ---
        canvas = self._render_all(norm, H, W, thickness)
        image = (1.0 - canvas / 255.0).astype(np.float32)  # 1=筆画, 0=背景
        image = image[np.newaxis]  # (1, H, W)

        # --- 前景マスク (各ストロークマスクの和) ---
        fg_mask = np.clip(sum(stroke_masks), 0, 1).astype(np.float32)
        if fg_mask.sum() < 5:
            return None

        # --- ストロークID マップ (最近傍中心線への帰属) ---
        stroke_id_map = self._stroke_id_map(norm, stroke_masks, H, W)

        # --- 接線方向場 ---
        orientation = self._orientation_field(norm, stroke_id_map, H, W)

        # --- 端点ヒートマップ ---
        endpoint_hm = self._endpoint_heatmap(norm, H, W)

        # --- 交差点ヒートマップ ---
        junction_hm = self._junction_heatmap(stroke_masks, H, W)

        return {
            "image":             image,           # (1, H, W) float32
            "fg_mask":           fg_mask,         # (H, W) float32
            "stroke_id_map":     stroke_id_map,   # (H, W) int64
            "orientation":       orientation,     # (2, H, W) float32
            "endpoint_heatmap":  endpoint_hm,     # (H, W) float32
            "junction_heatmap":  junction_hm,     # (H, W) float32
        }

    # ------------------------------------------------------------------ #
    #  座標正規化・拡張                                                   #
    # ------------------------------------------------------------------ #

    def _normalize(self, strokes):
        """全ストロークを (img_size - 2*margin) のキャンバスに収まるようスケール・中央配置。"""
        all_pts = [p for s in strokes for p in s]
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        ori_w = max(x_max - x_min, 1.0)
        ori_h = max(y_max - y_min, 1.0)
        canvas = self.img_size - 2 * self.margin
        if canvas <= 0:
            return None
        r = min(canvas / ori_w, canvas / ori_h)

        x_off = self.margin + (canvas - ori_w * r) / 2.0
        y_off = self.margin + (canvas - ori_h * r) / 2.0

        return [
            [((p[0] - x_min) * r + x_off, (p[1] - y_min) * r + y_off) for p in s]
            for s in strokes
        ]

    def _augment(self, strokes, H, W):
        """ランダム回転 (±20°) と水平反転。"""
        cx, cy = W / 2.0, H / 2.0
        if random.random() < 0.7:
            angle = random.uniform(-20, 20)
            rad = np.radians(angle)
            cos_a, sin_a = float(np.cos(rad)), float(np.sin(rad))
            strokes = [
                [
                    (
                        (p[0] - cx) * cos_a - (p[1] - cy) * sin_a + cx,
                        (p[0] - cx) * sin_a + (p[1] - cy) * cos_a + cy,
                    )
                    for p in s
                ]
                for s in strokes
            ]
        if random.random() < 0.5:
            strokes = [[(W - 1 - p[0], p[1]) for p in s] for s in strokes]
        return strokes

    # ------------------------------------------------------------------ #
    #  描画ユーティリティ                                                 #
    # ------------------------------------------------------------------ #

    def _to_int32(self, pts, H, W):
        arr = np.array(pts, dtype=np.float32)
        arr[:, 0] = np.clip(arr[:, 0], 0, W - 1)
        arr[:, 1] = np.clip(arr[:, 1], 0, H - 1)
        return arr.astype(np.int32)

    def _render_stroke(self, pts, H, W, thickness) -> np.ndarray:
        """単一ストロークを float32 バイナリマスク (0/1) として描画。"""
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.polylines(mask, [self._to_int32(pts, H, W)], False, 1, thickness, cv2.LINE_AA)
        return (mask > 0).astype(np.float32)

    def _render_all(self, strokes, H, W, thickness) -> np.ndarray:
        """全ストロークを白背景・黒線で描画した uint8 画像を返す。"""
        canvas = np.full((H, W), 255, dtype=np.uint8)
        for pts in strokes:
            cv2.polylines(canvas, [self._to_int32(pts, H, W)], False, 0, thickness, cv2.LINE_AA)
        return canvas.astype(np.float32)

    # ------------------------------------------------------------------ #
    #  ストロークID マップ                                                #
    # ------------------------------------------------------------------ #

    def _stroke_id_map(self, strokes, stroke_masks, H, W) -> np.ndarray:
        """
        各前景画素を最近傍の中心線を持つストロークに帰属させる。
        distance_transform_edt を各ストロークの中心線画素から計算して比較。
        """
        N = len(strokes)
        dist_stack = np.empty((N, H, W), dtype=np.float32)

        for i, pts in enumerate(strokes):
            # 中心線画素を 1, それ以外を 0 とした画像を作成
            cl = np.ones((H, W), dtype=np.uint8)
            for (x, y) in pts:
                xi, yi = int(round(x)), int(round(y))
                if 0 <= xi < W and 0 <= yi < H:
                    cl[yi, xi] = 0  # 中心線 = 0 (距離変換の「ゼロ」)
            dist_stack[i] = distance_transform_edt(cl)

        fg_mask = (sum(stroke_masks) > 0)
        nearest = np.argmin(dist_stack, axis=0)          # (H, W) 0-indexed
        stroke_id_map = np.zeros((H, W), dtype=np.int64)
        stroke_id_map[fg_mask] = nearest[fg_mask] + 1   # 1-indexed (0=背景)
        return stroke_id_map

    # ------------------------------------------------------------------ #
    #  接線方向場                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_tangents(pts) -> np.ndarray:
        """各座標点での単位接線ベクトル (N, 2) を計算。"""
        arr = np.array(pts, dtype=np.float32)   # (N, 2) x,y
        N = len(arr)
        tang = np.zeros((N, 2), dtype=np.float32)
        for t in range(N):
            if t == 0:
                v = arr[1] - arr[0] if N > 1 else np.array([1.0, 0.0])
            elif t == N - 1:
                v = arr[N - 1] - arr[N - 2]
            else:
                v = arr[t + 1] - arr[t - 1]
            norm = float(np.hypot(v[0], v[1]))
            tang[t] = v / (norm + 1e-8)
        return tang

    def _orientation_field(self, strokes, stroke_id_map, H, W) -> np.ndarray:
        """
        各前景画素に対して、帰属するストロークの最近傍中心線点の接線方向を割り当てる。
        出力: (2, H, W) float32 – (cos θ, sin θ)
        """
        orientation = np.zeros((2, H, W), dtype=np.float32)

        for i, pts in enumerate(strokes):
            pts_arr = np.array(pts, dtype=np.float32)  # (N, 2) x,y
            tangents = self._compute_tangents(pts)

            mask = stroke_id_map == (i + 1)
            if not mask.any():
                continue

            ys, xs = np.where(mask)
            pixel_xy = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)

            if len(pts_arr) == 1:
                orientation[0, ys, xs] = tangents[0, 0]
                orientation[1, ys, xs] = tangents[0, 1]
            else:
                tree = cKDTree(pts_arr)
                _, idx = tree.query(pixel_xy)
                orientation[0, ys, xs] = tangents[idx, 0]
                orientation[1, ys, xs] = tangents[idx, 1]

        return orientation

    # ------------------------------------------------------------------ #
    #  ヒートマップ                                                       #
    # ------------------------------------------------------------------ #

    def _gaussian_heatmap(self, points, H, W, sigma: float) -> np.ndarray:
        """点群をガウスヒートマップに変換。最大値で正規化して返す。"""
        hm = np.zeros((H, W), dtype=np.float32)
        for (x, y) in points:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < W and 0 <= yi < H:
                hm[yi, xi] = 1.0
        k = max(3, int(6 * sigma + 1) | 1)   # 奇数カーネル
        hm = cv2.GaussianBlur(hm, (k, k), sigma)
        if hm.max() > 0:
            hm /= hm.max()
        return hm

    def _endpoint_heatmap(self, strokes, H, W) -> np.ndarray:
        """各ストロークの始点・終点にガウスを置いたヒートマップ。"""
        endpoints = []
        for pts in strokes:
            endpoints.append(pts[0])
            endpoints.append(pts[-1])
        return self._gaussian_heatmap(endpoints, H, W, self.endpoint_sigma)

    def _junction_heatmap(self, stroke_masks, H, W) -> np.ndarray:
        """2本以上のストロークが重なる画素を中心としたガウスヒートマップ。"""
        if len(stroke_masks) < 2:
            return np.zeros((H, W), dtype=np.float32)
        count_map = sum(stroke_masks)
        junc = (count_map >= 2.0).astype(np.float32)
        k = max(3, int(6 * self.junction_sigma + 1) | 1)
        hm = cv2.GaussianBlur(junc, (k, k), self.junction_sigma)
        if hm.max() > 0:
            hm /= hm.max()
        return hm
