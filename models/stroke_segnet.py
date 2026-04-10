"""
stroke_segnet.py – ResUNet ベースのストローク分離ネットワーク

構造:
    Encoder : stem + 4 段の ResBlock (各 stride-2 ダウンサンプリング)
    Bridge  : 2 段の ResBlock
    Decoder : U-Net 型スキップ接続付きアップサンプリング
    Heads   :
        fg         1ch  – 前景確率 (logit)
        embedding  Dch  – 画素埋め込みベクトル
        orientation 2ch – 接線方向 (cos θ, sin θ)
        endpoint   1ch  – 端点ヒートマップ (logit)
        junction   1ch  – 交差点ヒートマップ (logit)

入出力:
    入力  (B, 1, H, W)  – グレースケール文字画像
    出力  dict:
        fg          (B, 1, H, W)
        embedding   (B, D, H, W)
        orientation (B, 2, H, W)
        endpoint    (B, 1, H, W)
        junction    (B, 1, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ #
#  Building Blocks                                                    #
# ------------------------------------------------------------------ #

class _ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class ResBlock(nn.Module):
    """Basic residual block with optional stride-2 downsampling."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            _ConvBnRelu(in_ch, out_ch, s=stride),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        return self.relu(self.body(x) + self.skip(x))


class _Head(nn.Module):
    """Conv3×3 → ReLU → Conv1×1 の出力ヘッド。"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            _ConvBnRelu(in_ch, in_ch),
            nn.Conv2d(in_ch, out_ch, 1),
        )

    def forward(self, x):
        return self.block(x)


# ------------------------------------------------------------------ #
#  Decoder block                                                      #
# ------------------------------------------------------------------ #

class _DecBlock(nn.Module):
    """バイリニアアップサンプリング → スキップ結合 → 2×ResBlock。"""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.res = nn.Sequential(
            ResBlock(in_ch + skip_ch, out_ch),
            ResBlock(out_ch, out_ch),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.res(torch.cat([x, skip], dim=1))


# ------------------------------------------------------------------ #
#  Main Network                                                       #
# ------------------------------------------------------------------ #

class StrokeSegNet(nn.Module):
    """
    Args:
        emb_dim: Embedding head の出力チャネル数 (論文推奨 8〜16)
    """

    def __init__(self, emb_dim: int = 8):
        super().__init__()
        self.emb_dim = emb_dim

        # --- Encoder ---
        # stem: (B,1,H,W) → (B,64,H,W)  ストライドなし
        self.stem = nn.Sequential(
            _ConvBnRelu(1, 64, k=7, p=3),
            ResBlock(64, 64),
        )
        # e1 → H,W  64ch (スキップ用)
        self.e1 = nn.Sequential(ResBlock(64, 64), ResBlock(64, 64))
        # e2 → H/2  128ch (スキップ用)
        self.e2 = nn.Sequential(ResBlock(64, 128, stride=2), ResBlock(128, 128))
        # e3 → H/4  256ch (スキップ用)
        self.e3 = nn.Sequential(ResBlock(128, 256, stride=2), ResBlock(256, 256))
        # e4 → H/8  512ch
        self.e4 = nn.Sequential(ResBlock(256, 512, stride=2), ResBlock(512, 512))

        # --- Bridge ---
        self.bridge = nn.Sequential(ResBlock(512, 512), ResBlock(512, 512))

        # --- Decoder ---
        # d4: up(H/8→H/4) + s3(256) → 256
        self.d4 = _DecBlock(512, 256, 256)
        # d3: up(H/4→H/2) + s2(128) → 128
        self.d3 = _DecBlock(256, 128, 128)
        # d2: up(H/2→H)   + s1(64)  → 64
        self.d2 = _DecBlock(128, 64, 64)
        # d1: up(H→H)     + stem(64)→ 64  (stem と同解像度)
        self.d1 = _DecBlock(64, 64, 64)

        # --- Output Heads (共有特徴 64ch から各タスクへ) ---
        self.head_fg        = _Head(64, 1)
        self.head_emb       = _Head(64, emb_dim)
        self.head_ori       = _Head(64, 2)
        self.head_endpoint  = _Head(64, 1)
        self.head_junction  = _Head(64, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (B, 1, H, W)
        Returns:
            dict with keys: fg, embedding, orientation, endpoint, junction
        """
        # Encoder
        s0 = self.stem(x)    # (B, 64, H, W)
        s1 = self.e1(s0)     # (B, 64, H, W)
        s2 = self.e2(s1)     # (B, 128, H/2, W/2)
        s3 = self.e3(s2)     # (B, 256, H/4, W/4)
        s4 = self.e4(s3)     # (B, 512, H/8, W/8)

        # Bridge
        b = self.bridge(s4)  # (B, 512, H/8, W/8)

        # Decoder
        d4 = self.d4(b, s3)  # (B, 256, H/4, W/4)
        d3 = self.d3(d4, s2) # (B, 128, H/2, W/2)
        d2 = self.d2(d3, s1) # (B, 64, H, W)
        feat = self.d1(d2, s0) # (B, 64, H, W) – shared features

        return {
            "fg":          self.head_fg(feat),        # (B, 1, H, W)
            "embedding":   self.head_emb(feat),       # (B, D, H, W)
            "orientation": self.head_ori(feat),       # (B, 2, H, W)
            "endpoint":    self.head_endpoint(feat),  # (B, 1, H, W)
            "junction":    self.head_junction(feat),  # (B, 1, H, W)
        }
