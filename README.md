# StrokeSegmentation

手書き文字画像からストロークごとの分離マスクを求めるネットワーク。

## 概要

- **入力**: 1文字の2値/グレースケール画像
- **出力**: ストロークごとの分離マスク（交差・接触を含む）
- **訓練データ**: PEN-Net の `data/data_lmdb/pot` (CASIA Pot1.1 由来, 約91万件)
- **参照設計書**: [stroke_separation_network_design_v2.md](stroke_separation_network_design_v2.md)

PEN-Net のストローク座標データを用いてレンダリングした文字画像を入力とし、
座標情報から自動生成した教師信号で訓練する。
テンプレートや参照ストロークは使用しない。

---

## プロジェクト構成

```
StrokeSegmentation/
├── configs/
│   └── config.yaml              # 全ハイパーパラメータ
├── datasets/
│   ├── lmdb_stroke_dataset.py   # PEN-Net data_lmdb を読む PyTorch Dataset
│   └── target_generator.py      # ストローク座標 → 訓練ターゲット生成
├── models/
│   └── stroke_segnet.py         # ResUNet + 5出力ヘッド
├── losses/
│   └── stroke_losses.py         # 各損失関数
├── utils/
│   └── postprocess.py           # 推論後処理・可視化
├── train.py                     # 訓練スクリプト
├── evaluate.py                  # 評価スクリプト
└── inspect_data.py              # データ確認用スクリプト
```

---

## モデルアーキテクチャ (StrokeSegNet)

設計書 Section 3〜5 に基づく ResUNet + マルチヘッド構成。パラメータ数約2,600万。

```
入力 (B, 1, H, W)
   ↓
Encoder (Stem + 4段 ResBlock, stride-2 ダウンサンプリング)
   64ch → 64ch → 128ch → 256ch → 512ch
   ↓
Bridge (2× ResBlock)
   ↓
Decoder (U-Net 型スキップ接続 + バイリニアアップサンプリング)
   512ch → 256ch → 128ch → 64ch → 64ch
   ↓
共有特徴 (B, 64, H, W)
   ├─ Head A: foreground     → (B, 1, H, W)
   ├─ Head B: embedding      → (B, 8, H, W)
   ├─ Head C: orientation    → (B, 2, H, W)  (cos θ, sin θ)
   ├─ Head D: endpoint       → (B, 1, H, W)
   └─ Head E: junction       → (B, 1, H, W)
```

---

## 訓練ターゲットの自動生成

`datasets/target_generator.py` が LMDB のストローク座標から以下を自動生成する。

| ターゲット | 形状 | 内容 |
|---|---|---|
| `image` | (1, H, W) | レンダリングした入力文字画像 (1=筆画, 0=背景) |
| `fg_mask` | (H, W) | 前景マスク (0/1) |
| `stroke_id_map` | (H, W) | 画素ごとのストロークID (0=背景, 1〜N) |
| `orientation` | (2, H, W) | 接線方向場 (cos θ, sin θ) |
| `endpoint_heatmap` | (H, W) | ストローク端点ガウスヒートマップ |
| `junction_heatmap` | (H, W) | 交差点ガウスヒートマップ |

---

## 損失関数

設計書 Section 6 の全体損失:

```
L = λ_fg * L_fg  +  λ_emb * L_emb  +  λ_ori * L_ori
      +  λ_end * L_end  +  λ_junc * L_junc
```

| 損失 | 手法 | 対象 |
|---|---|---|
| `L_fg` | BCE + Dice | 前景セグメンテーション |
| `L_emb` | Discriminative loss (variance + distance + reg) | 埋め込み空間でのストローク分離 |
| `L_ori` | Cosine similarity (絶対値で方向曖昧性を吸収) | 接線方向場 |
| `L_end` | Focal loss | 端点ヒートマップ |
| `L_junc` | Focal loss | 交差点ヒートマップ |

初期の重み: λ_fg=1.0, λ_emb=1.0, λ_ori=0.5, λ_end=0.5, λ_junc=0.5

---

## 推論・後処理

設計書 Section 10 の手順:

1. foreground を閾値処理して前景マスクを得る
2. 前景画素の embedding を抽出
3. DBSCAN でクラスタリングしてストローク候補を得る
4. 各クラスタをストロークマスクとして出力
5. endpoint / junction heatmap で特徴点を確認・補正

---

## セットアップ

```bash
pip install -r requirements.txt
```

主要依存ライブラリ: torch, torchvision, numpy, opencv-python, Pillow, scipy, scikit-learn, lmdb, pyyaml, tqdm, tensorboard

---

## 実行方法

### 1. データ確認

ターゲット生成が正しく動作しているかを可視化で確認する。

```bash
python inspect_data.py --n 8 --out_dir ./inspect_out
```

各サンプルについて入力画像・前景マスク・ストロークID・方向場・端点・交差点の6パネルが PNG として保存される。

### 2. 訓練

```bash
python train.py --config configs/config.yaml
```

- チェックポイントと TensorBoard ログは `outputs/` に保存される
- 途中再開:

```bash
python train.py --resume outputs/ckpt_ep020.pth
```

### 3. 評価

```bash
python evaluate.py --checkpoint outputs/ckpt_best.pth
```

可視化画像も保存する場合:

```bash
python evaluate.py --checkpoint outputs/ckpt_best.pth --visualize --vis_dir ./vis_output
```

---

## 評価指標

設計書 Section 12 に対応。

| 指標 | 内容 |
|---|---|
| `fg_iou` | 前景セグメンテーション IoU |
| `fg_dice` | 前景 Dice スコア |
| `stroke_precision` | ストロークレベル適合率 (IoU≥0.5 でマッチ) |
| `stroke_recall` | ストロークレベル再現率 |
| `stroke_f1` | ストロークレベル F1 |
| `over_seg_rate` | 過分割率 (1 GT に複数 pred がマッチ) |
| `under_seg_rate` | 過統合率 (1 pred に複数 GT がマッチ) |

---

## 設定ファイル (configs/config.yaml)

主要パラメータの意味:

| キー | デフォルト | 説明 |
|---|---|---|
| `DATA.img_size` | 256 | 入力・出力画像サイズ |
| `DATA.margin` | 12 | 正規化時の余白 px |
| `DATA.thickness_min/max` | 1 / 3 | 訓練時の線幅ランダム範囲 |
| `MODEL.emb_dim` | 8 | Embedding ベクトル次元数 |
| `TRAIN.batch_size` | 16 | バッチサイズ |
| `TRAIN.max_epochs` | 100 | 最大エポック数 |
| `TRAIN.lr` | 4e-4 | 初期学習率 (AdamW) |
| `INFER.dbscan_eps` | 0.3 | DBSCAN の近傍半径 (embedding 空間) |

---

## 実装フェーズ (設計書 Section 16)

| フェーズ | 内容 | 対応する損失 |
|---|---|---|
| 第1段階 | 基本的な分離性能の確認 | fg + emb |
| 第2段階 | 交差部での誤分離改善 | + ori |
| 第3段階 | 構造情報の有効性確認 | + end + junc |
| 第4段階 | 後段の trajectory 推定へ接続 | — |

第1段階のみ実施する場合は `configs/config.yaml` で `lambda_ori`, `lambda_end`, `lambda_junc` を 0 に設定する。
