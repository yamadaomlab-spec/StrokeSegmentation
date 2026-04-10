# ストローク分離ネットワーク設計案

## 1. 目的

1文字画像から各ストロークを分離するためのネットワークを設計する。  
本段階では各ストローク間の筆順は扱わず、まずは **ストローク分離** のみを目的とする。  
ただし、将来的に各ストロークの trajectory を復元することを見据え、後段処理につながりやすい表現も同時に学習する。

前提は次のとおりである。

- 入力: 1文字の2値画像またはグレースケール画像
- 出力: 各ストロークの分離マスク
- 交差・接触・一部の隠れを含む
- 参照ストロークやテンプレートは用いない

---

## 2. 基本方針

採用するのは、**画素埋め込み + 方向場 + 端点/交差点ヒートマップ** に基づく構成である。  
ネットワークは入力画像から次の4種類の情報を同時に出力する。

1. foreground map
2. embedding map
3. orientation field
4. endpoint / junction heatmap

すなわち、次の写像を学習する。

\[
I \rightarrow \{P_{\text{fg}}, E, O, H_{\text{end}}, H_{\text{junc}}\}
\]

---

## 3. 全体アーキテクチャ

基本骨格には **U-Net 系** を用いる。  
U-Net は encoder-decoder 構造と skip connection を持ち、細い線構造の位置精度を保ちやすいため、本課題に適している。

### 構成

```text
Input image
   ↓
Encoder
   ↓
Bottleneck
   ↓
Decoder + skip connections
   ├─ Head A: foreground segmentation
   ├─ Head B: pixel embedding
   ├─ Head C: orientation field
   ├─ Head D: endpoint heatmap
   └─ Head E: junction heatmap
```

### 推奨設定例

- Backbone: U-Net / ResUNet / HRNet
- 入力サイズ: 256×256 または 384×384
- 出力解像度: 入力と同じ
- Embedding 次元: 8〜16
- Orientation: 2ch（cosθ, sinθ）
- Endpoint / Junction: 各1ch

---

## 4. 各ヘッドの定義

### 4.1 Foreground head

出力:

\[
P_{\text{fg}}(x,y) \in [0,1]
\]

役割:

- 文字画素と背景の分離
- embedding loss を前景画素のみに適用するためのマスク

損失:

- BCE loss
- Dice loss（併用推奨）

---

### 4.2 Embedding head

出力:

\[
E(x,y) \in \mathbb{R}^{D}
\]

役割:

- 同じストロークに属する画素は embedding 空間で近くする
- 異なるストロークに属する画素は離す

学習には discriminative loss を用いる。

\[
L_{\text{emb}} = L_{\text{var}} + L_{\text{dist}} + \lambda_{\text{reg}} L_{\text{reg}}
\]

- **Variance loss**: 同一ストローク内の埋め込みを凝集
- **Distance loss**: 異なるストローク間の重心を分離
- **Regularization loss**: 重心の発散を抑制

注意点:

- 背景画素は loss 計算から除外する
- 細いストロークには重み補正を加えるとよい
- 交差近傍では高い重みを与えると有効

---

### 4.3 Orientation head

出力:

\[
O(x,y) = (\cos\theta(x,y), \sin\theta(x,y))
\]

役割:

- 局所接線方向を表す
- 交差部で「どちらの方向につながるべきか」を補助する
- 後段のグラフ構築や trajectory 復元に有利

損失の例:

\[
L_{\text{ori}} =
\frac{1}{|\Omega_{\text{fg}}|}
\sum_{(x,y) \in \Omega_{\text{fg}}}
\|O(x,y)-O^*(x,y)\|_1
\]

または cosine similarity loss を用いてもよい。

---

### 4.4 Endpoint / Junction heads

出力:

- \(H_{\text{end}}(x,y)\)
- \(H_{\text{junc}}(x,y)\)

役割:

- endpoint: ストローク端点の検出
- junction: 交差点・接触点の検出

教師信号は点そのものではなく、Gaussian heatmap として与える。

損失候補:

- MSE
- BCE with logits
- Focal loss

少数点検出であるため、focal loss が有効なことが多い。

---

## 5. 推奨ネットワーク詳細

### Encoder

- ResNet34 または軽量 CNN
- 各 stage の feature map を保持

### Decoder

- U-Net 型アップサンプリング
- skip connection により位置精度を保持

### Shared feature

最終 decoder 出力を共有特徴 \(F\) とする。

### Heads

- Foreground: Conv 3×3 → ReLU → Conv 1×1 → 1ch
- Embedding: Conv 3×3 → ReLU → Conv 1×1 → Dch
- Orientation: Conv 3×3 → ReLU → Conv 1×1 → 2ch
- Endpoint: Conv 3×3 → ReLU → Conv 1×1 → 1ch
- Junction: Conv 3×3 → ReLU → Conv 1×1 → 1ch

---

## 6. 総損失関数

全体損失は次のように定義する。

\[
L =
\lambda_{\text{fg}} L_{\text{fg}}
+ \lambda_{\text{emb}} L_{\text{emb}}
+ \lambda_{\text{ori}} L_{\text{ori}}
+ \lambda_{\text{end}} L_{\text{end}}
+ \lambda_{\text{junc}} L_{\text{junc}}
\]

初期値の例:

- \(\lambda_{\text{fg}} = 1.0\)
- \(\lambda_{\text{emb}} = 1.0\)
- \(\lambda_{\text{ori}} = 0.5\)
- \(\lambda_{\text{end}} = 0.5\)
- \(\lambda_{\text{junc}} = 0.5\)

交差部の扱いを重視する場合は、orientation と junction の重みを少し上げる。

---

## 7. 教師データの形式

### 必須

1. 元画像
2. 各ストロークのインスタンスマスク  
   - `stroke_id_map`
   - または `K枚のbinary mask`

### 推奨

3. 各ストロークの中心線
4. 各中心線上の接線方向
5. 端点座標
6. 交差点座標

---

## 8. 教師信号の生成

### 8.1 Foreground

各ストロークマスクの和集合を foreground とする。

### 8.2 Embedding label

各前景画素に対して `stroke_id` を付与する。

### 8.3 Orientation

各ストローク中心線の点列 \((x_t, y_t)\) に対し、接線方向を

\[
v_t = (x_{t+1} - x_{t-1}, y_{t+1} - y_{t-1})
\]

として近似し、正規化して教師信号とする。

### 8.4 Endpoint

各ストロークの始点・終点に Gaussian を置く。

### 8.5 Junction

ストローク同士が交差または接触する点に Gaussian を置く。

---

## 9. 交差部に強くする工夫

### 9.1 交差近傍重み付け

交差点から半径 \(r\) 以内の画素に高い重みを与える。

\[
w(x,y)=
\begin{cases}
w_{\text{junc}} & \text{junction近傍} \\
1 & \text{otherwise}
\end{cases}
\]

### 9.2 Skeleton 補助損失

foreground とは別に skeleton map を出力し、細線構造そのものに supervision を与える。

### 9.3 Affinity 学習

近傍画素ペアについて「同じストロークかどうか」を予測する auxiliary head を追加する。

例:

- 右隣との affinity
- 下隣との affinity
- 斜め隣との affinity

---

## 10. 推論時の処理

### Step 1

foreground を閾値処理して前景マスクを得る。

### Step 2

前景画素の embedding を抽出する。

### Step 3

embedding をクラスタリングしてストローク候補を得る。  
候補手法:

- DBSCAN
- MeanShift
- HDBSCAN

初期実装では DBSCAN が扱いやすい。

### Step 4

各クラスタを細線化し skeleton を得る。

### Step 5

endpoint / junction heatmap で特徴点を補正する。

### Step 6

orientation field を用いて、交差部で不自然な接続を修正する。

---

## 11. スコア統合の考え方

交差部では embedding のみでは不安定なことがある。  
したがって、最終的な接続判定には次の情報を統合するのが望ましい。

- embedding 類似度
- 空間距離
- orientation 整合性
- endpoint / junction 整合性

例えば画素 \(p,q\) の接続スコアを次で表せる。

\[
S(p,q)=
\alpha \cdot \mathrm{sim}(E_p,E_q)
+ \beta \cdot \mathrm{ori\_compat}(O_p,O_q)
- \gamma \cdot \mathrm{dist}(p,q)
\]

---

## 12. 評価指標

IoU だけでは不十分であるため、次を併用する。

### マスク系

- mIoU
- Dice
- instance AP

### ストローク分離系

- stroke-level precision
- stroke-level recall
- stroke-level F1
- over-segmentation rate
- under-segmentation rate

### 交差部特化

- junction 近傍での stroke assignment accuracy

---

## 13. 学習データが少ない場合の工夫

オンライン筆跡データから合成データを作る方法が有効である。

### 合成例

- ストローク列をラスタ化
- 各ストロークを別マスクとして保持
- 線幅を変える
- ぼかし
- かすれ
- 濃淡変動
- 回転
- 交差・接触を多く含む文字を増やす

この方法により、foreground / stroke instance / orientation / endpoint / trajectory を自動で得られる。

---

## 14. 初期実装の推奨仕様

### 入力

- 256×256 grayscale image

### Backbone

- ResUNet

### Heads

- foreground: 1ch
- embedding: 8ch
- orientation: 2ch
- endpoint: 1ch
- junction: 1ch

### Loss

- foreground: BCE + Dice
- embedding: discriminative loss
- orientation: L1
- endpoint / junction: focal loss

### Postprocess

- foreground threshold
- DBSCAN on embeddings
- skeletonization
- endpoint / junction refinement

---

## 15. 研究上の主張

この構成の利点は、単なるストローク分離に留まらず、

- stroke instance separation
- crossing-aware separation
- trajectory-ready representation

を同時に主張できる点にある。

したがって、研究としては次のように位置づけられる。

> 参照ストロークなしで、交差を含む文字画像からストロークを分離し、さらに後続の trajectory 復元に有利な中間表現を同時に学習する手法

---

## 16. 実装順序の提案

### 第1段階

foreground + embedding のみを実装し、基本的な分離性能を確認する。

### 第2段階

orientation を追加し、交差部での誤分離が減少するかを評価する。

### 第3段階

endpoint / junction を追加し、構造情報の有効性を確認する。

### 第4段階

後段の trajectory 推定へ接続する。

---

## 17. まとめ

本設計案は、参照データを導入せずにストローク分離を行うための、実装可能性と将来拡張性の両方を考慮した構成である。  
特に、画素埋め込みと方向場を組み合わせることで、交差や接触を含む難しい文字画像に対しても、単純なマスク分割より強い分離能力が期待できる。  
さらに、将来的な trajectory 推定への橋渡しとしても自然な中間表現を提供する。
