# HRAvatar における DECA / SMIRK / レンダリング / 損失関数の整理

このメモは、`demos/demo_1_train_subject.sh` と
`demos/demo_2_cross_reenactment.sh` を起点に、HRAvatar が DECA と SMIRK を
どのように使い分けているか、HRAvatar 自体が何を学習し、どのように
レンダリングしているかを整理したものです。

## 結論

HRAvatar は、DECA と SMIRK を同じ目的の代替トラッカーとして単純に
切り替えているわけではない。

- DECA は、オフライン前処理で FLAME 系パラメータを推定し、
  `tracked_params.json` を作るために使われる。
- SMIRK は、HRAvatar の学習・推論中に画像 crop から expression / jaw /
  eyelid を推定するオンライン expression encoder として使われる。
- HRAvatar はピクセル空間で直接画像を生成するニューラル画像生成器では
  ない。学習可能な 3D Gaussian avatar を、微分可能な
  Gaussian rasterization と明示的な material / environment shading で描画する。
- 主損失は、レンダリング画像と実画像を比較するピクセル空間の
  L1 + SSIM である。補助的に jaw pose、normal、albedo、roughness、
  environment map などの正則化が入る。

## 用語

### DECA

DECA は単一画像から FLAME の shape、expression、pose などを推定する
3D face reconstruction 手法である。HRAvatar のこの実装では、
動画の各フレームに対して DECA を実行し、その後 `optimize.py` で
landmark / iris / temporal などに基づいて FLAME パラメータを refinement
する。

公式実装上の前処理経路:

```text
video
  -> crop + matting
  -> DECA demo_reconstruct.py
  -> face landmark detection
  -> iris segmentation
  -> DECA optimize.py
  -> tracked_params.json
```

該当箇所:

- `demos/_preprocess_subject.sh`
- `preprocess/submodules/DECA/demos/demo_reconstruct.py`
- `preprocess/submodules/DECA/optimize.py`

### SMIRK

SMIRK は expressive 3D face reconstruction のためのモデルである。
この HRAvatar 実装では、SMIRK 全体をオフライン extractor として
`tracked_params.json` を生成する用途には使っていない。

実装上は、`assets/smirk/pretrained_models/SMIRK_em1.pt` から
expression encoder 部分をロードし、学習・推論中に以下を推定する。

- `expression_params`
- `eyelid_params`
- `jaw_params`

該当箇所:

- `net_modules/flame_params_net_smirk.py`
- `scene/data_loader.py`
- `scene/gaussian_head_model.py`

## demo_1_train_subject.sh 実行時に何が起きるか

ユーザーが次のように実行した場合:

```bash
bash demos/demo_1_train_subject.sh <root> <name> <video> hdtf
```

`hdtf` は DECA / SMIRK の選択ではなく、カメラ内部パラメータ
`fx, fy, cx, cy` のプリセットを選ぶための引数である。

実行される段階は次の通り。

### 1. 前処理

`demo_1_train_subject.sh` は `demos/_preprocess_subject.sh` を呼ぶ。

`_preprocess_subject.sh` は DECA を使って初期 FLAME パラメータを推定する。
その後、landmark / iris を使って `preprocess/submodules/DECA/optimize.py`
で最適化し、`tracked_params.json` を生成する。

この段階の特徴抽出は DECA + optimize.py 由来である。

### 2. HRAvatar 学習

その後 `train.py` が呼ばれる。

`arguments/__init__.py` では、

```python
self.with_param_net_smirk = True
```

となっているため、デフォルトでは SMIRK 由来の expression encoder が
有効である。

このため学習中は、`tracked_params.json` から読み込んだ
`expcode` や jaw pose をそのまま使うのではなく、画像 crop から
SMIRK encoder が expression / jaw / eyelid を推定し、それで一部を
上書きする。

つまり `demo_1_train_subject.sh` の通常実行では:

```text
DECA:
  オフライン前処理で tracked_params.json を作る
  shape / pose / camera / translation の基準を提供

SMIRK:
  train.py 内で画像 crop から expression / jaw / eyelid を推定
  DECA 由来の expression / jaw を学習中に置き換える
```

一言でいうと、前処理の特徴抽出は DECA、学習中の表情特徴抽出は SMIRK。

## demo_2_cross_reenactment.sh 実行時に何が起きるか

`demo_2_cross_reenactment.sh` は、source subject の動画から
cross-reenactment 用の駆動信号を作り、既に学習済みの target HRAvatar
でレンダリングするデモである。

このデモも、source の前処理には `demos/_preprocess_subject.sh` を使う。
したがって、source 側の `tracked_params.json` は DECA + optimize.py
由来である。

ただし、target model が `with_param_net_smirk=True` で学習されている場合、
render 時にも SMIRK encoder が使われる。`render.py` の
cross-reenactment では source の `TrackedData` を読み、各フレームを
`gaussian_renderer.render_with_deferred` に渡す。

`GaussianHeadModel.forward()` では、`with_param_net_smirk` が有効で
`warped_image` が存在する場合、SMIRK encoder の出力で expression / jaw /
eyelid を上書きする。

したがって `demo_2_cross_reenactment.sh` の通常の動作は:

```text
source video
  -> DECA + optimize.py で tracked_params.json 作成
  -> render.py に source data と target model を渡す
  -> target model が SMIRK ON なら、render 時に source image crop から
     expression / jaw / eyelid を SMIRK で再推定
  -> target avatar の Gaussian geometry / material / envmap で描画
```

このため、cross-reenactment の source 表情は DECA の `expcode` だけで
駆動されるとは限らない。SMIRK ON の学習済み target model では、
render 時に SMIRK encoder の expression / jaw が優先される。

## SMIRK をオフにする方法

現状の `demo_1_train_subject.sh` だけでは SMIRK をオフに切り替えられない。

理由は、`arguments/__init__.py` の引数生成が bool に対して
`action="store_true"` を使っているためである。

`with_param_net_smirk` はデフォルトで `True` であり、CLI の
`--with_param_net_smirk` は True にするだけのフラグである。
したがって、現在の実装では次のような指定はできない。

```bash
python train.py ... --with_param_net_smirk False
```

最も簡単な方法は、`arguments/__init__.py` を直接変更すること。

```python
self.with_param_net_smirk = False
```

この状態で `demo_1_train_subject.sh` を実行すると、学習中に SMIRK encoder
は使われず、DECA + optimize.py が生成した `tracked_params.json` の
expression / jaw / pose がそのまま使われる。

より良い実装としては、Python 3.9 以降の `argparse.BooleanOptionalAction`
を使って、

```bash
python train.py ... --no-with_param_net_smirk
```

のように指定できるようにするのが望ましい。

注意点として、`render.py` は `model_path/cfg_args` を読み込む。
そのため、SMIRK ON で学習されたモデルは render / cross-reenactment 時も
基本的に SMIRK ON の設定を引き継ぐ。後から demo_2 側だけで簡単に
OFF にする設計ではない。

比較実験をするなら、次のように別モデルとして再学習するのが明確である。

```text
1. arguments/__init__.py の with_param_net_smirk を False にする
2. demo_1_train_subject.sh で別 model_path に再学習する
3. そのモデルで demo_2_cross_reenactment.sh を実行する
```

この場合、DECA + optimize.py の `tracked_params.json` だけで駆動する
HRAvatar になる。

## HRAvatar は何を学習しているか

HRAvatar は、ピクセル空間の画像を直接出力するニューラル画像生成器では
ない。

学習される主なもの:

- 3D Gaussian 点群
- 各 Gaussian の位置、スケール、回転、不透明度
- albedo
- roughness
- reflectance
- environment map
- learnable FLAME deformation dirs
- learnable LBS weights
- SMIRK 由来 expression encoder の一部

固定的な処理:

- FLAME / LBS による変形
- Gaussian rasterization
- normal / depth / material attribute の rasterize
- environment map + BRDF 風 shading

したがって、フォトリアルさは次の組み合わせで出る。

```text
学習済み 3D Gaussian avatar
  + 学習済み material / environment map
  + 学習済み deformation field
  + 明示的な differentiable rasterization / shading
```

完全に固定された非機械学習レンダラだけが後段でフォトリアル化している
わけではない。レンダリング式は明示的で固定的だが、その入力となる
Gaussian geometry、material、lighting、deformation が学習されている。

## HRAvatar のレンダリング処理

概念的な処理は次の通り。

```text
DECA / SMIRK FLAME params
  -> FLAME / LBS で Gaussian 点群を表情・姿勢に変形
  -> Gaussian rasterizer で画面上に投影
  -> 各 Gaussian の albedo / normal / roughness / reflectance を集約
  -> environment map + BRDF 風 shading で RGB を生成
```

つまり、HRAvatar は画像を直接ニューラルネットで生成するのではなく、
学習可能な 3D avatar 表現を明示的な微分可能レンダリングパイプラインで
描画している。

## アルベドについて

HRAvatar における albedo は、最終的には HRAvatar が学習する avatar 内部の
材質パラメータである。

学習時には Intrinsic Anything による pseudo-albedo を補助教師として使う
ことがある。ただし、最終的な albedo は固定の外部出力ではなく、
Gaussian ごとに最適化された HRAvatar の内部パラメータである。

`WITH_ALBEDO=1` を指定すると、前処理時に Intrinsic Anything で
pseudo-albedo が生成され、学習時の補助 supervision として使われる。

## 損失関数はピクセル空間か、パラメータ空間か

主損失はピクセル空間で計算される。

`train.py` では、現在の Gaussian / material / deformation / envmap から
レンダリング画像を作り、それを実画像と比較する。

概念的には次の式である。

```python
render_pkg = render(viewpoint_cam_param, gaussians, all_args, bg, iteration=iteration)
image = render_pkg["render"]
gt_image = viewpoint_cam_param.original_image.cuda(device)

Ll1 = l1_loss(image, gt_image)
image_loss = (1.0 - lambda_dssim) * Ll1 \
           + lambda_dssim * (1.0 - ssim(image, gt_image))
loss += image_loss
```

学習の主経路:

```text
learnable parameters
  -> FLAME / LBS deformation
  -> Gaussian rasterization
  -> material / envmap shading
  -> rendered RGB image
  -> pixel-space loss vs GT image
  -> backpropagation
```

「パラメータから画像への処理が微分可能」という説明は、まさにここに
つながる。レンダリング結果の画像誤差から、Gaussian geometry、material、
lighting、deformation、場合によっては SMIRK encoder まで勾配が戻る。

ただし、損失はピクセル空間だけではない。補助損失として以下も使われる。

- RGB image loss: L1 + SSIM。主損失。
- jaw pose loss: SMIRK jaw と tracked jaw の差分。パラメータ空間。
- intrinsic albedo loss: rendered albedo と pseudo-albedo の L1。
- normal loss: rendered depth 由来 normal と rendered normal の整合。
- environment map consistency loss。
- roughness の TV loss。

正確には、主目的はピクセル空間の再構成損失であり、補助的に
パラメータ空間、材質空間、法線 / 深度空間の正則化を足している。

## SSIM とは何か

SSIM は `Structural Similarity Index Measure` の略である。

2枚の画像を単純なピクセル差だけでなく、局所領域ごとの以下の要素で
比較する。

- luminance: 輝度
- contrast: コントラスト
- structure: 構造

L1 loss は、各ピクセルの RGB 値がどれだけ違うかを見る。

SSIM は、局所パッチとして見たときに、明暗、濃淡、エッジ構造が似ているか
を見る。

HRAvatar では次の形で使われる。

```text
image_loss = (1 - lambda_dssim) * L1
           + lambda_dssim * (1 - SSIM)
```

`lambda_dssim` はデフォルトで `0.2` であるため、主成分は L1、
補助的に SSIM で局所構造の一致を狙っている。

## 2025 / 2026 年の 3DGS / neural rendering における損失関数の動向

分野名としては、文脈に応じて以下が近い。

- neural rendering
- differentiable rendering
- novel view synthesis
- 3D Gaussian Splatting
- Gaussian avatar / head avatar reconstruction

2025 / 2026 年の 3DGS 系では、L1 + SSIM だけでは人間の知覚上は
ぼやけやすいという問題意識が強い。

### WD-R / Wasserstein Distortion Regularized

2026 年の Apple の “Drop-In Perceptual Optimization for 3D Gaussian
Splatting” では、従来の 3DGS が L1 / SSIM のような pixel-level loss に
頼るため、perceptual quality が十分でないと指摘している。

この研究では Wasserstein Distortion 系の知覚損失 `WD-R` を提案している。
人間評価で元の 3DGS loss より 2.3 倍、Perceptual-GS より 1.5 倍好まれた
と報告されている。

参考:

- https://machinelearning.apple.com/research/drop-in
- https://apple.github.io/ml-perceptual-3dgs/

### Perceptual-GS

ICML 2025 の Perceptual-GS は、単に loss を差し替えるだけではなく、
人間の知覚感度を考慮して Gaussian の密度配分を変える方向の手法である。

視覚的に重要な領域に細かい Gaussian を割り当てることで、品質と効率を
上げる。

参考:

- https://proceedings.mlr.press/v267/zhou25s.html

### セマンティック / 知覚ガイド付き学習

2025 年の 3DGS 系では、低品質入力やブレた画像に対し、semantic feature や
unsupervised quality assessment を使って劣化画像に引きずられにくくする
方向も提案されている。

参考:

- https://link.springer.com/article/10.1007/s00371-024-03754-z

## HRAvatar に新しい損失を入れるなら

HRAvatar に直接入れるなら、現実的には主損失を完全に置き換えるより、
小さい重みで perceptual loss を足すのが安全である。

例:

```text
L = L1 + 0.2 * DSSIM + small_weight * perceptual_loss
```

候補:

- LPIPS
- DISTS
- MS-SSIM
- WD-R 系の perceptual loss
- 顔領域 mask weighting
- 口周り / 目周りの局所重み付け
- temporal consistency loss
- identity loss

ただし head avatar では、知覚損失を強く入れると以下の副作用が出やすい。

- PSNR / SSIM は下がるが見た目は良くなる
- identity が少し崩れる
- 口、歯、目の局所アーティファクトが増える
- 時系列ちらつきが増える

したがって、顔アバターでは RGB 知覚損失だけでなく、face mask、mouth /
eye region weighting、temporal consistency、identity consistency を併用する
方が安定しやすい。

## 主要な実装参照

### 前処理

- `demos/demo_1_train_subject.sh`
- `demos/demo_2_cross_reenactment.sh`
- `demos/_preprocess_subject.sh`
- `preprocess/submodules/DECA/demos/demo_reconstruct.py`
- `preprocess/submodules/DECA/optimize.py`

### データ読み込み

- `scene/data_loader.py`

`TrackedData` が `tracked_params.json` を読み、画像、mask、albedo、
FLAME パラメータ、camera、SMIRK 用 crop を準備する。

### SMIRK encoder

- `net_modules/flame_params_net_smirk.py`

`SMIRK_em1.pt` から expression encoder をロードする。

### Gaussian head model

- `scene/gaussian_head_model.py`

FLAME / LBS による Gaussian 変形、material parameter、SMIRK encoder の
適用、学習対象パラメータの登録を行う。

### レンダリング

- `gaussian_renderer/__init__.py`
- `render.py`

Gaussian rasterization、material attribute の集約、environment map shading、
cross-reenactment、multi-view、relighting などを行う。

### 学習

- `train.py`

L1 + SSIM の image loss、jaw pose regularization、albedo / normal /
roughness / envmap 関連の補助損失を計算する。

### 引数定義

- `arguments/__init__.py`

`with_param_net_smirk=True` のデフォルトや、各種 learning rate、
material / rendering / deformation 関連の設定が定義されている。

## 参考文献・外部資料

- HRAvatar paper: https://arxiv.org/pdf/2503.08224
- HRAvatar project / repository: https://github.com/Pixel-Talk/HRAvatar
- DECA: https://github.com/yfeng95/DECA
- SMIRK project: https://georgeretsi.github.io/smirk/
- SMIRK repository: https://github.com/georgeretsi/smirk
- Apple WD-R / Drop-In Perceptual Optimization for 3DGS:
  https://machinelearning.apple.com/research/drop-in
- Apple perceptual 3DGS project page:
  https://apple.github.io/ml-perceptual-3dgs/
- Perceptual-GS, ICML 2025:
  https://proceedings.mlr.press/v267/zhou25s.html
