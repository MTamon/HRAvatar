# HRAvatar デモ (Python 3.11 + PyTorch 2.9 + CUDA 12.8)

言語: [English](README.md) | 日本語

このディレクトリには、関連する MTamon ブランチ
(`MTamon/DECA@release/cuda128`, `MTamon/smirk@release/cuda128`) のデモ構成に
合わせた 3 つのデモがあります。

| # | スクリプト | 目的 |
|---|---|---|
| 1 | `demo_1_train_subject.sh` | 1 人の被写体動画から HRAvatar をエンドツーエンドで学習します。 |
| 2 | `demo_2_cross_reenactment.sh` | 学習済み HRAvatar を、**別人**の動画から抽出した FLAME で駆動します。 |
| 3 | `demo_3_overlay_tracking.py` | HRAvatar のレンダラが実際に受け取る **optimize 後** の特徴量 (FLAME 頂点 + ランドマーク + パラメータカード) を、元フレームに重ねて可視化します。 |

3 つのデモはいずれも、`bash setup.sh` と `bash download_assets.sh` が
正常に完了しており、`conda activate HRAvatar` が有効な状態を前提にしています。

---

## データ配置

動画は 1 つのディレクトリツリー配下に置きます。前処理後の構成は次のようになります。

```text
/data/subjects/
└── alice/
    ├── video.mp4                <- 入力動画 (必須)
    ├── image/                   <- フレーム (crop_and_matting が作成)
    │   ├── 000000.png
    │   └── ...
    ├── matted/                  <- アルファマット済み画像
    ├── mask/                    <- バイナリマスク
    ├── deca/                    <- DECA によるフレームごとの FLAME 予測
    ├── keypoints/               <- 2D 顔ランドマーク
    ├── iris/                    <- 虹彩セグメンテーション
    ├── albedo/                  <- IntrinsicAnything pseudo-GT (任意)
    └── tracked_params.json      <- 最適化済み FLAME パラメータ (optimize.py 由来)
```

クロス reenactment デモでは、表情をターゲットアバターへ転送する *source* 被写体にも
同じ配置を使います。

---

## 1. 自分の被写体で HRAvatar を学習する

`demos/demo_1_train_subject.sh`

```bash
bash demos/demo_1_train_subject.sh \
    --sbj-root /data/subjects \
    --sbj-name alice \
    --video /data/raw/alice.mp4 \
    --intrinsics hdtf
```

既存の自動化向けに、従来の位置引数形式も引き続き使用できます。

このスクリプトは次の処理を実行します。

1. `preprocess/crop_and_matting.py` - フレーム抽出、matting、512² crop。
2. `preprocess/submodules/DECA/demos/demo_reconstruct.py` - DECA による初期
   FLAME パラメータ (shape/expression/pose)。
3. `preprocess/keypoint_detector.py` - face-alignment による 68 点ランドマーク。
4. `preprocess/iris.py` - FDLite による虹彩セグメンテーション。
5. `preprocess/submodules/DECA/optimize.py` - photometric FLAME fit。
6. *(任意、`--with-albedo` で有効)*
   `preprocess/submodules/IntrinsicAnything/inference.py` - albedo pseudo-GT。
7. `train.py` - HRAvatar の学習 (デフォルト 15 epochs、`--epochs` で上書き可能)。
8. `render.py` - self-reenactment のレンダリングと metrics。

出力は `outputs/custom/<subject_name>/` に保存されます。

現状のベストプラクティスは以下のコマンド
```bash
bash demos/demo_1_train_subject.sh \
    --sbj-root /data/subjects \
    --sbj-name MK6cOpt \
    --video /data/raw/alice.mp4 \
    --intrinsics hdtf \
    --jitter-filter --jitter-filter-smirk \
    --resize 720 --image-size 512 \
    --lambda-image 1.2
```

## 2. オフライン cross-reenactment

`demos/demo_2_cross_reenactment.sh`

学習済み HRAvatar を、*別人* の動画から抽出した FLAME パラメータで駆動します。
必要に応じて、このスクリプトは source 動画に対して同じ前処理パイプラインを実行し
(`tracked_params.json` とランドマークを取得)、その後
`render.py --corss_source_paths ...` を呼び出して、ターゲット被写体の Gaussian head を
source 被写体の表情軌跡でレンダリングします。

**特徴量抽出バックエンド。** HRAvatar には DECA (`preprocess/submodules/DECA`) と
SMIRK (`net_modules/flame_params_net_smirk.py`) の両方が含まれています。
プロジェクトの慣例に従って使い分けます。

* **DECA** - フレームごとの標準 FLAME 抽出器です (学習パイプラインで使用)。
  `demo_2_cross_reenactment.sh` のデフォルトです。
* **SMIRK** - 学習済み HRAvatar **内部** のオンライン表情エンコーダとしてのみ使用します
  (`arguments/__init__.py` の `--with_param_net_smirk`)。
  リファレンスコードでは、オフライン特徴量抽出には使いません。

```bash
bash demos/demo_2_cross_reenactment.sh \
    --sbj-root /data/subjects \
    --sbj-name alice \
    --video /data/raw/alice.mp4 \
    --intrinsics hdtf \
    --target-model-dir outputs/custom/bob
```

既存の自動化向けに、従来の位置引数形式も引き続き使用できます。

結果は次の場所に書き込まれます。

```text
outputs/custom/bob/test_cross_reenactment/ours_<N>/alice_reenactment_bob/
├── 00000.png  ...  NNNNN.png
└── alice_reenactment_bob_video.mp4
```

## 3. HRAvatar の実際のレンダラ入力を可視化する

`demos/demo_3_overlay_tracking.py`

これは **オフライン特徴量インスペクタ** です。下流の Listening-Head-Generation モデルが
HRAvatar のレンダラが期待する正確な特徴量形状を生成しているか確認するために使います。

重要: HRAvatar が取り込む特徴量は生の DECA 出力**ではありません**。
`preprocess/submodules/DECA/optimize.py` による photometric + landmark + temporal
refinement の結果であり、`tracked_params.json` に保存されます。このデモは次を行います。

1. 前処理パイプラインが生成した `tracked_params.json` を読み込みます。
2. HRAvatar が内部で使うものと同じ LBS + offset chain
   (`scene.gaussian_head_model.GaussianHeadModel.lbs_v2`、`num_joints = J_regressor.shape[1] + 1`
   の 6-joint augmentation と `scene/__init__.py` の `lbs_weights` zero-column pad を含む)
   を再実装します。
3. データローダが構築するものと完全に同じカメラで頂点を投影します
   (`w2c = diag(1,-1,-1,1) @ world_mat`、pinhole
   `fo = image_w / (2·tan(½·fovx))`、`fovx = 2·arctan2(cx, fx)`)。
4. 結果を元フレームに重ね、mp4 として書き出します。

HRAvatar のチェックポイントは読み込みません。`download_assets.sh` でインストールされた
FLAME assets だけを使用します。この可視化で overlay が入力に追従していれば、
HRAvatar でも同じ動きが再現されます。

```bash
# 1) 一度だけ前処理する (demo 1 と同じ helper を再利用)
bash demos/_preprocess_subject.sh \
    --sbj-root /data/subjects \
    --sbj-name alice \
    --video /data/raw/alice.mp4 \
    --intrinsics hdtf

# 2) レンダラへ渡される特徴量を確認する
python demos/demo_3_overlay_tracking.py \
    --subject_dir /data/subjects/alice \
    --output      /tmp/alice_features.mp4 \
    --mode        all
```

Modes: `vertices` (投影された全 FLAME 頂点を cyan dots で表示)、
`wireframe` (triangle edges)、`landmarks` (optimizer が fit した 68 点 FAN points。
`keypoint.json` から読み込み)、`params_card` (`shape[:6]`, `exp[:6]`,
global/neck/jaw pose in degrees, eyelids, translation の数値サマリ)、
`all` (vertices + landmarks + card)。

overlay style は
`MTamon/smirk@release/cuda128/demos/demo_video.py --show_vertices` に合わせています。
