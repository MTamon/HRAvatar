# LHG (Listening Head Generation) preprocessing パイプライン

言語: 日本語 | (English version: not yet written — open an issue if needed)

このディレクトリは Listening Head Generation 用の preprocessing パイプラインです。
HRAvatar のアバター個人 fit (one-time) とは独立した、**Stage 1 / Stage 2 / Stage 3** の 3 段構成で動きます。各 Stage の責務と入出力は厳密に分離されています — 詳細は
[`doc/preprocessing_scope.md`](../doc/preprocessing_scope.md) と
`feedback_lhg_stage_separation` メモリを参照してください。

| Stage | スクリプト | 役割 | 頻度 |
|---|---|---|---|
| 1 | `demos/_preprocess_subject.sh --lhg-only` | DECA optimize による clip-wide joint 最適化で `world_mat` / `shapecode` / `intrinsics` を確定 | 1 被写体につき 1 回 |
| 2 | `demos/extract_lhg_features.sh --mode online` | per-frame 特徴量抽出 (MediaPipe → SMIRK → EPnP → causal Hampel → symmetric FIR LPF) | 各クリップごと |
| 3 | `demos/extract_lhg_features.sh --mode pseudo-online` | Stage 2 と同じ per-frame core + 双方向 Hampel + interpolation + 拡張 lookahead LPF | 教師データ作成時 (将来実装) |
| - | `demos/render_lhg_features.sh` | 学習済みアバター × `lhg_features.npz` で MP4 レンダリング (検証用) | 任意 |
| - | `python scripts/plot_lhg_jitter.py` | static / velocity / acceleration の 3×3 グラフ生成 (検証用) | 任意 |

> **frame rate 規約**
> feature stream = **25 FPS** (online/offline 共通)、LHG model inference = **10 FPS**
> (motion update rate)。本ドキュメントの "per-frame" / `--lookahead L` の絶対時間換算は
> 全て 25 FPS 基準 (`L * 40ms`)。詳細は `project_lhg_fps_convention` メモリ。

---

## 前提条件

- `bash setup.sh` + `bash download_assets.sh` 実行済み、`conda activate HRAvatar` 有効
- `assets/lhg/mediapipe_flame_landmarks.npz` (MediaPipe → FLAME 対応) があること
  - 無い場合は `python tools/build_mediapipe_flame_correspondence.py` で生成
  - FAN を使う場合は `assets/lhg/dlib_flame_landmarks.npz` も必要

---

## Stage 1: Offline calibration (1 被写体 1 回)

### 目的

DECA optimize.py の clip-wide joint Adam 最適化で被写体の **clip 定数** を確定:
- `world_mat` (4×4) — カメラ extrinsics
- `shapecode` (100,) — FLAME shape
- `intrinsics` (4,) — `[fx, fy, cx, cy]` (outer-crop 座標系)

per-frame entries (`expcode`, `fullposecode`, `translation`) も `tracked_params.json`
に書かれますが、Stage 2 はこれらを **読みません** — per-frame 値は Stage 2 の
オンライン推定が出します。

### コマンド

```bash
bash demos/_preprocess_subject.sh \
    --sbj-root /data/lhg \
    --sbj-name alice \
    --video /data/raw/alice.mp4 \
    --intrinsics hdtf \
    --lhg-only
```

`--lhg-only` flag が肝心:
- IntrinsicAnything (擬似 albedo) を skip
- RobustVideoMatting (matting) を skip
- face-parsing による衣服マスクを skip
- DECA optimize.py 自体は通常通り実行 (`tracked_params.json` 生成)

avatar 学習用と全く同じ shell script を再利用します。`--lhg-only` を付けずに
通常の avatar 学習用に走らせる場合と比べて壁時計時間がほぼ半減します。

### 入力

- `--video` — mp4/mov 入力
- `--intrinsics` — `hdtf` / `insta` / `custom:fx,fy,cx,cy` (outer-crop 512² 座標系)
- `--sbj-root` / `--sbj-name` — 出力ルート + サブディレクトリ名

### 出力 (`<sbj-root>/<sbj-name>/`)

```text
alice/
├── alice.mp4              <- 入力動画への symlink
├── image/                 <- outer-cropped 512×512 RGB フレーム
├── image_raw/             <- prescaled raw フレーム (任意、SMIRK 入力用)
├── outer_offset.json      <- raw video 座標での outer crop bbox
├── stable_bbox.npz        <- 224-crop の安定化済み tform
├── tracked_params.json    <- ★ Stage 2 が consume する calibration ファイル
└── deca/, keypoints/, iris/   <- DECA optimize の中間生成物
```

`tracked_params.json` の `world_mat[2,3]` が `-5.0` 前後の負値であれば
カメラ前方に物体が配置されている (HRAvatar 規約) ことを示します。

---

## Stage 2: Online per-frame extraction (各クリップ)

### 目的

clip-wide 最適化を **使わず**、causal pipeline で per-frame の FLAME 特徴量を抽出:
- MediaPipe FaceLandmarker (video running mode) で 478 pt landmark
- stable bbox (causal hysteresis follower) で SMIRK 224-crop
- SMIRK encoder で `expression` (50d), `jaw` (3d), `eyelid` (2d)
- cv2.solvePnP (EPnP, 1-shot 解析) で `global_rot` (3d), `translation` (3d)
- Causal Hampel で外れ値 reject
- Stage 1 の `world_mat` で per-frame pose を delta に再センター
- **Symmetric FIR LPF (zero-phase + lookahead L=4 default)** を `global_rot` /
  `translation` のみに適用。`expression` / `eyelid` は pass-through。
  `jaw` は `--lpf-jaw` で opt-in (default cutoff 10 Hz)。

LPF は `lhg/lpf.py:apply_offline_zero_phase` 経由 (offline / online 同一の filter
design、`taps = 2L+1`、`scipy.signal.firwin` で線形位相)。online 推論時には
`StreamingSymmetricFIR` で同じ係数を streaming 適用するため、抽出時と推論時の
出力は **bit-exact 一致** します (検証済み: `max diff = 2.2e-16`)。

### コマンド

最小例 (Stage 1 出力を直接 consume):

```bash
bash demos/extract_lhg_features.sh \
    --video /data/lhg/alice/image \
    --calibration /data/lhg/alice/tracked_params.json \
    --output /data/lhg/alice/lhg_features.npz \
    --mode online
```

LPF パラメータをチューニング:

```bash
bash demos/extract_lhg_features.sh \
    --video /data/lhg/alice/image \
    --calibration /data/lhg/alice/tracked_params.json \
    --output /data/lhg/alice/lhg_features_L8.npz \
    --mode online \
    --lookahead 8 \
    --lpf-cutoff-hz 3.0 \
    --lpf-jaw \
    --lpf-jaw-cutoff-hz 10.0
```

LPF 完全 OFF (生 EPnP のジッタを観察したいとき):

```bash
bash demos/extract_lhg_features.sh \
    --video /data/lhg/alice/image \
    --calibration /data/lhg/alice/tracked_params.json \
    --output /data/lhg/alice/lhg_features_L0.npz \
    --mode online \
    --lookahead 0
```

### 主要 CLI flag

| Flag | Default | 役割 |
|---|---|---|
| `--video` | (required) | mp4/mov OR 画像ディレクトリ |
| `--calibration` | (required) | Stage 1 の `tracked_params.json` (or `_v2`) |
| `--output` | (required) | 出力 `.npz` パス |
| `--mode` | (required) | `online` (実装済) / `pseudo-online` (Stage 3 で実装予定) |
| `--fps` | `25` | feature stream FPS |
| `--detector` | `mediapipe` | `mediapipe` / `fan` (Phase 4 A/B 用) |
| `--mediapipe-mode` | `video` | `video` (内部 Kalman tracker) / `image` (per-frame baseline) |
| `--lookahead L` | `4` | symmetric FIR の片側 lookahead frames。`taps = 2L+1` |
| `--lpf-cutoff-hz F` | `4.0` | rotation/translation cutoff (Hz) |
| `--lpf-jaw` | off | jaw に LPF を適用 (opt-in) |
| `--lpf-jaw-cutoff-hz F` | `10.0` | jaw 専用 cutoff (Hz)、`--lpf-jaw` 有効時のみ |
| `--lookahead-offline L` | `12` | Stage 3 の offline FIR lookahead (`taps=25`) |
| `--camera-convention` | `hravatar` | `hravatar` (renderer-ready) / `opencv` (raw EPnP) |
| `--bbox-scale F` | `1.6` | SMIRK 224-crop scale factor |
| `--intrinsics` | (calibration から継承) | 上書きしたい場合のみ |

### 出力 (`lhg_features.npz`)

```text
expression       (N, 50)  float32   SMIRK 出力
jaw              (N,  3)  float32   SMIRK 出力
eyelid           (N,  2)  float32   SMIRK 出力
global_rot       (N,  3)  float32   axis-angle delta around world_mat
translation      (N,  3)  float32   FLAME canonical 単位 delta
valid_mask       (N,)     bool      False = 検出器 miss
interpolated_mask(N,)     bool      pseudo-online interpolation 適用
rejected_mask    (N,)     bool      Hampel で reject されたフレーム
world_mat        (4, 4)   float32   ★ Stage 1 calibration からコピー
intrinsics       (4,)     float32   [fx, fy, cx, cy]
outer_bbox       (4,)     int32     [xmin, xmax, ymin, ymax] in raw video coord
mode, fps, image_size, flame_scale, camera_convention   メタデータ
```

---

## Stage 3: Pseudo-online teacher data (Stage 2 検証後の別 commit で実装予定)

現状: `--mode pseudo-online` を渡すと SystemExit で「Stage 3 未実装」を案内します。
Stage 2 が end-to-end で動作確認できたら、`lhg/extract.py` のゲートを外すだけで
有効化できます (pipeline.py の pseudo-online 経路は既に実装済み)。

実装後の使い方:

```bash
bash demos/extract_lhg_features.sh \
    --video /data/lhg/alice/image \
    --calibration /data/lhg/alice/tracked_params.json \
    --output /data/lhg/alice/lhg_features_teacher.npz \
    --mode pseudo-online \
    --lookahead-offline 12 \
    --lpf-cutoff-hz 4.0
```

Stage 2 との違い:
- 双方向 Hampel + linear interpolation で検出 dropout / 単発外れ値を除去
- 双方向 quaternion-flip 修正で global_rot の符号を統一
- LPF は同じ filter design (firwin + cutoff) で **lookahead だけ大きく** (L=12 → taps=25)

`expression` / `eyelid` は **online と完全同一の処理パス** (LPF 含む)。
詳細は `feedback_pseudo_online_for_lhg_teacher` メモリ。

---

## Render demo (検証用)

学習済み HRAvatar アバター × `lhg_features.npz` で MP4 を生成し、
Stage 2 出力の妥当性を視覚的に確認します。

```bash
bash demos/render_lhg_features.sh \
    --avatar  outputs/custom/alice \
    --source  /data/lhg/alice \
    --lhg-features /data/lhg/alice/lhg_features.npz
```

仕組み:
- `render.py --lhg-features <npz>` flag が `lhg/render_adapter.py` を呼び出し、
  `lhg_features.npz` を `tracked_params.json` 相当の in-memory dict に変換
- `shapecode` は `--source` の元 `tracked_params.json` から (アバター学習時の値で固定)
- `world_mat` / per-frame entries は `lhg_features.npz` で上書き
- `scene/data_loader.py:TrackedData` の `tracked_params_override` kwarg 経由で
  通常の JSON ロード経路を bypass

出力 MP4 は通常の `render.py` と同じ場所 (`<avatar>/{train|test}/ours_<epoch>/...`)。

### 比較セット (Stage 2 受け入れ検証)

同じ avatar に対して 3 種類の入力を rendering して並べて比較:

```bash
# (1) リファレンス: 学習時の tracked_params.json
python render.py -m outputs/custom/alice -s /data/lhg/alice

# (2) LPF off baseline: lookahead=0
bash demos/extract_lhg_features.sh \
    --video /data/lhg/alice/image \
    --calibration /data/lhg/alice/tracked_params.json \
    --output /tmp/L0.npz --mode online --lookahead 0
bash demos/render_lhg_features.sh \
    --avatar outputs/custom/alice --source /data/lhg/alice \
    --lhg-features /tmp/L0.npz

# (3) LPF on default: lookahead=4
bash demos/extract_lhg_features.sh \
    --video /data/lhg/alice/image \
    --calibration /data/lhg/alice/tracked_params.json \
    --output /tmp/L4.npz --mode online --lookahead 4
bash demos/render_lhg_features.sh \
    --avatar outputs/custom/alice --source /data/lhg/alice \
    --lhg-features /tmp/L4.npz
```

期待:
- (1) リファレンスに対し、(2) はジッタが目立つ
- (3) はジッタが減って (1) に近付くが、頭部運動の本質は保たれている
- 表情の lip-sync timing と blink dynamics は (1)(2)(3) で同等 (LPF 対象外のため)

---

## Jitter 可視化

`scripts/plot_lhg_jitter.py` で複数の `lhg_features.npz` を重ね描き、
static / velocity / acceleration の 3 グラフを生成します。

```bash
python scripts/plot_lhg_jitter.py \
    --inputs L0=/tmp/L0.npz \
             L4=/tmp/L4.npz \
             L12=/tmp/L12.npz \
    --output-dir verification/
```

出力:
- `verification/jitter_static.png` — 各フレームの値 (raw vs filtered overlay)
- `verification/jitter_velocity.png` — 1 次差分
- `verification/jitter_acceleration.png` — 2 次差分

各 PNG は 3×3 グリッド (rows: rotation / translation / jaw、cols: x / y / z)。
標準出力には velocity / acceleration の RMS が表示され、lookahead 増加に伴って
単調減少することを確認できます。

---

## トラブルシューティング

### `world_mat not found` エラー

`--calibration` に渡した `tracked_params.json` のスキーマが想定外。
`lhg/calibration.py:_resolve_world_mat` が以下のいずれかを期待します:
- top-level `world_mat`
- `frames[0].world_mat`
- per-image-key dicts (`"00000.png": {"world_mat": ...}`)

avatar 学習用 Stage 1 の出力ならそのまま consume できるはず。

### `correspondence.flame_canonical_xyz is None`

`assets/lhg/mediapipe_flame_landmarks.npz` (or FAN 用 `dlib_flame_landmarks.npz`)
を `--include-canonical` 付きで再生成してください:

```bash
python tools/build_mediapipe_flame_correspondence.py --include-canonical
```

### MediaPipe video mode で `timestamp_ms` の単調増加違反

video running mode は frame ごとに増加する整数 ms を要求します。
`lhg/pipeline.py:_detect_landmarks` は `int(frame_index * 1000 / fps)` を渡すので
通常は問題ありません。fps を 25 以外に変更したクリップで再現する場合は
`--fps` を必ず実 fps に合わせてください。

### `pseudo-online (Stage 3) mode is not yet implemented`

Stage 3 はまだゲート中です。Stage 2 を `--mode online` で先に動作確認してください。

---

## アーキテクチャ参照

- `lhg/calibration.py` — Stage 1 出力の loader
- `lhg/config.py` — `LHGConfig` データクラス、CLI <-> dataclass マッピング
- `lhg/detector.py` — MediaPipe FaceLandmarker (image / video mode)
- `lhg/detector_fan.py` — FAN wrapper (per-frame stable_bbox tracking 経路)
- `lhg/encoders.py` — SMIRK / DECA encoder
- `lhg/epnp.py` — `cv2.solvePnP(SOLVEPNP_EPNP)` + opencv→hravatar 変換
- `lhg/hampel.py` — causal / bidirectional Hampel
- `lhg/interpolate.py` — linear interpolation across dropouts
- `lhg/lpf.py` — symmetric FIR LPF (`design_fir`, `apply_offline_zero_phase`,
  `StreamingSymmetricFIR`)
- `lhg/pipeline.py` — `extract()` + `per_frame_core()`
- `lhg/output.py` — `LHGFeatures` データクラス + npz I/O
- `lhg/render_adapter.py` — `lhg_features.npz` → `tracked_params.json` schema
- `lhg/extract.py` — CLI エントリーポイント

メモリ参照:
- `feedback_lhg_stage_separation` — Stage 1/2/3 の責務分離原則
- `feedback_pseudo_online_for_lhg_teacher` — Stage 3 で許される / 禁止される処理
- `feedback_no_one_euro_in_lhg_pipeline` — symmetric FIR LPF の適用範囲
- `feedback_use_mediapipe_not_fan` — detector 選択基準
- `feedback_online_translation_via_epnp` — EPnP 採用理由
- `project_lhg_fps_convention` — 25 FPS / 10 FPS 規約
