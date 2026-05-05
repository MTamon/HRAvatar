# Stable BBox 前処理（SMIRK / DECA 入力 jitter 対策）

HRAvatar の crop は **二段スタック** になっています:

1. **Outer 512 crop (`crop_and_matting.py`)**: 元動画から学習用 512x512 RGBA
   フレームを切り出す段。**動画全体で固定 (frame 間不変)** の bbox で
   切り出します。crop offset が時間に対して動かないため、頭部の並進は
   FLAME `translation_code` 一箇所に集約されます (per-frame 追従だと
   並進が crop offset に吸収されて消えるため不採用)。詳細は
   後述「動画全体固定 outer crop」節。
2. **Inner 224 crop (`stable_bbox.py`)**: その 512 フレーム内で SMIRK
   encoder 用 224 crop と DECA 内部 FAN crop を駆動する bbox を計算する段。
   K-of-N hysteresis center + FIR LPF size の **時間軸スムージング** が掛かる。

このドキュメントは主に **inner 側 (2)** の jitter 対策について書いていますが、
outer 側 (1) は別系統 (動画全体固定 outer crop) で組まれているため、関連節
「動画全体固定 outer crop」も合わせて参照してください。

`crop_and_matting.py` の outer-crop の内側で実行される
**SMIRK encoder 入力 224 crop** と **DECA 内部の per-frame FAN crop** には
ともに「口の開閉・瞬きのたびに bbox が呼吸する」jitter が残ります。
これは bbox の `size` を MediaPipe 全478点（SMIRK 側）／FAN 全68点
（DECA 側）の min/max から導出しているためで、口の上下変位がそのまま
crop サイズに漏れ、SMIRK の弱透視カメラ `cam[s]` が振動し、最終的に
レンダリングされた FLAME メッシュが耳・頭頂で揺れる原因になります。

このモジュール群は、上流 [MTamon/smirk@release/cuda128 (PR #7)](https://github.com/MTamon/FlashAvatar/blob/claude/add-smirk-bbox-stabilization-4rvvR/docs/smirk.md#bbox-%E5%AE%89%E5%AE%9A%E5%8C%96---bbox-mode)
と同じアイデアを HRAvatar に移植したものです。

- **安定ランドマーク部分集合**: 目尻・鼻根・こめかみなど、発話/瞬きで動かない
  15 点だけを bbox の元に使う。**口/瞬きが bbox に漏れる構造を断つ**。
- **ゼロ位相 FIR 低域通過 (`size`)**: 上で得られた `size` 系列を
  `scipy.signal.firwin` + `valid` 畳み込みで両方向参照スムージング。
  **位相遅延ゼロ**。
- **K-of-N hysteresis center follower (`center`)**: 顔向きが大きくなると
  MediaPipe ランドマーク自体が瞬時にぶれるため、center を生値で渡すと
  そのノイズが crop 位置に伝播する。窓 N の中で K フレーム以上
  「動量 > 閾値 (px)」が成立した場合のみ持続的な動きと判定し、
  目標位置を更新して時定数 τ の指数追従で滑らかに合わせる。
  **閾値以下のノイズには反応しない**ため、頷きや瞬きでの crop 揺れが
  消える。bbox 自体は `scale=1.6` の余裕を確保しているので
  追従が遅れても顔は 224 crop から外れない。

これらの結果を **dataset 単位で 1 回だけ計算** して `stable_bbox.npz` に永続化し、
DECA preprocessing と SMIRK encoder の両方で再利用します。

## ファイル

| ファイル | 役割 |
|---|---|
| `preprocess/_smirk_constants.py` | 安定 15 点 mediapipe index + キャリブレーション係数 1.55 |
| `preprocess/stable_bbox.py` | 2-pass 計算と `stable_bbox.npz` 出力（CLI あり） |
| `preprocess/bbox_verify.py` | legacy / stable raw / smoothed の比較 mp4 + CSV 出力 |
| `scene/data_loader.py` | `stable_bbox.npz` を自動検出して per-iter MediaPipe を bypass |
| `demos/_preprocess_subject.sh` | 既定で前処理に組み込み（`--no-stable-bbox` で無効化、`--bbox-verify` で検証動画も生成） |
| `tools/patches/apply_deca_stable_bbox.py` | DECA 側に `--precomputed-bbox` を追加する冪等な Python パッチャ |

## 使い方

### 通常（demo_1 経由）

```bash
# 前提: DECA fork に stable_bbox 統合を適用しておく（冪等）
python tools/patches/apply_deca_stable_bbox.py

# あとは普段通り。stable bbox はデフォルトで有効。
bash demos/demo_1_train_subject.sh <root> <name> <video> hdtf
```

実行ログには以下が現れます。

```
[preprocess 1.5/5] stable bbox (FlashAvatar PR#7 port)
[stable_bbox] wrote .../stable_bbox.npz (N frames, M detected, taps=K, cutoff=2.5 Hz)
[deca/TestData] using precomputed stable bbox (N frames) from .../stable_bbox.npz
```

学習開始時にも:

```
[data_loader] using precomputed stable bbox (N frames) from .../stable_bbox.npz
```

### オプション

`demos/_preprocess_subject.sh` の `--help` で全フラグ一覧が出ます。stable bbox 関連は以下:

| フラグ | 既定 | 意味 |
|---|---|---|
| `--no-stable-bbox` | (off) | 指定するとこの段階を完全にスキップし、legacy 経路（DECA 内 FAN, data_loader 内 MediaPipe）に戻す |
| `--no-prescale` | (off) | ffmpeg フレーム抽出時の短辺 = `--resize` への自動スケーリングを無効化（既定は ON）。原寸動画解像度に依存して顔の絶対 px サイズが変動する従来挙動に戻す |
| `--bbox-cutoff-hz F` | `2.5` | FIR LPF カットオフ (Hz)。座位で上半身を動かす程度なら既定で十分 |
| `--bbox-scale F` | `1.6` | bbox 倍率。legacy は 1.4 だったが、center hysteresis の追従遅れ中も顔が 224 crop から出ないよう 1.6 に拡大 |
| `--bbox-center-deadzone-px F` | `4.0` | K-of-N 動量閾値 (ソース画像 px)。これ以下の `|raw - anchor|` は landmark ノイズとみなし無視 |
| `--bbox-center-window N` | `5` | K-of-N の窓長 (フレーム) |
| `--bbox-center-k-of-n N` | `3` | K-of-N の閾値。直近 N フレーム中 K 以上で動量超過 → 持続的動きと判定 |
| `--bbox-center-tau F` | `0.25` | 指数追従の時定数 (秒)。target → anchor を `1 - exp(-dt/τ)` で近づける |
| `--bbox-center-passthrough` | (off) | FIR・hysteresis を両方バイパスし center を生値で出力（A/B 比較用、legacy 挙動） |
| `--bbox-verify` | (off) | 指定すると `bbox_verify.mp4` + CSV を `<DATA_DIR>/` に書く |

### 単独実行

```bash
# stable_bbox.npz だけ作り直す
python preprocess/stable_bbox.py --source <data_dir> --fps 30

# 比較動画を生成して目視確認
python preprocess/bbox_verify.py --source <data_dir> --fps 30
```

## 数学的設計

### `taps` の自動導出

ユーザは cutoff (Hz) と fps だけ指定すれば良く、FIR タップ数は内部で:

```python
taps = next_odd(round(WINDOW_SECONDS * fps))   # WINDOW_SECONDS = 2.0
```

として自動算出されます。これにより **時間領域の窓幅が fps 不変**で、
30 fps では 61 タップ（FlashAvatar 既定と一致）、60 fps では 121 タップ、
25 fps では 51 タップが選ばれます。群遅延は `(taps-1)/(2*fps)` 秒で常に
~1 秒。エッジパディング + `valid` 畳み込みで群遅延は相殺されます。

### `size` は FIR LPF、`center` は K-of-N hysteresis

座位で上半身を動かすなど被写体が画面内を移動するケースでは、**頭部の
並進 (center) は本物の運動なので追従させたい**一方、顔向きが大きい
フレームでの MediaPipe ランドマーク自体のぶれを center に伝播させたく
ありません。これを区別するため、center 経路には離散イベント駆動の
**K-of-N hysteresis follower** を採用しています:

```
disp_i = ||raw_centers[i] - anchor||             # 動量
ring_i = (disp_i > deadzone_px)                  # 直近 N フレームの bool ring
flag_i = sum(ring) >= K                          # K-of-N 判定
target = mean(raw_centers[i-N+1 : i+1]) if flag_i else target
anchor = anchor + (target - anchor) * (1 - exp(-dt/τ))
smooth_centers[i] = anchor
```

FIR と同じく forward + reverse の双方向パスを取って 0.5×平均することで
ゼロ位相とします。閾値以下のノイズには反応しないため、頷きや瞬きの
タイミングで crop 位置がブレません。一方、顔の本物の並進は
持続的に閾値を超えるため、target が更新され τ で滑らかに追従します。

### bbox スケール 1.6 とマージン

center が target に追従するまでの間、anchor は target と離れます。
この乖離期間中も顔が 224 crop の中に収まっている必要があるため、
`scale` の既定値を 1.4 → **1.6** に引き上げ、安定 15 点 calibration
1.55 と合わせて約 2.48 倍の余裕を確保しました。レガシー挙動に
合わせたい場合は `--bbox-scale 1.4` で復元できます。

### 旧経路 (FIR center / passthrough)

`--smooth_center` を opt-in すると center 経路は FIR LPF になります。
`--bbox-center-passthrough` で hysteresis と FIR の両方を抜き、
center を生値で渡す legacy 挙動に戻すこともできます (A/B 比較用)。
size 経路は常に FIR LPF が掛かります — size の強い中央値ロックは
SMIRK の弱透視カメラ `Z = f_px * s_ff / (s * 112)` を介して
メッシュ深度を体系的にズラすため踏み込んでいません。

### キャリブレーション係数 1.55

安定 15 点だけだと bbox の縦方向 extent は全点の ~60% になるため、
`(width+height)/2` に **1.55 を掛けて** legacy `scale=1.4` と視覚的に
同等な crop サイズを保ちます。この係数は MTamon/smirk 側で経験的に
チューニング済みで、HRAvatar 側でも同じ値を使うことで DECA の crop
位置と SMIRK encoder への入力 crop 位置が一致します。

## 設計上の判断

- **SMIRK / DECA の両方で共通利用**: 1 つの `stable_bbox.npz` を両者が
  消費することで、tracker パラメータの座標系が一致します。DECA 側の
  `code.json` `tform` も smoothed crop を反映するため、`optimize.py`
  以降の経路もすべて整合します。
- **学習時 CLI フラグなし**: `stable_bbox.npz` の有無で `data_loader` が
  自動切替するため、`arguments/__init__.py` には新フラグを足していません。
  停止したいときは npz をリネーム/削除すれば legacy 経路に戻ります。
- **DECA 側はパッチ配布**: HRAvatar の git submodule を改変する代わりに、
  パッチファイルとして外部配布します。利用者側で適用してください。
- **後方互換**: `--no-stable-bbox` でこの仕組みを完全に無効化できます。
  `stable_bbox.npz` を持たない既存 dataset は、`data_loader` 側でも
  legacy MediaPipe + crop_face にフォールバックします。

### 動画全体固定 outer crop (`crop_and_matting.py`)

HRAvatar の output (`tracked_params.json`) は **head motion generation
モデルへの入力**として消費される予定で、頭部位置パラメータを FLAME
`translation_code` 一箇所に集約することが絶対要件です。outer crop の
設計はこの制約から逆算されています。

#### なぜ per-frame 追従でも first-frame-fixed でもないのか

- **first-frame-fixed (legacy)**: `crop_and_matting.py` 初版は frame 0
  だけで FAN bbox を計算し全 frame で流用していました。被写体が横に
  動く動画では後続 frame で顔が固定 bbox から外れて切り取られ、頭頂・
  側頭部が欠ける回帰がありました。
- **per-frame 追従 (試行版・採用せず)**: 上記回帰への対処として frame
  ごとに bbox を更新する設計を試しましたが、これは「frame ごとに crop
  offset が変わる」ことを意味し、**頭部の並進が crop offset に吸収
  されて FLAME translation がほぼゼロになる**致命的な副作用が生じ
  ました。head motion generation モデルへの入力としては使えません。

両者の中間として **動画全体で固定 (frame 間不変) の outer crop** を
採用しました。frame 間で crop offset が変わらないので頭部並進が
FLAME translation_code に正しく乗り、かつ動画全体での face 可動領域
を bbox に含めることで横移動でも顔が落ちません。

#### アルゴリズム (3-pass)

1. **Pass 1 — 全 frame FAN 検出**: 全フレームに `face_alignment` を
   走らせて 68-landmark から `(x_min, x_max, y_min, y_max)` を集計。
   検出失敗 frame は単に集計から除外 (last-valid 伝播はせず、union 計算
   のため後続 frame の検出値が代行する)。動画全体で 1 frame も検出でき
   なかった場合は明示的に `RuntimeError` を上げる。
2. **Pass 2 — union bbox 計算**: 全 frame の `min(x_min)`, `max(x_max)`,
   `min(y_min)`, `max(y_max)` を取って **動画全体での face 可動領域 +
   顔サイズの最大値**を覆う union bbox を 1 個だけ作る。
3. **Pass 3 — inflate / 正方化 / 偶数化**: union bbox を `--outer-bbox-scale`
   倍 (default 2.2) に inflate し、長辺採用で正方化、偶数化。これを
   全 frame に同じ bbox として適用 → `crop_image_bbox` + `squarefiy` で
   512x512 RGBA を書き出す。**bbox は image bounds で clip しません**
   — image をはみ出る領域は `crop_image` でゼロ padding されます (後述)。

#### bbox は image bounds で clip しない (zero-padding 設計)

legacy 実装では outer bbox を image bounds で clip していましたが、
動画内で頭部が大きく動く動画 (= union bbox の半幅が image 短辺の半分を
超える) では `--outer-bbox-scale` が短辺で **頭打ち** してしまい、
scale を下げても face/canvas 比が改善しない問題がありました。

そこで Plan B では `crop_image` 段でゼロ padding を行い、bbox を
image bounds で clip しない設計を採用しています:

- outer bbox = `bbox_scale * 2 * half_extent` の正方が **必ず** 確保
  される (= `--outer-bbox-scale` が線形に効く)
- image をはみ出た領域は黒 padding (matting 後は背景色に塗られるので
  実害なし)
- canvas 内 face 比率を operator が直接 dial できる

これにより `--outer-bbox-scale` 値ごとの canvas 内 face 比率は概ね:

| `--outer-bbox-scale` | canvas 内 face 比率 (talking head 想定) |
|---|---|
| 2.2 (default) | ~20% (顔小さめ、頭部動きの余裕大) |
| 1.5 | ~30% |
| **1.3** | **~35%** |
| 1.0 | ~45% (talking head dataset の標準) |
| 0.8 | ~55% (頭部が画面端で切れる手前) |

`bbox_scale < 1.0` は union bbox を縮小するため、頭部の輪郭が画面端で
切れるリスクがあります。**安全範囲は 1.0–1.5** です。

#### フラグ

| フラグ | 既定 | 意味 |
|---|---|---|
| `--outer-bbox-scale F` | `2.2` | union bbox を inflate する倍率。線形に効く (`bbox_scale * 2 * half_extent` 正方を outer crop として確保)。`crop_and_matting.py --bbox_scale` に渡される。 |

inner stable_bbox 側の `--bbox-scale` (default 1.6) とは独立です。
inner は安定 15 ランドマーク中心に 1.6 倍、outer は全 frame の face
union を 2.2 倍 (調整可) にそれぞれ拡大します。

#### Intrinsics の整合性

学習時のカメラ intrinsics は `optimize.py --cx --cy --fx --fy` で video
全体に対し単一値を渡します。動画全体固定 outer crop なので、union bbox
の中心は被写体の頭部全期間平均に近い位置に固定されます。`hdtf` プリセット
`(cx=261.44, cy=253.23)` などの「画面中央付近」前提は、被写体が画面中央
を向いて座っている標準シーンであれば成立します。被写体が偏ったところに
立っているシーンでは `--intrinsics custom:fx,fy,cx,cy` で union bbox の
中心に合わせて cx, cy を調整してください。

### 短辺 prescale (`crop_and_matting.py`)

`crop_and_matting.py` は 既定で **ffmpeg 抽出時に短辺 = `--resize`
（既定 512）にスケール** してからフレームを書き出します。これは原寸
動画の解像度に依存して顔の絶対 px サイズがばらつき、`--bbox-scale`
を一律にチューニングできない問題を回避するためです。
`--no-prescale` で無効化できます。

**Intrinsics 整合性の注意**:

- `hdtf` プリセット (`fx=1539.67, fy=1508.93, cx=261.44, cy=253.23`) と
  `insta` プリセット (`fx=1536, fy=1536, cx=256, cy=256`) は元から
  512x512 動画前提の値なので、prescale 後 (短辺 512) と整合します。
- `custom:fx,fy,cx,cy` を渡す場合、ユーザーは「prescale **後**の
  解像度に対応する intrinsics」を指定してください。原寸動画基準の値を
  そのまま渡すと prescale を有効にした瞬間に整合性が崩れます。
  `--no-prescale` を併用すれば原寸基準のまま動かせます。

## 既知の制限

- **MediaPipe が初フレームで顔を検出できる必要があります**。先頭が後ろ
  向き / 空フレームの動画は、`crop_and_matting.py` 段階の ffmpeg 抽出
  前にトリミングしてください。
- **シーケンス長 < taps（例: 30 fps で 60 フレーム = 2 秒未満）** では
  FIR LPF が `s.copy()` にフォールバックします（無音）。短い動画には
  この機構の恩恵が出ません。`bbox_verify.mp4` の `d|s|` 系列で平滑化が
  効いていることを 1 度は目視確認してください。
- **MediaPipe 検出失敗フレームは直前の有効ランドマークを伝播**します。
  数フレームの欠落なら問題ありませんが、長い欠落区間がある動画では
  動画自体を見直してください。

## 参考

- 上流 SMIRK 側ドキュメント: `docs/smirk.md`「BBox 安定化（`--bbox-mode`）」節
- 上流コア実装: `MTamon/smirk@release/cuda128:utils/bbox_tracker.py`
- HRAvatar 側全体設計: `doc/hravatar_deca_smirk_rendering_notes.md`
