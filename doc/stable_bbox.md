# Stable BBox 前処理（SMIRK / DECA 入力 jitter 対策）

`crop_and_matting.py` の固定 outer-crop だけでは、その内側で実行される
**SMIRK encoder 入力 224 crop** と **DECA 内部の per-frame FAN crop** に
ともに「口の開閉・瞬きのたびに bbox が呼吸する」jitter が残ります。
これは bbox の `size` を MediaPipe 全478点（SMIRK 側）／FAN 全68点
（DECA 側）の min/max から導出しているためで、口の上下変位がそのまま
crop サイズに漏れ、SMIRK の弱透視カメラ `cam[s]` が振動し、最終的に
レンダリングされた FLAME メッシュが耳・頭頂で揺れる原因になります。

このモジュール群は、上流 [MTamon/smirk@release/cuda128 (PR #7)](https://github.com/MTamon/FlashAvatar/blob/claude/add-smirk-bbox-stabilization-4rvvR/docs/smirk.md#bbox-%E5%AE%89%E5%AE%9A%E5%8C%96---bbox-mode)
と同じアイデアを HRAvatar に移植したものです。

- **安定ランドマーク部分集合**: 目尻・鼻根・こめかみなど、発話/瞬きで動かない
  15 点だけを bbox の元に使う。**口/瞬きが bbox に漏れる構造を断つ**。
- **ゼロ位相 FIR 低域通過**: 上で得られた `size` 系列を `scipy.signal.firwin`
  + `valid` 畳み込みで両方向参照スムージング。**位相遅延ゼロ**。

これらの結果を **dataset 単位で 1 回だけ計算** して `stable_bbox.npz` に永続化し、
DECA preprocessing と SMIRK encoder の両方で再利用します。

## ファイル

| ファイル | 役割 |
|---|---|
| `preprocess/_smirk_constants.py` | 安定 15 点 mediapipe index + キャリブレーション係数 1.55 |
| `preprocess/stable_bbox.py` | 2-pass 計算と `stable_bbox.npz` 出力（CLI あり） |
| `preprocess/bbox_verify.py` | legacy / stable raw / smoothed の比較 mp4 + CSV 出力 |
| `scene/data_loader.py` | `stable_bbox.npz` を自動検出して per-iter MediaPipe を bypass |
| `demos/_preprocess_subject.sh` | `STABLE_BBOX=1`（既定）で前処理に組み込み |
| `tmp/hravatar_stable_bbox/deca_stable_bbox.patch` | DECA 側で `--precomputed-bbox` を受け付けるパッチ |

## 使い方

### 通常（demo_1 経由）

```bash
# 前提: DECA fork に deca_stable_bbox.patch を当てておく
cd preprocess/submodules/DECA
git apply /tmp/hravatar_stable_bbox/deca_stable_bbox.patch
cd -

# あとは普段通り。STABLE_BBOX=1 がデフォルト。
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

| 環境変数 | 既定 | 意味 |
|---|---|---|
| `STABLE_BBOX` | `1` | `0` でこの段階を完全にスキップし、legacy 経路（DECA 内 FAN, data_loader 内 MediaPipe）に戻す |
| `BBOX_CUTOFF_HZ` | `2.5` | FIR LPF カットオフ。座位で上半身を動かす程度なら既定で十分 |
| `BBOX_VERIFY` | `0` | `1` で `bbox_verify.mp4` + CSV を `<DATA_DIR>/` に書く |

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

### `size` のみ平滑化、`center` は生値

座位で上半身を動かすなど被写体が画面内を移動するケースでは、**頭部の並進
(center) は本物の運動なので追従させたい**ため、デフォルトで center は
平滑化しません（FlashAvatar と同じ設計）。中央値ロックなど size 側の
強い制約を入れると、SMIRK の弱透視カメラ `Z = f_px * s_ff / (s * 112)`
を介してメッシュ深度が体系的にズレるため、この方向にも踏み込んでいません。

`--smooth_center` を opt-in すると center にも同じ FIR LPF が掛かります。
撮影が完全に静止している場合のみ意味があります。

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
- **後方互換**: `STABLE_BBOX=0` でこの仕組みを完全に無効化できます。
  `stable_bbox.npz` を持たない既存 dataset は、`data_loader` 側でも
  legacy MediaPipe + crop_face にフォールバックします。

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
