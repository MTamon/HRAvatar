# One-Euro Jitter フィルタ（オプトイン）

[MTamon/Gaussian-HS](https://github.com/MTamon/Gaussian-HS) を参考に、HRAvatar
の per-frame tracker パラメータと SMIRK encoder 出力に対して
**One-Euro filter (Casiez et al., 2012)** で時間方向に低域通過を掛ける機能を
追加しました。**デフォルトでは無効**で、すべて opt-in CLI フラグから有効化します。

## なぜ必要か

`stable_bbox.npz` は SMIRK 入力 224 crop の bbox `size` jitter（口・瞬き由来）
だけを抑えますが、以下は素通しです:

- DECA `optimize.py` 由来の `tracked_params.json` 中の per-frame
  `translation` / `fullposecode` / `expcode` / `eyelids`
- SMIRK encoder が render 時に出す per-frame
  `expression_params` / `jaw_params` / `eyelid_params`

これらの高周波ノイズは最終アバターの jitter に直結します。本機能はその両方を
**学習を始める前 (load 時) にオフラインでバッチ平滑化** + **render 時に
encoder 出力を causal 平滑化** の二段で抑えます。

## 使い方

最低限のフラグは `--jitter_filter` 1 個だけです。

```bash
# 学習: tracked_params の channels をオフライン双方向 One-Euro で平滑化
python train.py --source_path <data> --model_path <out> \
    --jitter_filter \
    --jitter_filter_min_cutoff 1.0 --jitter_filter_beta 0.0 \
    --jitter_filter_fps 30
```

```bash
# レンダリング: 上に加えて SMIRK encoder 出力も causal 平滑化
python render.py --model_path <out> \
    --jitter_filter \
    --jitter_filter_smirk \
    --jitter_filter_min_cutoff 1.0 --jitter_filter_beta 0.0 \
    --jitter_filter_fps 30
```

## フラグ一覧

| フラグ | 既定 | 意味 |
|---|---|---|
| `--jitter_filter` | off | 全機能のマスタースイッチ。これが off のときは何も起きない（後方互換）。 |
| `--jitter_filter_targets` | `translation fullpose expression eyelid` | TrackedData ロード時に平滑化する `tracked_params.json` 内チャンネル。`world_mat` は **opt-in**（カメラが動かない撮影なら無意味）。 |
| `--jitter_filter_smirk` | off | render.py で SMIRK encoder 出力 (expression / jaw / eyelid) に causal 平滑化を掛ける。学習には影響しない。 |
| `--jitter_filter_min_cutoff` | 1.0 | One-Euro の `min_cutoff` (Hz)。**小さいほど静止時に強くスムージング**。一般的な顔トラッキング jitter には 0.8〜1.5 Hz が落とし所。 |
| `--jitter_filter_beta` | 0.0 | One-Euro の `beta`（速度係数）。**大きいほど急動への追従が早い**。0.0 だと純粋な低域通過。発話中のあご動きや視線の素早い移動がある動画では `0.005`〜`0.05` 程度を試す。 |
| `--jitter_filter_d_cutoff` | 1.0 | derivative の低域通過カットオフ (Hz)。通常は 1.0 のままで OK。 |
| `--jitter_filter_fps` | 30 | One-Euro の時間ステップに使う fps。学習動画と一致させる。 |

## どこで何が起きるか

### 1. `scene/data_loader.py` (load 時、双方向)

`TrackedData.__init__` で `tracked_params.json` を読み込んだ直後に、
`--jitter_filter_targets` で指定された各チャンネルをフレーム順に
スタックし、**双方向 One-Euro (forward + backward 平均)** で平滑化します。
位相遅延ゼロです。学習でも render でも同じ平滑値が使われます。

ログ:
```
[data_loader] jitter_filter: smoothed N channel(s) (['eyelid', 'expression', 'fullpose', 'translation']) with min_cutoff=1.0 Hz, beta=0.0, fps=30.0
```

### 2. `scene/gaussian_head_model.py` (render 時、causal)

`GaussianHeadModel.forward()` で SMIRK encoder の出力を受け取った直後に、
`set_smirk_smoother()` でインストールされた `CausalOneEuroBuffer` が
expression / jaw / eyelid を causal 平滑化します。**勾配は学習時には流れません**
（render.py 側でだけ buffer をインストールするため）。

### 3. `render.py` (sequence ごとに reset)

`render_sets` で `_maybe_attach_smirk_smoother()` がフラグを見て buffer を
インストールします。各独立 sequence
（train / test / multi-view / cross-reenactment）の前に
`_reset_smirk_smoother()` で内部状態をクリアし、causal フィルタを再起動します。

## チューニング指針

| 症状 | 試す調整 |
|---|---|
| 静止時の頭揺れが残る | `--jitter_filter_min_cutoff 0.6` まで下げる |
| 発話中に口が遅れて動く | `--jitter_filter_beta 0.02`〜`0.05` を入れる |
| まばたきが鈍る | `--jitter_filter_targets translation fullpose expression`（eyelid を外す） or `--jitter_filter_beta 0.05`〜 |
| カメラ追従が遅い動画 | `--jitter_filter_targets translation fullpose expression eyelid world_mat` で `world_mat` も含める |
| 多視点 (multi_views) の側面崩壊 | 本フィルタでは直らない（単眼学習の構造的限界） |

## 既存 stable_bbox との関係

- 直交します。両方有効が推奨。
- stable_bbox: **SMIRK 入力 crop の jitter** を消す（preprocess 段階）。
- jitter_filter: **SMIRK 出力 / DECA 由来 tracker 値の jitter** を消す（学習・render 段階）。

## 参考

- One-Euro Filter: G. Casiez, N. Roussel, D. Vogel.
  *1€ filter: a simple speed-based low-pass filter for noisy input in
  interactive systems.* CHI 2012.
- 上流参考実装: [MTamon/Gaussian-HS](https://github.com/MTamon/Gaussian-HS)
- 関連: `doc/stable_bbox.md` (SMIRK 入力 crop 安定化)
