# DECA fork パッチ（自動適用）

HRAvatar の preprocess パイプラインは `MTamon/DECA@cuda128-HRAvatare` フォークを
**冪等な文字列パッチ** で改変して使います。`demos/_preprocess_subject.sh` が起動時に
`tools/patches/apply_deca_*.py` を 1 回ずつ呼ぶため、ユーザは通常意識する必要が
ありません。再適用済みの場合は `[skip]` が表示されます。

## パッチ一覧

### 1. `apply_deca_stable_bbox.py`

`decalib/datasets/datasets.py` と `demos/demo_reconstruct.py` を改変し、
`--precomputed-bbox <path>` で `stable_bbox.npz` を受け取れるようにします。
これにより DECA 内部の per-frame FAN crop を bypass し、HRAvatar 側で計算済み
の安定 bbox（口・瞬きの jitter を除去済み）と DECA crop の座標系を完全一致させます。

詳細: `doc/stable_bbox.md`

### 2. `apply_deca_optimize_regularizer.py`（新規）

DECA フォークの `optimize.py` で **shape / expression 正則化重みを CLI 化** します。
オリジナルは以下のようにハードコード:

```python
total_loss = landmark_loss2 + torch.mean(torch.square(shape)) * 1e-2 \
                            + torch.mean(torch.square(exp)) * 1e-2
```

パッチ適用後:

```python
total_loss = landmark_loss2 + torch.mean(torch.square(shape)) * args.lambda_shape \
                            + torch.mean(torch.square(exp)) * args.lambda_exp
```

新しい CLI フラグ:

| フラグ | 既定 | 意味 |
|---|---|---|
| `--lambda_shape` | `1e-2` | shape**2 正則化重み。**既定値はオリジナル踏襲**。 |
| `--lambda_exp` | `1e-2` | exp**2 正則化重み。同上。 |

#### なぜ必要か（エイリアン頭問題）

`optimize.py` は **5000 frame 程度の landmark 誤差を合算** vs **1 つの shape
ベクトル正則化** という非対称な戦いを 1000 iter 走らせます。デフォルトの
`1e-2` は landmark loss に圧倒され、**FLAME shape が unrealistic な領域**
（頭頂部肥大、顔面中央陥没）に流れます。これが
`data/subjects/<name>/optimize_vis.jpg` の右端グレーメッシュの「エイリアン化」の
正体です。

`vanilla DECA` は 1 枚画像の regression（学習済みネット推論のみ）なので
この問題は起きません — HRAvatar フォーク固有の挙動です。

#### 推奨値

| 用途 | `--lambda_shape` | コメント |
|---|---|---|
| デフォルト | `1e-2` | 既存挙動。エイリアン化リスクあり |
| 軽い安定化 | `0.5` 〜 `1.0` | 顔形状を identity 範囲に保ちやすい |
| 強い安定化 | `3.0` 〜 `5.0` | landmark との fit は若干緩むが安定 |

`exp` の方は通常デフォルトで問題ありません。

## 使い方

### 既定（自動適用）

```bash
bash demos/demo_1_train_subject.sh \
    --sbj-root ./data/subjects --sbj-name MK6c \
    --video ./data/raw/mikawa6c.mp4 --intrinsics hdtf
```

ログ:
```
[preprocess 0/5] DECA patches (idempotent)
[skip] .../optimize.py already patched (...)
[skip] .../datasets.py already patched (...)
[preprocess 1/5] crop + matting
...
```

### 強い shape 正則化を使う

```bash
bash demos/demo_1_train_subject.sh \
    --sbj-root ./data/subjects --sbj-name MK6c \
    --video ./data/raw/mikawa6c.mp4 --intrinsics hdtf \
    --lambda-shape 1.0
```

### パッチを手動管理したい

```bash
bash demos/demo_1_train_subject.sh ... --skip-deca-patches
# その後、自分で
python tools/patches/apply_deca_stable_bbox.py
python tools/patches/apply_deca_optimize_regularizer.py
```

## 単独実行

```bash
# パッチ単独適用（冪等。複数回 OK）
python tools/patches/apply_deca_stable_bbox.py
python tools/patches/apply_deca_optimize_regularizer.py

# 別 DECA チェックアウトに対して
python tools/patches/apply_deca_optimize_regularizer.py /path/to/DECA
```

## マーカ

各パッチは識別マーカをコメントで埋め込み、再適用時に検出します。

| パッチ | マーカ |
|---|---|
| `apply_deca_stable_bbox.py` | `# HRAVATAR_STABLE_BBOX` |
| `apply_deca_optimize_regularizer.py` | `# HRAVATAR_OPTIMIZE_REGULARIZER` |

確認:

```bash
grep -n HRAVATAR_ preprocess/submodules/DECA/optimize.py
grep -rn HRAVATAR_ preprocess/submodules/DECA/decalib/datasets/
```

## トラブルシュート

- **`[error] anchor missing in ...`**: DECA フォークが想定 (`cuda128-HRAvatare`)
  からドリフトしています。正しいブランチに rebase するか、パッチ手動マージ。
- **`No module named 'numpy'` etc.**: パッチ自体は標準ライブラリだけで動きます。
  もし import エラーが出たら Python パスの問題か別ファイルが import されています。
- **`set -u: unbound variable: DECA_OPTIMIZE_EXTRA_ARGS[@]`**: bash < 4.4
  での既知の問題。リポジトリは bash 5.x 想定で `${ARR[@]+"${ARR[@]}"}` の
  defensive form を使っています。

## 参考

- `doc/jitter_filter.md` — One-Euro による jitter 軽減（独立機能）
- `doc/stable_bbox.md` — bbox 安定化（独立機能）
- `tools/patches/apply_deca_stable_bbox.py`
- `tools/patches/apply_deca_optimize_regularizer.py`
