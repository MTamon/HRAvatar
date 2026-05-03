# DECA fork パッチ（自動適用）

HRAvatar の preprocess パイプラインは `MTamon/DECA@cuda128-HRAvatare` フォークを
**冪等な文字列パッチ** で改変して使います。`demos/_preprocess_subject.sh` が起動時に
`tools/patches/apply_deca_*.py` を 1 回ずつ呼ぶため、ユーザは通常意識する必要が
ありません。再適用済みの場合は `[skip]` が表示されます。

## パッチ一覧

| パッチ | 役割 | マーカ |
|---|---|---|
| `apply_deca_stable_bbox.py` | DECA に `--precomputed-bbox` を追加 | `# HRAVATAR_STABLE_BBOX` |
| `apply_deca_optimize_regularizer.py` | shape / exp 正則化重みの CLI 化 | `# HRAVATAR_OPTIMIZE_REGULARIZER` |
| `apply_deca_pose_anchor.py` | per-frame pose を DECA 初期値にアンカー | `# HRAVATAR_POSE_ANCHOR` |

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

### 3. `apply_deca_pose_anchor.py`（新規）

DECA フォークの `optimize.py` に **pose anchor 正則化** を追加します。

```python
# 追加される項
total_loss += torch.mean(torch.square(pose - pose_init)) * args.lambda_pose_anchor
```

`pose_init` は `pose = nn.Parameter(pose)` 直後にスナップショットを取った
**DECA per-frame 推定値**（`code.json` の `pose` フィールド由来）です。

新しい CLI フラグ:

| フラグ | 既定 | 意味 |
|---|---|---|
| `--lambda_pose_anchor` | `0.0` | pose を DECA 初期値に引き戻す重み。**既定 0 = 無効**（後方互換）。 |

#### なぜ必要か（顔向きの過剰回転問題）

`--lambda_shape` を強くすると、optimizer は landmark 誤差を埋めるために
**rigid transform**（pose / translation）を過剰に動かしてつじつまを合わせる
傾向に流れます。特に **global rotation が映像本来の向きより大きく振れる**
症状が出ます。これは shape の自由度を奪った副作用です。

`--lambda_pose_anchor` で per-frame pose を **DECA の per-frame 推定値**
（DECA 単体では信頼できる出発点）に弱く引き戻すことで、shape を強く制約
しつつ pose の過剰補償を抑えられます。

#### 推奨値

| `--lambda_shape` | `--lambda_pose_anchor` | 効果 |
|---|---|---|
| `1e-2`（既定） | `0.0`（既定） | 完全にオリジナル挙動 |
| `0.5` 〜 `1.0` | `0.05` 〜 `0.1` | shape を軽く絞り、pose 過剰回転を弱く抑える |
| `3.0` 〜 `5.0` | `0.2` 〜 `0.5` | shape を強く絞り、pose もしっかり anchor |

**注意**: `lambda_pose_anchor` が大きすぎると、被写体が首を振っても
optimizer が DECA 初期推定の周囲に固定されてしまい、頭の動きが追従しなく
なります。`lambda_shape` の上げ幅に対して `lambda_pose_anchor` も控えめに
(0.1〜0.2 程度から) 探るのが安全です。

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

### shape を強く絞った副作用（顔向き過剰回転）も同時に抑える

```bash
bash demos/demo_1_train_subject.sh \
    --sbj-root ./data/subjects --sbj-name MK6c \
    --video ./data/raw/mikawa6c.mp4 --intrinsics hdtf \
    --lambda-shape 1.0 --lambda-pose-anchor 0.1
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
python tools/patches/apply_deca_pose_anchor.py

# 別 DECA チェックアウトに対して
python tools/patches/apply_deca_optimize_regularizer.py /path/to/DECA
python tools/patches/apply_deca_pose_anchor.py /path/to/DECA
```

## マーカ

各パッチは識別マーカをコメントで埋め込み、再適用時に検出します（一覧は
冒頭表参照）。確認:

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
