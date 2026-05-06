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

### 3. `apply_deca_optimize_iters.py`（新規）

DECA フォークの `optimize.py` で **iter 上限と早期終了条件を CLI 化** します。
オリジナルはメイン最適化を `for k in range(1, 1001)`、iris 最適化を
`for k in range(1, 501)` でハードコードしており、loss が plateau に入っても
固定回数まで走り切ります。

新しい CLI フラグ:

| フラグ | 既定 | 意味 |
|---|---|---|
| `--max_iters` | `1000` | メイン最適化ループの iter 上限。**既定値はオリジナル踏襲**。 |
| `--max_iris_iters` | `500` | iris 最適化ループの iter 上限。同上。 |
| `--early_stop_rel_tol` | `0.0` | `landmark_loss` の相対改善 tolerance（100 iter ごとに判定）。`0.0` で早期終了無効。`0.005`〜`0.01` を opt-in 時の典型値とする。 |
| `--early_stop_patience` | `2` | tolerance 未達の 100-iter ウィンドウが連続して何個続いたら break するか。`early_stop_rel_tol > 0` のときのみ有効。 |

#### なぜ必要か

ユーザの典型的な訓練ログを観察すると:

- **iris ループ**: iter 300 で `landmark_loss=0.03189`、iter 400 で `0.03188`、
  iter 500 で `0.03190`。300 iter 以降は実質 4桁目以下の noise で **完全 plateau**。
  最後の 200 iter は純粋に時間の無駄。
- **メインループ**: iter 900→1000 で 0.0596→0.0556（~7% 改善）。
  改善は緩慢ながらまだ続いていることが多く、**動画依存**で plateau タイミングは
  読みにくい。

固定 iter は上記の前者に対して特に過剰。動画長 N に対して計算量は
`O(N × iter)` の線形なので、無駄な iter を切るとそのまま秒数が削れます。

#### 推奨設定

| 用途 | flag 例 | コメント |
|---|---|---|
| デフォルト（変更なし） | flag を渡さない | 旧挙動。iters=1000/500 完走 |
| iris のみ短縮（安全） | `--early-stop-rel-tol 0.005 --early-stop-patience 2` | iris は plateau しやすいので opt-in しても劣化リスク小。main は完走 |
| 両方短縮（積極的） | `--early-stop-rel-tol 0.01 --early-stop-patience 3` | main も plateau になったら break。改善が緩慢な動画で効果大 |
| iter 上限を直接下げる | `--max-iters 700 --max-iris-iters 300` | 早期終了より管理が容易だが、品質確認は必須 |

ユーザログから推定した時短スケール（参考）：

- iris 500 → ~300 iter（plateau 検出）≒ -25 秒
- main 1000 → ~700 iter ≒ -40 秒

#### 検出ロジック

`landmark_loss` を 100 iter ごとに サンプル（既存ロギング cadence と同じ）し、
`best_loss × (1 - early_stop_rel_tol)` を下回る改善があればカウンタを
リセット、無ければカウンタを進めます。`patience` ウィンドウ連続で改善が
無ければ `[early-stop] ... plateau at iter=K (best=..., cur=...)` を出力して
`break` します。判定は GPU sync を伴いますが、もともと `avg_lmk_loss +=
landmark_loss2.item()` を毎 iter 行っているので追加コストは事実上ゼロです。

### 4. `apply_deca_optimize_lr.py`（新規）

DECA フォークの `optimize.py` で **メインループの初期 lr と step decay を
CLI 化** します。オリジナルは Adam `lr=1e-2` をハードコードし、step decay
は **コメントアウト** されたまま:

```python
# if k%300==0:
#     lr_opt/=2
#     for param_group in opt_p.param_groups:
#         param_group['lr'] = lr_opt
```

オリジナルのコメント形式は **全 param group の lr を一律に上書き** する
ため、`eyelid` (1e-3) / `translation` (1e-4) / `translation_p` (1e-2) の
個別 lr が消え、これがコメントアウトされた理由と推測されます。

新しい CLI フラグ:

| フラグ | 既定 | 意味 |
|---|---|---|
| `--main_lr` | `1e-2` | pose / exp / shape の初期 Adam lr。**既定値はオリジナル踏襲**。eyelid / translation / translation_p の lr は触らない |
| `--main_lr_decay_step` | `0` | `>0` で N iter ごとに全 param group の lr を `--main_lr_decay_factor` 倍する。0 で無効。**乗算なので per-group 比率を保つ**（オリジナルコメント形式の問題を回避） |
| `--main_lr_decay_factor` | `0.5` | decay 倍率 |

#### なぜ必要か

ユーザログの相転移特性（iter 600-700 で 0.152 → 0.099、~35% 改善）から
推測される問題:

- 初期 lr=1e-2 は相転移到達まで 600 iter かかる → **lr を上げれば前倒しに
  なる可能性**
- 相転移後（iter 700 以降）の細かい fine-tune では lr=1e-2 が大きすぎ
  振動し、改善が緩慢になる → **decay で lr を絞れば収束加速の可能性**

#### 推奨設定

| 用途 | flag 例 | 期待効果 | リスク |
|---|---|---|---|
| デフォルト | flag を渡さない | 既存挙動 | なし |
| **lr 軽め昇圧** | `--main-lr 1.5e-2` | 相転移までの iter 数を 1〜2 割削減 | 振動、収束精度低下 |
| **step decay 単体** | `--main-lr-decay-step 700 --main-lr-decay-factor 0.3` | 相転移後（iter 700）に lr×0.3、後段 fine-tune 加速 | decay 早すぎると相転移直前で停滞 |
| **昇圧 + decay 併用**（推奨） | `--main-lr 1.5e-2 --main-lr-decay-step 500 --main-lr-decay-factor 0.5` | 相転移を前倒し → 中盤で減衰し精密化 | flag 多い、要試行錯誤 |

`--main-lr` を上げる場合、`2e-2` を超えると Adam でも振動が観察されやすく
なります。最初は `1.5e-2` から試すのが安全です。

#### 効果の見方

`optimize_vis.jpg` の右端「shape mesh」が崩れていない（FLAME identity を
保っている）こと、最終 `landmark_loss` がデフォルト走と同水準（±10%）に
収まっていること、を確認してください。loss が下がっても mesh が崩れたら
shape regularizer の方を上げる（`--lambda-shape 1.0` など）必要があります。

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

### iris の早期終了を opt-in（時短例）

```bash
bash demos/demo_1_train_subject.sh \
    --sbj-root ./data/subjects --sbj-name MK6c \
    --video ./data/raw/mikawa6c.mp4 --intrinsics hdtf \
    --early-stop-rel-tol 0.005 --early-stop-patience 2
```

### main loop の lr を上げて相転移を前倒し + decay で fine-tune 加速

```bash
bash demos/demo_1_train_subject.sh \
    --sbj-root ./data/subjects --sbj-name MK6c \
    --video ./data/raw/mikawa6c.mp4 --intrinsics hdtf \
    --main-lr 1.5e-2 --main-lr-decay-step 500 --main-lr-decay-factor 0.5 \
    --early-stop-rel-tol 0.01 --early-stop-patience 2
```

### パッチを手動管理したい

```bash
bash demos/demo_1_train_subject.sh ... --skip-deca-patches
# その後、自分で
python tools/patches/apply_deca_stable_bbox.py
python tools/patches/apply_deca_optimize_regularizer.py
python tools/patches/apply_deca_optimize_iters.py
python tools/patches/apply_deca_optimize_lr.py
```

## 単独実行

```bash
# パッチ単独適用（冪等。複数回 OK）
python tools/patches/apply_deca_stable_bbox.py
python tools/patches/apply_deca_optimize_regularizer.py
python tools/patches/apply_deca_optimize_iters.py
python tools/patches/apply_deca_optimize_lr.py

# 別 DECA チェックアウトに対して
python tools/patches/apply_deca_optimize_regularizer.py /path/to/DECA
python tools/patches/apply_deca_optimize_iters.py /path/to/DECA
python tools/patches/apply_deca_optimize_lr.py /path/to/DECA
```

## マーカ

各パッチは識別マーカをコメントで埋め込み、再適用時に検出します。

| パッチ | マーカ |
|---|---|
| `apply_deca_stable_bbox.py` | `# HRAVATAR_STABLE_BBOX` |
| `apply_deca_optimize_regularizer.py` | `# HRAVATAR_OPTIMIZE_REGULARIZER` |
| `apply_deca_optimize_iters.py` | `# HRAVATAR_OPTIMIZE_ITERS` |
| `apply_deca_optimize_lr.py` | `# HRAVATAR_OPTIMIZE_LR` |

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
- `tools/patches/apply_deca_optimize_iters.py`
- `tools/patches/apply_deca_optimize_lr.py`
