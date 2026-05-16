# LHG オンライン経路 FLAME メッシュずれ — 原因分析

> 作成: 2026-05-16 / 改訂: 2026-05-16(ユーザー回答反映) /
> ブランチ `claude/analyze-gpu-issues-exDQA`
> 本書はコード変更を含まない **原因候補の分析メモ** です。実行環境に
> CUDA/GPU が無くレンダリング検証ができないため、`lhg/` および
> `doc/` のコード読解だけに基づいて推論しています。各候補には
> 確度（HIGH / MEDIUM / LOW）を付けています。HIGH でも「コードから
> 読み取れる構造的不整合」であって、実測で確定したものではありません。

---

## 0. 環境制約

| 項目 | 状態 |
|---|---|
| `nvidia-smi` | コマンド無し（GPU 非搭載コンテナ） |
| `python -c "import torch"` | `ModuleNotFoundError`（torch 未インストール） |

したがって `extract` / `render` / `overlay_online_features.py` は実行
不能。会話ログの overlay 動画も再生成できません。本書は静的解析のみ。

---

## 1. グランドデザインの理解（2026-05-16 ユーザー補足反映、要確認）

`doc/preprocessing_scope.md` / `lhg/README.md` / 各 `*.py` の docstring・
コミットメッセージに、**ユーザー補足（2026-05-16）**を加えて再構成した
理解。ユーザーのコメント待ち。

### 1.1 グランドデザイン＝LHG モデルにおける HRAvatar 利用のビジョン

LHG（Listening Head Generation）モデルは、話し手に対する**聞き手の頭部
動作を生成**する。モデルの入力は 2 系統:

| 入力 | 性質 | 必要な符号化 |
|---|---|---|
| **話し手（speaker）の頭部動作** | ライブ。リアルタイムに取得が必須 | **オンライン経路**。厳密 causal・lookahead=0 が原則 |
| **聞き手（listener）の頭部動作** | モデルが聞き手について**自己回帰**するためリアルタイム取得は不要 | **オフライン経路**。最大限最適化したデータを使える |

HRAvatar は renderer とアバター個人 fit を提供し、LHG モデル出力（FLAME
パラメータ）が HRAvatar アバターを駆動して聞き手頭部を描画する。

### 1.2 二つの符号化経路（用語の整理）

| 経路 | 実体 | アルゴリズム |
|---|---|---|
| **オフライン経路** | Stage 1 DECA `optimize.py` clip-wide joint fit → `lhg/teacher.py` で `lhg_features.npz` 化 | clip 全体・双方向・未来可視を使い切る。**ゴールドスタンダード** |
| **オンライン経路** | Stage 2（`extract` の `--mode online`） | detector → stable bbox → SMIRK → EPnP → causal Hampel → symmetric FIR LPF |

> **`pseudo-online`（README で言う Stage 3）の扱いは不明確。** ユーザーに
> よれば現行で使うのは「オフライン経路」と「オンライン経路」の 2 つで、
> `pseudo-online` は古いコードの可能性があり立場が曖昧。本書は
> `pseudo-online` を分析対象から外し、オフライン経路＝
> `Stage 1 + lhg/teacher.py` として扱う。
> （本書初版は `pseudo-online` を「teacher 生成」と誤同定していた — 訂正済み。）

### 1.3 中核テーゼ — オンラインはオフラインの「制約付き複製」

> **オンライン経路はオフライン経路と可能な限り同一にする。同一でないのは、
> オンラインでアルゴリズム的に実装不可能な部分のみ。**

理由: ① 話し手と聞き手の頭部動作は同一スキーマ・同一分布に乗らねば
モデルが一貫処理できない。② 学習時は話し手も動画からオフライン符号化
されるため、「学習時オフライン符号化の話し手」と「推論時オンライン
符号化の話し手」が一致しないと **train/test 分布ミスマッチ**になる。

online/offline の差として **「許される差」は 2 カテゴリ**ある
（ユーザー補足 2026-05-16 で明確化）:

**カテゴリ A — アルゴリズム的に不可能:**

- clip-wide joint Adam 最適化（未来全体を見る）→ 不可能
- 双方向フィルタ（未来フレーム参照）→ 不可能
- per-frame Adam 最適化（リアルタイム予算超過）→ 不可能 ⇒ EPnP（1-shot
  解析解）で代替
- 任意 lookahead → 予算が許す範囲のみ

**カテゴリ B — 意図的に取得要件を緩和したパラメータ:**

実環境利用時に **カメラとユーザの位置関係を調整・キャリブレーション
できる**場合や、**事前取得データを使い回せる**場合、無理にオンラインで
取得すべきでないパラメータが存在する。それらは online で per-frame 推定
**せず**、Stage 1 キャリブレーション／clip 定数／事前取得値を使う。

> **重要**: カテゴリ B は「online でその値を出さなくてよい」という意味
> ではなく「**online でその値を per-frame 推定しなくてよい — 代わりに
> 較正済み clip 定数を使え**」という意味。**ゼロを出すのは誤り**。
> 緩和の正しい実装は「Stage 1 / 事前取得の clip 定数を焼き込む」こと。

**この 2 カテゴリ以外の online/offline の差はすべて「不当な乖離
（＝バグ／設計違反）」**。本書 §3 の候補①②はこの「不当な乖離」、
あるいは「カテゴリ B の緩和をゼロ出力で誤実装したもの」に該当する（後述）。

### 1.4 派生する設計上の取り決め

- per-frame EPnP は clip-wide joint fit の精度を再現できない（約 10× 差）。
  → 絶対 pose は Stage 1（clip 定数）、per-frame delta は Stage 2 が担当。
  per-frame `global_rot` / `translation` は `world_mat` まわりの小 delta。
- **clip 定数を最大化し、per-frame オンライン推定を最小化する。**
- 抽出時の平滑化は必ず streaming 等価物を持つ
  （`apply_offline_zero_phase` ⇔ `StreamingSymmetricFIR`、bit-exact）。
- **teacher = renderer が実消費する値**。teacher の expression/jaw/eyelid は
  アバター自身の学習済み SMIRK を 224 crop に掛けた値。online も同じ
  SMIRK 重みを共有する。
- expression/eyelid は LPF 対象外（lip-sync 保護）。jaw は opt-in。LPF は
  `global_rot` / `translation` が基本対象。

### 1.5 未確定・注意点

1. **lookahead**: 話し手は原則 lookahead=0 のはずだが、実装の online
   デフォルトは `lpf_lookahead=4`（160ms、`lhg/config.py:110`）。
   「lookahead>0 の将来余地」はパラメータとして実装に反映されているが、
   デフォルト 4 は lookahead=0 ではない。会話ログの抽出が L=0 か L=4 かは
   不明。
2. **ゴールドスタンダード自体の品質**: 設計は「オフライン＝正解」を前提に
   するが、ユーザー観察（FAN jitter、大ヨー overshoot、本書 §4・§5）は
   オフライン経路自身に欠陥がある可能性を示す。online がオフラインより
   良い側面があるなら、それは「online がオフラインに一致していない（乖離）」
   と「正解側が不完全」の両方を意味する。**online を正解に合わせる前に
   正解側（オフライン経路）の品質改善も課題**になり得る。

---

## 2. 現状の問題

会話ログ末尾の到達点と、ユーザー確認（2026-05-16）:

- online EPnP の `translation` に systematic bias（teacher 比で
  y ≈ −8.8cm, z ≈ −23.6cm、全フレーム同方向）。
- 候補 A（clip 定数 offset 補正）で systematic bias の**数値**は完全ゼロ、
  候補 C（expression-aware landmark）で jitter がやや減、A+C が数値上最良。
- **しかしユーザー確認では、A / A+C 適用後も FLAME メッシュと顔のずれが
  視覚的に解消していない。**

### 2.1 ユーザーによる「ずれ」の性質確認（決定的）

| 質問 | 回答 |
|---|---|
| ずれは一定 or pose 依存か | **全フレーム一定方向・一定量** |
| ずれの種類 | **「顔から数センチ浮いているお面のような状態」**＝平行移動的な浮き、傾き（回転）は目立たない |
| 大ヨー時の FAN 赤点の様子 | 記憶になく現在確認不可 |

**この 2 つの回答が原因を強く絞り込む。** 候補 A は「定数 offset を引く」
処理なので、**定数の浮きが A 後も残るなら、原因は『A が触れていない
別の定数チャンネル』**でなければならない。A が触るのは `translation`
チャンネルのみ。よって原因は `translation` 以外の定数成分。さらに
「傾きが目立たない」ことは、原因が**頭ジョイント近傍まわりの回転
（＝`global_rot`）ではない**ことを示唆する（頭ジョイントまわりの回転は
顔面そのものを傾けるため）。

---

## 3. 問題A: オンライン経路のメッシュずれが A 適用後も残る

### 3.1 原因候補（確度順、ユーザー回答反映後）

#### 候補① online が `neck_pose` を **ゼロ**で出力している 【HIGH — 最有力】

`lhg/pipeline.py:843-850`:

```python
# Stage 2 online backends (epnp / deca_encoder) do not currently
# estimate neck or eye pose ...
neck_pose = np.zeros((n_total, 3), dtype=np.float32)
eye_pose  = np.zeros((n_total, 6), dtype=np.float32)
```

**online 経路は `neck_pose` を全フレーム ゼロで出力する。** 一方
teacher（`lhg/teacher.py:338-348`）は DECA `fullposecode[3:6]` の
**実際の neck 回転**を格納する。

renderer も overlay も、この neck を使う:
- overlay: `scripts/overlay_online_features.py:296` →
  `full[3:6] = features.neck_pose[i]` を `pose_flame` に渡す。
- 実 renderer: `lhg/render_adapter.py:122` →
  `fullposecode[3:6] = features.neck_pose[i]`。

`pose_flame`（`demos/demo_3_overlay_tracking.py:165-181`）は neck を
FLAME キネマティックツリーの**首ジョイント**まわりの回転として適用し、
LBS で頭部全体に伝播させる。

**帰結 — これが「お面が数cm浮く・傾かない・一定量」を一発で説明する:**

- 被写体は通常、首を厳密に canonical ゼロ姿勢では保持しない（軽い前傾・
  あおむき等）。teacher はその**平均 neck 姿勢**を焼き込み、online は
  ゼロにする。両者の頭部位置は定数だけ食い違う。
- 首ジョイントは顔より十数〜数十 cm **下・後方**にある。そこを支点に
  数度回しても、**顔領域（小さな領域）ではほぼ平行移動**として現れ、
  顔面の傾きはほとんど知覚されない → **「お面のように浮く」**。
- neck の平均姿勢は clip 内でほぼ一定 → **「全フレーム一定方向・一定量」**。
- 候補 A は `translation` チャンネルしか補正しない → **neck 由来の浮きには
  まったく無効** → 「A で数値ゼロ・見た目ずれたまま」と完全に整合。

これは overlay の検証アーティファクトであると同時に、**実 renderer 出力
でも同じ浮きが出る実体的なパイプライン欠落**である（`render_adapter` も
neck を使うため）。

> グランドデザイン整合: §1.3 の「clip 定数を最大化」テーゼに照らすと、
> neck 姿勢は本来 **Stage 1 由来の clip 定数（clip-mean neck）**として
> online に焼き込むべきもの。`pipeline.py` のコメント自身も「those are
> clip-baseline or supplied by a different module」と書いている。
> **ゼロを出すのではなく clip-baseline neck を出す**のが設計意図に沿う
> 修正方向。

#### 候補② `_recenter_against_world_mat` が translation を回転していない 【MEDIUM】

`lhg/pipeline.py:489-499`:

```python
R_delta = R_ref.T @ R_total          # rotation は world_mat 系へ回す
...
trans_recentered = translations - t_ref_canonical   # translation は単純減算のみ
```

renderer の合成（`pose_flame`：translation を **FLAME 標準空間**で頂点に
加算）を展開すると、正しい変換は

```
translation = R_ref^T @ (t_total − t_ref)
```

のはずだが、現コードは `R_ref^T` を掛けていない。EPnP の `tvec` は
**カメラ空間**の並進なので、online の translation はカメラ系 delta、
teacher は FLAME 系 delta となり `R_ref` 回転だけ食い違う。

ただし**ユーザー回答「ずれは一定」を踏まえると、これは A 後の残差の
主因ではない**。この不整合が生む残差は
`(I − R_ref^T) @ (d − mean(d))`（`d` は per-frame translation、平均除去
済み）で、**pose 依存の変動成分**であって定数ではないため。頭の並進変動が
小さければ残差も小さい。**潜在的な正しさの問題としては残る**が、観測
されている定数の浮きの説明にはならない。

#### 候補③ EPnP の深度条件数（中心寄り・準平面な 16 点アンカー）【MEDIUM】

EPnP は中央 16 点（眉・目・鼻、`correspondence.mp_indices`）だけで
`cv2.solvePnP` を解く。準平面な点群の PnP は深度（z）が弱条件で、小さな
2D ノイズが大きな z 誤差に増幅される（実測 bias 支配項が z だったことと
整合）。

ただし**この成分は `translation` チャンネルに乗る**ため、定数成分は
候補 A がほぼ除去できる。残るのは pose 依存の z 変動。よって「一定量の
浮き」の主因ではないが、**A 後にわずかに残る pose 依存のゆらぎ**として
寄与している可能性がある。

#### 候補④ rotation チャンネルの bias 【LOW（今回の症状に対して）】

会話ログは「rotation の bias/jitter は A/B/C 未改善」とする。ただし
`global_rot` は**頭ジョイントまわり**の回転で、定数 bias があれば顔面
そのものが**傾いて**見えるはず。ユーザー回答「傾きは目立たない」は
**rotation 定数 bias が主因ではない**ことを示す。jitter 成分としては
別途残る（問題 B 参照）。

#### 候補⑤ EPnP テンプレートと実アバター形状の不一致 【LOW-MEDIUM】

EPnP 物体点は FLAME 形状テンプレート（Stage 1 `shapecode`）由来。実顔と
テンプレートのサイズ差は深度バイアスを生むが、概ね clip 定数なので
`translation` 経由で候補 A に吸収される。残差は小。

#### 候補⑥ overlay ツールの投影 【除外】

teacher(01) は同じ overlay で顔に整合するので投影系は正しい。online の
ずれは実際の特徴量値の差。**ただし候補① の通り、overlay が
`neck_pose` を使う点は「online npz をそのまま描くと teacher と
非対称比較になる」という検証手法上の落とし穴**であることに注意。

### 3.2 まとめ（問題 A）

ユーザー回答（一定・無傾き・お面浮き）を反映した確度順:

1. **候補①（online の `neck_pose`=0）— 最有力**。定数・無傾き・A 無効を
   すべて説明。コード位置 `lhg/pipeline.py:849`。
2. 候補②（translation の `R_ref^T` 欠落）— 潜在バグだが今回の定数浮きの
   主因ではない（pose 依存残差を生む）。
3. 候補③（EPnP 深度条件数）— A 後の pose 依存ゆらぎとして寄与し得る。
4. 候補④（rotation bias）— 傾きが無い以上、主因ではない。

会話ログの候補 A〜C の位置づけ:

| 会話ログ候補 | 効く対象 | 本分析での評価 |
|---|---|---|
| A: clip 定数 offset 補正 | translation の clip 平均のみ | neck 由来の浮き（候補①）には**原理的に無効**。translation 内の定数 bias は消すが症状は残る |
| B: DECA encoder backend | translation+rotation を別系統に | neck は B でも未推定（同じく zeros）。候補① は B でも残る |
| C: expression-aware landmark | per-frame jitter | 候補①②③④いずれにも無効 |

**＝ A/B/C はいずれも候補①（neck=0）を直さない。** これが「全モデルで
ずれが解消しない」ことの最有力説明。

### 3.3 グランドデザイン上の位置づけ（2 カテゴリ・モデルで再評価）

§1.3 の「許される差は カテゴリ A（アルゴリズム的に不可能）／カテゴリ B
（意図的に緩和）の 2 つだけ」に照らすと:

- **候補①（neck=0）**: neck は **カテゴリ B（緩和対象）**である公算が高い
  — `pipeline.py:843` のコメント自身が「clip-baseline or supplied by a
  different module」と書く。だが §1.3 の重要注記の通り、**緩和の正しい
  実装は「Stage 1 / 事前取得の clip 定数 neck を焼き込む」ことであって、
  ゼロを出すことではない**。現コードはゼロを出している → **カテゴリ B の
  緩和をゼロ出力で誤実装したもの**。修正は任意改善ではなく設計遵守の
  必須要件で、修正方向は「clip-baseline neck を焼き込む」。
- **候補②（`R_ref^T` 欠落）**: 単なる座標変換の実装ミス。カテゴリ A にも
  B にも当たらない → **純然たる不当な乖離＝グランドデザイン違反**。

**候補③（EPnP 深度条件数）の再評価 — 重要**: 深度（translation z）は
**カテゴリ B の有力候補**。ユーザー補足「カメラとユーザの位置関係を
調整可能」はまさに**カメラ・ユーザ間距離（≒絶対深度）が較正可能**で
あることを意味する。だとすれば:

> 会話ログの候補 A／B／C／cam→z proxy は **「online で深度を正確に
> 推定する」努力**だが、深度がカテゴリ B なら、この努力は設計趣旨から
> 見て**部分的に方向違い**である。設計に沿った答えは「**深度を online で
> EPnP 推定しない — Stage 1 較正の clip 定数深度を使う**」。EPnP の準平面
> 16 点による深度の悪条件（§3.1 候補③）と systematic bias は、本来
> 緩和すべきパラメータを無理に online 推定したことの症状とも解釈できる。

ただし「深度が完全に clip 定数でよいか／ユーザの前後移動を live で取る
必要があるか」は設計判断であり、ユーザー確認が必要（§7.1）。

### 3.4 ユーザー追加観察（2026-05-16）と再評価

ユーザー観察:
- 可視化メッシュは顔のヨー回転に追従して向きを正しく変える（確認済み）。
- 「数センチ浮いたお面」は、回転軸を顔と共有しつつメッシュが浮くため、
  より大きい半径で動いて見える（「厳密未確認、そう見える」との見立て）。

含意:

1. **向き（`global_rot`）はほぼ正しい。** メッシュがヨーに追従する＝
   EPnP の回転推定は機能。問題は純粋に**位置**。候補④（rotation bias）は
   本症状の主因でないことがさらに裏付けられる。
2. **「より大きい半径」は 2D 投影効果として最も自然に説明できる。**
   メッシュがカメラに数 cm 近く描画されると投影スケールが増し、同じヨー角
   でも像面上の掃引が大きくなる（pixels-per-cm 増）。＝核心的事実は
   「**メッシュが実顔より一定量カメラ寄りに描画されている**」。
3. **消去法**: A 後も残る・無傾き・一定 の位置誤差は、(a) A が触れない
   チャンネルで、かつ (b) `global_rot` ではない（傾くため）もの。該当は
   `neck_pose`（online=0, teacher≠0）のみ → 候補① を消去法的に支持。
4. **正直な留保**: ただし neck=0 が「カメラ寄りの浮き」を生むという符号
   までは FLAME を実行せずに確証できない（首の習慣姿勢と関節幾何に依存。
   純ヨーでは neck/head 関節が縦軸をほぼ共有するため neck=0 は半径を
   ほとんど変えず、neck は主にピッチに効く）。本症状が純粋にヨー連動なら
   neck=0 単独では不足で、候補②（translation 座標系）・③（深度）が
   併存している可能性が高い。

**判別テスト（GPU 復帰時、1 観察で切り分け可能）:**
- 大きくヨーした（〜45°）フレームで、メッシュの浮きの向きは
  「**カメラ方向**（カメラ系一定）」か「**鼻の前方向**（頭部系一定）」か。
  - カメラ系 → 候補②（translation 座標系不整合）・③（EPnP 深度）。
  - 頭部系 → 候補①（neck）または pose の pivot 誤り。
- あわせて、A の offset 計算（`lhg/pipeline.py:826-841` が
  `tracked_params.json` の全画像キーを読む）が、抽出クリップとフレーム数・
  範囲が一致しているか確認。不一致なら A の平均がずれ、定数残差を残す
  （＝「A で数値ゼロのはずが見た目ずれたまま」の別経路の説明になり得る）。

---

## 4. 問題B: オフライン経路の方が jitter が多い

ユーザー観察「オンラインの方が jitter が小さい」「FAN の jitter が
オフライン jitter につながっていないか」は、コード構造から**妥当**です。

### 4.1 処理の非対称性（最大の要因）【HIGH】

- **online**（`lhg/pipeline.py:802-808`）は `global_rot` / `translation` に
  **明示的な symmetric FIR LPF**（cutoff 4Hz, lookahead 4）を掛ける。
- **teacher**（`lhg/teacher.py`）は `global_rot` / `translation` を
  DECA `tracked_params.json` から**生コピー**し **LPF を一切掛けない**。
  teacher の平滑化は DECA optimize 内部の時間正則化のみ。

→ online には設計された低域通過があり、teacher には無い。teacher の
平滑度は loss 重み次第で、控えめなら FAN の per-frame jitter が teacher に
残る。**「online < teacher の jitter」は設計上の非対称性の必然。**

### 4.2 検出器の差【HIGH】

- online は **MediaPipe FaceLandmarker の video running mode**。内部
  Kalman tracker で時間方向に安定化（`lhg/config.py:62-67`）。
- offline（DECA optimize）は **FAN 68 点**。per-frame 推論で時間モデル無し。
- `lhg/config.py:55-61` のコメント自身が「FAN が低 jitter という歴史的
  主張は seed-once-bbox 構成でしか成り立たない」と認めている。

→ ユーザーの仮説「FAN の jitter → オフライン jitter」は妥当。

> 注意: teacher(01) の expression/jaw/eyelid は FAN 由来ではなく
> avatar-SMIRK 由来。FAN jitter が効くのは teacher の
> **global_rot / translation / neck / eye** チャンネルのみ。

---

## 5. 問題C: オフライン経路が大ヨーで過剰にずれる

ユーザー観察「正面から大きく外れるとオフラインは顔がさらに大きく逸れ、
欠損区間の 2 次補間オーバーシュートに酷似」。原因候補:

#### 候補C-1 FAN 輪郭点の silhouette スライド【HIGH】

FAN は顔向きに関わらず 68 点を必ず返す。顎・顔輪郭点はシルエット点で、
大ヨーでは**可視シルエット上を滑り**、隠れた側の点は事実上「捏造」される。
DECA optimize は global_rot/translation をこれら 68 点全部に fit する
ため、滑った輪郭に像を合わせようとして**回転を回しすぎる** →「顔がさらに
逸れて見える」。online EPnP は中央 16 点のみで輪郭・顎を使わないため、
このスライドの影響を受けない。

#### 候補C-2 弱データ区間で時間正則化が補間器として振る舞う【MEDIUM／要確認】

これがユーザーの「2 次補間オーバーシュート」アナロジーの核心。大ヨーで
FAN landmark 信頼度が落ちる → その区間の per-frame データ項が実効的に
弱まる → clip-wide joint 最適化では時間正則化項が支配 → 解は弱データ
区間を滑らかに繋ぐ「補間器」のように振る舞う。

決定的なのは**時間正則化の階数**:
- **1 階差分**（速度ペナルティ）なら過渡 excursion のピークを**減衰**させる
  （オーバーシュートしない）。
- **2 階差分**（加速度ペナルティ）なら最小曲率スプライン的に振る舞い、
  弱データ区間の端で**オーバーシュート**する — ユーザーのアナロジー通り。

`doc/preprocessing_scope.md` は「`pose[1:]−pose[:-1]`」と **1 階差分**と
記述。記述通りなら C-2 は単純なオーバーシュート機構としては成立しにくく、
**主因は C-1（輪郭スライド）**の可能性が高い。ただし DECA fork
（`preprocess/submodules/DECA`）は本環境に未チェックアウトで `optimize.py`
実物を確認できていない。**§7 要確認事項①**。

#### 候補C-3 clip-wide joint 最適化の境界・未来リーク【MEDIUM】

`doc/preprocessing_scope.md` 自身が「平滑化正則化が未来をリークする」
「最適化の最終フレーム状態が clip 長に依存する」と明記。過渡的な極端
pose では双方向結合（未来フレームからの引き込み）が excursion を歪め得る。
online は per-frame 独立 EPnP なのでフレーム間結合が無く、この種の
オーバーシュートは原理的に起きない。

### 5.1 統一的な描像

| 性質 | online EPnP（中央 16 点・per-frame 独立） | offline DECA（FAN 68 点・clip-wide joint） |
|---|---|---|
| 大ヨーでの輪郭スライド | 受けない → **頑健** | 受ける → 回転を回しすぎ |
| フレーム間結合のオーバーシュート | 無い → **頑健** | 有り得る |
| 深度（z）の安定性 | **弱い**（準平面 16 点）→ 問題 A 候補③ | 強い（前後に厚い 68 点） |
| jitter | 明示 FIR LPF + Kalman → **小** | LPF 無し + FAN per-frame → 大 |

→ **「中央アンカーのみ・per-frame 独立」は遮蔽と大ヨーに強いが深度に弱い。
「全 68 点・clip-wide joint」は深度に強いが大ヨーと jitter に弱い。**
online が C 問題で勝ち A 問題（深度・並進）で負けるのは同じトレードオフ
の表と裏。

---

## 6. まとめ — 推奨される次アクション（GPU 復帰後）

優先度順:

0. **設計判断の確定（最優先・コード変更前）**: どのパラメータがカテゴリ B
   （緩和対象）かを確定する（§7.1）。特に `neck` / `eye` / **translation z
   (深度)** の扱い。これが決まらないと 1・4 の修正方針が定まらない。
1. **候補①の修正**: online 経路の `neck_pose`（および `eye_pose`）の
   ゼロ出力（`lhg/pipeline.py:849`）を、**Stage 1 / 事前取得の clip 定数
   neck**（例: DECA optimize の per-frame neck の clip 平均）に置き換える。
   これがカテゴリ B 緩和の正しい実装。修正後、teacher と online の overlay
   を再比較し「お面浮き」が消えるか確認。
2. **検証手法の是正**: overlay 比較は teacher(neck≠0) vs online(neck=0) で
   非対称になっている。当面は online overlay を「teacher の neck を借りて」
   描くか、両者とも neck=0 で描くと、neck 以外の差（②③④）が切り分け
   られる。
3. **候補②の検証・修正**: `tracked_params.json` の `world_mat` 回転成分を
   測り、単位行列から数度以上ずれていれば `_recenter_against_world_mat`
   の translation を `R_ref^T @ (translations − t_ref)` に修正。
4. **候補③＝深度の扱い**: 深度がカテゴリ B なら、EPnP の深度を補強する
   のではなく **online で深度推定をやめ、Stage 1 較正の clip 定数深度を
   使う**（会話ログの A/B/C/cam→z proxy は不要になる）。深度を live で
   取る必要があると判断するなら、EPnP に前後に厚みのある点（顎・鼻先）を
   足すか別系統で安定化。**0 番の設計判断に従う。**
5. **問題C（§7 要確認①）**: `preprocess/submodules/DECA/optimize.py` を
   チェックアウトして時間正則化の階数を確認。

---

## 7. ユーザーへの確認事項

§2.1 で「ずれは一定・無傾き・お面浮き」「大ヨー時の FAN 点は記憶なし」を
確認済み。残りの未確定点:

1. **DECA `optimize.py` の時間正則化の階数** — `preprocess/submodules/DECA`
   は本環境に未チェックアウト。`optimize.py` の `total_loss` で pose/
   translation に掛かる時間項が **1 階差分** か **2 階差分（加速度）** か
   で候補 C-2 の成否が決まる。GPU 復帰時に確認をお願いします。
2. **teacher の `neck_pose` が実際に非ゼロか** — 候補① の前提は「teacher の
   neck 平均が無視できない大きさ」。`tracked_params.json` の数フレームの
   `fullposecode[3:6]` を見て、neck がほぼゼロか、数度オーダーかを教えて
   ください。ほぼゼロなら候補① は外れ、候補②③へ重心が移ります。
3. **大ヨー時のオフライン崩れ方（再観察可能になったら）** — overlay の
   赤点（FAN 68pt）が顔から外れるフレームで崩れるか、赤点は顔に乗って
   いるのにメッシュだけ逸れるか。前者なら検出器側（C-1）、後者は
   最適化側（C-2/C-3）。

### 7.1 グランドデザインの理解についての確認（§1 の妥当性）

§1 はユーザー補足（2026-05-16）を反映して再構成したが、以下が未確定:

4. **オフライン経路の実体** — 「Stage 1 DECA `optimize.py` ＋
   `lhg/teacher.py`」で合っているか。
5. **train/inference の対応** — 学習時は話し手チャンネルもオフライン
   符号化、推論時のみオンライン符号化、という対応で合っているか
   （＝ online が offline に一致すべき根拠）。
6. **`pseudo-online`（Stage 3）の現況** — 完全に未使用の旧コードか、
   将来有効化予定のものか。本書は分析対象から外している。
7. **話し手アバターと聞き手アバターの異同** — 同一アバターか別か
   （teacher の avatar-SMIRK 重みの扱いに影響）。
8. **カテゴリ B（緩和対象）パラメータの確定（最重要）** — どのチャンネルを
   「online で per-frame 推定せず Stage 1 / 事前取得の clip 定数を使う」
   と意図しているか。具体的に:
   - `neck_pose` / `eye_pose` はカテゴリ B か（＝ clip 定数 neck を焼き込む
     のが正しい実装か）。
   - **translation z（カメラ・ユーザ間の絶対深度）はカテゴリ B か。**
     カテゴリ B なら、会話ログの候補 A/B/C・cam→z proxy は設計趣旨に
     対して方向違いで、「較正 clip 定数の深度を使う」が正解になる。
     ユーザの前後移動を live で取りたい場合はカテゴリ B から外れる。
   - `world_mat` / `shapecode` / `intrinsics` は既にカテゴリ B（clip 定数）
     という理解で合っているか。
