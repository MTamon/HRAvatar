# LHG オンライン経路 FLAME メッシュずれ — 原因分析

> 作成: 2026-05-16 / ブランチ `claude/analyze-gpu-issues-exDQA`
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

## 1. グランドデザインの理解（要確認）

`doc/preprocessing_scope.md` / `lhg/README.md` / 各 `*.py` の docstring と
コミットメッセージ（"2026-05-14 / 2026-05-15 grand design"）から、本ブランチの
設計意図を以下のように理解しました。**誤解があればご指摘ください。**

### 1.1 二つの独立パイプライン

このリポジトリには出力フォーマットもアルゴリズム保証も異なる 2 つの
前処理系が同居している:

1. **HRAvatar アバター個人 fit**（1 被写体 1 回）— `tracked_params.json` +
   matting + albedo を生成。clip 全体の joint Adam 最適化。
2. **LHG (Listening Head Generation) 前処理** — LHG モデルが推論時に
   生成すべき per-frame FLAME パラメータ列を作る。

### 1.2 LHG 3-Stage 構成と責務分離

| Stage | 役割 | アルゴリズム |
|---|---|---|
| **Stage 1** | clip 定数（`world_mat` / `shapecode` / `intrinsics`）の確定 | DECA `optimize.py` の clip-wide joint Adam fit |
| **Stage 2** (`--mode online`) | per-frame FLAME 特徴量の **causal** 抽出 | detector → stable bbox → SMIRK → EPnP → causal Hampel → symmetric FIR LPF |
| **Stage 3** (`--mode pseudo-online`) | teacher データ生成 | Stage 2 core + 双方向 Hampel + dropout 補間 + quaternion-flip + 大 lookahead LPF |

### 1.3 設計の根本テーゼ

- **per-frame EPnP は clip-wide joint fit の精度を再現できない**（精度差は
  概ね 10×）。だから「無理に再現しない」。Stage 1 が**絶対 pose**を、
  Stage 2 が**per-frame delta**を担当する分業。Stage 2/3 は Stage 1 の
  per-frame 出力を**意図的に読まない**（clip 定数のみ読む）。
- per-frame `global_rot` / `translation` は `world_mat` まわりの
  **小さな delta** として出力する。LHG モデルには「絶対 pose」ではなく
  「運動」だけを学習させる。
- **train/inference の一致**: 抽出時に掛ける平滑化は必ず streaming 等価物を
  持つ（`apply_offline_zero_phase` ⇔ `StreamingSymmetricFIR`、bit-exact）。
- **teacher = renderer が実際に消費する値**。2026-05-14/15 の明確化で、
  teacher の expression/jaw/eyelid は **アバター自身の学習済み SMIRK** を
  224 crop に掛けた値とする（`scene/gaussian_head_model.py` が学習時に
  外部 expression を SMIRK 出力で上書きするため）。online と teacher は
  同じ SMIRK 重みを共有し、両者の差は「online 固有の制約（causality・
  clip-wide joint 最適化の不在）」のみであるべき。
- expression/eyelid は LPF 対象外（lip-sync を壊すため）。jaw は opt-in。
  LPF は `global_rot` / `translation` のみが基本対象。
- LHG モデルは「online 入力」と「teacher 目標」の**差**を学習で吸収する。
  この差は理想的には lookahead=0 で取りきれない jitter 程度に収まるべき。

> 要するに「**clip 定数を最大化し、per-frame オンライン推定を最小化する**」
> 「**teacher と online の唯一の差を『因果性』に局所化する**」が背骨。

---

## 2. 現状の問題（会話ログから）

会話ログ末尾の到達点:

- online EPnP の `translation` に **systematic bias**（teacher 比で
  y ≈ −8.8cm, z ≈ −23.6cm, world scale 換算、全フレーム同方向）。
- 候補 A（clip 定数 offset 補正）で systematic bias の**数値**は完全ゼロ、
  候補 C（expression-aware landmark）で jitter がやや減、A+C が数値上最良。
- **しかしユーザー確認では、オンライン経路の全モデル（A / A+C 含む）で
  FLAME メッシュと実際の顔のずれが視覚的に解消していない。**

つまり「**メトリクス（clip 平均）はゼロになったが、見た目のずれは残る**」。
これは「A が機構を直さず平均を打ち消しているだけ」であることの強い兆候です
（後述 3.1 候補③）。

加えてユーザーの観察:

- オンライン経路の方が **jitter が小さい**。
- オンライン経路の方が **顔の向きが正面から大きく外れた時のメッシュ精度が
  高い**。オフライン経路は大きな向き変化で「顔がさらに大きく逸れて見える」。
  これは欠損区間を多項式・2 次補間した時の**オーバーシュート**に酷似。

---

## 3. 問題A: オンライン経路のメッシュずれが A 適用後も残る

### 3.1 原因候補（確度順）

#### 候補① `_recenter_against_world_mat` が translation を回転させていない 【HIGH】

`lhg/pipeline.py:463-500` の `_recenter_against_world_mat`:

```python
R_ref = world_mat[:3, :3]
t_ref_canonical = world_mat[:3, 3] / flame_scale
...
R_delta = R_ref.T @ R_total          # ← rotation は world_mat 系へ回す
...
trans_recentered = translations - t_ref_canonical   # ← translation は単純減算のみ
```

**回転は `R_ref^T` で world_mat 相対系へ変換しているのに、translation は
単純減算しかしていない（`R_ref^T` を掛けていない）。** これは座標系の
不整合になり得ます。

renderer 側の合成規則を `demos/demo_3_overlay_tracking.py:143-186` の
`pose_flame`（docstring に「`scene.gaussian_head_model.GaussianHeadModel.lbs_v2`
の鏡像」と明記）で確認すると:

```python
if translation is not None:
    v_world = v_world + translation     # FLAME 標準空間で頂点に加算
return v_world * flame_scale
```

→ `translation` は **FLAME 標準空間のベクトル**として頂点に足され、その後に
`project()` 内で `world_mat`（カメラ変換）が掛かる。

一方 EPnP（`lhg/epnp.py:solve_epnp`）が返す `tvec` は **カメラ空間**での
物体並進。renderer の合成を展開すると:

```
p_cam = R_ref @ (R_pose @ v + translation) + t_ref
      = (R_ref @ R_pose) @ v + (R_ref @ translation + t_ref)
```

EPnP の `p_cam = R_total @ v + t_total` と係数比較すると:

```
R_pose      = R_ref^T @ R_total                 … 現コードは正しい
translation = R_ref^T @ (t_total − t_ref)       … 現コードは R_ref^T が欠落
```

**translation の正しい変換は `R_ref^T @ (t_total − t_ref)` のはずだが、
現コードは `t_total − t_ref`（カメラ系の delta のまま）を出力している。**

帰結:
- online の `translation` はカメラ系の delta、teacher の `translation`
  （DECA `translation_param` の生コピー）は FLAME 系の delta。両者は
  `R_ref` の回転だけ食い違う。
- 候補 A は「clip 平均の差」しか引かない。回転は並進ではないので、
  **A は平均を打ち消せても、回転で歪んだ per-frame の軌道形状は残す**。
  → 「数値ゼロなのに見た目はずれたまま」と完全に整合する。
- ずれの大きさは `R_ref` が単位行列からどれだけ離れているかに比例。
  `world_mat` の回転成分が小さければ影響は小さく、傾きがあれば顕著。

**確認方法（GPU 不要）**: Stage 1 の `tracked_params.json` の `world_mat`
左上 3×3 を見て、単位行列からの乖離（軸角換算で何度か）を測る。乖離が
数度以上なら本候補は有力。teacher 経路はこのバグの影響を受けない
（DECA は最初から FLAME 系で `translation_param` を fit するため）。

#### 候補② EPnP の深度条件数の悪さ（中心寄り・準平面な 16 点アンカー）【HIGH】

EPnP は `correspondence.mp_indices`（会話ログでは 16 点の MICA anchor、
眉・目・鼻）だけで `cv2.solvePnP(SOLVEPNP_EPNP)` を解く
（`lhg/pipeline.py:360`、`lhg/epnp.py`）。

この 16 点は顔中央に集中し、**ほぼ同一平面上**にある。準平面な点群に対する
PnP は **深度（z）と回転が弱く結合し条件数が悪い**古典的に知られた問題で、
小さな 2D ノイズが大きな z 誤差に増幅される。会話ログで支配的だった bias が
z（−23.6cm）だったことと整合する。

対して DECA optimize が使う FAN 68 点は顎・輪郭まで含み前後に厚みがあるため
深度が安定する。

重要なのは **この z 誤差は pose 依存（非定数）になり得る**こと。顔の向きが
変わると 16 点の見かけの平面性・配置が変わり z 誤差も変動する。
**候補 A の clip 定数 offset では pose 依存成分を除去できない。**

> 会話ログの AI は systematic bias を「16 点 vs 68 点の重心差」とだけ説明
> しているが、重心差は主に**像面内（x/y）**のずれを生む。実測の支配項が
> **z** だった事実は、重心差より**準平面 PnP の深度条件数**が主因である
> ことを示唆する。A の「平均を引く」は症状を隠すだけで機構を直していない。

#### 候補③ 候補 A は機構修正ではなく数値マスキング 【HIGH／②①の系】

`lhg/pipeline.py:826-841` の A は

```python
translation_offset = _deca_mean - _epnp_mean
translation += translation_offset
```

で **clip 平均だけ**を一致させる。per-frame の構造（①の回転歪み、②の
pose 依存 z 誤差）はそのまま残る。「A で数値ゼロ・見た目ずれたまま」は
A の定義上の必然。会話ログの A の評価（"systematic bias 完全ゼロ"）は
**clip 平均という 1 メトリクスがゼロ**という意味でしかなく、フレーム毎の
整合を保証しない。

#### 候補④ rotation チャンネルが未補正 【MEDIUM】

会話ログ自身が「rotation の jitter / bias は A/B/C のいずれも未改善」と
明記。深度 `world_mat[2,3] ≈ −5·flame_scale` の距離では、わずか数度の
回転誤差でもメッシュは像面上で大きく平行移動して見える
（「浮き」と区別がつかない）。online の `global_rot` は per-frame EPnP +
Hampel + LPF で、teacher の clip-wide joint fit とは生成過程が根本的に
異なる。translation を完全に直しても rotation 由来のずれは残る。

#### 候補⑤ EPnP テンプレートと実アバター形状の不一致 【MEDIUM】

EPnP の 3D 物体点は FLAME 形状テンプレート（Stage 1 `shapecode`）から
`build_shape_aware_landmarks` で作る（`lhg/correspondence.py:159`）。
baseline は expression-neutral テンプレート。C は SMIRK expression を
加味するが、**アバター学習後の実ジオメトリと FLAME 推定形状の差**は
どちらも補正しない。テンプレートが実顔より広い/狭いと EPnP は深度で
辻褄を合わせ、深度バイアスを生む。これは概ね clip 定数なので一部は A に
吸収されるが、expression 連動の残差は残る。

#### 候補⑥ overlay ツール自体の投影 【LOW（主因ではない）】

`scripts/overlay_online_features.py` の投影が間違っていれば teacher(01)も
ずれるはず。ユーザー確認で **01 は顔に概ね整合**しているので、overlay の
投影系は概ね正しく、online のずれは**実際の特徴量値の差**である。
→ overlay バグは主因から除外してよい（teacher と online で同一ツール）。

### 3.2 候補 A〜C と本分析の対応

| 会話ログの候補 | 効く対象 | 本分析での評価 |
|---|---|---|
| A: clip 定数 offset 補正 | translation の clip 平均 | 候補③ — 機構を直さず平均をマスク。①②の per-frame 成分は残る |
| B: DECA encoder backend | translation+rotation を別系統に | EPnP の①②を回避するが、cam→z proxy 自体のノイズが残る（log: z jitter 増） |
| C: expression-aware landmark | per-frame jitter | 候補⑤の expression 成分を一部改善。①②④には無効 |

**示唆**: 真の修正は A の「平均引き」ではなく、候補①（`_recenter_against_world_mat`
の translation に `R_ref^T` を掛ける）と候補②（EPnP に深度を安定させる点を
足す、もしくは深度を別系統で推定する）に踏み込む必要がある。

---

## 4. 問題B: オフライン経路の方が jitter が多い

ユーザー観察「オンラインの方が jitter が小さい」「FAN の jitter が
オフライン jitter につながっていないか」は、コード構造から**妥当**です。

### 4.1 処理の非対称性（最大の要因）【HIGH】

- **online**（`lhg/pipeline.py:802-808`）は `global_rot` / `translation` に
  **明示的な symmetric FIR LPF**（`apply_offline_zero_phase`、cutoff 4Hz、
  lookahead 4）を掛ける。
- **teacher**（`lhg/teacher.py`）は `global_rot` / `translation` を
  DECA `tracked_params.json` から**生コピー**するだけで **LPF を一切
  掛けない**。teacher の平滑化は DECA optimize 内部の時間正則化
  （`pose[1:]−pose[:-1]` 等）のみ。

→ online には設計された低域通過があり、teacher には無い。teacher の
平滑度は loss 重み次第で、その重みが控えめなら FAN の per-frame jitter が
そのまま teacher に残る。**「online < teacher の jitter」は設計上の
非対称性の必然であって、謎ではない。**

### 4.2 検出器の差【HIGH】

- online は **MediaPipe FaceLandmarker の video running mode**。内部 Kalman
  tracker で時間方向に安定化（`lhg/config.py:62-67`）。
- offline（DECA optimize）は **FAN 68 点**。per-frame 推論で時間モデル無し。
- `lhg/config.py:55-61` のコメント自身が「FAN が 5–10× 低 jitter という
  歴史的主張は seed-once-bbox 構成でしか成り立たず、被写体が動くと破綻する」
  と認めている。

→ ユーザーの仮説「FAN の jitter がオフライン jitter につながる」は妥当。

> 注意: teacher(01) の expression/jaw/eyelid は FAN 由来ではなく
> avatar-SMIRK 由来（`lhg/teacher.py:_recompute_..._via_smirk`）。FAN jitter が
> 効くのは **teacher の global_rot / translation / neck / eye** チャンネルのみ。

---

## 5. 問題C: オフライン経路が大きな顔向きで過剰にずれる

ユーザー観察「正面から大きく外れるとオフラインは顔がさらに大きく逸れて
見える／欠損区間の 2 次補間オーバーシュートに酷似」。原因候補:

#### 候補C-1 FAN 輪郭点の silhouette スライド【HIGH】

FAN は顔向きに関わらず 68 点を**必ず**返す。顎・顔輪郭点（17–26 等）は
シルエット点で、大きなヨー回転では**可視シルエット上を滑り**、隠れた側の
点は事実上「捏造」される。DECA optimize は global_rot / translation を
これら 68 点全部に fit するため、滑った輪郭に像を合わせようとして
**回転を回しすぎる** → 「顔がさらに大きく逸れて見える」。

online EPnP は中央 16 点（眉・目・鼻）のみで輪郭・顎を使わないため、
このスライドの影響を受けない。**「中央アンカーのみ」は遮蔽・大ヨーに
頑健**（C 問題で online が勝つ理由）だが、同じ性質が**深度を不安定に
する**（問題 A 候補② の z 誤差）。トレードオフが表裏一体。

#### 候補C-2 弱データ区間で時間正則化が補間器として振る舞う【MEDIUM／要確認】

これがユーザーの「2 次補間オーバーシュート」アナロジーの核心。

大きな顔向きでは FAN landmark の信頼度が落ちる → その区間で **per-frame の
landmark データ項が実効的に弱くなる** → clip-wide joint 最適化では時間
正則化項がその区間を支配 → 解は「弱データ区間を滑らかに繋ぐ補間器」の
ように振る舞う。

ここで決定的なのは**時間正則化の階数**:
- **1 階差分**（速度ペナルティ `pose[1:]−pose[:-1]`）なら、過渡的な
  excursion の**ピークを減衰**させる（オーバーシュートしない）。
- **2 階差分**（加速度ペナルティ）なら、最小曲率スプライン的に振る舞い、
  弱データ区間の端で**オーバーシュート**する — ユーザーのアナロジー通り。

`doc/deca_patches.md` に載る `total_loss` は `landmark_loss2 + shape*λ +
exp*λ` の 1 行のみで、時間正則化の式は見えない。`doc/preprocessing_scope.md`
は「`exp[1:]−exp[:-1]`, `pose[1:]−pose[:-1]`, `translation[1:]−translation[:-1]`」
と **1 階差分**と記述している。記述通り 1 階差分のみなら C-2 は単純な
オーバーシュート機構としては成立しにくく、**主因は C-1（輪郭スライド）**
の可能性が高い。ただし DECA fork (`preprocess/submodules/DECA`) は本環境に
チェックアウトされておらず `optimize.py` 実物を確認できていない。
**要確認事項①**（§7）。

#### 候補C-3 clip-wide joint 最適化の境界・未来リーク【MEDIUM】

`doc/preprocessing_scope.md` 自身が「平滑化正則化が未来をリークする」
「最適化の最終フレーム状態が clip 長に依存する」と明記。過渡的な極端 pose
では双方向の結合（未来フレームからの引き込み）が excursion を歪め得る。
online は per-frame 独立 EPnP なのでフレーム間結合が無く、この種の
オーバーシュートは原理的に起きない。

#### 候補C-4 透視投影の前後短縮 + 固定 world_mat【LOW-MEDIUM】

極端 pose では深度が弱拘束。clip 定数 `world_mat` は「平均的 pose」に
fit されるため、平均から大きく外れた excursion では固定参照が悪く、
per-frame delta が大きく振れる。

### 5.1 統一的な描像

| 性質 | online EPnP（中央 16 点・per-frame 独立） | offline DECA（FAN 68 点・clip-wide joint） |
|---|---|---|
| 大ヨーでの輪郭スライド | 受けない（輪郭点を使わない）→ **頑健** | 受ける → 回転を回しすぎ |
| フレーム間結合のオーバーシュート | 無い（per-frame 独立）→ **頑健** | 有り得る（時間正則化・未来リーク） |
| 深度（z）の安定性 | **弱い**（準平面 16 点）→ 問題 A | 強い（前後に厚い 68 点） |
| jitter | 明示 FIR LPF + MediaPipe Kalman → **小** | LPF 無し + FAN per-frame → 大 |

→ **「中央アンカーのみ・per-frame 独立」は遮蔽と大ヨーに強いが深度に弱い。
「全 68 点・clip-wide joint」は深度に強いが大ヨーと jitter に弱い。**
online が C 問題（大ヨー精度・jitter）で勝ち、A 問題（深度ずれ）で負ける
のは同じトレードオフの表と裏。

---

## 6. まとめ — 推奨される次アクション（GPU 復帰後）

優先度順:

1. **候補①の検証と修正**: `tracked_params.json` の `world_mat` 回転成分を
   測る。単位行列から数度以上ずれているなら、`_recenter_against_world_mat`
   の translation を `R_ref^T @ (translations − t_ref)` に修正。これは A の
   「平均引き」より原理的に正しく、A を不要にし得る。
2. **候補② への対処**: EPnP に深度を安定させる点（顎/鼻先など前後に厚みを
   出す点）を追加するか、深度のみ別系統（B の cam→z proxy など）で推定。
   中央 16 点だけの PnP では z は構造的に不安定。
3. **rotation チャンネル（候補④）**: online と teacher の rotation 軌道を
   直接比較し、bias 成分の有無を確認。
4. **C 問題（要確認①）**: `preprocess/submodules/DECA/optimize.py` を
   チェックアウトして時間正則化の階数を確認（§7）。

---

## 7. ユーザーへの確認事項

以下を教えていただけると分析を確定できます。

1. **DECA `optimize.py` の時間正則化の階数** — `preprocess/submodules/DECA`
   は本環境に未チェックアウトです。`optimize.py` の `total_loss` で
   pose/translation に掛かる時間項が **1 階差分**（`x[1:]-x[:-1]`）か
   **2 階差分**（加速度）かで、候補 C-2 の成否が決まります。
2. **A/A+C 適用後（動画 03/06）のずれの見え方** — ずれは全フレームで
   **一定方向・一定量**ですか、それとも**顔の向きが正面から外れるほど
   大きくなる pose 依存**ですか。前者なら候補①、後者なら候補②④が主因。
3. **ずれの種類** — メッシュが顔に対して「平行移動して浮く」だけですか、
   それとも「傾いて（回転して）ずれる」成分もありますか。傾きがあれば
   候補④（rotation 未補正）が効いています。
4. **大ヨー時のオフライン崩れ方** — overlay の赤点（FAN 68pt）が顔から
   外れているフレームで崩れますか、それとも赤点は顔に乗っているのに
   メッシュだけ逸れますか。前者なら検出器側、後者は最適化側の問題。
5. **`world_mat` の回転成分**（手元に `tracked_params.json` があれば）—
   左上 3×3 が単位行列に近いか、軸角換算で何度ずれているか。候補① の
   影響度を直接見積もれます。
