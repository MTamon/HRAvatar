# 実装担当へのレビュー — online FLAME メッシュずれと neck-slot プランの評価

> 2026-05-16 / `claude/verify-gpu-analysis-L0HCm`
> 本書はユーザー報告の症状を整理し、`lhg/` `scene/` `demos/` の
> コード検証に基づいて原因候補を示し、現行 neck-slot プラン
> (`stage11jigglyhoney.md`)への疑義を述べる。各原因候補には
> 「可能性」ではなく**コードで確認できた事実**を根拠として付す。

---

## 1. 観察された症状(整理)

ユーザー観察を時系列・幾何学的性質で整理する。

**当初症状:**
- online 経路で FLAME メッシュが実顔から数センチ浮く。
- メッシュは**常に顔の正面**にある(=正面フレームでも浮いている)。
- 顔が回転すると、メッシュは**より大きい回転半径**で回る。
- カメラ系の平行移動ではない。**頭部ローカル座標系で +z(鼻方向)に
  一定平行移動**している、という表現が正しい。

**実装担当による後続修正の後:**
- 顔とメッシュの距離は縮んだが**完全には埋まらない**。
- 頭部ローカル座標系で**上下(y)方向のずれ**も出現(当初から在った
  可能性もあり — 浮きが大きく気づかなかっただけかもしれない)。
- 別の修正では、メッシュが頭部ローカルの **x 軸回転(pitch)**を
  伴って顔に追従した。

**全症状に共通する性質:** すべて**定常的**。特定フレーム・特定状況
(大ヨー時など)に限らず、常時発生している。

---

## 2. 症状の幾何学的含意(これが診断を絞る)

「メッシュは顔と回転を共有しつつ、頭部ローカルで一定ベクトル `d` だけ
ずれている」= カメラ系で書くと **ずれ = `R·d`(`d` は頭部固定の定数)**。

この性質は次を**意味する**:

1. **正面フレームでもずれている** → ずれは「回転 R に比例して増減する
   pose 依存項」ではない。R に依らない定数成分が主体。
2. **頭部と一緒に回る(カメラ系で固定されていない)** → ずれはカメラ系
   の誤差(カメラ深度バイアス、カメラ系並進)ではない。

この 2 点が、後述の通り**現行プランの土台と複数の従来候補を否定する**。

---

## 3. 現行 neck-slot プランへの疑義 — 【疑義 HIGH】

プランは「ずれの主因は、EPnP の回転を pose スロットに入れた際の
**回転中心ずれ `(I−R)@J`** を並進で相殺していないこと」と診断し、
neck スロット配置 + `t_neck = t_e − (I−R_e)@J[1]` で修正する。

**コード検証の結果、この診断は主症状を説明できない:**

- `scene/flame.py:663-691` `batch_rigid_transform` を精読した。
  `joints_homogen` は `F.pad(joints,[0,0,0,1])`(4 要素目 0)、
  `rel_transforms = transforms − pad(transforms @ joints_homogen, …)`。
  これにより**全関節 k(関節 0 を含む)**の LBS ブロックは
  `[[R, (I−R)@J[k]], [0,1]]` となる。
- したがって **`(I−R)@J[k]` は `R = I`(=正面・無回転)のとき厳密に
  ゼロ**。プランが主因とする `(I−R)@J` 機構は、**正面フレームで
  ずれを生まない**。
- しかし症状(§1)は**正面フレームでも数センチ浮く**。

→ **プランの中核機構 `(I−R)@J` は、正面でも残る定常的な浮き(=主症状)
の原因ではあり得ない。** `(I−R)@J` は実在する効果だが、回転とともに
増減する **pose 依存の二次的項**であり、症状の一部(大ヨー時に増える分)
にしか対応しない。

加えて、プランの式自体にも前回レビュー(`lhg_neckslot_third_party_review.md`
§4・§8.3)で指摘した不完全さ(`world_mat` 回転 `R_ref` の欠落、joint 項に
カメラ系 `R_e` を使う不整合)がある。

**結論:** プランを実装しても**主症状は解消しない**。プランは pose 依存の
副次的ずれには部分的に有効だが、ユーザーが報告している「正面でも残る
定常的な浮き」には無効。プラン単独での着手は推奨しない。

---

## 4. 原因候補 — コード検証に基づく根拠

### 4.0 統一原理(これがすべての候補の土台)

`demos/demo_3_overlay_tracking.py:143-215`(renderer と bit-parity と
明記)を精読して確認した renderer の合成:

```
v_world = LBS(pose を適用した FLAME 頂点) + translation      (L181-184)
cam     = FLIP_YZ @ world_mat @ (flame_scale · v_world)       (L203-206)
```

一方 EPnP(`lhg/epnp.py`)は `p_cam = R_e·p_obj + t_e` を解く。すなわち:

> **EPnP は「テンプレート `T_epnp`」を 2D 顔ランドマークにピン留めして
> pose を求める。renderer は別の「メッシュ `T_render`」を描く。
> `T_epnp` と `T_render` が頭部座標系で定数 `d` だけ食い違えば、
> その差は renderer 出力にそのまま `R·d`(頭部固定の定数浮き)として
> 現れる。**

§2 の症状(頭部固定の定数 = `R·d`)は、この `T_epnp ≠ T_render` で
**しか**説明できない。以下、`T_epnp` と `T_render` が食い違う箇所を
**コードで確認できたものだけ**列挙する。

### 4.1 expression 状態の不一致 【根拠: code-verified】

- `lhg/correspondence.py:159-189` `build_shape_aware_landmarks`:
  docstring(L176-178)とコードが明示する通り、EPnP のテンプレートは
  **`expression = 0`、`pose = 0`** で評価される。`lhg/pipeline.py:596`
  でこれが `epnp_object_points` として EPnP に渡る。
- 一方 renderer は per-frame の SMIRK expression を適用する
  (`scene/gaussian_head_model.py:261-269`:`betas = expression_params`,
  `blend_shapes(betas, expression_dirs)`;`demo_3:163` も同じ)。
- → `T_epnp`(expression=0)と `T_render`(SMIRK expression あり)は
  expression 変位 `E·e` だけ食い違う。被写体の**平均 expression**
  成分はそのまま頭部固定の定数 `d` になる。
- `config.py:170` `epnp_expression_aware` は**デフォルト False**。
  True にすると per-frame でテンプレートを SMIRK expression から
  組み直すため、この不一致は解消する(ただし会話ログでは C 候補=
  jitter 対策として扱われ、定数ずれの文脈では評価されていない)。

### 4.2 pose blendshape の不一致 【根拠: code-verified】

- `lhg/correspondence.py:205-209`(`ExpressionAwareLandmarkComputer`
  docstring)が明示:「Pose blendshape … is intentionally omitted」。
  base の `build_shape_aware_landmarks` も `pose = 0` 評価。
- renderer は pose blendshape を適用する
  (`gaussian_head_model.py:276-280`:`pose_feature = rot_mats[:,1:]−I`,
  `pose_offsets = blend_shapes(pose_feature, pose_dirs_t)`;
  `demo_3:165-167` も同じ)。
- → 頭部が neck/jaw を曲げている分、`T_render` は `T_epnp` から
  pose blendshape 変位ぶん食い違う。被写体の平均 neck/jaw 姿勢ぶんが
  定数成分。

### 4.3 EPnP テンプレートと renderer メッシュが別 FLAME ファイル経由 【根拠: パスの相違は code-verified、中身の異同は要確認】

- EPnP テンプレート: `correspondence.py:159` →
  `_load_flame_canonical(flame_model_path)`、デフォルトは
  `correspondence.py:117` `DEFAULT_FLAME_MODEL_PATH =
  ./assets/FLAME2020/generic_model.pkl`。
- renderer の FLAME: `scene/flame.py:101` `load_flame_mesh(...,
  file_name="assets/flame_model/flame2020.pkl")`、`demo_3:128` も同じ。
- → **2 つは別パス・別ファイル名**。両者が同一の FLAME モデルを指すか
  どうかは本環境に両ファイルとも存在せず**未検証**。`v_template` が
  異なれば、その差がそのまま定数 `d` になる。**実装前に両ファイルの
  同一性(または `v_template` の一致)を必ず確認すること。**

### 4.4 shape(identity)の不一致 【根拠: code-verified な参照経路、値の異同は要確認】

- EPnP テンプレートの shape: `pipeline.py:596` は
  `build_shape_aware_landmarks(correspondence, calibration.shapecode)` —
  **Stage 1 calibration の shapecode**を使う。
- renderer のメッシュ shape: `render_adapter.py` docstring(L24-26)が
  明示する通り**アバター学習時の shapecode** を使い、「renderer は
  per-frame の shape override を無視する」。
- → EPnP テンプレートが Stage 1 shapecode、描画メッシュがアバター
  shapecode。**両 shapecode が一致するか要確認**。一致しなければ顔の
  サイズ・形が食い違い、EPnP は深度で補償するため(`correspondence.py:165-178`
  が説明する機構)頭部ローカル z のずれを生む。

### 4.5 EPnP の準平面 16 点による深度の悪条件 【根拠: code-verified、ただしカメラ系】

- EPnP は中央 16 点(`correspondence.mp_indices`)のみで `cv2.solvePnP`
  を解く(`epnp.py:122-125`)。準平面点群の PnP は深度 z が弱条件。
- これは `t_e` の**カメラ z** 成分の誤差 → カメラ系のずれ。
- §2 の通り症状は頭部固定であってカメラ系ではないため、**主症状の
  主因ではない**。ただし正面フレームでは頭部 +z 軸とカメラ深度軸が
  ほぼ一致するため、正面だけ見ると 4.1-4.4 と区別がつかない点に注意。

### 補足: 従来候補の位置づけ

- 候補②(`_recenter` の `R_ref` 欠落、`pipeline.py:495` vs `497-499`
  の非対称)はコード上実在するバグだが、生む誤差は
  `(I−R_ref^T)(t_total−t_ref)` で**頭部並進依存・ほぼカメラ系**。
  頭部固定の定数ではないため**主症状の主因ではない**。独立した既存
  バグとして別途修正対象。
- プランの `(I−R)@J`(§3)は正面でゼロ。

---

## 5. 推奨する次の一手(GPU 不要で主因を確定できる)

**レンダリングや学習に着手する前に**、以下の数値検証で `d` を確定する。

1. **§4.3 の確認:** `assets/FLAME2020/generic_model.pkl` と
   `assets/flame_model/flame2020.pkl` が同一 FLAME モデルか
   (`v_template` 一致か)を確認する。
2. **§4.4 の確認:** EPnP に渡る `calibration.shapecode` と、描画に使う
   アバター shapecode が一致するか確認する。
3. **テンプレート直接差分(主因確定):** FLAME canonical 座標で
   - `T_epnp` = `build_shape_aware_landmarks` の出力(16/31 点)
   - `T_render` = renderer 側の FLAME(`flame2020.pkl`)+ アバター
     shapecode + 同一バリセントリック対応で再構成したランドマーク点、
     さらに被写体の**平均 expression / 平均 pose blendshape を適用**した点

   この 2 つを引き算する。差がほぼ一定ベクトルなら、それが `d` で
   主因確定。**この検証は GPU 不要・レンダリング不要**で、`d` の
   z(深度)・y(上下)成分を数値で出せる(症状の +z 浮きと y ずれに
   直接対応)。
4. **reprojection 検証は正面フレームで行う:** 正面では `(I−R)@J`・
   候補②・候補4.5 が最小化され、`d` を単離できる。

修正方針は「`d` を消す」= **EPnP の object_points を、renderer が実際に
描くメッシュと幾何的に同一の FLAME モデル・shape・(必要なら平均
expression/pose を織り込んだ)状態から生成する**こと。これは neck-slot
プランとも候補②修正とも**独立した別レイヤの修正**であり、主症状は
これでしか消えない。

neck-slot プラン・候補②修正は、`d` を消した後に残る pose 依存成分
(大ヨー時に増える分)に対して着手すればよい。**順序として、まず §5 の
テンプレート整合を先に解決すること。**

---

## 6. スコープ外(本書では扱わない)

ユーザー観察のうち「online の方が jitter が小さい」「オフラインが大ヨーで
過剰にずれる(2 次補間オーバーシュート様)」は別問題(`doc/
lhg_online_offline_misalignment_analysis.md` §4・§5)であり、neck-slot
プラン・本レビューの対象外。
