# LHG neck-slot プラン — 第三者独立レビュー

> 作成: 2026-05-16 / ブランチ `claude/verify-gpu-analysis-L0HCm`
> 本書は、実装担当 AI のプラン (`stage11jigglyhoney.md`) と、それを
> 評価した分析メモ (`doc/lhg_online_offline_misalignment_analysis.md`、
> 特に §8) を、**第三者として独立に検証**した結果です。コードは
> `lhg/` `scene/` `demos/` `scripts/` を精読しました。実行環境に
> GPU・FLAME モデルファイルが無いため、数値実測ではなくコード上の
> 構造的検証です。確度は HIGH / MEDIUM / LOW で示します。

---

## 0. 総括

- **プランの中核(回転中心ずれ `(I−R)@J` を並進で相殺)も、分析メモ §8
  の主要な指摘(§8.3 の `R_ref` 欠落、§8.4 の表現不一致、§8.6 の回転域、
  §8.7 のリプロジェクション検証)も、コード上おおむね妥当**と確認した。
- ただし **プランには分析メモが見落とした原理的な誤りが 1 点**あり、
  さらに **両文書が触れていない実装上のリスクが 3 点**ある(§2)。
- 加えて、ユーザー観察「ずれは全フレームで一定」に対し、**プラン・
  分析メモが挙げる全機構が pose 依存であって真の定数を説明しない**と
  いう論理的ギャップが残る(§3)。

---

## 1. コードで確認できた事実(分析メモの引用は正確)

| 主張 | 検証結果 |
|---|---|
| online は `neck_pose`/`eye_pose` をゼロ出力 (`pipeline.py:849-850`) | ✅ 正確 |
| `_recenter_against_world_mat` は translation を単純減算、`R_ref` を掛けない (`pipeline.py:497-499`) | ✅ 正確。rotation 側は `R_ref.T @ R_total` (L495)、translation 側だけ非対称 |
| teacher は `fullposecode[3:6]` を実 neck として格納 (`teacher.py:343-344`) | ✅ 正確。teacher は LPF も recenter も掛けない(生コピー) |
| overlay は `neck_pose` を slot[3:6] に置く (`overlay_online_features.py:298`) | ✅ 正確。online は neck=0 なので現状 overlay は総回転を **global slot** に置いている |
| `batch_rigid_transform` は `joints_homogen` の 4 要素目が 0 → `A_k=[[R,(I−R)J[k]],[0,1]]` (`flame.py:688-691`) | ✅ 正確 |
| renderer は `v_world = LBS(v)+translation`、その後 `cam = FLIP_YZ@world_mat@(flame_scale·v_world)` (`demo_3:181-203`) | ✅ 確認。translation は world_mat **回転の内側**に入る → §8.3 の構造的根拠 |
| `lpf_lookahead` デフォルト 4 (`config.py:110`) | ✅ 正確 |

EPnP→hravatar 変換式 `R_hr = M@R_cv`(`epnp.py:44-73`)も導出は正しい
(object frame 不変、camera frame のみ反転)。`solve_epnp` の scale 処理も
一貫(`epnp.py:117,130`)。

**`_recenter` の並進バグ(分析メモ候補②/§8.3)を独立に再導出して確認した。**
renderer 整合条件 `world_mat@(flame_scale·(LBS(v)+translation)) = R_e·p_obj+t_e`
を解くと、正しい最終並進は

```
translation = R_ref^T @ (t_e − t_ref) − (I − R_pose) @ J[k]   (R_pose = R_ref^T @ R_e)
```

これは分析メモ §8.3 の式と一致する。プランの式 `t_e−(I−R_e)@J[1]`
(→ `_recenter` が `−t_ref`)との差は `(I−R_ref^T)@[(t_e−t_ref)+R_e@J[1]]`
で、§8.3 の残差式とも一致。**プランの式が厳密に正しいのは `R_ref≈I` の
ときだけ**、という §8.3 の結論は妥当(HIGH)。`_recenter` の docstring 自身が
「world_mat の回転は FLAME-canonical→camera の pre-rotation を吸収する」と
書いており、`R_ref` は単位行列ではない蓋然性が高い。

---

## 2. 見落とし — プランの誤りと両文書が触れていないリスク

### 2.1 【HIGH】プラン §2-1「global slot はモデル原点を中心に回る」は誤り

`batch_rigid_transform` は **関節 0 を含む全関節**について
`rel_transforms[k] = [[R_k,(I−R_k)@J[k]],[0,1]]` を計算する(§1 で確認)。
したがって **global slot(関節 0)もモデル原点ではなく `J[0]` を中心に
回る**。原点中心になるのは `J[0]=(0,0,0)` のときだけ。

分析メモ §8.2 は「各関節 k の回転は J[k] 固定点」と**正しく**書いている
が、それが**プラン §2-1/§4-1 の前提と矛盾する**ことを指摘していない。
帰結としてプランの診断ナラティブには次の誤りが残る:

- **global slot で「浮いた」真因は slot 選択ではなく『並進補正の不在』**。
  旧コードは global slot に回転を入れつつ `(I−R)@J[0]` も `R_ref^T` も
  補正していなかった。slot を neck に替えなくても、**global slot のまま
  `t = R_ref^T(t_e−t_ref) − (I−R_pose)@J[0]` を入れればメッシュは張り付く**。
- すなわちプランの「neck slot にすると張り付く / global slot だと浮く」と
  いう対比は機構の説明として不正確。neck slot を選ぶ正当な理由は
  (a) pose blendshape の物理的正しさ、(b) アバター学習時の pivot との
  整合 であって、**浮きの解消そのものではない**。

実害: 実装者が「neck slot にしたから並進補正は軽くてよい」と誤読する
恐れがある。**並進補正(§8.3 の `R_ref` 込み)は slot 選択と独立に必須**。

### 2.2 【HIGH】J[1] の算出元 FLAME モデルが renderer と別ファイル

プラン Step 1 は `correspondence.py` の `_load_flame_canonical` を使って
`flame_neck_joint_position` を実装する。`correspondence.py:117` の
`DEFAULT_FLAME_MODEL_PATH = ./assets/FLAME2020/generic_model.pkl`。
一方 renderer の FLAME は `scene/flame.py:101` で
`assets/flame_model/flame2020.pkl` を読む。**2 つは別パス・別ファイル**。

`(I−R)@J[1]` が renderer の pivot を正確に打ち消すには、J[1] は
renderer の `flame_joint_center[1]`(= `J_regressor @ flame_vertexes_shaped`、
`gaussian_head_model.py:157`)と**同一**でなければならない。プランは
`generic_model.pkl` 由来で計算するため、2 ファイルが topology・
`J_regressor`・`v_template` レベルで一致しない場合、補正が pivot と
ずれて**新たな定数残差**を生む。

なお既存の EPnP object_points も `build_shape_aware_landmarks` 経由で
`generic_model.pkl` を使っており、**object_points と renderer mesh が
既に別モデルである潜在的不整合**が存在する。分析メモはこの点に触れて
いない。**実装前に 2 ファイルの同一性(または J_regressor の一致)を
確認すること。**

### 2.3 【MEDIUM】§8.4 の不一致は「数値」だけでなく「幾何」

分析メモ §8.4 は online/teacher の表現不一致を schema・分布の問題として
扱うが、**幾何的帰結**も明示すべき。teacher は 2 関節チェーン
(R_g を J[0] 中心、R_n を J[1] 中心)、プラン後の online は 1 関節
(R_total を J[1] 中心)。同じ頭部**向き**でも、頭部**位置**は

```
(R_g − I) @ (J[1] − J[0])
```

だけ食い違う(per-frame、R_g 依存)。LHG モデルが teacher で学習し
推論時に online 値を食わせる経路では、この差が renderer 出力の幾何に
直接出る。→ §8.4 の schema 統一は「単なる整理」ではなく、同一アバターを
両系統で駆動する限り**先送りの安全性が低い**。

### 2.4 【MEDIUM】LBS ウェイト混合の影響(再学習要否)

§8.5 は pose blendshape のみ挙げるが、より基本的な点として: 顔・頭部
頂点の `lbs_weights` が関節 0 と関節 1 に**分散**している場合、
「総回転を neck 単独」と「global+neck 分割」は、pose blendshape を
別にしても**異なる頂点変形**を生む(pivot の異なる 2 剛体変換の線形
混合 ≠ 単一剛体変換)。プラン §6「再学習不要」は楽観的。
**実装前に FLAME `lbs_weights` の顔・頭部領域が関節 1 にほぼ単一
重みか、関節 0 と混合かを確認する**こと。単一重みなら §2.3 の
`(R_g−I)(J[1]−J[0])` のみ、混合ならさらに混合アーティファクトが乗る。

---

## 3. 未解決の論理ギャップ — 「定数ずれ」を説明する機構が無い

ユーザーは「ずれは全フレーム一定方向・一定量、お面のような平行移動的
浮き」と観察している(分析メモ §2.1)。しかし候補に挙がる機構を整理
すると **すべて回転 R 依存(= pose 依存)** である:

| 機構 | A 適用後の残差 | 性質 |
|---|---|---|
| 候補②(`R_ref` 欠落) | `(I−R_ref^T)@(t_total−mean)` | 頭の並進変動依存 |
| joint-center(global slot, `(I−R)@J[0]` 未補正) | `(I−R)@J[0] − mean` | 回転依存 |
| neck 分割不一致 | `(R_g−I)@(J[1]−J[0])` | 回転依存 |

**真に回転非依存な定数ずれは、上記のどれでも説明できない。** プラン
§4-3 は「観察区間で R の変化が小さく定数に見えただけ」と説明するが、
これは検証で確認すべき**仮説**であって確証ではない。分析メモ §3.4 も
同じ留保を正直に書いている。

真の定数を生む機構は別にある: clip 定数の誤り(world_mat / shapecode /
intrinsics)か、候補 A の offset を**抽出クリップと異なるフレーム集合**で
平均したことによる誤差(分析メモ §3.4 末尾が指摘。`pipeline.py:831` は
`tracked_params.json` の全画像キーを読む)。プランは候補 A を廃止する
ので後者はプランでは消えるが、**前者(clip 定数の誤り)は残る**。

→ **推奨**: プラン+§8.3 修正の検証時、残差を頭部回転角と相関プロット
すること。回転と無相関な定数成分が残るなら、診断は未完であり clip
定数を疑う。分析メモ §8.9 の「候補①撤回」は機構としては妥当(EPnP の
R_e は総頭部向きを既に含むので neck=0 自体は pose 誤差を生まない)だが、
**「定数ずれの説明」という宿題は撤回後も未解決のまま**である点に注意。

---

## 4. プランの式・実装順序の細部(§8.3 を補強)

§8.3 の指摘に加えて独立に確認した点:

1. プランは `t_neck` を `per_frame_core` 内で**カメラ系の `R_e`** を
   使って計算する。しかし `(I−R)@J[1]` の `R` は renderer が J[1]
   まわりに適用する **FLAME 系回転 `R_pose = R_ref^T@R_e`** でなければ
   ならない。プランは `R_e` を使う → §8.3 が言う通り二重に不整合
   (並進フレーム + joint 項の回転)。
2. したがってプランの「`_recenter_against_world_mat` は無改修で動作」は
   成立しない。正しくは **`_recenter` で translation も `R_ref^T` で
   回し、joint 補正は recenter 後・FLAME 系で `R_pose` を使って適用**
   する。実装はこの順序にすべき。
3. scale 整合は問題なし: `t_e`(=`tvec_canonical`)も J[1] も translation
   フィールドも FLAME canonical(未 scale, メートル系)で一貫。EPnP
   landmark 資産から FLAME canonical は実寸メートル(顔幅 ~10.6cm)と
   確認した。
4. joint 補正後の translation は回転連動成分を含むため、translation の
   LPF(4Hz)が回転由来の動きも平滑化する。global_rot も同じ 4Hz なので
   desync は生じないが、cutoff を将来変える場合は要注意(LOW)。

---

## 5. 推奨する検証順序(GPU/FLAME 復帰後、コード変更前)

0. **`J[0]` と `J[1]` の数値を print する(最優先)。** `J[1]−J[0]` の
   大きさが、プランの機構が量的に有意か、global/neck の slot 差が
   そもそも問題になるかを決める。本環境に FLAME モデルファイルが無く
   検証不能だった。FLAME canonical 原点は顔表面(z≈+0.05m)より内側に
   あることまでは landmark 資産から確認済み。
1. **`generic_model.pkl` と `flame2020.pkl` の同一性**を確認(§2.2)。
   不一致なら J[1] は renderer 側の `flame2020.pkl`+`J_regressor` から
   計算するよう修正。
2. **`world_mat[:3,:3]` を軸角換算**し単位行列からの乖離を測る(§8.3)。
   非自明なら `_recenter` の translation に `R_ref^T` を入れ、joint 補正を
   recenter 後・FLAME 系で適用(§4 の順序)。
3. **FLAME `lbs_weights` の顔・頭部領域**が関節 1 単一か関節 0 と混合かを
   確認(§2.4)。
4. プラン §8.7 の**1 フレーム数値リプロジェクション検証を主観評価の前に
   必ず実施**(強く支持)。式の誤り(§4 の 1・2)は 2x2 動画の主観評価
   では検出しにくい。
5. 検証時、残差を頭部回転角と相関プロットし、回転非依存の定数成分が
   残らないか確認(§3)。
6. プラン §6 の「実効回転 10〜30°」は **teacher の global∘neck 合成**で
   見積もり直す(§8.6 を支持。最大 60〜90° に達し得る)。

---

## 6. 守るべき制約の確認

プランの「DECA submodule 不可触」「LHGFeatures schema 不変」「既存コミット
non-revert」「`--neck-slot-rotation` 未指定時の後方互換」は妥当。本レビューの
推奨(特に `_recenter` 修正)も `--neck-slot-rotation` ゲート内に閉じれば
後方互換は保てる。ただし `_recenter` の translation バグ(候補②)は
**neck-slot とは独立した既存バグ**であり、フラグ無指定の現行 epnp 経路にも
存在する。修正範囲を neck-slot に閉じるか、候補②を独立に直すかは設計判断。
