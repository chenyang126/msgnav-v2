# MSGNav idea 分析（Idea Analysis）

> **本文只做「方向判断与改进方案」**：哪些方向已证伪不要再碰、哪些待做、教训、未决问题。原始数字见 [`experiment_results.md`](./experiment_results.md)；支撑这些判断的失败拆解见 [`data_analysis.md`](./data_analysis.md)。
>
> 核心校准（来自 `data_analysis.md §1`，当前版本 exp007）：头部矛盾是**假阳性确认**（86/127＝68%），中位最终距 5.55m、51/86 从未感知到 GT——是**自信地在错东西上确认到达**，不是差一点没走近。**修复必须发生在「锁定目标」这一步之前，而不是停止之后。**

---

## 1. 方向状态看板

| 方向 | 来源 | 状态 | 备注 |
| --- | --- | --- | --- |
| 精对准 refine（到达后走近） | Layer-1 MVP | ❌ **已证伪（3次）** | 001/004/004b 全败，根因「近center≠近GT viewpoint」 |
| 保守停止 strict stop-check | Layer-1/2 | ❌ **已证伪，现全局关** | 「驳回→重探索」代价远超收益；`stop_check.enabled=false` |
| Episode Memory（跨subtask记忆） | 数据分析 | ✅ **已降级 log-only（EXP-008）** | EXP-007 效果净负→改默认不干预 prompt/导航，退回观测层；决策干预需窄开关 `live_center_fallback`。见 §4.2 |
| VVD 移除 + 目标点直取物体中心 | EXP-006 | 🆕 已切换，**未单独消融** | 与口径变更、island fix 绑一起跑 |
| island 约束（目标点限可达连通域） | EXP-006 | 🆕 已切换，**未单独消融** | 同上，image +9pp（口径B）但无法归因 |
| **目标保持 + 到达即确认（方向1）** | 数据分析 §1.2 | 🆕 **就近确认已实现（默认关），待 A/B** | opportunistic_stop：靠近同类检测即触发 end_check；exp008(关) vs exp009(开)。目标锁定/view-pose bias 未做 |
| 候选消歧 + 多实例去重（方向2） | 数据分析 §1.2 | ⬜ 未实现 | 对应「远处认错」~23–35；候选 CLIP 排序 + 建图聚类去重，切断记忆错传播 |
| 硬类别检测召回（方向3） | Layer-2 | ⬜ 未实现 | planar/reflective 降阈值+top-k+复核；结构类「可视即成功」。**不要**严格拒绝 |
| 记忆改造：存成功视角+防传播护栏（方向4） | §4.2/4.3 | ⬜ 未实现 | 补回成功 view pose、漂移护栏；迭代期可默认回退 legacy |
| 语义 frontier（方向5，room prior） | Layer-3 | ⬜ 未实现 | SPL 导向，针对超时 31（desc 为主） |
| **仪表盘：每步记录承诺目标是否 match GT** | 数据分析 §1.2 | ⬜ **解锁方向1，优先** | 现无法区分「锁对却放弃」vs「路过真GT」，改动小 |
| 修复 records 重复写入 | 工程 | ⬜ 未实现 | 每 subtask 重复写 ~3 次，不影响指标 |

---

## 2. 已证伪，不要再做（附根因）

| 方向 | 实验 | 结果 | 根因 |
| --- | --- | --- | --- |
| 停止后精对准 / refine | EXP-001/004/004b | 3 次全败，最差 **−6.4pp** | 无 GT viewpoint oracle，几何 proxy（可见性、距中心）不能可靠逼近 GT；「近 center」常是「近 GT」的反方向（尤其 image） |
| 停止判定收紧为「驳回→重探索」 | EXP-002 | **−1.80pp**，17 例灾难走失 | 把 0.06–0.24m 的必成功判「没到」后放弃重探索，走失到 1–11m，代价远大于收益 |

→ **两条硬规则**：
1. **停止之后不要再动 agent**（refine/reject 都不行）。「收紧确认」本身有信号（EXP-002 在 ≤1.0m 口径反超 baseline），但收紧后的动作不能是「放弃目标」。
2. **hard target（mirror/picture/glass/plant）靠检测召回增强，不靠严格拒绝**（EXP-002 对它们的拒绝是净负）。

---

## 3. 待做方案（按优先级）

> 全部方向必须发生在**确认之前**——EXP-001/004/004b 证伪「停止后走近」、EXP-002 证伪「驳回→重探索」，见 §2。优先级依据 `data_analysis.md §1.2` 的根因量级。

### ★★★ 方向 1：目标保持 + 到达即确认（对症「到过 GT 又走开」，50% 失败，最高杠杆）
50% 失败 agent 已到过 GT viewpoint <1m 却走开（49 例走到 >1.5m 外确认）。把这部分转化成功是当前最大且最省的收益。三个抓手：
- **✅ 就近确认（已实现，默认关）**：`opportunistic_stop`——agent 靠近任一同类检测物体（XZ<radius，默认 1.5m）即触发现有 end_check，VLM 仍把关；带触发配额/冷却/护栏（不污染负向记忆、正确归因）。默认关＝exp008 基线，打开＝exp009，构成 A/B。⚠️ 风险：可能停在错实例上（多实例），靠 A/B 验证。
- **⬜ 目标锁定稳定性**：object/image 目标本就持久，收益有限，暂不做。
- **⬜ 接 view-pose bias / 找回成功站位**（§4.3）：记忆把成功 view pose 喂回 planner，对症 SPL 回落，下一步。

### ★★ 方向 2：候选消歧 + 多实例去重（对症「远处认错」~23–35 + 记忆错误传播）
- 承诺目标前对同类候选做**参考图/描述 vs 候选框 CLIP 排序**，而非「第一个过阈值就选」。
- 建图阶段对同类多实例**聚类/去重**，让 GT-id 映射到正确簇——同时切断方向 4「记忆跨子任务传播错锚点」的链条（首步准了，exact_id 复用才安全）。

### ★★ 方向 3：硬类别检测召回（对症「检测漏」~18 + 平面反光 38.8%/结构 9.1%）
- planar/reflective（mirror/picture/glass）**单独降检测阈值 + top-k + CLIP/VLM 复核**（不全局降，避免给方向 2 添噪）。
- 结构类（stair/handrail，SR 0–20%）改「可视即成功 / 更宽到达半径」，当区域而非紧致 bbox。

### ★ 方向 4：记忆改造（对症记忆传播 + 承接方向 1）
- ✅ **已做（EXP-008）**：记忆默认降为 log-only（不干预 prompt/导航），窄开关 `live_center_fallback` 保留 opt-in（仅 exact_id + 同 node ≥2 prior positive）。这撤掉了 EXP-007 净负的两处干预。
- ⬜ 记忆存回**成功 view pose**（补 redesign 丢掉的部分），接 view-pose bias（§4.3）——这是记忆重新参与导航的**首选方式**，优于恢复 live-center override。
- ⬜ **防传播护栏**：即便走 fallback，exact_id 复用前检查活中心相对记录中心的**漂移量**，漂移过大或原锚点 final_dist 贴边则不复用。

### ★ 方向 5：探索效率（对症超时 31，desc 为主，SPL 导向）
frontier 选择引入房间语义先验（M3DSG 已有 room label）。得分示意：`score = w_room·room_prior + w_obj·nearby_objs + w_area·frontier_area − w_dist·path − w_repeat·visited`。room-goal 先验：fridge/microwave/sink→kitchen，pillow/bed→bedroom，toilet/shower→bathroom，sofa/TV/picture→living room，stair→hallway。`scene.objects[id]["room_label"]` 已存在可直接用。

### 仪表盘 / 工程（解锁方向 1，优先）
- **✅ 每步记录「当前承诺目标是否 match GT」**（`run_goatbench_evaluation.py` 已加 `gt_perceived_now` / `committed_target_is_gt`）：object/desc 按 target_index∈target_obj_ids_estimate，image 按 nav 点到 GT 检测中心 ≤success_distance。**下一轮跑完即可**区分方向 1 里「锁对却放弃」vs「路过真 GT」。
- 修复 `records_*.jsonl` 重复写入（每 subtask 写 ~3 次，只影响体积）。
- 跑全集 `--splits 10` 对齐论文 Table1（当前仅对齐消融 Table3）；换 Qwen-VL-Max 做同口径对比。

---

## 4. 未决问题（Open Issues）

### 4.1 EXP-006 三处改动未消融
image +9pp 但（a）口径从 viewpoint 变物体中心本就抬数字，（b）VVD 移除/口径/island 三件事绑一起跑。**下一步**：同口径（1.0m 物体中心）下单跑「只加 island、不改 VVD」，隔离 island 真实增量；有余力再补「只移除 VVD、不加 island」。

### 4.2 EXP-007 记忆重设计：机制成功但效果净负 ❌（A/B 已事实完成）
之前担心「效果层无法复现 A/B」的问题**已经被 EXP-006 vs EXP-007 事实性地回答了**：两次跑导航栈/口径/config 全相同（仅差被删的死 flag），唯一区别是记忆实现 legacy(exp006) vs redesign(exp007)，构成一次事实上的记忆 A/B（判定依据是 log 签名，非存档 config，详见 `data_analysis.md §3.2`）。
- **机制层**：✅ 完全达成——spatial 误配 28→3，VLM 身份门调用→仅 5，命中数不变(88)。
- **效果层**：❌ 净负——同口径下 redesign 比 legacy **SR −1.80pp / SPL −4.30pp**，且**最想帮的 image 反而最惨（SR −5.68 / SPL −6.63）**。SPL 全线跌指向：redesign 丢掉「冻结成功站位」、命中后只给「活中心」，几何看似更优却让路径变长。
- **决策（已定，EXP-008）**：采用了「不删记忆、但默认停止一切决策干预」的路线——把 EpisodeMemory 降级为**默认 log-only / 观测型**：
  1. ✅ 默认不干预 prompt（`annotate_positive`/`memory_block`/`surface_negatives_in_prompt` 全 false，`explore_utils.py` 按 `MEMORY_HINT_POLICY` 控制注入）。
  2. ✅ 默认不覆盖导航点（live-center override 移到 `_should_apply_live_center_fallback()` gate 之后，`live_center_fallback.enabled` 默认 false）。
  3. ✅ 保留 record/query/分析能力，只是不再影响决策。
  - ❌ **实测未如预期回到 legacy 水平**（EXP-008 已跑完 278）：SR **52.88%**（比 legacy exp006 56.12、redesign exp007 54.32 **更低**，三者最低），SPL **32.01%**（比 exp007 +1.82pp，但仍低于 legacy 34.49）。说明 legacy-nav 的一部分 SR 收益本就来自记忆决策，全关后 SR 掉下来；log-only 只买回了路径效率（SPL）。
  - **exp007 vs exp008（baseline）关键分裂**：记忆干预是「用 SPL 换 SR」的不均衡交易，且**方向按 task_type 分裂**——image（SR+3.41/SPL+2.21）、object（+3.03/+1.50）双赢，唯独 **description 双输（SR −2.20 / SPL −9.33）**，全量 SPL 转负几乎全由 description 拖累。→ 下一步应把干预做成**按 task_type 条件启用**（image/object 开、description 关），细节见 `experiment_results.md` EXP-008。
  - 若日后要重新让记忆干预导航，走窄开关 `live_center_fallback`（仅 exact_id + 同 node ≥2 prior positive）或**优先接 view-pose bias**（§4.3），而非恢复 broad hint/override。
- ⚠️ 归因谨慎：exp006/007 两次均单跑、非确定（±~1.4pp/278），overall −1.80 接近噪声，但 image/SPL 回落超纯噪声。

### 4.3 Phase-2 view-pose bias 未接（未来功能，非旧机制）—— EXP-007 后升为对症首选
memory 命中时除了导向物体中心，还可把 planner 落脚点/朝向朝「上次成功的 view pose」偏置，直接复现能成功的观察几何。当前状态：`use_view_pose_bias` 默认关、flag 已留、`query()` 已暴露 `view_position`/`view_angle`，但 **planner 偏置逻辑未接**（`use_view_pose_bias` 读进来后全代码无消费点，开关当前无效）。要实现需：命中分支读出 view pose → 传入 `set_next_navigation_point` → planner 加偏置逻辑 → 切 true 跑 A/B 验证。
- **EXP-007 后动机变强**：redesign 相对 legacy 的 SPL 全线回落（§4.2）很可能正因为它**丢掉了 legacy 的成功站位、命中后只给活中心**。view-pose bias 把成功 view pose 补回来，是对症 SPL 回落最直接的一步——从「可选实验」升级为「若保留 redesign 的首选修复」。

---

## 5. 关键教训（Lessons，反复看）

1. **非确定性**：43 子任务重跑就翻 3–4 个。**结论只在 278 全首集判定**，小样本只能看趋势/debug。
2. **「推近」要用 refine（走向目标），绝不用 reject（释放目标重探索）**——后者把 0.08m 必成功变 11m 彻底失败（EXP-002）。
3. **refine 只对「还没到」的目标做，已进成功半径的绝不动**（EXP-001 的下行全来自动了已成功案例）。
4. **hard target 靠检测召回增强，不靠严格拒绝**（EXP-002 对它们的拒绝是净负）。
5. **改动必须可追踪**：把触发/动作写进 `records` note 或 trajectory（EXP-001 靠 `n_refine_steps` 才复盘出触发点错）。
6. **几何 proxy 不能替代 GT**：可见性饱和(0.93+)、dist-to-center 与 dist-to-GT 无因果。无 GT oracle 的 post-hoc refinement 都是赌博；end_check 通过时的 pose 已是系统最优，不要画蛇添足。
7. **跨 subtask 记忆要检查「归因时刻」是否与「决策时刻」一致**：`choose_every_step` 下只有 frontier 会被强制重选，object/image 选中后不再重置——循环结束前若切回 frontier，记录用的 `last_*` 可能是已放弃的旧目标（EXP-005b stale-target bug）。这类 bug 可能长期潜伏（275 里 8 个可达但 0 个真写错），**不能靠「没观察到坏结果」证明逻辑没问题**。
8. **改判定口径 ≠ 改导航逻辑，要分开验证**：口径一变跨口径数字就失去可比性（EXP-006 把三件事一起跑，无法归因）。想同时换口径+换逻辑时，至少在同一口径下补一组只变一个变量的对照，否则宁可分两次提交。
9. **「机制生效」≠「效果为正」，两者分开验证**（EXP-007，已被实测坐实）：smoke 确认机制按设计动作在跑（命中、CLIP 短路、负标注注入、正确重访），但机制触发也可能拖累。EXP-007 全量跑给出结论：redesign 机制层完全成功（spatial 误配 28→3、VLM 门→5），**效果层却净负（SR −1.80 / SPL −4.30 / image −5.68 vs legacy）**——效率优化不等于指标提升。判断效果方向唯一可靠办法是同导航栈同口径的 A/B（本次靠 exp006/exp007 事实凑出），全新路径、无 baseline 的跑只能验机制、不能验效果。
