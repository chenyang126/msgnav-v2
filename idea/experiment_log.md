# MSGNav 改进实验日志（Experiment Log）

> **用途**：记录每一次改动的动机、口径、结果和结论，避免重复实现已被证伪或方向错误的改动。
> **维护规则**：每做一次改动/一次评测，就在下方"实验记录"里**追加一条**（不要删旧记录）；同时更新顶部"结论速查表"和"方向状态看板"。
>
> **统一口径**：GOAT-Bench Val Unseen，`--splits 1`（每 scene 仅 episode 0），278 subtasks，gpt-5.4，4×RTX5090。分析以 `records_*.jsonl` 去重（按 subtask_id 保留最后一条）为准，SPL 用 msgnav conda 环境算（普通 python 缺 numpy 会显示 SPL=0）。
>
> ⚠️ **成功判定口径在 EXP-006 变了**：EXP-000~005b 用 `success_distance=0.25m`，量的是"到 GT viewpoint 的距离"；EXP-006 起改成 `success_distance=1.0m`，量的是"到 GT 物体中心的距离"。**跨这条线的 Success/SPL 数字不可直接比较**，对比时先看清楚在哪个口径下。

---

## 0. 结论速查表（先看这里）

口径 A（`success_distance=0.25m`，到 GT viewpoint 距离）：

| 改动 | 结果目录 | n | Success | SPL | 相对 baseline | 结论 |
| --- | --- | --- | --- | --- | --- | --- |
| **Baseline**（无改动） | `exp000_baseline_gpt-5.4_278subtasks` | 278 | **55.04%** | 33.65% | — | 基准 |
| Refine v1（到达后走近，EXP-001） | `exp001_refine_v1_gpt-5.4_43subtasks_partial` | 43(仅重叠) | 65% vs 67% | — | 净 −1 | ❌ 证伪（触发点错，见下方合并结论） |
| Strict stop-check（EXP-002） | `exp002_stopcheck_strict_gpt-5.4_278subtasks` | 278 | 53.24% | 32.96% | **−1.80pp** | ❌ 证伪（灾难性走失） |
| Refine v2 可见性微移（EXP-004） | `exp004_refine_v2_vis_gpt-5.4_34subtasks_partial` | 34(terminated) | 67.6% | — | — | ❌ 证伪（可见性饱和，no-op） |
| Refine v2b 距离驱动微移（EXP-004b） | `exp004b_refine_v2_dist_gpt-5.4_113subtasks_partial` | 74 | 48.6% | ~30% | **−6.4pp** | ❌ 证伪（靠近center≠靠近GT） |
| Episode Memory v1（spatial 有 bug，EXP-005） | `exp005_episode_memory_v1_gpt-5.4_275subtasks` | 275 | 净0(raw)/+7(因果) | — | 噪声抵消 | ⚠️ exact_id 安全，spatial 参照系错误 |
| Episode Memory v2（修复版，EXP-005b） | `exp005b_episode_memory_v2_gpt-5.4` | 278 | 54.32% | 38.22% | −0.72pp / **SPL +4.57pp** | ✅ Success 打平，路径效率明显提升 |

口径 B（`success_distance=1.0m`，到 GT 物体中心距离，EXP-006 起）：

| 改动 | 结果目录 | n | Success | SPL | 结论 |
| --- | --- | --- | --- | --- | --- |
| +VVD 移除 / +island 约束（EXP-006） | `exp006_nav_island_fix_gpt-5.4` | 278 | 56.12% | 34.11% | ⬜ 待消融，不能跟口径A的数字比 |

> ⚠️ **方法论警告**：本评测**非确定**。43 个子任务上光是重跑就会随机翻 3–4 个成功/失败。**任何结论都必须在 278 全首集上得出，不要用几十个子任务下判断。**

---

## 1. 方向状态看板

| 方向 | 来源 | 状态 | 备注 |
| --- | --- | --- | --- |
| 精对准 refine（到达后走近） | idea Layer-1 MVP | ❌ **已证伪**（3次尝试） | v1/v4/v4b 全部失败，根因："靠近center"≠"靠近GT viewpoint" |
| 保守停止 strict stop-check | idea Layer-1/2 | ❌ **已证伪**，现全局关闭 | "驳回→重探索"代价远超收益；EXP-005 起 `stop_check.enabled=false` |
| Episode Memory（跨subtask记忆锚点） | 数据分析 | ✅ 已重设计上线（EXP-007） | v1 spatial bug→v2 修复（EXP-005b，SPL +4.57pp）→v2 冻结快照重设计为「场景图节点薄标注层」（EXP-007，机制已确证，效果层待受控 A/B） |
| VVD 移除 + 目标点直取物体中心 | EXP-006 | 🆕 已切换，**未单独消融** | 与成功口径变更、island fix 绑在一起跑，见 EXP-006 |
| island 约束（目标点限制在可达连通域） | EXP-006 | 🆕 已切换，**未单独消融** | 同上，image +9pp（口径B），因归带未拆分无法确认 |
| Hard target 检测增强（planar/反光/结构） | idea Layer-2 B1/B2 | ⬜ 未实现 | 应走"降检测阈值+top-k CLIP 复核"，**不要**用严格拒绝 |
| 语义 frontier（room prior 注入） | idea Layer-3 | ⬜ 未实现 | SPL 导向，超时仍 34/130 |
| 修复 records 重复写入 | idea D2 / 工程 5.1 | ⬜ 未实现 | 每 subtask 重复写 ~3 次，不影响指标 |

---

## 2. 实验记录（按时间追加，勿删）

### [EXP-000] Baseline — gpt-5.4 首集 278
- **日期**：2026-07-01/02
- **目录**：`results/exp000_baseline_gpt-5.4_278subtasks`
- **改动**：无（原始 MSGNav 流程）
- **结果**：Success **55.04%**（153/278），SPL **33.65%**
  - object 62.63% ｜ description 54.95% ｜ image 46.59%（image 最难）
- **失败构成**：提前放弃(<10步) 62(50%) ｜ 中途 33 ｜ 超时(≥50) 30
- **阈值消融**：≤0.25m 55.0% → ≤0.5m 59.4% → ≤1.0m 60.8%
- **详细分析**：见 `experiment_analysis_gpt5.4.md`

### [EXP-001] Refine（到达后精对准，走近再停）— 已证伪
- **日期**：2026-07-02　**目录**：`results/exp001_refine_v1_gpt-5.4_43subtasks_partial`（仅 43 子任务）
- 到达后无条件重采样更近的 0.45m corner 走 1–3 步再停。结果净 −1，但样本太小 + 触发逻辑有明显缺陷（18 次触发里绝大多数打在 baseline 已经 <0.25m 的已成功案例上，属于"没事找事"）——**非严格证伪，但暴露的教训（只对"还没到"的目标 refine）被后续 EXP-004/004b 继承，代码未保留**。

### [EXP-002] Strict stop-check（保守停止，原始全局设计）— 已证伪
- **日期**：2026-07-03　**目录**：`results/exp002_stopcheck_strict_gpt-5.4_278subtasks`
- 到达后叫 VLM 二次复核，不通过就**驳回停止、释放目标、回frontier重探索**。278 全集 Success **53.24%（−1.80pp）**，17 例"灾难走失"（baseline 已站在 0.06–0.24m，被误判后放弃目标走到 1–11m 外）。❌ **证伪**：驳回→重探索的代价远超收紧本身的收益（≤1.0m 口径下其实反超 baseline，说明"推近"有信号，但要用走近而非拒绝去兑现）。

### [EXP-003] Stop-check 收窄到 image-only — 未验证即被取代
- **日期**：2026-07-04
- 把 EXP-002 的复核范围收窄到只对 image 任务生效，减少误伤 object/description。预估 +1.1pp，但未在 278 全集重跑验证，后续实验（EXP-005 起）直接把 `stop_check.enabled` 全局设为 `false`，这条分支未再启用。

### [EXP-004] Refine v2（到达后可见性微移）— 已证伪
- **日期**：2026-07-04　**目录**：`results/exp004_refine_v2_vis_gpt-5.4_34subtasks_partial`
- 用系统自带的 `compute_visibility` 在更近的候选环上找更优可见 viewpoint，有 headroom 才微移、否则回退。实测可见性在 0.75m 环已饱和（mean vis0=0.935），32 次触发几乎全是 `skip:no_headroom`，34/278 subtask 后终止。❌ **证伪**：可见性作为 proxy 无梯度，refine 等于 no-op。

### [EXP-004b] Refine v2b（距离驱动 + 可见性保底）— 已证伪，方向性结论
- **日期**：2026-07-04　**目录**：`results/exp004b_refine_v2_dist_gpt-5.4_113subtasks_partial`
- 改成朝目标中心靠近（可见性只做保底不再是目标）。74/278 时确认 **回归 −6.4~−6.8pp**（image −13.7pp 最重），10 个新增失败里 7 个由 refine 直接导致（0.1m→0.3m）。❌ **证伪**：GT viewpoint 通常不在物体中心方向上（尤其 image 类，如正对 picture/mirror 的位置常与"靠近中心"反向）。**至此三次 refine 尝试（001/004/004b）全部失败，"停止后精对准"整个方向放弃**，详见教训 #6。

### [EXP-005] Episode Memory（跨 subtask 记忆锚点）
- **日期**：2026-07-04
- **目录**：`results/exp005_episode_memory_v1_gpt-5.4_275subtasks`
- **动机**：Baseline 分析发现 19 个 subtask 曾到达 GT (<0.25m) 但最终失败（占失败的 15.2%）。同一 episode 中多个 subtask 常指向同一物理目标。数据验证：60 组 (episode, GT obj) 有 ≥2 subtask 共享目标，22 组混合成功+失败，其中有 11-23 个 subtask 理论上可通过记忆拯救。
- **改动**：新增 `EpisodeMemory` 类（`run_goatbench_evaluation.py`），两级匹配：
  1. **Level 1** — target_index 精确匹配：VLM 选中的 scene graph obj_id 与记忆完全相同
  2. **Level 2** — 空间近邻匹配：target bbox.center 距记忆 position < 1.5m（覆盖同物体不同 detected ID 的情况）
  - Subtask 成功时记录 (target_id, position, angle, success, dist)
  - 后续 subtask VLM 识别同一目标时，用记忆 viewpoint 替代 VVD（跳过 0.75m 环计算，直接导航到已知成功位置）
- **安全性**：仅替代 viewpoint 选择，end_check 仍正常运行。记忆位置 <0.20m 才启用（min_confidence_dist 门控）。不动 = baseline（worst case）。
- **参数**：`spatial_match_radius=1.5m`, `min_confidence_dist=0.20m`
- **预估**：Level 1 可救 11 个，Level 2 额外 ~12 个，保守 +4-6pp → 59-61%
- **运行命令**：`python start_multiprocess.py --task goatbench --devices 0,1,2,3 --total_scenes 36 --splits 1`
- **结果**：✅ **已运行完成**（275/278 subtasks，1 scene 因 worker 卡死终止，已重命名为 `..._v1_gpt-5.4_275subtasks`），见下方数据分析。

### [EXP-005 数据分析] Spatial 匹配污染根因 + 按 match type / 模态分解
- **日期**：2026-07-04/05
- **口径**：exp005 v1（275 subtasks，去重后）vs `baseline_no_refine_gpt-5.4_278subtasks`（278），仅取重叠子任务做因果分解；用 trajectory 里的 `memory_hit` 字段区分"记忆命中导致的翻转" vs "无关重跑噪声导致的翻转"。
- **原始净效应**：重叠子任务上 raw gained 25 / lost 25，**净 0**——被非确定性噪声完全掩盖。
- **因果分解**（按 `memory_hit` 是否存在）：memory 命中导致的翻转 gained 13 / lost 6，**净 +7**；无关噪声翻转 gained 12 / lost 19，净 −7。两者相加抵消为表面上的净 0。
- **按 match type**：`exact_id`（Level 1，精确 obj_id 匹配）58 次命中，0 次导致回归，净 +6——**完全安全**。`spatial`（Level 2，空间近邻）50 次命中，6 次回归，净 +1——**回归的唯一来源，勉强打平**。
- **按模态**：image 受益最大（+8.33pp raw）；object（−4.08pp）、description（−3.30pp）反而回归；`spatial + image` 组合是最差子集（13 次命中仅 53.8% 成功）。
- **根因**（已用具体案例定位）：Level 2 旧实现比较的是"过去成功时 agent 的站立位置"（`position`，因 VVD 通常站在离目标物体中心 ~0.75m 的可见性环上）vs "新目标的物体中心"（`target_center`）——**参照系不一致**，实际上放大了名义 1.5m 匹配半径，导致跨物体/跨类别误配（案例：`00831-yr17PDCnDDW` 场景 plant↔vase 互相污染；`00821-eF36g7L6Z9M` 场景 mirror↔picture 互相污染）。
- **附带确认**：`stop_check.enabled: false` 使 `_strict_stop_reasons()` 在第一行就短路返回 `[]`，config 里配置的 `strict_for_task_types:["image"]`（EXP-003 的收窄设计）当前完全休眠、未生效——exp005 里 image 模态的提升**全部来自 episode memory**，与 stop-check 无关。

### [EXP-005b] Episode Memory 修复 v2（空间匹配四点修复 + record() stale-target 归因修复）
- **日期**：2026-07-05
- **目录**：`results/exp005b_episode_memory_v2_gpt-5.4`（重新启动，见下）
- **改动 1 — 空间匹配四点修复**（`EpisodeMemory` 类，`run_goatbench_evaluation.py`）：
  1. Level 2 比较对象从 `position`（agent 站立点）改为 `target_center`（物体 bbox 中心 vs 物体 bbox 中心），修正参照系不一致的根本 bug。
  2. Level 2 新增 `target_class` 门控：必须与查询目标同一归一化类别才允许匹配（`_normalize_text` 处理大小写/空白）。
  3. Level 2 新增置信度门控：候选 anchor 必须 `final_dist <= min_confidence_dist`（此前只有 Level 1 有这个门控）。
  4. `spatial_match_radius` 从 1.5m 收紧到 0.6m。
  - **验证**：`py_compile` 通过；`msgnav` conda 环境下 5 组合成测试用例全部通过（跨类别近距离物体→拒绝；同类别半径内→spatial 命中；同类别但超出半径→拒绝；exact_id→命中；低置信度 anchor→拒绝）。
- **改动 2 — record() stale-target 归因修复**（同文件，2026-07-05 新发现）：
  - **问题**：`last_target_index`/`last_target_center` 只在 VLM 决策为 `object`/`image` 时更新（`choose_every_step` 下，只有 frontier 类型的 pending target 会被强制重选，object/image 类型选定后不会被重置）。若某 subtask 在最后一次 object/image 决策之后，又发生了 frontier 决策（目标被放弃/切换回探索），循环结束时 `record()` 仍会用**旧的、与最终 `pts` 无关的** `last_target_index`/`last_target_center` 写入记忆锚点——若这次 subtask 恰好几何巧合地成功（`success_by_distance=True`），就会把成功错误地归因到一个不相关的物体上，污染后续 subtask 的 Level 1/2 匹配。
  - **实证检查**：在 exp005 v1 的 275 个 subtask 轨迹上排查"最后一次 object/image 决策之后又出现 frontier 决策"的案例，共 **8 例**，全部 `success_by_distance=False`——**在已跑完的数据里这个 bug 尚未真正触发过错误归因**，但逻辑上是可达的、需要修复（不能依赖运气）。
  - **修复**：新增 `last_decision_type` 变量，在每次 VLM 查询后无条件更新（不管返回 object/image/frontier）。`record()` 调用处新增门控：只有当 `last_decision_type in ("object", "image")`（即最后一次决策仍是该目标本身，没有被后续 frontier 决策取代）才把 `last_target_index`/`last_target_center` 传给 `record()`；否则传 `None`（`EpisodeMemory.record()` 对 `target_obj_id is None` 已有 no-op 保护，天然安全）。
  - **验证**：`py_compile` 通过；对 exp005 v1 数据重放确认这 8 例（均为失败）在修复后行为不变（仍不产出锚点），修复对已有数据是无损的防御性修正。
- **运行**：旧 v1 目录有 1 个 scene（`00871-VBzV5z6i1WS`）worker 卡死 3+ 小时（GPU 显存占用但 0% 利用率），已 kill 干净。08:24 曾用**四点修复但未含 stale-target 修复**的代码启动过一次 `exp005b`，13 分钟后（仅 2/36 scene 落盘）发现 stale-target bug 并修复，遂 kill 重启，避免用不完整修复的代码产出可能需要重新分析的结果。**当前运行**：4×RTX 5090（GPU 0-3），conda env `msgnav`，36 scenes × 1 split = 278 subtasks，08:52 启动。
- **结果**：⬜ **待运行完成后重新做因果分解**（对比新版 spatial/exact_id 拆分 + 按模态拆分，验证净效应是否从"+7 causal / 0 raw"提升）。
- **补记（2026-07-06）**：已跑完 278 subtasks，口径A（0.25m viewpoint）下 Success **54.32%**（−0.72pp vs baseline）、SPL **38.22%**（**+4.57pp**）——Success 基本打平，但 SPL 明显提升（记忆锚点跳过重复探索，路径更短）。

### [EXP-006] 移除 VVD（可见性环）+ 成功判定改为 GT 物体中心 + 目标点加 island 约束
- **日期**：2026-07-06　**目录**：`results/exp006_nav_island_fix_gpt-5.4`（对照 `exp005b_episode_memory_v2_gpt-5.4`）
- **改动**（三处一起做，**未单独消融**）：
  1. **移除 VVD**：删掉 `Visibility_based_Viewpoint_Decision`/`generate_candidate_viewpoints`/`select_navigation_corner`（`src/query_vlm.py`+`src/utils.py`，约180行），导航目标直接用物体 **bbox 中心**，交给 planner 的最近可导航点逻辑处理。动机：EXP-004/004b 已证明"0.75m 可见性环 viewpoint"本身就是"锁对物体但停远"的根源之一，源头简化比停止后修正更直接。
  2. **成功判定口径改变**：`success_distance` 从 0.25m（到 GT viewpoint）改成 **1.0m（到 GT 物体中心）**，`logger_goatbench.py` 新增 `goal_positions` 字段，`agent_subtask_distance` 改用它计算——配合改动1，目标既然不再瞄准 viewpoint，评判也同步换成物体中心距离。
  3. **island 约束**：`tsdf_planner.set_next_navigation_point` 里 object/image 目标点从"全图最近的 unoccupied 点"改成"当前可达连通域内最近点"（`unoccupied & self.island`），避免落到墙对面/不可达房间。
- **结果**（⚠️ 与 EXP-005b 判定口径不同，**不可直接比较**，仅供参考）：

  | | 判定口径 | Overall | description | image | object | SPL |
  | --- | --- | --- | --- | --- | --- | --- |
  | exp005b | 0.25m viewpoint | 54.32% | 49.45% | 48.86% | 63.64% | 38.22% |
  | exp006 | 1.0m 物体中心 | 56.12% | 48.35% | **57.95%** | 61.62% | 34.11% |

- **结论：⬜ 待消融，暂不能下因果结论**。image 涨了 9pp，但口径变宽（1.0m vs 0.25m）本身就会抬高数字，且三处改动绑在一起跑，无法归因是 VVD 移除、口径变化还是 island fix 起的作用。
- **下一步**：在同一口径（1.0m 物体中心）下单独跑一次"只加 island 约束、不改 VVD"的对照组，隔离出 island fix 的真实增量；有余力再补一次"只移除 VVD、不加 island 约束"做完整消融。

### [EXP-007] EpisodeMemory 重设计（冻结快照 → 场景图节点薄标注层）
- **日期**：2026-07-07
- **改动文件**：`run_goatbench_evaluation.py`（`EpisodeMemory` 类重写）、`src/query_vlm.py`、`src/explore_utils.py`、`cfg/eval_goatbench.yaml`
- **动机**：EXP-005b 的 EpisodeMemory 有三个结构性问题——(1) `record()` 存**冻结几何快照**、命中后硬覆盖导航点，而场景图节点 `bbox.center` 会随后续视角合并持续精修，快照反而更差；(2) 手写 XZ 空间匹配 + 每个 image/description 候选都调 VLM 身份门，完全没用场景图已有的 CLIP/room 信号，昂贵；(3) 命中后「VLM 决策后硬覆盖」与 CLR「注入 prompt 让 VLM 自己推理」的哲学割裂。
- **重设计**（利用「同 episode 内场景图节点 id 稳定」这一前提，把 EpisodeMemory 变成挂在节点上的薄结果标注层）：
  1. **几何委托活节点**：丢弃 authoritative `target_center`，query 时一律从 `scene.objects[id]["bbox"].center` 活取（`_live_center`）。
  2. **图信号先筛、VLM 兜底**：`_graph_identity_prefilter` 用 CLIP 余弦（`>=0.92` pass / `<=0.70` reject / 中间 uncertain）+ room 一致性；只有 uncertain 且非 object 任务才落 VLM 身份门（复用 `_passes_identity_gate`）。
  3. **两层软先验耦合（VLM 可否决）**：(a) prompt 软先验——`annotate_nodes` 在物体候选行标注「previously reached successfully / previously rejected」，另加 memory 汇总块（`MEMORY_HINTS` 经 `query_vlm.py`→`explore_utils.py` 贯通）；(b) planner 先验——命中时用**活节点中心**作导航点，不再硬覆盖为快照。
  4. **负向记忆**：记 confirmed-negative（子任务末 `distance_fail` + mid-loop `rejected_strict`），`query()` surface 出来，prompt 里注入软标注（**不硬排除**，防误伤）。
  5. **作用范围 = episode 内**（保持每 episode 重置）。
- **机制验证**（4 场景 smoke，29 subtasks，**全部新路径**，⚠️**非受控 A/B、非 benchmark 数字**）：
  - ✅ 0 崩溃，端到端跑通。
  - ✅ **11 次 exact_id 正向命中，全部是同一物理物体的合法多模态复访**——典型案例 `00820-mL8ThkuaVTM` Episode 0 全 9 个子任务都指向同一台冰箱（node 128），description/image/object 三模态交替查询，VLM 每次落到同一节点、内存正确识别「曾到达」、走活中心，8 次重访全部成功（0.02–0.08m），锚点择优正确（引用 final_dist 最小的 subtask 1）。
  - ✅ **VLM 身份门调用归零**——CLIP 预筛把 exact_id 的身份判别全部短路（余弦 ~1.0），这是重设计的主要效率诉求，实测不是"减少"而是"消除"。
  - ✅ **41 次负向标注正确注入 prompt**（2 个场景）。
- **结论：机制层已确证生效（同 episode 内曾到达的地方能被正确重访），效果层未验证**。smoke 4 进程全是新路径、无同代码 baseline，SR/SPL（合计 55.2/44.0）不可归因到重设计；场景间差异（100% vs 25%）主要是场景难度（object 类 SR 仍只 33%，老问题）。**要判断 SPL 是否优于 legacy 冻结快照，需另跑同代码同场景切开关的受控对照**（本轮未做）。
- **代码清理**：机制确证后，**移除了 legacy A/B 路径**（`_record_legacy`/`_query_legacy`/`_best_success` 及 `redesign_enabled` 分支），新路径成为唯一实现；config 删除 `redesign_enabled` 开关。`_passes_identity_gate` 保留（新路径 uncertain 兜底仍用）。删除 smoke 临时文件。
- **未做**：Phase-2 view-pose bias（`use_view_pose_bias` 默认关，flag 已留，`query()` 已暴露 view_position/view_angle，planner 偏置逻辑未接）。

---

## 3. 关键教训（Lessons，反复看）

1. **非确定性**：43 子任务上重跑就翻 3–4 个。**结论只在 278 全首集下判定**，小样本只能看趋势/debug。
2. **"推近"要用 refine（走向目标），绝不能用 reject（释放目标重探索）**——后者会把 0.08m 的必成功变成 11m 的彻底失败（EXP-002 实证）。
3. **refine 只对"还没到"的目标做，已进成功半径的绝不动**（EXP-001 的下行全来自动了已成功案例）。
4. **hard target（mirror/picture/glass/plant）要靠检测召回增强,不要靠严格拒绝**（EXP-002 里对它们的拒绝是净负）。
5. **改动必须可追踪**：把触发/动作写进 `records` 的 note 或 trajectory（EXP-001 的 `n_refine_steps` 就是靠这个才复盘出触发点错了）。
6. **几何 proxy 不能替代 GT 信息**：可见性饱和（0.93+），dist-to-center 与 dist-to-GT 无因果关系。任何 post-hoc refinement 在没有 GT oracle 的条件下都是赌博。Agent end_check 通过时的 pose 已经是系统最优，不要画蛇添足。
7. **跨 subtask 记忆类改动要检查"归因时刻"是否与"决策时刻"一致**：`choose_every_step` 下，只有 frontier 类型的 pending target 会被强制重选，object/image 一旦选中就不再重置——若循环结束前又切回 frontier，记录时用的 "last_*" 变量可能是已被放弃的旧目标（EXP-005b 的 stale-target 归因 bug）。这类 bug 在数据里可能长期"潜伏不触发"（275 个里 8 个可达但 0 个真的写错），**不能靠"没观察到坏结果"证明逻辑没问题**，要专门检查决策序列本身。
8. **改判定口径（success_distance/goal 参照点）跟改导航逻辑要分开验证**：口径一变，跨口径的 Success/SPL 数字就失去可比性（EXP-006 把 VVD 移除、口径从 viewpoint 改物体中心、island 约束三件事一起跑，结果无法归因）。以后每次既想换口径又想换逻辑时，至少要在**同一口径**下补一组只变一个变量的对照，否则宁可先分两次提交分别验证。
9. **「机制生效」≠「效果为正」，两者要分开验证**（EXP-007）：smoke 能确认机制按设计动作在跑（exact_id 命中、CLIP 预筛短路 VLM、负向标注注入 prompt、同物体正确重访），但**不能**判断改动是否提升 SR/SPL——机制触发也可能拖累（如活中心漂移）。判断效果方向唯一可靠办法是同代码同场景切开关的受控 A/B；全是新路径、无 baseline 的多进程跑只能验机制、不能验效果，其 SR/SPL 绝对值不可归因。
