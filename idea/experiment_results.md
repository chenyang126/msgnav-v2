# MSGNav 实验结果（Experiment Results）

> **本文只记录「跑了什么、什么口径、什么数字、什么结论」**。失败模式的深入拆解见 [`data_analysis.md`](./data_analysis.md)；方向判断、改进方案、教训见 [`idea_analysis.md`](./idea_analysis.md)。
>
> **维护规则**：每做一次评测就在「实验记录」追加一条（勿删旧记录），同步更新顶部速查表。
>
> **统一口径**：GOAT-Bench Val Unseen，`--splits 1`（每 scene 仅 episode 0），278 subtasks，gpt-5.4，4×RTX5090。指标以 `records_*.jsonl` 去重（按 subtask_id 保留最后一条）为准，SPL 须在 `msgnav` conda 环境算（普通 python 缺 numpy 会显示 SPL=0）。
>
> ⚠️ **成功判定口径在 EXP-006 变过**：EXP-000~005b 用 `success_distance=0.25m`（到 GT viewpoint 距离）；EXP-006 起改 `success_distance=1.0m`（到 GT 物体中心距离）。**跨这条线的 Success/SPL 不可直接比较**。
>
> ⚠️ **非确定性**：43 子任务上光重跑就会翻 3–4 个成败。**任何结论只在 278 全首集下判定**，小样本只能看趋势/debug。

---

## 0. 结论速查表

**口径 A**（`success_distance=0.25m`，到 GT viewpoint）：

| 实验 | 结果目录 | n | Success | SPL | vs baseline | 结论 |
| --- | --- | --- | --- | --- | --- | --- |
| **EXP-000 Baseline** | `exp000_baseline_gpt-5.4_278subtasks` | 278 | **55.04%** | 33.65% | — | 基准 |
| EXP-001 Refine v1 | `exp001_refine_v1_..._43subtasks_partial` | 43 | 65 vs 67 | — | 净 −1 | ❌ 证伪（触发点错） |
| EXP-002 Strict stop-check | `exp002_stopcheck_strict_..._278subtasks` | 278 | 53.24% | 32.96% | **−1.80pp** | ❌ 证伪（灾难走失） |
| EXP-004 Refine v2 可见性 | `exp004_refine_v2_vis_..._34subtasks_partial` | 34 | 67.6% | — | — | ❌ 证伪（可见性饱和 no-op） |
| EXP-004b Refine v2b 距离 | `exp004b_refine_v2_dist_..._113subtasks_partial` | 74 | 48.6% | ~30% | **−6.4pp** | ❌ 证伪（近center≠近GT） |
| EXP-005 Episode Memory v1 | `exp005_episode_memory_v1_..._275subtasks` | 275 | 净0(raw)/+7(因果) | — | 噪声抵消 | ⚠️ exact_id 安全、spatial 参照系错 |
| EXP-005b Episode Memory v2 | `exp005b_episode_memory_v2_gpt-5.4` | 278 | 54.32% | **38.22%** | −0.72pp / **SPL +4.57pp** | ✅ Success 打平、路径效率明显提升 |

**口径 B**（`success_distance=1.0m`，到 GT 物体中心，EXP-006 起）：

| 实验 | 结果目录 | n | Success | SPL | 结论 |
| --- | --- | --- | --- | --- | --- |
| EXP-006 移除VVD+island（**legacy 记忆**） | `exp006_nav_island_fix_gpt-5.4` | 278 | 56.12% | 34.49% | ⬜ 三处改动未消融，不能跟口径A比 |
| EXP-007 Memory 重设计（**redesign 记忆**） | `exp007_epmem_redesign_gpt-5.4` | 278 | 54.32% | 30.19% | ❌ vs exp006 **−1.80pp SR / −4.30pp SPL**（image −5.68/−6.63 最惨）；机制成功但效果净负 |
| EXP-008 Memory 降级 log-only（**默认不干预**，**当前 baseline**） | `exp008_epmem_logonly_gpt-5.4` | 278 | **52.88%** | **32.01%** | vs exp007 **+1.82pp SPL / −1.44pp SR** | ⚠️ 记忆退回纯观测层；SPL 回升但 SR 反而是三者最低（低于 exp006 56.12 / exp007 54.32），未如预期回到 legacy-nav 水平 |
| EXP-009 Opportunistic Stop v1（same-class 近邻触发 end_check） | `exp009_opportunistic_stop_gpt-5.4` | 278 | **47.12%** | 29.47% | ❌ vs exp007 **−7.19pp SR / −0.72pp SPL**；触发子集 125 个 SR **56.0→40.0**，class-only 近邻触发会系统性误停 |

> exp006/exp007 导航栈、口径、config 全相同（仅差被删的 `redesign_enabled` 死 flag），唯一实质区别是记忆实现 legacy(exp006) vs redesign(exp007)——**这实际构成了 legacy↔redesign 的记忆 A/B**，见 EXP-007 记录与 `data_analysis.md §3.2`。SPL 用统一 jsonl 平均法重算（exp006 原记 34.11 为 pkl 法，差 ~0.4pp，不影响结论）。

---

## 1. 实验记录（按时间追加，勿删）

### [EXP-000] Baseline — gpt-5.4 首集 278
- **日期** 2026-07-01/02 ｜ **目录** `results/exp000_baseline_gpt-5.4_278subtasks`
- **改动** 无（原始 MSGNav 流程）
- **结果** Success **55.04%**（153/278），SPL **33.65%**；object 62.63% ｜ description 54.95% ｜ image 46.59%
- **阈值消融** ≤0.25m 55.0% → ≤0.5m 59.4% → ≤1.0m 60.8%
- **深入分析** 见 `data_analysis.md`（失败模式、分场景、分物体）

### [EXP-001] Refine v1（到达后无条件走近再停）— ❌ 证伪
- **日期** 2026-07-02 ｜ **目录** `results/exp001_refine_v1_gpt-5.4_43subtasks_partial`（仅 43）
- **改动** 到达后无条件重采样更近的 0.45m corner 走 1–3 步再停
- **结果** 净 −1，但样本太小 + 18 次触发绝大多数打在 baseline 已 <0.25m 的已成功案例上（没事找事）
- **结论** 非严格证伪，但教训「只对还没到的目标 refine」被 004/004b 继承；代码未保留

### [EXP-002] Strict stop-check（到达后驳回→释放目标重探索）— ❌ 证伪
- **日期** 2026-07-03 ｜ **目录** `results/exp002_stopcheck_strict_gpt-5.4_278subtasks`
- **改动** 到达后 VLM 二次复核，不通过就驳回停止、释放目标、回 frontier 重探索
- **结果** Success **53.24%（−1.80pp）**，17 例灾难走失（已站 0.06–0.24m 被误判后走到 1–11m 外）
- **结论** 驳回→重探索代价远超收益；但 ≤1.0m 口径下反超 baseline，说明「推近有信号，但要用走近而非拒绝兑现」。EXP-005 起 `stop_check.enabled=false` 全局关闭

### [EXP-003] Stop-check 收窄到 image-only — 未验证即被取代
- **日期** 2026-07-04
- 把 EXP-002 复核范围收窄到只对 image 生效，预估 +1.1pp，但未在 278 全集重跑；后续（EXP-005 起）直接全局关 stop-check，本分支未再启用

### [EXP-004] Refine v2（到达后可见性微移）— ❌ 证伪
- **日期** 2026-07-04 ｜ **目录** `results/exp004_refine_v2_vis_gpt-5.4_34subtasks_partial`
- **改动** 用 `compute_visibility` 在更近候选环找更优可见 viewpoint，有 headroom 才微移
- **结果** 可见性在 0.75m 环已饱和（mean vis0=0.935），32 次触发几乎全 `skip:no_headroom`，34/278 后终止
- **结论** 可见性 proxy 无梯度，refine ≈ no-op

### [EXP-004b] Refine v2b（距离驱动 + 可见性保底）— ❌ 证伪
- **日期** 2026-07-04 ｜ **目录** `results/exp004b_refine_v2_dist_gpt-5.4_113subtasks_partial`
- **改动** 改朝目标中心靠近（可见性只做保底）
- **结果** 74/278 时确认回归 **−6.4~−6.8pp**（image −13.7pp 最重），10 个新增失败里 7 个由 refine 直接导致（0.1m→0.3m）
- **结论** GT viewpoint 通常不在物体中心方向（尤其 image 正对 picture/mirror 常与靠近中心反向）。**三次 refine（001/004/004b）全败，「停止后精对准」整方向放弃**

### [EXP-005] Episode Memory v1（跨 subtask 记忆锚点）— ⚠️
- **日期** 2026-07-04 ｜ **目录** `results/exp005_episode_memory_v1_gpt-5.4_275subtasks`（1 scene worker 卡死终止）
- **动机** baseline 有 19 个 subtask 曾到达 GT(<0.25m) 但最终失败（占失败 15.2%）；同 episode 多个 subtask 常指同一物理目标
- **改动** 新增 `EpisodeMemory` 两级匹配：L1 target_index 精确匹配；L2 空间近邻（center 距记忆 position <1.5m）。成功时记 (target_id, position, angle, success, dist)，后续复用记忆 viewpoint 替代 VVD
- **参数** `spatial_match_radius=1.5m`, `min_confidence_dist=0.20m`
- **运行** `python start_multiprocess.py --task goatbench --devices 0,1,2,3 --total_scenes 36 --splits 1`
- **结果** raw 净 0，因果分解净 +7（详见 `data_analysis.md`：spatial 参照系污染根因）

### [EXP-005b] Episode Memory v2（空间匹配四点修复 + stale-target 归因修复）— ✅
- **日期** 2026-07-05/06 ｜ **目录** `results/exp005b_episode_memory_v2_gpt-5.4`
- **改动 1（空间匹配四点修复）** L2 比较对象 position→target_center（修参照系 bug）；L2 加 target_class 门控；L2 加置信度门控（`final_dist<=min_confidence_dist`）；`spatial_match_radius` 1.5m→0.6m
- **改动 2（stale-target 归因修复）** 新增 `last_decision_type`，每次 VLM 查询后无条件更新；`record()` 仅当最后一次决策仍是 object/image 才归因，否则传 None（防把成功错误归到已放弃的旧目标）。在 v1 275 轨迹上排查该 bug 8 例可达但均未真触发
- **结果** Success **54.32%（−0.72pp）**、SPL **38.22%（+4.57pp）**——Success 打平、SPL 明显提升（记忆跳过重复探索，路径更短）

### [EXP-006] 移除 VVD + 成功判定改 GT 物体中心 + 目标点 island 约束
- **日期** 2026-07-06 ｜ **目录** `results/exp006_nav_island_fix_gpt-5.4`（对照 exp005b）
- **改动（三处一起做，未单独消融）**
  1. 移除 VVD（删 `Visibility_based_Viewpoint_Decision` 等约 180 行），导航目标直接用 bbox 中心
  2. 成功判定 `success_distance` 0.25m(viewpoint)→**1.0m(物体中心)**，`logger_goatbench.py` 加 `goal_positions`
  3. island 约束：object/image 目标点从「全图最近 unoccupied」改「可达连通域内最近点」（`unoccupied & self.island`）
- **结果**（⚠️ 与 EXP-005b 口径不同，不可直接比较）

  | | 口径 | Overall | desc | image | object | SPL |
  | --- | --- | --- | --- | --- | --- | --- |
  | exp005b | 0.25m viewpoint | 54.32% | 49.45% | 48.86% | 63.64% | 38.22% |
  | exp006 | 1.0m 物体中心 | 56.12% | 48.35% | **57.95%** | 61.62% | 34.11% |

- **结论** ⬜ 待消融。image +9pp 但口径变宽本就抬数字，三处绑一起无法归因。**下一步**：同口径下单跑「只加 island、不改 VVD」隔离 island 增量

### [EXP-007] EpisodeMemory 重设计（冻结快照 → 场景图节点薄标注层）
- **日期** 2026-07-07 ｜ **改动文件** `run_goatbench_evaluation.py`（`EpisodeMemory` 重写）、`src/query_vlm.py`、`src/explore_utils.py`、`cfg/eval_goatbench.yaml`
- **改动**（利用「同 episode 内场景图节点 id 稳定」，把 memory 变成挂在节点上的薄结果标注层）
  1. **几何委托活节点**：丢弃冻结 `target_center`，query 时一律 `scene.objects[id]["bbox"].center` 活取（`_live_center`）
  2. **图信号先筛、VLM 兜底**：`_graph_identity_prefilter` 用 CLIP 余弦（≥0.92 pass / ≤0.70 reject / 中间 uncertain）+ room 一致性；仅 uncertain 且非 object 才落 VLM 身份门
  3. **两层软先验（VLM 可否决）**：prompt 软先验（`annotate_nodes` 标注「previously reached/rejected」+ `MEMORY_HINTS` 块）；planner 先验（命中用活节点中心，不再硬覆盖快照）
  4. **负向记忆**：记 confirmed-negative（末 `distance_fail` + mid-loop `rejected_strict`），surface 到 prompt 软标注（不硬排除）
- **机制验证**（4 场景 smoke，29 subtasks，全新路径）
  - ✅ 0 崩溃端到端跑通；✅ exact_id 正向命中全是同一物理物体合法多模态复访（典型 `00820-mL8ThkuaVTM` Ep0 全 9 子任务指向同一冰箱 node 128，多次重访全成功 0.02–0.08m）；✅ VLM 身份门调用几乎归零（CLIP 预筛全短路）；✅ 负向标注正确注入 prompt
- **全量结果**（`exp007_epmem_redesign_gpt-5.4`，278 subtasks，口径 B，2026-07-08 跑完）

  | | Overall | object | description | image | SPL |
  | --- | --- | --- | --- | --- | --- |
  | exp006（legacy 记忆） | 56.12% | 61.62% | 48.35% | 57.95% | 34.49% |
  | exp007（redesign 记忆） | 54.32% | 60.61% | 49.45% | 52.27% | 30.19% |
  | **Δ（redesign−legacy）** | **−1.80** | −1.01 | +1.10 | **−5.68** | **−4.30** |

  （SPL 为各任务类型 SPL：object 27.72 / desc 28.07 / image 35.16。全量 SPL 30.19。）
- **记忆行为对比**（从 log/trajectory 聚合，实锤 exp006=legacy、exp007=redesign）：命中总数都是 88，但 **spatial 命中 28(exp006)→3(exp007)**、**VLM 身份门调用 →仅 5 次**、live_center 覆盖 88 次——重设计的效率诉求（图信号短路 VLM、收紧 spatial 误配）**机制上完全达成**。
- **结论** ✅ **机制层已确证**（重访正确、效率大增）；❌ **效果层净负**——同导航栈同口径下，redesign 比 legacy **回落 SR −1.80pp / SPL −4.30pp，image 最惨（SR −5.68 / SPL −6.63）**。「机制生效 ≠ 效果为正」被坐实（详见 `idea_analysis.md §4.2`）。
  - ⚠️ **口径/归因说明**：exp006 存档 config 于 07-07 被覆盖（显示 `redesign_enabled:true`），exp006=legacy 是由 **log 签名**判定（28 次 spatial 命中、无 "using live target center"），非 config；且两次均单跑、非确定（±~1.4pp/278），overall −1.80 接近噪声带，但 image −5.68 / SPL −4.30 超出纯噪声。
- **代码清理** 机制确证后移除 legacy A/B 路径（`_record_legacy`/`_query_legacy`/`_best_success` 及 `redesign_enabled` 分支），新路径成为唯一实现，config 删 `redesign_enabled` 开关。`_passes_identity_gate` 保留（uncertain 兜底）
- **遗留** ⚠️ 清理已删 `redesign_enabled`，**再想切开关做受控 A/B 需回退或重加 flag**（见 `idea_analysis.md §4.2`）；Phase-2 view-pose bias 未接（`use_view_pose_bias` 默认关、flag 已留、`query()` 已暴露 view_position/view_angle，planner 偏置未接）

### [EXP-008] EpisodeMemory 降级 log-only（默认零决策干预）— ⚠️ 当前 baseline
- **日期** 2026-07-07/08 ｜ **目录** `results/exp008_epmem_logonly_gpt-5.4` ｜ **状态** ✅ 278/278 全量跑完
- **动机** EXP-007 确证「记忆机制生效但 prompt/导航干预净负、image 最惨」。EXP-008 把记忆彻底退回**纯观测层**：只 `_live_center` 活取几何 + `_graph_identity_prefilter`（CLIP/room）落盘 anchor/candidate 供分析，**默认不向 prompt 或导航点注入任何 broad prior**。目的是隔离出「不带记忆干预的干净基线」，供后续 EXP-009+ 在其上叠加单点机制（同 config 已被 EXP-009 继承）。
- **改动**（相对 EXP-007，仅 `cfg/eval_goatbench.yaml` 开关，导航栈/口径不变）
  1. `surface_negatives_in_prompt: true→false`（关负记忆 prompt 软标注）
  2. `prompt_prior.annotate_positive: true→false`、`prompt_prior.memory_block: true→false`（关正向 prompt 先验与 MEMORY_HINTS 块）
  3. 新增 `live_center_fallback`（默认 `enabled:false`；仅当显式开启且 exact_id 多次成功才允许覆盖 VLM 导航点）
- **全量结果**（口径 B，278 subtasks，以 EXP-008 为 baseline，Δ = exp007 − exp008）

  | | Overall SR | description | image | object | SPL | 平均步数 | 平均探索距离 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | **exp008（log-only，baseline）** | **52.88%** | 51.65% | 48.86% | 57.58% | **32.01%** | 12.72 | 12.23 |
  | exp007（redesign 记忆干预） | 54.32% | 49.45% | 52.27% | 60.61% | 30.19% | 13.21 | 12.78 |
  | **Δ（exp007 − exp008）** | **+1.44** | −2.20 | **+3.41** | +3.03 | **−1.82** | +0.49 | +0.55 |

  分类型 SPL（exp008 / exp007）：description **37.40 / 28.07（Δ−9.33）**、image 32.95 / 35.16（Δ+2.21）、object 26.22 / 27.72（Δ+1.50）。

- **分维度对比**
  - **按输入类型**：记忆干预对 **image / object 双赢**（SR+SPL 均升），但对 **description 净伤害**（SR −2.20、SPL **−9.33**）。全量 SPL 转负几乎全由 description 一类拖累——与 EXP-007「记忆干预让轨迹绕远」一致，且伤害集中面从 image 修正到 description。
  - **按难度（GT 探索距离）**：中距离 3–6m 记忆干预增益最大（SR +7.14），但远距离 10m+ 反而拖累（SR −4.88 / SPL −2.66）；SPL 在几乎所有距离段都因干预下降。
  - **配对翻转（278 一一配对）**：exp007 相对 baseline **新赢 19 / 新丢 15，净 +4**；其中 description 是唯一净负（+4 / −6）、image（+7/−4）与 object（+8/−5）净正。
  - **类别层面（n≥5）**：干预帮 microwave(+14.3)/decorative plant(+14.3)/piano(+25)/flowerpot(+20)；坑 statue(SR−14.3/SPL−10.1)/nightstand(−20)/radiator(−33)/mirror(SPL−8.0)——statue 步数 23→48、探索距离翻倍，典型被错误锚点先验误导绕路。
- **结论** ⚠️ **log-only 并非"回到 legacy 水平"**：SR 52.88% 是 exp006(56.12)/exp007(54.32)/exp008 三者**最低**，说明 legacy-nav 的一部分收益本就来自记忆决策，全关后 SR 掉下来；但 SPL 32.01 反超 exp007（+1.82），干净基线路径更短。**exp007 vs exp008 本质是"用 SPL 换 SR"的不均衡交易，且方向按 task_type 分裂**（image/object 该开、description 该关）。
- **后续建议** 把记忆干预做成**按 task_type 条件启用**（image/object 开、description 关），有望同时保住 exp007 的 SR 增益与 exp008 的 SPL；EXP-009 已在本 log-only 基线上叠加 opportunistic stop（另见该记录）。

### [EXP-009] Opportunistic Stop v1（same-class 近邻触发 end_check）— ❌ 证伪
- **日期** 2026-07-08/09 ｜ **目录** `results/exp009_opportunistic_stop_gpt-5.4` ｜ **状态** ✅ 278/278 全量跑完
- **动机** `data_analysis.md §1.2` 指出大量失败轨迹曾经从 GT 附近经过但没有停：原流程只有「到达已承诺的 object/image 目标点」才触发 `end_check`，如果机器人在 frontier/exploration 过程中路过正确物体，或当前承诺目标不是该物体，就会错过确认窗口。EXP-009 试图把「身体已经靠近同类物体」转成一次现有 `end_check` 机会。
- **主要改动**
  1. **Memory 先退回 log-only 基线**：继承 EXP-008 的降级方向，`surface_negatives_in_prompt:false`、`prompt_prior.annotate_positive:false`、`prompt_prior.memory_block:false`、`live_center_fallback.enabled:false`。也就是说 EpisodeMemory 默认只记录/分析，不再给 prompt 或导航点施加 broad prior，避免把 EXP-007 的记忆干预负效应混进来。
  2. **新增 same-class 近邻扫描**：`_nearest_target_class_object(scene, pts, target_class, radius)` 在每步导航后用 agent 当前 XZ 位置扫描场景图中 class 名完全等于当前 subtask class 的最近节点。
  3. **opportunistic stop gate**：配置 `opportunistic_stop.enabled:true`，`radius:1.5m`，`max_triggers_per_subtask:3`，`cooldown_steps:2`，`require_identity_gate:false`。当 agent 未正常 arrive，但 1.5m 内有同类节点时，也进入原有 end_check 分支；VLM end_check 仍是最终 gate。
  4. **frontier 触发归因修复**：opportunistic 触发可能发生在 frontier step，此时 `last_target_index` 可能是旧目标。代码为 opportunistic break 记录 `opportunistic_break_obj_id`，最终 `EpisodeMemory.record()` 归因到触发的 same-class node；strict reject 时跳过负向记忆，避免污染 stale node。
- **全量结果**（与 EXP-007 同为口径 B、同 278 subtask，可按 `subtask_id` 一一配对）

  | 实验 | Success | SPL | 平均步数 | 平均探索距离 |
  | --- | ---: | ---: | ---: | ---: |
  | EXP-007 epmem redesign | **151/278 = 54.32%** | 30.19% | 13.21 | 12.78 |
  | EXP-009 opportunistic stop | **131/278 = 47.12%** | 29.47% | 11.90 | 11.55 |
  | **Δ（009−007）** | **−20 / −7.19pp** | −0.72pp | −1.31 | −1.23 |

  配对变化：两边都失败 113；**007 fail→009 succ 14**；**007 succ→009 fail 34**；两边都成功 117。也就是说 v1 确实救回一些路过案例，但额外制造了更多误停，净损失 20 个成功。
- **分维度失败分布**

  | 维度 | N | EXP-007 SR | EXP-009 SR | Δ |
  | --- | ---: | ---: | ---: | ---: |
  | object | 99 | 60.61% | 50.51% | **−10.10pp** |
  | description | 91 | 49.45% | 41.76% | **−7.69pp** |
  | image | 88 | 52.27% | 48.86% | −3.41pp |
  | GT dist 0–2m | 121 | 71.90% | 66.94% | −4.96pp |
  | GT dist 2–5m | 61 | 54.10% | 47.54% | −6.56pp |
  | GT dist 5–10m | 59 | 38.98% | 23.73% | **−15.25pp** |
  | GT dist 10m+ | 37 | 21.62% | 18.92% | −2.70pp |
  | frames 100+ | 183 | 49.73% | 40.98% | **−8.74pp** |

  Scene 退化最重：`00815-h1zeeAwLh9Z` 100→20（−80pp）、`00808-y9hTuugGdiq` 40→0（−40pp）、`00813-svBbv1Pavdk` 50→10（−40pp）。类别退化最重：`rug` 78.6→35.7（−42.9pp）、`piano` 75.0→37.5（−37.5pp）、`statue` 28.6→0（−28.6pp）。少数收益场景存在（`00824-Dd4bFSTQ8gi` +28.6pp、`00831-yr17PDCnDDW` +22.2pp、`00861-GLAQ4DNUx5U` +20pp），但不足以抵消整体损失。
- **触发行为复盘**
  - log 中共出现 **183 次** `Opportunistic stop`，覆盖 **125 个 subtask**（33/36 scenes 有触发）；每个触发 subtask 通常 1 次，少数 2–3 次。
  - 触发距离分布：min 0.37m，mean 1.04m，max 1.50m；≤1.0m 仅 75/183，说明 1.5m radius 经常在物体中心较远处就触发。
  - 触发类别主要是 `mirror` 43、`picture` 40、`refrigerator` 29、`rug` 15、`microwave` 11。
  - **关键证据**：被 opportunistic 触发覆盖的 125 个 subtask，SR 从 EXP-007 的 **56.0% 降到 40.0%（−16pp）**，配对 **+4 win / −24 loss**；未触发的 153 个 subtask 则 **52.9%→52.9%（净 0）**。负效应几乎集中在这个新机制的触发子集上。
  - 触发子集内 object 最惨：object 65.0→37.5（−27.5pp），description 54.8→45.2（−9.5pp），image 48.8→37.2（−11.6pp）。
- **失败经验 / 根因判断**
  1. **class-only 不是 identity**：`require_identity_gate:false` 使 same-class 近邻只按类别匹配，多个 mirror/picture/rug/refrigerator 的 scene 中极易把「同类但非目标实例」当成确认候选。VLM end_check 看的是最近若干 egocentric views，不知道该 same-class node 的实例身份；一旦画面里出现一个同类物体，就可能 yes。
  2. **1.5m 到 bbox center 过宽**：成功口径是到 GT object center ≤1.0m，但 opportunistic 在 XZ 1.5m 就触发，且没有要求目标在正前方、可见、mask 足够大或与当前候选一致。很多新增失败 final distance 落在 1.0–1.5m 边界外（例如 rug/piano/refrigerator），说明它常在「还差一点」时过早停。
  3. **提前停缩短轨迹但牺牲 SR**：平均步数 13.21→11.90、探索距离 12.78→11.55，说明机制确实让 agent 更早结束；但 SR −7.19pp，证明节省主要来自 premature stop，而不是更高效命中。
  4. **object query 反而最脆弱**：object 没有 description/image 的实例约束，只要类别相同就更容易被 opportunistic 触发误停；这与 object −10.1pp、触发子集 object −27.5pp 一致。
  5. **中距离/长 episode 更容易被误截断**：5–10m GT distance −15.25pp、100+ frames −8.74pp，说明复杂探索中路过 distractor 的机会更多，class-only 触发更容易截断本来还能继续修正的轨迹。
- **结论** ❌ **Opportunistic Stop v1 证伪**。方向里的「路过正确目标时应该给一次确认机会」有信号（14 个新增成功），但 v1 的 class-only、宽 radius、无可见性/实例门控实现过于激进，制造 34 个新增失败，净负明显。
- **后续建议**
  - 默认关闭 `opportunistic_stop`，不要把 v1 合入主线默认配置。
  - 若继续做 v2，必须变成 **high-precision trigger**：radius ≤1.0m 或按 object size 自适应；要求目标在前向视野且检测/分割可见；要求 same-class node 与当前 VLM 选择或 memory exact-id 一致；description/image 必须过 CLIP/VLM identity gate；object 类至少需要「当前 committed target 也是同一 node」或连续多帧同一 node 才允许停。
  - 建议先离线重放 183 个触发点，标注 end_check yes/no 与 GT/distractor 关系，再设计 gate；否则只调半径容易继续在 win/loss 间震荡。
