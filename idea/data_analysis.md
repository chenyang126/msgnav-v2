# MSGNav 数据分析（Data Analysis）

> **本文只做「对已跑结果的深入拆解」**：失败模式归因、记忆命中成败、分物体分布。原始数字与实验清单见 [`experiment_results.md`](./experiment_results.md)；由分析得出的方向判断与改进方案见 [`idea_analysis.md`](./idea_analysis.md)。
>
> **主分析对象＝当前版本 `exp007_epmem_redesign_gpt-5.4`**（口径 B，`success_distance=1.0m` 到 GT 物体中心，278 subtasks，redesign 记忆）。EXP-000 baseline（口径 A）作历史方法参考保留在 §3。分析脚本以 `records_*.jsonl` 去重（按 `subtask_id` 保留最后一条）+ `trajectory_*.jsonl` 逐步事件为准。

---

## 1. exp007 轨迹失败归因（当前版本）★

**总体**：151 成功 / 127 失败，SR 54.32% / SPL 30.19%；object 60.61% ｜ description 49.45% ｜ image 52.27%。

### 1.1 失败三分类（按停止那一刻发生了什么）

用每步 `events`（是否 `end_check_yes`）+ `n_steps` vs `num_step`(50) 把 127 个失败分三类：

| 类别 | n | 占比 | fd 均值/中位 | 从未感知到 GT | fd<1.5m(差一点) | 主要 task |
| --- | --- | --- | --- | --- | --- | --- |
| **A. 假阳性确认**（喊了 yes 但失败） | 86 | **68%** | 7.02 / 5.55m | **51/86** | 14 | image 37 / object 29 / desc 20 |
| B. 超时耗尽（≥49 步） | 31 | 24% | 7.87 / 6.77m | 27/31 | 0 | desc 19 / object 7 / image 5 |
| C. 中途停/从未 yes | 10 | 8% | 7.60 / 9.91m | 10/10 | desc 7 / object 3 |

> 「从未感知到 GT」＝整条轨迹 `goal_obj_ids_mapping` 全为空，即 agent 场景图从没把 GT 物体映射到任何检测实例。

**核心结论**：头部矛盾是**假阳性确认（68%）**，中位最终距 **5.55m**（不是「差一点没走近」，86 例里仅 14 个 <1.5m）。但按「是否感知到 GT」拆开后，真实机制比「grounding 认错」更细，见 §1.2。

### 1.2 ★ 一半的失败是「到过 GT 又走开」，不是「没找到」

用每步的 `dist_to_gt_viewpoint`（agent 到 GT viewpoint 的距离，成功案例此值中位仅 0.12m，可信）取整条轨迹的**最小值**，问一个新问题：agent 的身体到底有没有到过 GT 旁？

- **127 个失败里，63 个（50%）某一步已走到离 GT viewpoint <1.0m，然后仍失败**。其中 14 个是边界差一点（最终 ≤1.5m），**49 个明确走开**（最终 >1.5m，中位 3.61m，最狠 microwave 0.06m→17.9m、carpet 0.13m→16.4m、mirror 0.02m→10m）。

即**一半失败不是「没找到」，而是「找到了、走到了、却没停下」**——系统反复把 agent 送到目标脚边却转化不成成功。进一步分层（下表按假阳性 86 的两个来源）：

| 子类 | n | agent 到 GT 最近距 | 解读 |
| --- | --- | --- | --- |
| 假阳性·感知到 GT | 35 | 中位 0.15m（31 例 <1m） | 多为**到过 GT 又走开**：24 例到 <1m 后走到 >1.5m 外确认；7 例边界差一点 |
| 假阳性·从未感知 GT | 51 | 中位 3.32m | 18 例站到 GT 旁(<1m)却没检测出→**检测召回漏**；23 例从没靠近(≥4m)→**覆盖不到/远处认错** |

**三种不同机制，修复方向完全不同**：
1. **到过又走开 / 目标不稳定（~49–63，含边界）**：已到目标却没锁定/确认，或路过真 GT 后走向已承诺的错实例。`choose_every_step` 每步重决策 + redesign 丢了成功站位，都会让 agent 在同类多实例间漂移。**最大杠杆、最省**。
2. **检测召回漏（~18）**：身体到了 GT 旁，检测器（planar/reflective）没认出来。
3. **覆盖不到 / 远处认错（~23–35）**：从没靠近 GT，在远处别的东西上确认——初始候选选错 + 探索没覆盖到 GT 区域。

> ⚠️ **仪表盘缺口**：机制 1 里「锁对了却放弃」vs「锁错实例、只是路过真 GT」当前**分不清**（`target_obj_ids_estimate` 全空、image 的 target 是 png）。已加轻量日志记录「每步承诺目标是否 match GT」来解锁这个判断（见 `idea_analysis.md §3` 方向 1 与仪表盘条目）。

### 1.3 记忆命中的成败（本版关键）

> 本节数字基于 **exp007**（redesign 记忆，命中即覆盖导航点）。⚠️ **EXP-008 起记忆降为默认 log-only**，`n_memory_positive_hits` 语义变为 candidate 数、`n_live_center_overrides` 变为实际 fallback 次数（默认 0）——分析 exp008+ 时勿沿用下表口径。

| 分组 | n | SR | SPL |
| --- | --- | --- | --- |
| 有记忆正向命中 | 78 | **82.1%** | 50.5% |
| 无记忆命中 | 200 | 43.5% | 22.3% |

⚠️ **强选择偏差**：正向命中的前提是「同一物体在更早子任务已被 confident 成功过」（exact_id 引用 positive anchor），这些本就是可达易复访的目标。**高 SR 主要是选择效应，不能当作「记忆提升成功率」的证据**——效果方向已由 legacy↔redesign A/B 给出（净负，见 §3.2）。

**14 例记忆命中却失败（全部 exact_id）**——这才是记忆主动参与的因果案例：

| subtask | 类型/类别 | fd | 形态 |
| --- | --- | --- | --- |
| 00880-Nfvxx8J5NCo_0_1/0_3 | desc/image mirror | 1.26m | 差一点（记忆活中心停在 1.0m 门槛外） |
| 00844-q5QZSEeHe5g_0_3 | image calendar | 1.37m | 同上 |
| 00831-yr17PDCnDDW_0_4/0_5 | image/object plant | 1.19m | 同上 |
| 00821-eF36g7L6Z9M_0_6/0_7 | desc/image picture | 2.25m | 锚点略偏 |
| 00829-QaLdnwvtxbs_0_6 | desc mirror | 6.37m | 锚点指向错实例 |
| **00831-yr17PDCnDDW_0_3** | **desc plant** | **15.10m** | **锚点传播灾难（见下）** |
| 00831-yr17PDCnDDW_0_8 | desc vase | 5.76m | 同一场景污染 |

**案例 00831-yr17PDCnDDW（plant/vase 多实例，回落机理实锤）**：这正是 EXP-005 里 plant↔vase 互相污染的老场景。redesign 的 exact_id 把**首个子任务锁定的错误/不精确 plant 锚点，忠实地传播到后续所有同物体子任务**（0_3 plant 15.1m、0_4/0_5 plant 1.19m、0_8 vase 5.76m）。记忆在这里**放大而非纠正**首步误差——首步对，后续省事；首步错，错误被复用到整个 episode。这是 redesign 相对 legacy 回落的具体机制之一（legacy 用冻结成功站位，至少不会把「活中心漂移」跨子任务传播）。

### 1.4 分物体类别（exp007）

| 大类 | n | SR |
| --- | --- | --- |
| 大型/其他（refrigerator 92.9% ｜ hanging clothes 80% ｜ dresser 63.6%） | 118 | 68.6% |
| 小型装饰（rug 78.6% ｜ statue 28.6% ｜ plant 20% ｜ flowerpot 20%） | 69 | 55.1% |
| **平面/反光**（mirror 41.9% ｜ picture 37.9% ｜ glass 0%） | 80 | **38.8%** |
| **结构**（stair 0% ｜ handrail 20%） | 11 | **9.1%** |

**单类失败 Top**：glass 0%(0/4)、stair 0%(0/6)、plant 20%(3/15)、handrail 20%、nightstand 20%、statue 28.6%、picture 37.9%、mirror 41.9%。

→ 与 baseline 完全一致的两大失败源：**平面/反光（mirror/picture/glass）** 和 **结构（stair/handrail）**——记忆重设计对这两类毫无改善（结构类甚至只有 9.1%）。

---

## 2. exp007 主要错误原因总结

按可操作性排序，当前轨迹的具体错误原因：

1. **到过 GT 又走开 / 目标不稳定（最大头，50% 失败到过 <1m 仍失败，§1.2）**：已到目标却没锁定/确认，或路过真 GT 后走向已承诺的错实例。`choose_every_step` 漂移 + redesign 丢成功站位。修复在**确认之前**做目标保持/就近确认（不违反已证伪的「停止后走近」），并接 view-pose bias。
2. **grounding 认错 / 覆盖不到（~23–35 从没靠近 GT）**：给 image/类名找实例选错候选，或探索没到 GT 区域就在远处确认。修复：候选 CLIP 相似度排序 + 多实例去重 + 语义探索。
3. **检测召回漏（~18 站到 GT 旁没认出）+ 平面反光/结构类失效**：mirror42%/picture38%/glass0%/stair0%，检测器与深度不可靠。修复：硬类别降阈值+top-k+复核；结构类改「可视即成功」。
4. **记忆跨子任务传播首步误差**：exact_id 忠实复用首个子任务锚点，首步错则整 episode 错（case 00831）。near-miss（1.1–1.4m）与灾难（15m）并存。
5. **description 超时**（超时 31 里 19 个是 desc）：大场景探索效率不足，SPL 低。

---

## 3. 历史与演进参考

### 3.1 EXP-000 baseline 轨迹级归因（口径 A，方法源头）
baseline 125 失败按轨迹重新归因：**A 锁对停远 36(29%) ｜ B 锁错物体 50(40%，image 占 58%) ｜ C 从未确认 39(31%)**。A+B=69% 是「假阳性确认」，与 exp007 的 68% 同源同性质。B 类高度集中 image、根因在 grounding 的结论，在 exp007 中被进一步放大（假阳性里 image 37 最多、51 例从未感知 GT）。

> ⚠️ baseline 是口径 A（0.25m viewpoint），数字不与 exp007（1.0m 物体中心）直接可比，仅供定性方法参照。

### 3.2 记忆 legacy↔redesign A/B（EXP-006 vs EXP-007，口径 B）

exp006 与 exp007 导航栈/口径/config 全相同（仅差被删的 `redesign_enabled` 死 flag），唯一实质区别是记忆实现——事实上的 legacy↔redesign A/B（判定靠 log 签名，非存档 config）：

| | 命中总数 | exact_id | spatial | VLM 身份门 | "using live target center" |
| --- | --- | --- | --- | --- | --- |
| exp006 (legacy) | 88 | 60 | **28** | 每候选都调 | 0 |
| exp007 (redesign) | 88 | 85 | **3** | **仅 5** | 88 |

机制层完全达成（误配 28→3、VLM 门近乎归零）；**但效果层净负**：

| 指标 | exp006 legacy | exp007 redesign | Δ |
| --- | --- | --- | --- |
| Overall SR | 56.12% | 54.32% | **−1.80pp** |
| Overall SPL | 34.49% | 30.19% | **−4.30pp** |
| image SR/SPL | 57.95 / 41.79 | 52.27 / 35.16 | **−5.68 / −6.63** |

SPL 全线跌 + image 最惨，与 §1.2 的锚点传播/活中心漂移机制吻合：redesign 丢弃冻结成功站位、命中后只给活中心，几何看似更优却让路径变长、把首步误差跨子任务传播。**归因谨慎**：两次均单跑、非确定（±~1.4pp/278），overall −1.80 接近噪声，但 image/SPL 超纯噪声，方向可信。

### 3.3 EXP-005 v1 spatial 污染因果分解（口径 A，历史）
exp005 v1 raw 净 0，因果分解 memory 命中翻转净 +7、无关噪声净 −7 抵消。按 match type：exact_id 58 命中 0 回归净 +6（安全）；spatial 50 命中 6 回归净 +1（回归唯一来源）。根因：L2 旧实现比较「agent 站立位置」vs「新目标物体中心」参照系不一致，放大匹配半径致跨类误配（plant↔vase、mirror↔picture）。→ 修复见 `experiment_results.md` EXP-005b。

---

## 附：速查
- exp007：278（去重），成功 151 / 失败 127（口径 B）
- **★ 50% 失败（63/127）到过 GT viewpoint <1m 仍失败**：49 例明确走开(final>1.5m,中位3.61m)、14 例边界差一点
- 失败三分类：假阳性确认 86(68%) ｜超时 31(24%) ｜中途 10(8%)；假阳性再分：感知到GT 35(多为到过又走开)、从未感知GT 51(检测漏18/覆盖不到23)
- 记忆命中 78 SR82.1%/SPL50.5%（强选择偏差）vs 未命中 200 SR43.5%/SPL22.3%；命中失败 14 例全 exact_id
- 分类：平面反光 38.8% ｜结构 9.1% ｜小装饰 55.1% ｜大型 68.6%；glass/stair 0%、plant 20%、picture 38%、mirror 42%、refrigerator 93%
- legacy↔redesign A/B：机制更优（spatial 28→3、VLM 门→5）但效果净负 SR−1.80/SPL−4.30/image−5.68
- baseline(口径A,参考)：A 锁对停远 29% ｜B 锁错物体 40%(image58%) ｜C 从未确认 31%
