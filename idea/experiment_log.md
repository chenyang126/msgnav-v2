# MSGNav 改进实验日志（Experiment Log）

> **用途**：记录每一次改动的动机、口径、结果和结论，避免重复实现已被证伪或方向错误的改动。
> **维护规则**：每做一次改动/一次评测，就在下方"实验记录"里**追加一条**（不要删旧记录）；同时更新顶部"结论速查表"和"方向状态看板"。
>
> **统一口径**：GOAT-Bench Val Unseen，`--splits 1`（每 scene 仅 episode 0），278 subtasks，gpt-5.4，4×RTX5090。成功阈值 `success_distance = 0.25m`。分析以 `records_*.jsonl` 去重（按 subtask_id 保留最后一条）为准，SPL 用 msgnav conda 环境算（普通 python 缺 numpy 会显示 SPL=0）。

---

## 0. 结论速查表（先看这里）

| 改动 | 结果目录 | n | Success | SPL | 相对 baseline | 结论 |
| --- | --- | --- | --- | --- | --- | --- |
| **Baseline**（无改动） | `baseline_no_refine_gpt-5.4_278subtasks` | 278 | **55.04%** | 33.65% | — | 基准 |
| Refine（到达后走近） | `example_goatbench_refine_gpt-5.4` | 43(仅重叠) | 65% vs 67% | — | 净 −1 | ⚠️ **无结论**（样本太小+触发点错），非证伪 |
| Strict stop-check | `example_goatbench_gpt-5.4` | 278 | 53.24% | 32.96% | **−1.80pp** | ❌ **回归**（灾难性走失） |
| Stop-check 收窄到 image-only | *(未重跑)* | — | 预估 ~56% | — | 预估 +1.1pp | 🔧 代码已改，**待验证** |

> ⚠️ **方法论警告**：本评测**非确定**。43 个子任务上光是重跑就会随机翻 3–4 个成功/失败。**任何结论都必须在 278 全首集上得出，不要用几十个子任务下判断。**

---

## 1. 方向状态看板

| 方向 | 来源 | 状态 | 备注 |
| --- | --- | --- | --- |
| 精对准 refine（到达后走近） | idea Layer-1 MVP | ⚠️ 无结论，需重做触发逻辑 | 旧实现打在已成功案例上，无上行、有下行 |
| 保守停止 strict stop-check | idea Layer-1/2 | ❌ 已证伪（原始设计） | "驳回→重探索"导致灾难走失；收窄到 image 后待验证 |
| Hard target 检测增强（planar/反光/结构） | idea Layer-2 B1/B2 | ⬜ 未实现 | 应走"降检测阈值+top-k CLIP 复核"，**不要**用严格拒绝 |
| 语义 frontier（room prior 注入） | idea Layer-3 | ⬜ 未实现 | SPL 导向，超时仍 34/130 |
| 修复 records 重复写入 | idea D2 / 工程 5.1 | ⬜ 未实现 | 每 subtask 重复写 ~3 次，不影响指标 |

---

## 2. 实验记录（按时间追加，勿删）

### [EXP-000] Baseline — gpt-5.4 首集 278
- **日期**：2026-07-01/02
- **目录**：`results/baseline_no_refine_gpt-5.4_278subtasks`
- **改动**：无（原始 MSGNav 流程）
- **结果**：Success **55.04%**（153/278），SPL **33.65%**
  - object 62.63% ｜ description 54.95% ｜ image 46.59%（image 最难）
- **失败构成**：提前放弃(<10步) 62(50%) ｜ 中途 33 ｜ 超时(≥50) 30
- **阈值消融**：≤0.25m 55.0% → ≤0.5m 59.4% → ≤1.0m 60.8%
- **详细分析**：见 `experiment_analysis_gpt5.4.md`

### [EXP-001] Refine（到达后精对准，走近再停）
- **日期**：2026-07-02
- **目录**：`results/example_goatbench_refine_gpt-5.4`（**仅跑了 43 个子任务**）
- **改动**：config 加 `refine:` 块（enabled, max_steps=3, stop_radius=0.35, target_radius=0.45, min_improvement=0.02）。VLM 判"到达"后不立即停，重采样更近的观察点走 1–3 步。**（该实现当前已被 stop-check 版本覆盖，代码未保留。）**
- **结果**：重叠 43 子任务上 baseline 29 成功 → refine 28 成功（**净 −1**，gained 3 / lost 4）
- **结论：⚠️ 无结论，且触发逻辑有明显缺陷 —— 不能据此判定 refine 无效**
  1. **样本太小 + 高噪声**：4 个 LOST 里 2 个 `nref=0`（refine 根本没触发）也翻车（00862_0_3/0_6：0.06→2.94）；1 个 GAINED 也是 `nref=0`。→ 光重跑就翻 3–4 个，43 个测不出 ±4pp 的效应。
  2. **触发点全错**：`n_refine_steps` 分布 = {0:25, 1:17, 2:1}。触发的 18 次里绝大多数打在 baseline **已经 <0.25m（已成功）** 的目标上（如 00820 场景 ×9 全是 0.15→0.12）。对已成功目标走近**没有上行、只有被推出的下行**，两个 LOST 正是这么来的（00803_0_5 0.13→1.73；00862_0_7 0.11→1.17）。
  3. **真近失打中就救回，命中率 2/2**：00862_0_1 1.31→0.08(nref=2) GAINED；00803_0_8 1.08→0.16(nref=1) GAINED。
- **下次若重做 refine 的修正点**：
  - 触发门控反过来：**仅当估计距目标 > 成功半径（如 >0.5m）才 refine，已在半径内绝不动**。
  - 每步须单调靠近，否则立即停（防 0.13→1.73 overshoot）。
  - **必须在 278 全首集测**（近失案例：≤1.0m 有 27 个、≤0.5m 有 16 个，43 里几乎抽不到）。

### [EXP-002] Strict stop-check（保守停止，原始全局设计）
- **日期**：2026-07-03
- **目录**：`results/example_goatbench_gpt-5.4`
- **改动**：config 加 `stop_check:`，`strict_for_task_types:["image"]`, `strict_for_min_steps:2`, `hard_targets:[picture,mirror,window,glass,plant,decorative plant,rug,carpet]`。逻辑（旧）：多条件 **OR**，任一命中就叫 VLM 二次复核；复核"no"则**驳回停止、释放目标、回到 frontier 重探索**。
- **结果**：Success **53.24%**（148/278），SPL **32.96%** → **相对 baseline −1.80pp** ❌
  - object 58.59%(−4.04) ｜ description 50.55%(−4.40) ｜ image 50.0%(**+3.41**)
- **翻转**：净 −5（gained 23 / lost 28）。image +3 / object −4 / description −4。
- **结论：❌ 原始设计证伪。两个根因：**
  1. **"驳回→重探索"是灾难性动作**：被打破的 28 个里 **17 个是"灾难走失"**——baseline 本已站在目标 0.06–0.24m，被判"没到"后 agent 放弃目标全局重探索，走到 1–11m 外（00821 picture 0.13→10.96；00871 piano 0.08→11.86；00862 picture 0.06→6.41）。误判的代价是把必成功变成彻底失败。
  2. **hard_targets/min_steps 全局触发**：跟 task_type 无关,导致 object/description 撞到 rug/picture/mirror/plant 或前 2 步时被误伤。被打破的目标恰好是 hard_targets 列表本身。
- **意外发现**：阈值消融里 stop-check 在 **≤1.0m 口径反超 baseline**（62.9% vs 60.8%）——它确实把分布推近了,但 0.25m 严格口径 + 走失把收益吃光。→ 说明"推近"有真信号,但要用**走近(refine)**而不是**拒绝(reject)**去兑现。

### [EXP-003] Stop-check 收窄到 image-only（两层门控）
- **日期**：2026-07-04
- **目录**：*(待重跑)*
- **改动**：重写 `_strict_stop_reasons`（`run_goatbench_evaluation.py`）为两层：① **SCOPE 门控**（`strict_for_task_types`/`strict_for_target_types` 决定谁有资格被复核，不在作用域直接返回）；② **REFINEMENT**（hard_targets/min_steps 仅在作用域内细化，不再全局触发）。config 收窄 `strict_for_task_types:["image"]`、`strict_for_min_steps:0`。删除冗余包装函数 `_should_run_strict_stop_check`。
- **验证（单元级）**：image 任务仍复核；object/description 撞 plant/rug、或前 2 步——**不再复核**（改前会）。
- **预估**：object/description 回到 baseline，image 保留 +3 → ~156/278 ≈ **56.1%**（+1.1pp）。仅止损,非大增益。
- **结果**：⬜ **待在 278 全首集重跑填写**
- **注意**：即使收窄，image 上 strict check 的"驳回→重探索"走失风险仍在（image 曾 lost 9）。根治仍需把"拒绝"改成"局部走近/再观察"。

---

## 3. 关键教训（Lessons，反复看）

1. **非确定性**：43 子任务上重跑就翻 3–4 个。**结论只在 278 全首集下判定**，小样本只能看趋势/debug。
2. **"推近"要用 refine（走向目标），绝不能用 reject（释放目标重探索）**——后者会把 0.08m 的必成功变成 11m 的彻底失败（EXP-002 实证）。
3. **refine 只对"还没到"的目标做，已进成功半径的绝不动**（EXP-001 的下行全来自动了已成功案例）。
4. **hard target（mirror/picture/glass/plant）要靠检测召回增强,不要靠严格拒绝**（EXP-002 里对它们的拒绝是净负）。
5. **改动必须可追踪**：把触发/动作写进 `records` 的 note 或 trajectory（EXP-001 的 `n_refine_steps` 就是靠这个才复盘出触发点错了）。
