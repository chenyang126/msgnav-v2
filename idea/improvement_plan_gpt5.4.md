# MSGNav 改进方案设计（基于 gpt-5.4 首集实验）

> 基于 `idea/experiment_analysis_gpt5.4.md` 中的实验分析，本文给出一个可落地的改进方案。目标是在不大改整体架构的前提下，优先提升 GOAT-Bench 首集评测的 Success Rate 和 SPL。
>
> 当前基线：`gpt-5.4`，GOAT-Bench Val Unseen first episode per scene，278 subtasks，Success **55.04%**，SPL **33.65%**。

---

## 0. 设计目标

本轮改进不追求重写系统，而是针对实验暴露出的主要瓶颈做最小有效改动：

1. **减少提前放弃**：当前 125 个失败中有 62 个（50%）在 10 步内失败。
2. **挽救近距离失败**：至少 12–16 个失败最终距离在 0.25–1.0m，可通过精对准挽救。
3. **提升平面/反光/结构目标能力**：mirror/picture/glass/stair/handrail 是失败高发类别。
4. **提高探索效率与 SPL**：减少无效 frontier 选择和 50 步超时。
5. **保留可解释调试信息**：所有改动都必须在 `records_all.jsonl` / trajectory 中可追踪。

---

## 1. 总体方案：三层修复

```text
当前 MSGNav
  ↓
Layer 1: Conservative Stop + Near-target Refinement
  修复：过早停止、0.25m 阈值卡边界
  预计收益：+4~6 pp Success，SPL 小幅提升

Layer 2: Hard Target Type Handling
  修复：mirror/picture/glass/stair/handrail 等特殊目标
  预计收益：+3~8 pp Success，取决于检测召回

Layer 3: Semantic Frontier Selection
  修复：大场景盲目探索、50 步超时、SPL 低
  预计收益：SPL 提升为主，Success 中等提升
```

建议按 Layer 1 → Layer 2 → Layer 3 顺序实现和消融。

---

## 2. Layer 1：Conservative Stop + Near-target Refinement（优先级最高）

### 2.1 问题定位

当前主循环在 `run_goatbench_evaluation.py` 中：

1. VLM 选择 `target_type` / `target_index`
2. `tsdf_planner.set_next_navigation_point(...)`
3. `tsdf_planner.agent_step(...)`
4. 若 `target_type != "frontier" and target_arrived`，调用 `query_vlm_for_response_end(...)`
5. 若 VLM 回答 `yes`，直接 `break`
6. 最终用 `calc_agent_subtask_distance(...) < cfg.success_distance` 判成功

问题是：

- VLM 的 end check 只判断“看起来像目标”，不保证 agent 距离目标 viewpoint < 0.25m。
- 一旦 `yes` 就立刻结束，导致很多 `0.25m~1.0m` 的近距离失败。
- 分析显示阈值从 0.25m 放到 0.5m 可挽救 12 个失败，到 1.0m 可挽救 16 个失败。

### 2.2 改进思路

新增一个 **refinement phase**：

> VLM 认为目标已到达后，不立即结束，而是执行 1–3 步“精对准/贴近”动作，直到估计已经足够接近再结束。

具体逻辑：

```python
if target_type != "frontier" and target_arrived:
    vlm_response = query_vlm_for_response_end(...)

    if vlm_response == "yes":
        # 新逻辑：不要立刻 break
        if should_refine_target(...):
            run_refinement_steps(max_refine_steps=3)
            # refinement 后再次 check 或直接进入最终距离判定
        break
```

### 2.3 `should_refine_target` 规则

建议先不依赖 GT distance（测试时不能用 GT），用可观测信号判断：

满足任一条件就进入 refinement：

1. 当前目标是 `object` 或 `image`，且 planner 的 `target_point` 与 agent 当前位置距离 > `cfg.refine_stop_radius`。
2. 目标 bbox / object pcd 存在，但 agent 到 bbox center 的平面距离 > `cfg.refine_min_object_dist`。
3. 最近一次 `query_vlm_for_response_end` 是 `yes`，但目标图像中 bbox 面积偏小，说明还较远。

建议配置：

```yaml
refine:
  enabled: true
  max_steps: 3
  stop_radius: 0.35        # 比 success_distance=0.25 略宽，作为 refinement 触发线
  target_observe_dist: 0.8 # 期望最终停在目标附近 0.8m 内可观察位置
  require_second_check: true
```

### 2.4 `run_refinement_steps` 做什么

根据目标类型不同：

#### object 目标

- 使用 `scene.objects[target_index]["bbox"]` 或 `pcd` 估计目标中心。
- 在目标周围采样候选观察点（半径 0.6–1.0m），选择 pathfinder 可达且距离当前位置最近的点。
- 将该点设置为下一导航点，连续执行 1–3 次 `agent_step`。

#### image 目标

- image 目标最终也会通过检测映射到 object / pcd（`query_vlm_for_response` 已经有从 image 检测 object 的逻辑）。
- 若 image 目标无法稳定映射到 object，则退化为朝当前 `target_point` 再走 1–2 步，而不是立即停。

#### frontier 目标

- frontier 不做 refinement，因为它本来是探索目标，不是终止目标。

### 2.5 避免负面影响

refinement 可能降低 SPL（多走几步），但如果换来成功率提升，通常值得。为避免过度绕路：

- 最多 3 步。
- 如果连续两步到目标估计点的距离不下降，停止 refinement。
- 如果 VLM 第二次 check 回答 no，则恢复探索，不终止。

### 2.6 预期收益

从离线阈值消融看：

| 可挽救范围 | 对应失败数 | 理论成功率提升 |
| --- | --- | --- |
| final distance < 0.5m | 12 | +4.3 pp |
| final distance < 1.0m | 16 | +5.8 pp |

因此 Layer 1 预计可把 Success 从 **55.0% 提升到 59–61%**。如果 refinement 还减少部分提前放弃，则可能更高。

---

## 3. Layer 2：Hard Target Type Handling

### 3.1 问题定位

不同目标类别成功率差异极大：

| 类别 | n | 成功率 |
| --- | --- | --- |
| 大型家具 | 81 | 81% |
| 小型装饰 | 51 | 45% |
| 平面/反光（mirror/picture/glass） | 80 | 38% |
| 结构（stair/handrail） | 11 | 18% |

说明当前 “YOLOWorld + SAM + pcd bbox + viewpoint” 对普通大型物体有效，但对平面/反光/结构目标不适配。

### 3.2 目标类别分组

新增 hard target registry：

```python
PLANAR_TARGETS = {"mirror", "picture", "glass", "window", "frame"}
STRUCTURE_TARGETS = {"stair", "stairs", "staircase", "handrail", "railing"}
SMALL_DECO_TARGETS = {"plant", "decorative plant", "statue", "pillow", "vase", "rug"}
```

根据 `subtask_metadata["class"]` 或 question 中关键词启用特殊策略。

### 3.3 平面/反光目标策略

#### B1. 检测召回增强

当前 `cfg/eval_goatbench.yaml` 中：

```yaml
mask_conf_threshold: 0.95
```

这个阈值对 mirror/picture/glass 可能过高。建议：

- 对 hard planar targets 单独降低检测阈值，如 0.6–0.8。
- 保留多个候选，不要只相信 top-1。
- 用 CLIP / VLM 对候选 crop 做二次确认。

伪逻辑：

```python
if target_class in PLANAR_TARGETS:
    detection_conf = cfg.hard_target.planar_conf_threshold  # e.g. 0.65
    keep_topk = 5
    rerank_by_clip_or_vlm(candidates)
```

#### B2. 到达判定改为“可视 + 近旁”

mirror/picture 是墙面目标，agent 不需要贴到 0.25m 的 viewpoint。可以用以下组合判定：

- 目标在当前视野中可见。
- bbox 面积超过阈值。
- agent 到目标平面/pcd 的距离 < 1.0m 或 path distance < 1.5m。

注意：最终 benchmark 仍用 GT viewpoint distance 判分，不能直接改评估标准。但策略上可以让 agent 选择更接近 wall object 的可达点，而不是 bbox center。

### 3.4 结构目标策略

stair/handrail 不是普通 object，建议：

1. 不使用紧 bbox center 作为目标点，而使用 candidate viewpoint / frontier around structure。
2. 若检测到 stair/handrail 的 pcd 稀疏或 bbox 异常，优先选择其附近 frontier，而不是 object target。
3. 对结构目标提高 max refinement steps，因为它们往往跨楼层/走廊，需要更长探索。

---

## 4. Layer 3：Semantic Frontier Selection

### 4.1 问题定位

失败中 30 个（24%）是 steps >= 50 的探索超时，平均最终距目标 7.3m；SPL 只有 33.65%。这说明 frontier 选择缺少语义引导。

### 4.2 思路

在 VLM 选择 frontier 之前，对 frontier 进行语义打分，作为 prompt 信息或直接作为排序 prior。

得分函数：

```text
score(frontier) =
  w_room * room_goal_prior
+ w_object * nearby_related_objects
+ w_unexplored * frontier_area
- w_dist * path_distance
- w_repeat * visited_penalty
```

### 4.3 room-goal prior

根据目标类别给房间先验：

| 目标 | 优先房间 |
| --- | --- |
| refrigerator / microwave / sink | kitchen |
| pillow / bed / dresser | bedroom |
| toilet / shower / towel | bathroom |
| sofa / TV / picture / plant | living room |
| stair / handrail | hallway / stair area |

`scene.objects[obj_id]["room_label"]` 已经存在，可直接使用。

### 4.4 prompt 注入

在 `explore_two_step` 的 frontier 描述中，为每个 frontier 增加：

- Estimated room label
- Nearby object classes
- Distance from current position
- Whether this frontier moves toward likely target room

让 VLM 不再只看 frontier image，而是看到结构化语义。

### 4.5 预期收益

- 对超时失败（30 个）和中途失败（33 个）有帮助。
- SPL 应该提升更明显，因为减少绕路。
- Success 预计 +2~5 pp，SPL 预计 +3~8 pp（需实测）。

---

## 5. 工程改进：修复重复记录与实验可复现

### 5.1 records JSONL 重复写入

本次分析发现：

- 原始 `records_*.jsonl` 共有 885 条。
- 去重后唯一 subtask 只有 278。
- 每个 subtask 都被写入多次（最多 5 次），但 success 没有翻转。

这说明记录逻辑存在重复追加问题。建议：

- Logger 内维护 `self._written_record_ids`。
- `_write_live_record` 前检查：若 subtask_id 已写过，则更新 CSV 内存记录，但不再追加 JSONL。
- 或 JSONL 改为 append-only event log，另行命名为 `events.jsonl`；subtask summary 用 `records_current.json` 覆盖写。

### 5.2 失败复盘工具

建议增加一个脚本：

```bash
python analyze_goatbench_records.py results/example_goatbench_gpt-5.4 \
  --by-scene --by-object --failure-mode --threshold-sweep
```

输出：

- scene 排名
- 目标物体失败率
- threshold sweep
- steps=50 超时列表
- 近距离失败列表

这能支撑之后每次改动后的快速对比。

---

## 6. 推荐实施路线

### Stage 1：低成本验证（1 天内）

1. 修复 JSONL 重复写入。
2. 增加离线分析脚本。
3. 实现 Layer 1 的 Conservative Stop + Refinement。
4. 在 `--splits 1` 上重跑 278 subtasks。

验收指标：

- Success >= 59%。
- SPL 不低于当前 33.65%，最好提升。
- 近距离失败（<1m）数量明显下降。

### Stage 2：特殊目标增强（1–2 天）

1. 加 hard target registry。
2. planar targets 降低检测阈值 + top-k rerank。
3. structure targets 改目标点选择逻辑。
4. 只对失败高发 scene/target 做小规模回归，再跑 full first-episode。

验收指标：

- mirror/picture/glass 成功率从 38% 提升到 45%+。
- stair/handrail 至少不再 0/极低。

### Stage 3：语义 frontier（2–4 天）

1. 给 frontier 打 room/object prior 分。
2. 把 frontier score 注入 prompt 或直接排序。
3. 比较 steps=50 超时数和 SPL。

验收指标：

- steps>=50 的失败从 30 降到 <20。
- SPL +3pp 以上。

### Stage 4：全集验证（长跑）

在首集指标稳定后，再跑：

```bash
python start_multiprocess.py --task goatbench --devices 0,1,2,3 --splits 10
```

预估 4 卡约 7 天。只有在首集收益稳定后才值得投入全集。

---

## 7. 最推荐的 MVP 改动

如果只做一个改动，建议做：

> **到达后 refinement + 更保守的 end check**。

原因：

1. 数据支持最强：0.5m threshold sweep 可直接看到 +4.3pp 潜力。
2. 实现侵入最小：主要改 `run_goatbench_evaluation.py` 主循环和少量 planner helper。
3. 风险可控：最多 3 步 refinement，不会显著拖慢评测。
4. 便于验证：看近距离失败数是否下降即可。

MVP 伪代码：

```python
if target_type != "frontier" and target_arrived:
    vlm_response = query_vlm_for_response_end(...)
    if vlm_response == "yes":
        if cfg.refine.enabled and target_type in ["object", "image"]:
            refined = refine_near_target(
                tsdf_planner=tsdf_planner,
                scene=scene,
                pts=pts,
                angle=angle,
                target_type=target_type,
                target_index=max_point_choice,
                max_steps=cfg.refine.max_steps,
            )
            log_refine_result(refined)
        break
```

refinement 本质是：**VLM 说“看到了”以后，不要马上停；再走进一点、对准一点。**
