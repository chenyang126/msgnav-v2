# msgnav 现有导航逻辑整理

本文档整理当前代码库中 HM3D/ObjectNav 评估流程使用的导航逻辑、关键模块和主要策略。对应主入口为 `run_hm3d_evaluation.py`，核心导航循环位于 `main()` 中的 episode step loop。

## 1. 总体架构

当前 msgnav 的导航系统可以概括为：

> 多视角 RGB-D 感知 + 3D scene graph 记忆 + TSDF 占用/探索地图 + VLM 高层决策 + frontier/object/image 目标规划。

整体不是单纯的 frontier exploration，也不是只依赖 object detector 的目标导航，而是把语义记忆、前沿探索和 VLM 决策结合起来。

整体流程：

```text
加载配置、数据和模型
  ↓
遍历 scene
  ↓
遍历 episode
  ↓
初始化起点、目标、TSDFPlanner、Logger
  ↓
循环执行导航 step
  ├─ 多视角观察
  ├─ 更新 scene graph
  ├─ 融合 TSDF 占用地图
  ├─ 更新 frontier map
  ├─ VLM 选择下一导航目标
  ├─ planner 执行一步移动
  └─ 必要时 VLM 判断是否完成任务
  ↓
episode 结束后按 GT viewpoint distance 计算 success
```

## 2. 关键文件与职责

| 文件 | 主要职责 |
| --- | --- |
| `run_hm3d_evaluation.py` | 评估入口、scene/episode 遍历、导航主循环 |
| `src/multimodal_3d_scene_graph.py` | Habitat 观察、目标检测/分割、3D object 和 scene graph 维护 |
| `src/tsdf_planner.py` | TSDF 地图、frontier 提取、目标点设置、单步路径执行 |
| `src/query_vlm.py` | 组织 VLM 输入、解析 VLM 输出、将语义选择转成导航目标 |
| `src/explore_utils.py` | KSS 子图筛选、prompt 构造、探索决策、终止检查 prompt |
| `src/logger_hm3d.py` | episode/subtask 日志、路径和评估指标记录 |
| `cfg/eval_hm3d.yaml` | HM3D 评估和导航相关配置 |

## 3. Episode 初始化逻辑

每个 episode 开始时，系统会：

1. 读取目标类别和目标元信息。
2. 根据数据集给出的 `start_position` 和 `start_rotation` 初始化 agent 的位置 `pts` 和朝向 `angle`。
3. 根据当前楼层高度和场景边界初始化 TSDF 地图范围。
4. 根据场景大小计算最大步数：

```python
num_step = int(math.sqrt(scene_size) * cfg.max_step_room_size_ratio)
num_step = max(num_step, 50)
```

5. 初始化 `TSDFPlanner`：

```python
tsdf_planner = TSDFPlanner(
    vol_bnds=tsdf_bnds,
    voxel_size=cfg.tsdf_grid_size,
    floor_height=floor_height,
    floor_height_offset=0,
    pts_init=pts,
    init_clearance=cfg.init_clearance * 2,
    save_visualization=cfg.save_visualization,
)
```

6. 初始化历史决策记忆：

```python
his_decision = {}
subtask_metadata['CLR'] = his_decision
```

这里的 `CLR` 会作为历史决策上下文传给 VLM prompt，用于减少重复选择或无效探索。

## 4. 单步导航循环

主循环位于 `run_hm3d_evaluation.py` 中：

```python
while cnt_step < num_step - 1:
```

每一步包含以下阶段。

### 4.1 多视角观察

系统不是只看当前正前方，而是围绕当前朝向采集多个视角：

- 第 0 步使用 `extra_view_phase_2` 和 `extra_view_angle_deg_phase_2`。
- 后续步骤使用 `extra_view_phase_1` 和 `extra_view_angle_deg_phase_1`。

当前配置中：

```yaml
extra_view_phase_1: 6
extra_view_angle_deg_phase_1: 40
extra_view_phase_2: 6
extra_view_angle_deg_phase_2: 40
```

每个视角通过 `Scene.get_observation(pts, angle=ang)` 获取：

- RGB 图像：`color_sensor`
- 深度图：`depth_sensor`
- 语义图：`semantic_sensor`
- 相机位姿：`cam_pose`

这些观察会被用于三件事：

1. 更新 scene graph。
2. 融合 TSDF 地图。
3. 作为 egocentric views 输入 VLM。

### 4.2 更新 3D scene graph

每个视角都会调用：

```python
scene.update_scene_graph(...)
```

该函数负责把当前 RGB-D 观察转成结构化语义记忆，主要步骤包括：

1. 可选的房间识别：根据 CLIP 相似度判断当前图像更像哪个 room label。
2. YOLO-World 目标检测。
3. SAM 根据检测框做 mask 分割。
4. 使用 depth、mask 和 camera pose 生成 3D object point cloud 和 bounding box。
5. 将新检测到的 object 合并进 `scene.objects`。
6. 更新 object/image/edge 相关的 scene graph 信息。
7. 如果检测到 GT target semantic id，则维护 GT object id 到检测 object id 的映射。

相关配置包括：

```yaml
use_room_filter: true
use_room_det: true
yolo_model_name: yolov8x-world.pt
sam_model_name: sam_l.pt
obj_min_detections: 3
merge_overlap_thresh: 0.7
merge_visual_sim_thresh: 0.8
merge_text_sim_thresh: 0.8
```

### 4.3 融合 TSDF 占用地图

每个 RGB-D 视角都会被融合到 TSDFPlanner：

```python
tsdf_planner.integrate(
    color_im=rgb,
    depth_im=depth,
    cam_intr=cam_intr,
    cam_pose=pose_habitat_to_tsdf(cam_pose),
    obs_weight=1.0,
    margin_h=int(cfg.margin_h_ratio * img_height),
    margin_w=int(cfg.margin_w_ratio * img_width),
    explored_depth=cfg.explored_depth,
)
```

TSDFPlanner 维护的关键状态包括：

| 状态 | 含义 |
| --- | --- |
| `occupied` | 障碍区域 |
| `unoccupied` | 当前认为可通行的区域 |
| `unexplored` / `_explore_vol_cpu` | 已探索/未探索状态 |
| `frontiers` | 探索边界候选 |
| `island` | 当前 agent 所在连通可达区域 |
| `max_point` | 当前高层目标点或 frontier 对象 |
| `target_point` | planner 实际要走到的可行走点 |

相关配置：

```yaml
explored_depth: 1.7
tsdf_grid_size: 0.1
margin_w_ratio: 0.25
margin_h_ratio: 0.6
```

### 4.4 更新 frontier map

每一步完成观察和地图融合后，系统调用：

```python
tsdf_planner.update_frontier_map(
    pts=pts,
    cfg=cfg.planner,
    scene=scene,
    cnt_step=cnt_step,
    save_frontier_image=cfg.save_visualization,
    eps_frontier_dir=eps_frontier_dir,
    prompt_img_size=(cfg.prompt_h, cfg.prompt_w),
)
```

frontier 表示“已知可通行空间”和“未知空间”的边界。它用于回答：如果当前没有足够证据直接去目标，下一步应该往哪里探索。

frontier 策略大致包括：

1. 基于 TSDF 的 unexplored/unoccupied 区域提取候选边界。
2. 过滤面积过小、形状不合适或不可达的 frontier。
3. 为每个 frontier 生成可供 VLM 参考的观测图 `frontier.feature`。
4. 将 frontier 作为候选目标传给 VLM。

相关配置：

```yaml
planner:
  frontier_edge_area_min: 4
  frontier_edge_area_max: 6
  frontier_area_min: 8
  frontier_area_max: 9
  min_frontier_area: 10
  max_frontier_angle_range_deg: 150
  region_equal_threshold: 0.95
```

## 5. VLM 高层决策策略

### 5.1 查询时机

配置项：

```yaml
choose_every_step: true
```

当 `choose_every_step` 为 true 时，系统每一步都会允许 VLM 重新选择 frontier 目标：

```python
if cfg.choose_every_step:
    if tsdf_planner.max_point is not None and type(tsdf_planner.max_point) == Frontier:
        tsdf_planner.max_point = None
        tsdf_planner.target_point = None
```

这意味着：

- 对 frontier 探索目标，系统倾向于每步重新评估。
- 对 object/image 精确目标，如果还没到达，通常不会每步强制重选。

当 planner 当前没有目标时，会调用：

```python
query_vlm_for_response(...)
```

### 5.2 VLM 输入内容

`query_vlm_for_response()` 会构造 `step_dict`，主要包含：

| 输入 | 说明 |
| --- | --- |
| `question` | 当前导航任务问题 |
| `task_type` | 任务类型 |
| `class` | 目标类别/答案类别 |
| `image` | 图像目标信息，如果有 |
| `objects` | 当前 scene graph 中的 objects |
| `obj_map` | object id 到类别名的映射 |
| `object_id_to_room` | object id 到房间标签及置信度的映射 |
| `edges` | scene graph 关系 |
| `all_imgs` | 历史观察图像 |
| `image_to_edges` | 图像和 graph edge 的对应关系 |
| `frontier_imgs` | frontier 候选图像 |
| `egocentric_views` | 当前多视角观察 |
| `CLR` | 历史决策上下文 |

### 5.3 KSS：Key Subgraph Selection

在真正构造 prompt 之前，会先调用：

```python
Key_Subgraph_Selection(step, verbose, cfg.use_ollama, use_room_filter)
```

KSS 的作用是压缩输入上下文，只保留和当前任务最相关的信息，避免 prompt 过长。它输出：

- 当前问题
- 图像目标
- 当前 egocentric 图像
- 筛选后的 object
- 筛选后的 graph edge
- 处理后的历史图像
- frontier 图像

相关配置：

```yaml
prefiltering: true
top_k_categories: 20
use_room_filter: true
```

### 5.4 AVU + CLR prompt

如果开启：

```yaml
use_AVU: true
```

则使用：

```python
Prompt_with_AVU_and_CLR(...)
```

否则使用：

```python
Prompt_without_AVU(...)
```

这里的两个策略是：

- **AVU**：Active Visual Update。VLM 如果选择某张 image 并指定目标类别，系统会对该图重新检测和分割目标，然后恢复 3D 导航点。
- **CLR**：历史决策记忆。把之前的选择、失败判断等信息放入 prompt，帮助 VLM 避免重复或无效动作。

### 5.5 VLM 输出类型

VLM 决策有三类主要输出。

#### 5.5.1 选择已有 object

格式：

```text
object <id>
```

含义：VLM 认为 scene graph 中的某个 object 就是目标，系统应朝这个 object 导航。

#### 5.5.2 选择历史/当前 image

格式：

```text
image <id>, <object_class>
```

含义：VLM 认为某张图里包含目标，但 scene graph 中可能还没有可靠 object。系统会根据 `<object_class>` 对该图做 AVU 重新感知。

#### 5.5.3 继续探索

格式：

```text
continue exploration
```

如果 VLM 判断当前 object/image 信息不足，就进入 frontier-only prompt，再输出：

```text
frontier <id>
```

然后 planner 朝对应 frontier 移动。

如果 VLM 输出格式错误、目标 id 不合法或 API 失败，代码会 fallback 到随机 frontier。

## 6. 目标类型到导航点的转换

VLM 输出的高层选择不会直接用于移动，而是先转成 planner 可执行的导航目标。

核心函数：

```python
TSDFPlanner.set_next_navigation_point(...)
```

### 6.1 Frontier 目标

当目标类型是 `frontier` 时，`choice` 是一个 `Frontier` 对象。

planner 会：

1. 取 frontier 的位置和朝向。
2. 从 frontier 位置沿其 orientation 反方向回退。
3. 找到一个满足条件的点：
   - 在 TSDF 地图边界内；
   - 不在 obstacle 中；
   - 属于当前可达 island。
4. 将该点设为 `target_point`。

这样做的原因是 frontier 本身通常位于已知和未知区域的边界上，直接站到边界点可能不可达或不稳定，需要退回到可行走区域。

### 6.2 Object 目标

当目标类型是 `object` 时，系统先通过 object point cloud / bbox 估计一个适合观察目标的位置。

优先策略：

```python
Visibility_based_Viewpoint_Decision(...)
```

即根据目标点云、其他物体点云、当前 agent 位置和 TSDFPlanner，选择一个可见性较好的观察点。

fallback 策略：

```python
select_navigation_corner(aabb=scene.objects[target_index]["bbox"], robot_position=pts)
```

即从目标 bbox 周围选择一个相对合适的角点。

随后 `set_next_navigation_point()` 会把该 world/habitat 坐标转换到 voxel，并找到最近的可行走点：

```python
target_point = self.habitat2voxel(choice)[:2]
self.target_point = get_nearest_true_point(target_point, self.unoccupied)
```

### 6.3 Image 目标

当目标类型是 `image` 时，系统会执行 AVU：

1. 找到 VLM 选择的历史图像或当前 egocentric view。
2. 用 VLM 输出的 object class 临时设置 YOLO-World 检测类别。
3. 重新检测该图中的目标。
4. 使用 SAM 分割 mask。
5. 结合 depth 和 camera pose 生成 3D 点云和 bbox。
6. 优先调用 `Visibility_based_Viewpoint_Decision()` 选取观察点。
7. 如果失败，则退化为 bbox corner 策略。

这类目标最终也会转成 voxel 下的 `target_point`，并投影到最近可行走区域。

## 7. 低层移动策略

核心函数：

```python
TSDFPlanner.agent_step(...)
```

### 7.1 阶段化步长

planner 根据目标类型使用不同步长：

```python
max_dist_from_cur = (
    cfg.max_dist_from_cur_phase_1
    if type(self.max_point) == Frontier
    else cfg.max_dist_from_cur_phase_2
)
```

配置：

```yaml
planner:
  max_dist_from_cur_phase_1: 1.8
  max_dist_from_cur_phase_2: 1
```

含义：

- **探索阶段**：目标是 frontier，步长较大，快速扩大探索范围。
- **定位阶段**：目标是 object/image，步长较小，避免越过目标或视角不稳定。

### 7.2 路径计算

planner 通过 Habitat pathfinder 计算当前位置到 `target_point` 的距离和路径：

```python
dist, path_to_target = self.get_distance(
    cur_point[:2], self.target_point, height=pts[2], pathfinder=pathfinder
)
```

如果目标距离大于当前阶段最大步长：

- 如果 pathfinder 找到路径，则沿路径前进 `max_dist_from_cur`。
- 如果没有路径，则沿目标方向直线前进，并在发现非法点时不断回退到合法位置。

如果目标距离小于等于当前阶段最大步长：

- 直接移动到 `target_point`。
- 标记 `target_arrived = True`。

### 7.3 导航点修正

当尚未到达目标，或目标是 frontier 时，会调用：

```python
adjust_navigation_point(...)
```

用于把下一步位置从障碍附近挪到更安全的位置。

### 7.4 朝向更新

移动后会重新计算 agent 朝向：

- 如果正在路上：朝向下一步移动方向。
- 如果到达 frontier：朝向 frontier 目标点。
- 如果到达 object/image：朝向语义目标点 `max_point`。

最后转换成 yaw：

```python
next_yaw = np.arctan2(direction[1], direction[0]) - np.pi / 2
```

### 7.5 到达后清空目标

如果 `target_arrived` 为 true：

```python
self.max_point = None
self.target_point = None
```

这样下一轮循环可以重新查询 VLM，选择新的高层目标。

## 8. 任务终止与成功判定

### 8.1 在线 VLM 终止检查

当满足以下条件时：

```python
target_type != "frontier" and target_arrived
```

系统会回看最近若干帧：

```yaml
frames_to_check: 5
```

然后调用：

```python
query_vlm_for_response_end(...)
```

该函数内部通过 `task_check()` 构造终止判断 prompt，要求 VLM 输出：

```text
yes
```

或：

```text
no
```

如果 VLM 返回 `yes`，当前 episode 的 step loop 会提前结束。

注意：frontier 目标到达后不会直接终止任务，因为 frontier 只表示探索方向，不表示任务目标。

### 8.2 最终成功判定

episode 结束后，最终成功不是由 VLM 直接决定，而是由 agent 到 GT target viewpoints 的最短路径距离决定：

```python
agent_subtask_distance = calc_agent_subtask_distance(
    pts, subtask_metadata["viewpoints"], scene.pathfinder
)
```

如果小于：

```yaml
success_distance: 1.0
```

则记为成功：

```python
success_by_distance = True
```

因此当前系统的在线终止逻辑是 VLM-based，但评估指标仍是 GT-distance-based。

## 9. 当前使用到的主要策略总结

### 9.1 多视角主动观察

每一步采集多个角度的 RGB-D 观察，增强局部感知覆盖，减少单视角遗漏。

### 9.2 语义 scene graph 记忆

系统持续维护 objects、images 和 edges，将过去看到的信息结构化保存，供后续 VLM 决策使用。

### 9.3 TSDF 占用建图

使用深度图不断更新可通行区域、障碍区域和探索状态，为 low-level navigation 提供几何基础。

### 9.4 Frontier 探索

当当前语义信息不足以定位目标时，系统通过 frontier 候选引导探索未知区域。

### 9.5 KSS 上下文压缩

使用 Key Subgraph Selection 从完整 scene graph 中筛出和任务相关的对象、图像和关系，控制 prompt 大小。

### 9.6 VLM 高层语义决策

VLM 在 object、image 和 continue exploration 之间做选择，实现语义层面的下一目标决策。

### 9.7 AVU 主动视觉更新

当 VLM 指向一张图像时，系统不会直接相信图像，而是重新检测指定类别，恢复 3D 目标点，再导航过去。

### 9.8 CLR 历史决策记忆

历史决策会被记录并传回 prompt，帮助 VLM 利用之前选择的上下文，降低重复或错误探索概率。

### 9.9 分阶段步长控制

frontier 探索阶段步长较大，object/image 定位阶段步长较小。

### 9.10 VLM 在线完成确认 + GT 最终评估

到达 object/image 目标后，用 VLM 判断任务是否已经完成；最终实验成功率则按 GT viewpoint distance 计算。

## 10. 可调整的关键配置

| 配置项 | 作用 |
| --- | --- |
| `choose_every_step` | 是否每步重新让 VLM 决策 frontier 目标 |
| `egocentric_views` | 是否把当前多视角图像加入 prompt |
| `prefiltering` | 是否使用 KSS/prefiltering 控制上下文大小 |
| `use_AVU` | 是否启用 image 选择后的主动视觉更新 |
| `use_room_filter` | 是否用房间标签辅助筛选和 prompt |
| `use_room_det` | 是否检测当前观察对应的房间类型 |
| `top_k_categories` | KSS 中保留的相关类别数量 |
| `frames_to_check` | 终止判断时回看最近多少帧 |
| `dicision_radius` | 可见性视点选择半径，当前拼写为代码中的 `dicision_radius` |
| `success_distance` | 最终 GT 距离成功阈值 |
| `tsdf_grid_size` | TSDF voxel 分辨率 |
| `explored_depth` | 单次深度观察标记探索的深度范围 |
| `planner.max_dist_from_cur_phase_1` | frontier 探索阶段单步最大距离 |
| `planner.max_dist_from_cur_phase_2` | object/image 定位阶段单步最大距离 |
| `planner.surrounding_explored_radius` | 移动后周围标记为 explored 的半径 |
| `planner.min_frontier_area` | frontier 最小面积过滤阈值 |

## 11. 代码级主流程伪代码

```python
for scene in scenes:
    scene = Scene(...)

    for episode in episodes:
        pts, angle = init_start_pose(episode)
        tsdf_planner = TSDFPlanner(...)
        subtask_metadata = logger.init_subtask(...)
        subtask_metadata["CLR"] = {}

        while cnt_step < num_step:
            # 1. observe
            rgb_egocentric_views = []
            for ang in multi_view_angles(angle):
                obs, cam_pose = scene.get_observation(pts, angle=ang)
                scene.update_scene_graph(obs, cam_pose, ...)
                tsdf_planner.integrate(obs, cam_pose, ...)
                rgb_egocentric_views.append(obs.rgb)

            # 2. update memory and frontier
            scene.del_unused_scene_graph_edges()
            tsdf_planner.update_frontier_map(...)

            # 3. high-level decision
            if choose_every_step and current_target_is_frontier:
                tsdf_planner.clear_target()

            if tsdf_planner.has_no_target():
                target_type, choice, _, target_index = query_vlm_for_response(...)
                tsdf_planner.set_next_navigation_point(target_type, choice, ...)

            # 4. low-level step
            pts, angle, pts_voxel, fig, _, target_arrived = tsdf_planner.agent_step(...)
            logger.log_step(pts_voxel)

            # 5. task completion check
            if target_type != "frontier" and target_arrived:
                answer = query_vlm_for_response_end(recent_frames, ...)
                if answer == "yes":
                    break

        # 6. final metric
        success = calc_distance_to_gt_viewpoints(pts) < cfg.success_distance
        logger.log_subtask_result(success)
```

## 12. 后续修改导航逻辑时建议关注的位置

如果要改高层策略，优先看：

- `src/query_vlm.py::query_vlm_for_response`
- `src/explore_utils.py::explore_two_step`
- `src/explore_utils.py::Key_Subgraph_Selection`
- prompt 构造相关函数：`Prompt_with_AVU_and_CLR`、`Prompt_without_AVU`、`format_exploreonly_prompt`

如果要改低层移动和 frontier，优先看：

- `src/tsdf_planner.py::update_frontier_map`
- `src/tsdf_planner.py::set_next_navigation_point`
- `src/tsdf_planner.py::agent_step`
- `src/tsdf_planner.py::get_island_around_pts`

如果要改感知和记忆，优先看：

- `src/multimodal_3d_scene_graph.py::get_observation`
- `src/multimodal_3d_scene_graph.py::update_scene_graph`
- `src/multimodal_3d_scene_graph.py::periodic_cleanup_objects`
- `src/multimodal_3d_scene_graph.py::del_unused_scene_graph_edges`

如果要改评估和日志，优先看：

- `src/logger_hm3d.py`
- `run_hm3d_evaluation.py` 中 episode 结束后的 `calc_agent_subtask_distance` 和 `logger.log_subtask_result`

## 13. 为失败轨迹分析建议补充保存的数据

上面的内容主要整理了“系统如何导航”。如果后续要从失败轨迹中定位原因，还需要额外保存“每一步为什么这么走、当时有哪些候选、VLM 和 planner 分别做了什么”。建议把导航日志分成轻量结构化日志和重型可视化/数组快照两层。

### 13.1 最小必要日志

如果只做最小改动，建议至少为每个 episode 保存以下几个 JSONL 文件。

```text
episode_xxx/
  trajectory.jsonl
  decisions.jsonl
  planner.jsonl
  frontiers.jsonl
  objects.jsonl
  end_checks.jsonl
  events.jsonl
```

#### `trajectory.jsonl`

用于复盘 agent 实际轨迹。

建议字段：

```json
{
  "step": 12,
  "pts_before": [0.0, 0.0, 0.0],
  "pts_after": [0.0, 0.0, 0.0],
  "pts_voxel_before": [0, 0],
  "pts_voxel_after": [0, 0],
  "angle_before": 0.0,
  "angle_after": 0.0,
  "step_distance": 0.0,
  "total_distance": 0.0,
  "target_arrived": false,
  "dist_to_gt_viewpoint": 0.0
}
```

其中 `dist_to_gt_viewpoint` 很关键，可以判断每一步是否在接近真实目标。

#### `decisions.jsonl`

用于复盘 VLM 决策。

建议字段：

```json
{
  "step": 12,
  "vlm_called": true,
  "raw_response": "frontier 2",
  "parsed_target_type": "frontier",
  "parsed_target_index": 2,
  "reason": "...",
  "fallback_used": false,
  "fallback_reason": null,
  "n_filtered_snapshots": 3
}
```

需要特别记录 fallback，因为很多失败可能不是 VLM 真正选择了错误目标，而是格式错误、API 返回空、id 越界或 AVU 失败后退化到了随机 frontier。

#### `planner.jsonl`

用于分析从高层目标到实际可行走点的转换是否出错。

建议字段：

```json
{
  "step": 12,
  "target_type": "object",
  "max_point": [0, 0],
  "target_point": [0, 0],
  "target_point_habitat": [0.0, 0.0, 0.0],
  "dist_to_target": 0.0,
  "pathfinder_success": true,
  "path_length": 0.0,
  "agent_step_success": true,
  "agent_step_failure_reason": null
}
```

这能区分“VLM 选错”和“VLM 选对但 planner 投影/路径失败”。

#### `frontiers.jsonl`

用于分析探索候选是否合理。

建议字段：

```json
{
  "step": 12,
  "chosen_frontier_id": 2,
  "frontiers": [
    {
      "id": 0,
      "position_voxel": [0, 0],
      "orientation": [0.0, 1.0],
      "area": 20,
      "distance_from_agent": 3.5,
      "reachable": true,
      "chosen": false,
      "image_path": "frontiers/12_0.png"
    }
  ]
}
```

这能判断 frontier 是否缺失、是否不可达、VLM 是否反复选择无效 frontier。

#### `objects.jsonl`

用于分析 scene graph 中当时有哪些 object，以及 VLM 是否漏选了正确候选。

建议字段：

```json
{
  "step": 12,
  "objects": [
    {
      "id": 5,
      "class_name": "chair",
      "room_label": "living room",
      "room_conf": 0.71,
      "bbox_center": [0.0, 0.0, 0.0],
      "bbox_extent": [0.0, 0.0, 0.0],
      "num_points": 1200,
      "num_detections": 3,
      "last_seen_step": 12,
      "chosen_by_vlm": true
    }
  ]
}
```

如果 GT target 已经出现在 scene graph 中，但 VLM 一直没选，问题更可能在 KSS/prompt/VLM 决策；如果目标压根没进 scene graph，问题更可能在感知或合并。

#### `end_checks.jsonl`

用于分析在线终止判断是否错误。

建议字段：

```json
{
  "step": 12,
  "called": true,
  "frames_checked": ["12-view_0.png", "12-view_1.png"],
  "raw_response": "no",
  "parsed_response": "no",
  "reason": "..."
}
```

#### `events.jsonl`

用于记录异常、退化和关键事件。

建议事件：

```json
{"step": 4, "event": "vlm_invalid_format", "raw_response": "..."}
{"step": 7, "event": "random_frontier_fallback", "reason": "frontier index out of range"}
{"step": 9, "event": "avu_failed", "reason": "no detection in selected image"}
{"step": 13, "event": "pathfinder_failed", "from": [0, 0], "to": [1, 1]}
```

该文件适合后续做批量统计，例如统计失败中 AVU 失败、VLM 格式错误、pathfinder 失败分别占多少。

### 13.2 推荐但可选的重型数据

结构化 JSONL 足够做大部分批量分析，但如果要逐帧复盘，还建议按 debug 开关保存以下数据。

```text
episode_xxx/
  observations/
    step_0012_view_0_rgb.png
    step_0012_view_0_depth.npy
    step_0012_view_0_semantic.npy
    step_0012_view_0_annotated.png
    step_0012_view_0_detections.json
  maps/
    step_0012_topdown.png
    step_0012_map.npz
  prompts/
    step_0012_prompt_summary.json
    step_0012_vlm_response.txt
```

其中：

- RGB/annotated 图用于肉眼检查目标是否可见、检测是否正确。
- depth/semantic 用于分析感知失败和 GT 可见性。
- map npz 用于脚本化分析 occupied、unoccupied、explored、frontier、agent、target 的关系。
- prompt summary 用于确认 KSS 后到底给了 VLM 哪些 objects/images/frontiers。

不建议默认保存完整 base64 prompt；更推荐保存结构化摘要和图像文件路径，避免日志过大。

### 13.3 AVU 过程需要单独记录

当 VLM 输出 `image <id>, <object_class>` 时，建议在 `decisions.jsonl` 或单独 `avu.jsonl` 中记录 AVU 细节：

```json
{
  "step": 12,
  "selected_image": "8-view_3.png",
  "requested_class": "chair",
  "detection_success": true,
  "max_confidence": 0.37,
  "bbox_xyxy": [0, 0, 0, 0],
  "mask_area": 2341,
  "pointcloud_points": 812,
  "selected_viewpoint": [0.0, 0.0, 0.0],
  "fallback_to_bbox_corner": false,
  "failure_reason": null
}
```

这能判断：VLM 是否选对了图、检测器是否识别出指定类别、SAM mask 是否足够、从 depth 恢复出的 3D 点是否可靠。

### 13.4 Oracle debug 信息

为了做失败归因，可以保存一些不参与决策、只用于离线分析的 GT 信息：

| 字段 | 作用 |
| --- | --- |
| `gt_goal_object_ids` | 当前 episode 的真实目标对象 id |
| `gt_viewpoints` | 真实目标可达视点 |
| `dist_to_nearest_gt_viewpoint_per_step` | 每步是否接近目标 |
| `gt_target_visible` | 当前 semantic obs 中是否能看到目标 |
| `gt_target_pixel_area` | 目标在图像中的像素面积 |
| `detected_target_mapping` | GT object id 到检测 object id 的映射 |

这些字段只应用于 debug/analysis，不应该影响导航决策。

## 14. 失败归因建议

补充上述日志后，可以把失败初步分成以下类型：

| 失败类型 | 判断依据 |
| --- | --- |
| 感知失败 | GT target visible，但 detector/scene graph 没有目标 |
| 记忆失败 | 目标曾被看到，但后续 object 被清理、合并错或 id 漂移 |
| KSS 失败 | 目标在完整 scene graph 中，但没有进入 selected objects/images |
| VLM 决策失败 | 候选中有正确目标，但 VLM 选择了错误 object/image/frontier |
| AVU 失败 | VLM 选对 image，但重新检测、分割或 3D 恢复失败 |
| Frontier 失败 | 没有生成通向目标区域的 frontier，或反复选择无效 frontier |
| Planner 失败 | 高层目标正确，但 target_point 投影、pathfinder 或 agent_step 失败 |
| 终止失败 | 已接近目标但 VLM end check 返回 no，或未到目标却返回 yes |
| 评估不一致 | VLM 判断完成，但最终 GT distance 大于 `success_distance` |

## 15. 推荐落地顺序

建议不要一开始就保存所有 RGB-D 和地图数组，否则磁盘压力会很大。推荐分三阶段实现：

1. **第一阶段：结构化轻量日志**
   - `trajectory.jsonl`
   - `decisions.jsonl`
   - `planner.jsonl`
   - `frontiers.jsonl`
   - `objects.jsonl`
   - `events.jsonl`

2. **第二阶段：关键帧可视化**
   - 只在 VLM 决策步、target_arrived 步、fallback 步、episode 失败时保存 RGB、annotated RGB 和 topdown。

3. **第三阶段：完整 debug 模式**
   - 通过配置开关保存每步 depth、semantic、map npz、prompt summary 和 AVU 细节。

对应代码上，建议优先在 `src/logger_hm3d.py` 增加通用 JSONL 写入接口，然后在 `run_hm3d_evaluation.py` 主循环中按阶段调用。这样可以尽量不侵入 `Scene`、`TSDFPlanner` 和 VLM 逻辑本身。
