import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # disable warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)

# Force headless mode
os.environ["HABITAT_SIM_HEADLESS"] = "1"
os.environ["DISPLAY"] = ""
os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"
import argparse
from omegaconf import OmegaConf
import random
import numpy as np
import torch

# torch>=2.6 changed torch.load default to weights_only=True, which breaks loading
# the trusted local YOLO/SAM checkpoints. Restore the old default.
_orig_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_compat

import math
import time
import json
import logging
import base64
import matplotlib.pyplot as plt

import open_clip
from ultralytics import SAM, YOLOWorld

from src.habitat import pose_habitat_to_tsdf
from src.geom import get_cam_intr, get_scene_bnds
from src.tsdf_planner import TSDFPlanner, Frontier
from src.multimodal_3d_scene_graph import Scene
from src.utils import resize_image, calc_agent_subtask_distance, get_pts_angle_goatbench
from src.dataset_utils import prepare_goatbench_navigation_goals
from src.query_vlm import (
    query_vlm_for_response,
    query_vlm_for_response_end,
    query_vlm_for_response_end_strict,
    query_vlm_same_target_check,
)
from src.logger_goatbench import Logger
import time


class Timer:
    def __init__(self):
        self.start_time = {}
        self.total_time = {}
        self.count = {}

    def start(self, module_name):
        self.start_time[module_name] = time.time()
        if module_name not in self.total_time:
            self.total_time[module_name] = 0.0
            self.count[module_name] = 0

    def stop(self, module_name):
        if module_name not in self.start_time:
            raise ValueError(f"Module '{module_name}' was not started.")
        
        end_time = time.time()
        elapsed_time = end_time - self.start_time[module_name]
        
        self.total_time[module_name] += elapsed_time
        self.count[module_name] += 1

    def get_average_time(self, module_name):
        if module_name not in self.total_time:
            raise ValueError(f"Module '{module_name}' does not exist.")
        
        if self.count[module_name] == 0:
            return 0.0
        return self.total_time[module_name] / self.count[module_name]

    def reset(self, module_name=None):
        if module_name is None:
            self.start_time = {}
            self.total_time = {}
            self.count = {}
        elif module_name in self.total_time:
            del self.start_time[module_name]
            del self.total_time[module_name]
            del self.count[module_name]
        else:
            raise ValueError(f"Module '{module_name}' does not exist.")

    def print_summary(self, logger):
        logging.info("Time counter:")
        idx = 0
        sum = 0
        for module in self.total_time:
            avg_time = self.get_average_time(module)
            sum += avg_time

        logging.info(f" total average {sum:.6f}s / per step")
        for module in self.total_time:
            avg_time = self.get_average_time(module)
            idx += 1
            logging.info(f" {idx} {module}: average {avg_time:.6f} s")


def _summarize_frontiers(tsdf_planner, chosen_frontier=None):
    frontiers = []
    for idx, frontier in enumerate(tsdf_planner.frontiers):
        frontiers.append(
            {
                "id": idx,
                "position": getattr(frontier, "position", None),
                "orientation": getattr(frontier, "orientation", None),
                "image": getattr(frontier, "image", None),
                "chosen": chosen_frontier is not None and frontier is chosen_frontier,
            }
        )
    return frontiers


def _summarize_objects(scene):
    objects = []
    for obj_id, obj in scene.objects.items():
        bbox = obj.get("bbox")
        pcd = obj.get("pcd")
        try:
            num_points = len(pcd.points) if pcd is not None else None
        except Exception:
            num_points = None
        objects.append(
            {
                "id": obj_id,
                "class_name": obj.get("class_name"),
                "room_label": obj.get("room_label"),
                "room_conf": obj.get("room_conf"),
                "bbox_center": getattr(bbox, "center", None),
                "num_points": num_points,
            }
        )
    return objects


def _planner_point_summary(tsdf_planner):
    return {
        "max_point": tsdf_planner.max_point,
        "target_point": tsdf_planner.target_point,
        "target_type": getattr(tsdf_planner, "target_type", None),
    }


def _safe_goal_distance(pts, viewpoints, pathfinder):
    try:
        return calc_agent_subtask_distance(pts, viewpoints, pathfinder)
    except Exception as exc:
        logging.warning(f"Failed to calculate trajectory goal distance: {exc}")
        return None


def _finish_step_trajectory(
    logger,
    step_log,
    pts_after,
    angle_after,
    scene,
    tsdf_planner,
    subtask_metadata,
    target_arrived=None,
):
    step_log.update(
        {
            "pts_after": pts_after,
            "angle_after": angle_after,
            "target_arrived": target_arrived,
            "explore_dist": logger.subtask_explore_dist,
            "dist_to_gt_viewpoint": _safe_goal_distance(
                pts_after, subtask_metadata["viewpoints"], scene.pathfinder
            ),
            "planner_after": _planner_point_summary(tsdf_planner),
        }
    )
    logger.log_step_trajectory(step_log)


def _cfg_get(cfg, name, default):
    try:
        return getattr(cfg, name)
    except Exception:
        return default


def _cfg_list(cfg, name, default):
    value = _cfg_get(cfg, name, default)
    if value is None:
        return []
    return list(value)


def _normalize_text(text):
    return str(text or "").strip().lower()


def _load_image_b64(image_path):
    """Read an image file into a base64 string, for embedding directly into an
    EpisodeMemory anchor. Anchors must be self-contained (not just a path) --
    the subtask directory that held the original file gets deleted (see
    `rm -r ... subtask_id` near the end of the per-subtask loop) once
    `save_visualization` is off, long before a later subtask might want to
    compare against this anchor's image.
    """
    if not image_path or not os.path.exists(image_path):
        return None
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logging.warning(f"EpisodeMemory: failed to read reference image {image_path}: {e}")
        return None


class EpisodeMemory:
    """Cross-subtask memory anchors within one episode.

    A thin outcome-annotation layer keyed by scene-graph node id: it records
    the navigation *outcome* (positive/negative), the successful view pose,
    and the subtask query used for identity disambiguation. Geometry is always
    resolved live from scene.objects[id].bbox.center when a candidate is queried,
    but EpisodeMemory is observational by default: prompt hints and live-center
    navigation fallback are disabled unless explicit config knobs enable them.
    Identity is pre-filtered with cheap graph signals (CLIP/room) before any VLM
    same-target check.
    """

    def __init__(self, cfg):
        self.anchors = {}  # node_id -> list[record dict]
        mc = _cfg_get(cfg, "episode_memory", None)
        self.enabled = mc is not None and _cfg_get(mc, "enabled", False)
        self.spatial_match_radius = float(_cfg_get(mc, "spatial_match_radius", 0.6)) if mc else 0.6
        self.min_confidence_dist = float(_cfg_get(mc, "min_confidence_dist", 0.20)) if mc else 0.20
        # ---- behavior flags (see cfg/eval_goatbench.yaml: episode_memory) ----
        self.record_negatives = bool(_cfg_get(mc, "record_negatives", False)) if mc else False
        self.record_strict_rejections = bool(_cfg_get(mc, "record_strict_rejections", False)) if mc else False
        self.surface_negatives_in_prompt = bool(_cfg_get(mc, "surface_negatives_in_prompt", False)) if mc else False
        self.use_view_pose_bias = bool(_cfg_get(mc, "use_view_pose_bias", False)) if mc else False
        pf = _cfg_get(mc, "identity_prefilter", None) if mc else None
        self.pf_enabled = bool(_cfg_get(pf, "enabled", False)) if pf else False
        self.pf_clip_high = float(_cfg_get(pf, "clip_high", 0.92)) if pf else 0.92
        self.pf_clip_low = float(_cfg_get(pf, "clip_low", 0.70)) if pf else 0.70
        self.pf_use_room = bool(_cfg_get(pf, "use_room", False)) if pf else False
        self.pf_room_conf_min = float(_cfg_get(pf, "room_conf_min", 0.5)) if pf else 0.5
        self.pf_snapshot_clip = bool(_cfg_get(pf, "snapshot_clip", False)) if pf else False
        pp = _cfg_get(mc, "prompt_prior", None) if mc else None
        self.pp_annotate_positive = bool(_cfg_get(pp, "annotate_positive", True)) if pp else False
        self.pp_memory_block = bool(_cfg_get(pp, "memory_block", True)) if pp else False
        lf = _cfg_get(mc, "live_center_fallback", None) if mc else None
        self.lcf_enabled = bool(_cfg_get(lf, "enabled", False)) if lf else False
        self.lcf_allow_match_types = set(_cfg_list(lf, "allow_match_types", ["exact_id"])) if lf else {"exact_id"}
        self.lcf_min_distinct_positive_subtasks = int(_cfg_get(lf, "min_distinct_positive_subtasks", 2)) if lf else 2
        self.lcf_allowed_task_types = {
            _normalize_text(x) for x in _cfg_list(lf, "allowed_task_types", [])
        } if lf else set()
        # VLM-verified identity check result cache, keyed by (subtask_idx, id(anchor)).
        # query()/annotate_nodes() run every step while a target is active, so without
        # caching the same anchor would get re-checked by the VLM on every single step
        # of a subtask instead of once.
        self._identity_cache = {}

    # ------------------------------------------------------------------
    # Small helpers shared by the redesigned path
    # ------------------------------------------------------------------
    def _live_center(self, scene, node_id):
        """Current scene-graph center of a node, or None if the node no longer
        exists (merged/cleared). Geometry is always resolved live -- never from
        a frozen snapshot -- so a memory hit benefits from the continued
        refinement of bbox.center as more detections merge in."""
        try:
            if scene is not None and isinstance(node_id, int) and node_id in scene.objects:
                return np.array(scene.objects[node_id]["bbox"].center, dtype=float)
        except Exception:
            pass
        return None

    def _get_clip_ft(self, scene, node_id, anchor=None):
        """1-D CPU tensor of a node's CLIP feature. Prefer the live node; fall
        back to the anchor's stored snapshot only if the node is gone."""
        try:
            if scene is not None and isinstance(node_id, int) and node_id in scene.objects:
                ft = scene.objects[node_id].get("clip_ft")
                if ft is not None:
                    return torch.as_tensor(ft).detach().float().cpu().reshape(-1)
        except Exception:
            pass
        if anchor is not None and anchor.get("clip_ft") is not None:
            try:
                return torch.as_tensor(anchor["clip_ft"]).detach().float().cpu().reshape(-1)
            except Exception:
                pass
        return None

    def _graph_identity_prefilter(self, scene, query_node_id, anchor):
        """Cheap, VLM-free identity signal between the queried node and an
        anchor's node, from scene-graph features. Returns "pass" (same
        instance), "reject" (different instance), or "uncertain" (fall through
        to the VLM gate). For an exact-id match the CLIP cosine is ~1.0, so the
        common case resolves here without any VLM call."""
        if not self.pf_enabled:
            return "uncertain"
        verdict = "uncertain"
        q_ft = self._get_clip_ft(scene, query_node_id)
        a_ft = self._get_clip_ft(scene, anchor.get("node_id"), anchor=anchor)
        if q_ft is not None and a_ft is not None and q_ft.shape == a_ft.shape:
            sim = torch.nn.functional.cosine_similarity(
                q_ft.unsqueeze(0), a_ft.unsqueeze(0)
            ).item()
            if sim >= self.pf_clip_high:
                verdict = "pass"
            elif sim <= self.pf_clip_low:
                return "reject"
        # room signal: a confident room disagreement pushes an otherwise
        # "pass" down to "uncertain" (never upgrades, never hard-rejects alone)
        if self.pf_use_room and verdict == "pass":
            try:
                obj = scene.objects.get(query_node_id) if scene is not None else None
                if obj is not None and anchor.get("room_label"):
                    q_room = _normalize_text(obj.get("room_label"))
                    a_room = _normalize_text(anchor.get("room_label"))
                    q_conf = float(obj.get("room_conf", 0.0) or 0.0)
                    if q_room and a_room and q_room != a_room and q_conf >= self.pf_room_conf_min:
                        verdict = "uncertain"
            except Exception:
                pass
        return verdict

    def _identity_ok(self, scene, query_node_id, anchor, task_type,
                     description, image_b64, subtask_idx):
        """Unified identity gate: cheap graph pre-filter first, VLM only when
        genuinely uncertain. "object" task-type queries carry no distinguishing
        text (fixed template), so they never trigger the VLM -- they are judged
        on the CLIP/room pre-filter alone (a stricter signal than the legacy
        unconditional pass)."""
        verdict = self._graph_identity_prefilter(scene, query_node_id, anchor)
        if verdict == "pass":
            return True
        if verdict == "reject":
            return False
        # uncertain
        if task_type == "object":
            return True
        return self._passes_identity_gate(anchor, task_type, description, image_b64, subtask_idx)

    def record(self, scene, node_id, subtask_idx, task_type, target_class,
               success, final_dist, view_position=None, view_angle=None,
               outcome=None, neg_reason=None, description=None, image_b64=None):
        """Record a subtask outcome against a scene-graph node. Stores only what
        the scene graph lacks -- the navigation outcome, the successful view
        pose, and the subtask query used for identity disambiguation. Geometry
        is NOT stored (resolved live at query time)."""
        if not self.enabled or node_id is None:
            return

        if outcome is None:
            outcome = ("positive" if (success and final_dist is not None
                                      and final_dist <= self.min_confidence_dist)
                       else "negative")
        if outcome == "negative" and not self.record_negatives:
            return

        clip_snapshot = None
        room_snapshot = None
        try:
            if isinstance(node_id, int) and node_id in scene.objects:
                obj = scene.objects[node_id]
                room_snapshot = obj.get("room_label")
                if self.pf_snapshot_clip and obj.get("clip_ft") is not None:
                    clip_snapshot = torch.as_tensor(obj["clip_ft"]).detach().float().cpu().numpy().copy()
        except Exception:
            pass

        record = {
            "node_id": node_id,
            "subtask_idx": subtask_idx,
            "task_type": task_type,
            "target_class": _normalize_text(target_class),
            "outcome": outcome,
            "success": bool(success),
            "final_dist": (float(final_dist) if final_dist is not None else None),
            "neg_reason": neg_reason if outcome == "negative" else None,
            "view_position": (np.array(view_position, dtype=float).copy()
                              if view_position is not None else None),
            "view_angle": (float(view_angle) if view_angle is not None else None),
            "description": description,
            "image_b64": image_b64,
            "clip_ft": clip_snapshot,
            "room_label": room_snapshot,
        }
        self.anchors.setdefault(node_id, []).append(record)

    def _count_distinct_positive_subtasks(self, node_id):
        """Number of distinct previous subtasks that produced a confident
        positive anchor for this scene-graph node."""
        positive_subtasks = set()
        for anchor in self.anchors.get(node_id, []):
            if anchor.get("outcome") != "positive":
                continue
            fd = anchor.get("final_dist")
            if fd is not None and fd <= self.min_confidence_dist:
                positive_subtasks.add(anchor.get("subtask_idx"))
        return len(positive_subtasks)

    def _should_apply_live_center_fallback(self, positive, task_type):
        """Conservative opt-in gate for memory-driven navigation changes.

        By default EpisodeMemory is observational: it records/query/logs memory
        candidates but does not change the VLM-selected navigation point. This
        gate is intentionally narrow for experiments that want to test repeated
        exact-id fallback without broad prompt/geometry side effects.
        """
        if not self.lcf_enabled or positive is None:
            return False
        if positive.get("live_center") is None:
            return False
        if positive.get("match") not in self.lcf_allow_match_types:
            return False
        if self.lcf_allowed_task_types and _normalize_text(task_type) not in self.lcf_allowed_task_types:
            return False
        return int(positive.get("n_prior_positive_subtasks", 0)) >= self.lcf_min_distinct_positive_subtasks

    def query(self, scene, target_obj_id, target_class=None, task_type=None,
              description=None, image_b64=None, subtask_idx=None):
        """Layered cheap->expensive match returning both a positive hit and any
        surfaced negatives. All geometry is resolved live from the scene graph.

        Returns:
            {"positive": {node_id, match, from_subtask, final_dist,
                          live_center, view_position, view_angle} | None,
             "negatives": [{node_id, neg_reason, from_subtask, match}, ...]}
        """
        if not self.enabled:
            return {"positive": None, "negatives": []}

        query_class = _normalize_text(target_class)
        positive = None
        negatives = []

        def _consider(anchor, node_id, match):
            nonlocal positive
            if not self._identity_ok(scene, node_id, anchor, task_type,
                                     description, image_b64, subtask_idx):
                return
            if anchor["outcome"] == "positive":
                fd = anchor["final_dist"]
                if fd is None or fd > self.min_confidence_dist:
                    return
                n_prior_positive_subtasks = self._count_distinct_positive_subtasks(node_id)
                if positive is None or fd < positive["final_dist"]:
                    positive = {
                        "node_id": node_id,
                        "match": match,
                        "from_subtask": anchor["subtask_idx"],
                        "final_dist": fd,
                        "live_center": self._live_center(scene, node_id),
                        "view_position": anchor.get("view_position"),
                        "view_angle": anchor.get("view_angle"),
                        "n_prior_positive_subtasks": n_prior_positive_subtasks,
                    }
            elif self.surface_negatives_in_prompt:
                negatives.append({
                    "node_id": node_id,
                    "neg_reason": anchor.get("neg_reason"),
                    "from_subtask": anchor["subtask_idx"],
                    "match": match,
                })

        # Level 1: exact node-id match (cheapest; CLIP pre-filter ~1.0 -> no VLM)
        if target_obj_id in self.anchors:
            for anchor in self.anchors[target_obj_id]:
                _consider(anchor, target_obj_id, "exact_id")

        # Level 2: spatial proximity using LIVE centers of both nodes
        q_center = self._live_center(scene, target_obj_id)
        if q_center is not None:
            tc_xz = np.array([q_center[0], q_center[2]], dtype=float)
            for a_node, anchor_list in self.anchors.items():
                if a_node == target_obj_id:
                    continue
                a_center = self._live_center(scene, a_node)
                if a_center is None:
                    continue
                d = np.linalg.norm(np.array([a_center[0], a_center[2]]) - tc_xz)
                if d >= self.spatial_match_radius:
                    continue
                for anchor in anchor_list:
                    if query_class and anchor["target_class"] != query_class:
                        continue
                    _consider(anchor, a_node, "spatial")

        return {"positive": positive, "negatives": negatives}

    def annotate_nodes(self, scene, task_type, target_class, description,
                       image_b64, subtask_idx):
        """Build per-node memory hints for the VLM prompt (soft prior), for
        every recorded node that still exists in the scene graph and passes the
        identity gate against the current subtask. Called once per step BEFORE
        the VLM query (the VLM decision precedes query()). Returns
        {node_id: {"positive", "negative", "from_subtask", "final_dist",
                   "neg_reason"}}."""
        hints = {}
        if not self.enabled or not (
            self.pp_annotate_positive or self.surface_negatives_in_prompt
        ):
            return hints
        for node_id, anchor_list in self.anchors.items():
            if not (isinstance(node_id, int) and node_id in scene.objects):
                continue
            for anchor in anchor_list:
                if not self._identity_ok(scene, node_id, anchor, task_type,
                                         description, image_b64, subtask_idx):
                    continue
                h = hints.setdefault(node_id, {"positive": False, "negative": False,
                                               "from_subtask": None, "final_dist": None,
                                               "neg_reason": None})
                if anchor["outcome"] == "positive" and self.pp_annotate_positive:
                    fd = anchor["final_dist"]
                    if fd is not None and fd <= self.min_confidence_dist:
                        h["positive"] = True
                        h["from_subtask"] = anchor["subtask_idx"]
                        h["final_dist"] = fd
                elif self.surface_negatives_in_prompt:
                    h["negative"] = True
                    h["neg_reason"] = anchor.get("neg_reason")
                    if h["from_subtask"] is None:
                        h["from_subtask"] = anchor["subtask_idx"]
        # drop nodes that ended up with no active hint
        return {k: v for k, v in hints.items() if v["positive"] or v["negative"]}

    def _passes_identity_gate(self, anchor, task_type, description, image_b64, subtask_idx):
        """Reject a candidate anchor that looks like a different physical
        instance than the one currently being queried, per a VLM comparison
        of the two subtasks' own descriptions/images.

        "object" task-type queries carry no distinguishing info beyond the
        class name (the question is a fixed template, e.g. "Can you find the
        mirror?", identical for every mirror in the home) -- asking a VLM to
        compare two copies of that same generic sentence can't discriminate
        and risks an arbitrary/hallucinated rejection of a legitimate reuse,
        so the gate is skipped (passes through) for that task type and this
        candidate is judged solely on the existing target_obj_id/class/
        spatial checks, same as before this gate existed.

        Results are cached per (subtask_idx, anchor) since query() is called
        every step while a target is active -- without this the same anchor
        would trigger a fresh VLM call on every single step.
        """
        if task_type == "object":
            return True
        cache_key = (subtask_idx, id(anchor))
        if cache_key in self._identity_cache:
            return self._identity_cache[cache_key]
        same = query_vlm_same_target_check(
            target_class=anchor.get("target_class"),
            old_description=anchor.get("description"),
            new_description=description,
            old_image=anchor.get("image_b64"),
            new_image=image_b64,
        )
        self._identity_cache[cache_key] = same
        return same


def _strict_stop_reasons(cfg, subtask_metadata, target_type, cnt_step, num_step):
    """Decide whether the strict (second-opinion) stop-check should run this step.

    Two-tier logic:

      1. SCOPE (gate) -- `strict_for_task_types` / `strict_for_target_types`.
         These define *which subtasks are eligible at all*. When a scope is
         configured and the subtask is outside it, we return [] immediately and
         the strict check never runs. This keeps strict stopping from leaking
         into task types it was never meant for (the earlier version OR'd every
         condition, so a `plant`/`rug` object-task or any step<=min_steps got
         strict-checked regardless of task type, which regressed object &
         description success).

      2. REFINEMENTS (within scope) -- `strict_for_min_steps`,
         `strict_for_early_fraction`, `hard_targets`. These only add reasons for
         subtasks that already passed the gate; they never expand the scope.

    Backward-compat: if NO scope is configured (both lists empty), the
    refinements act as the sole triggers, matching the old behaviour.
    """
    stop_cfg = _cfg_get(cfg, "stop_check", None)
    if stop_cfg is None or not _cfg_get(stop_cfg, "enabled", False):
        return []
    if not _cfg_get(stop_cfg, "strict_enabled", False):
        return []

    task_type = _normalize_text(subtask_metadata.get("task_type"))
    target_type_norm = _normalize_text(target_type)
    question = _normalize_text(subtask_metadata.get("question"))
    target_class = _normalize_text(subtask_metadata.get("class"))

    # ---- tier 1: scope gate ----
    strict_task_types = [_normalize_text(x) for x in _cfg_list(stop_cfg, "strict_for_task_types", [])]
    strict_target_types = [_normalize_text(x) for x in _cfg_list(stop_cfg, "strict_for_target_types", [])]

    scope_reasons = []
    if task_type in strict_task_types:
        scope_reasons.append(f"task_type:{task_type}")
    if target_type_norm in strict_target_types:
        scope_reasons.append(f"target_type:{target_type_norm}")

    scope_configured = bool(strict_task_types or strict_target_types)
    if scope_configured and not scope_reasons:
        # a scope is set and this subtask is outside it -> never strict-check
        return []

    reasons = list(scope_reasons)

    # ---- tier 2: refinements (only reached for in-scope subtasks, or when no
    #      scope is configured at all) ----
    min_steps = int(_cfg_get(stop_cfg, "strict_for_min_steps", 0) or 0)
    if min_steps > 0 and cnt_step + 1 <= min_steps:
        reasons.append(f"early_step:{cnt_step + 1}<={min_steps}")

    early_fraction = float(_cfg_get(stop_cfg, "strict_for_early_fraction", 0.0) or 0.0)
    if early_fraction > 0 and num_step > 0 and (cnt_step + 1) / num_step <= early_fraction:
        reasons.append(f"early_fraction:{(cnt_step + 1) / num_step:.2f}<={early_fraction:.2f}")

    hard_targets = [_normalize_text(x) for x in _cfg_list(stop_cfg, "hard_targets", [])]
    target_text = f"{target_class} {question}"
    matched_hard_targets = [x for x in hard_targets if x and x in target_text]
    if matched_hard_targets:
        reasons.append("hard_target:" + ",".join(sorted(set(matched_hard_targets))))

    return reasons




def main(cfg, start_ratio=0.0, end_ratio=1.0, split=1, specific = None):
    # load the default concept graph config
    cfg_cg = OmegaConf.load(cfg.concept_graph_config_path)
    OmegaConf.resolve(cfg_cg)

    img_height = cfg.img_height
    img_width = cfg.img_width
    cam_intr = get_cam_intr(cfg.hfov, img_height, img_width)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load dataset
    scene_data_list = ['4ok3usBNeis.json', '5cdEh9F2hJL.json', '6s7QHgap2fW.json', '7MXmsvcQjpJ.json', 'BAbdmeyTvMZ.json', 'CrMo8WxCyVb.json', 'DYehNKdT76V.json', 'Dd4bFSTQ8gi.json', 'GLAQ4DNUx5U.json', 'HY1NcmCgn3n.json', 'LT9Jq6dN3Ea.json', 'MHPLjHsuG27.json', 'Nfvxx8J5NCo.json', 'QaLdnwvtxbs.json', 'TEEsavR23oF.json', 'VBzV5z6i1WS.json', 'XB4GS9ShBRE.json', 'a8BtkwhxdRV.json', 'bCPU9suPUw9.json', 'bxsVRursffK.json', 'cvZr5TUy5C5.json', 'eF36g7L6Z9M.json', 'h1zeeAwLh9Z.json', 'k1cupFYWXJ6.json', 'mL8ThkuaVTM.json', 'mv2HUxq3B53.json', 'p53SfW6mjZe.json', 'q3zU7Yy5E5s.json', 'q5QZSEeHe5g.json', 'qyAac8rV8Zk.json', 'svBbv1Pavdk.json', 'wcojb4TFT35.json', 'y9hTuugGdiq.json', 'yr17PDCnDDW.json', 'ziup5kvtCCR.json', 'zt1RVoi7PcG.json']
    num_scene = len(scene_data_list)
    random.shuffle(scene_data_list)
    # split the test data by scene
    if specific is not None:
        scene_data_list = scene_data_list[specific:specific + 1]
        logging.info(f"Specific scene {specific} selected for evaluation.")
    else:
        scene_data_list = scene_data_list[
            int(start_ratio * num_scene) : int(end_ratio * num_scene)
        ]
    num_episode = 0
    for scene_data_file in scene_data_list:
        with open(os.path.join(cfg.test_data_dir, scene_data_file), "r") as f:
            num_episode += len(json.load(f)["episodes"])
    logging.info(
        f"Total number of scenes: {len(scene_data_list)}; Total number of episodes: {num_episode}"
    )
    val_scene_path = os.path.join(cfg.scene_data_path, "val")
    all_scene_ids = os.listdir(val_scene_path)

    # load detection and segmentation models
    detection_model = YOLOWorld(cfg.yolo_model_name)
    logging.info(f"Load YOLO model {cfg.yolo_model_name} successful!")

    sam_predictor = SAM(cfg.sam_model_name)  # UltraLytics SAM
    logging.info(f"Load SAM model {cfg.sam_model_name} successful!")

    # clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    #     "ViT-B-32", "laion2b_s34b_b79k"  # "ViT-H-14", "laion2b_s32b_b79k"
    # )
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14-quickgelu", "dfn5b"  # "ViT-H-14", "laion2b_s32b_b79k"
    )
    clip_tokenizer = open_clip.get_tokenizer("ViT-H-14-quickgelu")
    logging.info(f"Load CLIP model successful!")

    # Initialize the logger
    logger = Logger(
        cfg.output_dir, start_ratio, end_ratio, split, voxel_size=cfg.tsdf_grid_size, specific=specific
    )
    timer = Timer()
    for scene_data_file in scene_data_list:
        # load goatbench data
        scene_name = scene_data_file.split(".")[0]
        scene_id = [scene_id for scene_id in all_scene_ids if scene_name in scene_id][0]
        scene_data = json.load(
            open(os.path.join(cfg.test_data_dir, scene_data_file), "r")
        )
        # selecat the episodes according to the split
        scene_data["episodes"] = scene_data["episodes"][split - 1 : split]
        total_episodes = len(scene_data["episodes"])

        all_navigation_goals = scene_data[
            "goals"
        ]  # obj_id to obj_data, apply for all episodes in this scene

        for episode_idx, episode in enumerate(scene_data["episodes"]):
            logging.info(f"Episode {episode_idx + 1}/{total_episodes}")
            logging.info(f"Loading scene {scene_id}")
            episode_id = episode["episode_id"]

            all_subtask_goal_types, all_subtask_goals = (
                prepare_goatbench_navigation_goals(
                    scene_name=scene_name,
                    episode=episode,
                    all_navigation_goals=all_navigation_goals,
                )
            )

            # check whether this episode has been processed
            finished_subtask_ids = list(logger.success_by_distance.keys())
            finished_episode_subtask = [
                subtask_id
                for subtask_id in finished_subtask_ids
                if subtask_id.startswith(f"{scene_id}_{episode_id}_")
            ]
            if len(finished_episode_subtask) >= len(all_subtask_goals):
                logging.info(f"Scene {scene_id} Episode {episode_id} already done!")
                continue

            pts, angle = get_pts_angle_goatbench(
                episode["start_position"], episode["start_rotation"]
            )

            # load scene
            try:
                del scene
            except:
                pass
            scene = Scene(
                scene_id,
                cfg,
                cfg_cg,
                detection_model,
                sam_predictor,
                clip_model,
                clip_preprocess,
                clip_tokenizer,
            )

            # initialize the TSDF
            floor_height = pts[1]
            tsdf_bnds, scene_size = get_scene_bnds(scene.pathfinder, floor_height)
            num_step = int(math.sqrt(scene_size) * cfg.max_step_room_size_ratio)
            num_step = max(num_step, 50)
            tsdf_planner = TSDFPlanner(
                vol_bnds=tsdf_bnds,
                voxel_size=cfg.tsdf_grid_size,
                floor_height=floor_height,
                floor_height_offset=0,
                pts_init=pts,
                init_clearance=cfg.init_clearance * 2,
                save_visualization=cfg.save_visualization,
            )

            episode_dir, eps_frontier_dir = logger.init_episode(
                episode_id=f"{scene_id}_ep_{episode_id}"
            )

            logging.info(f"\n\nScene {scene_id} initialization successful!")

            episode_memory = EpisodeMemory(cfg)

            # run questions in the scene
            global_step = -1
            for subtask_idx, (goal_type, subtask_goal) in enumerate(
                zip(all_subtask_goal_types, all_subtask_goals)
            ):
                subtask_id = f"{scene_id}_{episode_id}_{subtask_idx}"
                logging.info(
                    f"\nScene {scene_id} Episode {episode_id} Subtask {subtask_idx + 1}/{len(all_subtask_goals)}"
                )

                subtask_metadata = logger.init_subtask(
                    subtask_id=subtask_id,
                    goal_type=goal_type,
                    subtask_goal=subtask_goal,
                    pts=pts,
                    scene=scene,
                    tsdf_planner=tsdf_planner,
                )
                # This subtask's own description/reference image, captured once here
                # (before this subtask's directory can be deleted) and reused by every
                # episode_memory.query() call in the step loop below plus the final
                # episode_memory.record() call -- they're what lets EpisodeMemory tell
                # apart two different physical instances of the same class (e.g. two
                # mirrors) that a target_obj_id/spatial match alone would conflate.
                subtask_image_b64 = None
                if subtask_metadata.get("task_type") == "image" and subtask_metadata.get("image"):
                    subtask_image_b64 = _load_image_b64(subtask_metadata["image"])
                logger.init_trajectory_logging(
                    subtask_id,
                    metadata={
                        "scene_id": scene_id,
                        "episode_id": episode_id,
                        "subtask_idx": subtask_idx,
                        "question": subtask_metadata.get("question"),
                        "task_type": subtask_metadata.get("task_type"),
                        "class": subtask_metadata.get("class"),
                        "goal_obj_ids": subtask_metadata.get("goal_obj_ids"),
                        "gt_subtask_explore_dist": subtask_metadata.get("gt_subtask_explore_dist"),
                        "num_step": num_step,
                    },
                )

                # mapping from the obj id in habitat to the id assigned by concept graph
                # this mapping/alignment is done by heuristic matching between object masks
                goal_obj_ids_mapping = {
                    obj_id: [] for obj_id in subtask_metadata["goal_obj_ids"]
                }

                # run steps
                task_success = False
                task_check_obs = []
                cnt_step = -1
                n_filtered_frames = 0
                last_target_index = None
                last_target_type = None
                last_target_center = None
                last_decision_type = None  # type of the MOST RECENT vlm decision (incl. frontier)
                # EpisodeMemory counters (per subtask): candidates are log-only by
                # default; live-center fallback is an explicit opt-in intervention.
                n_memory_positive_candidates = 0
                n_memory_fallback_applied = 0
                n_memory_positive_hits = 0  # backward-compatible alias: candidates
                n_live_center_overrides = 0  # backward-compatible alias: applied fallback
                n_vlm_identity_calls_start = len(episode_memory._identity_cache)
                his_decision = {}
                subtask_metadata['CLR'] = his_decision
                # reset tsdf planner
                tsdf_planner.max_point = None
                tsdf_planner.target_point = None
                max_point_choice = None

                if cfg.clear_up_memory_every_subtask and subtask_idx > 0:
                    scene.clear_up_detections()
                    tsdf_planner = TSDFPlanner(
                        vol_bnds=tsdf_bnds,
                        voxel_size=cfg.tsdf_grid_size,
                        floor_height=floor_height,
                        floor_height_offset=0,
                        pts_init=pts,
                        init_clearance=cfg.init_clearance * 2,
                        save_visualization=cfg.save_visualization,
                    )

                while cnt_step < num_step - 1:
                    decision = {}

                    cnt_step += 1
                    global_step += 1
                    pts_before = pts.copy()
                    angle_before = angle
                    step_log = {
                        "step": cnt_step,
                        "global_step": global_step,
                        "pts_before": pts_before,
                        "angle_before": angle_before,
                        "events": [],
                        "vlm_called": False,
                        "planner_had_target_before_query": (
                            tsdf_planner.max_point is not None
                            or tsdf_planner.target_point is not None
                        ),
                        "planner_before": _planner_point_summary(tsdf_planner),
                        "end_check_called": False,
                        "end_check_response": None,
                        "strict_stop_check_required": False,
                        "strict_stop_check_reasons": [],
                        "strict_stop_check_called": False,
                        "strict_stop_check_response": None,
                        "strict_stop_check_reason": "",
                        "stop_validation_final": None,
                    }
                    his_decision['cnt_step'] = cnt_step
                    his_decision['max_step'] = num_step
                    logging.info(
                        f"\n== step: {cnt_step}, global step: {global_step} =="
                    )
                    timer.start("Observe the surroundings")
                    # (1) Observe the surroundings, update the scene graph and occupancy map
                    # Determine the viewing angles for the current step
                    if cnt_step == 0:
                        angle_increment = cfg.extra_view_angle_deg_phase_2 * np.pi / 180
                        total_views = 1 + cfg.extra_view_phase_2
                    else:
                        angle_increment = cfg.extra_view_angle_deg_phase_1 * np.pi / 180
                        total_views = 1 + cfg.extra_view_phase_1
                    all_angles = [
                        angle + angle_increment * (i - total_views // 2)
                        for i in range(total_views)
                    ]
                    # Let the main viewing angle be the last one to avoid potential overwriting problems
                    main_angle = all_angles.pop(total_views // 2)
                    all_angles.append(main_angle)

                    rgb_egocentric_views = []
                    all_added_obj_ids = (
                        []
                    )  # Record all the objects that are newly added in this step
                    for view_idx, ang in enumerate(all_angles):
                        # For each view
                        obs, cam_pose = scene.get_observation(pts, angle=ang)
                        rgb = obs["color_sensor"]
                        depth = obs["depth_sensor"]
                        semantic_obs = obs["semantic_sensor"]

                        # collect all view features
                        obs_file_name = f"{global_step}-view_{view_idx}.png"
                        with torch.no_grad():
                            # Concept graph pipeline update
                            annotated_rgb, added_obj_ids, target_obj_id_mapping = (
                                scene.update_scene_graph(
                                    image_rgb=rgb[..., :3],
                                    depth=depth,
                                    intrinsics=cam_intr,
                                    cam_pos=cam_pose,
                                    pts=pts,
                                    pts_voxel=tsdf_planner.habitat2voxel(pts),
                                    img_path=obs_file_name,
                                    frame_idx=cnt_step * total_views + view_idx,
                                    semantic_obs=semantic_obs,
                                    gt_target_obj_ids=subtask_metadata["goal_obj_ids"],
                                )
                            )
                            scene.all_observations[obs_file_name] = rgb
                            scene.all_depths[obs_file_name] = depth
                            scene.all_cam_poses[obs_file_name] = cam_pose
                            scene.all_obs_point[obs_file_name] = tsdf_planner.habitat2voxel(pts)
                            scene.global_step_cnt = global_step
                            rgb_egocentric_views.append(
                                resize_image(rgb, cfg.prompt_h, cfg.prompt_w)
                            )
                            for (
                                gt_goal_id,
                                det_goal_id,
                            ) in target_obj_id_mapping.items():
                                goal_obj_ids_mapping[gt_goal_id].append(det_goal_id)
                            all_added_obj_ids += added_obj_ids

                        # Clean up or merge redundant objects periodically
                        scene.periodic_cleanup_objects(
                            frame_idx=cnt_step * total_views + view_idx,
                            pts=pts,
                            goal_obj_ids_mapping=goal_obj_ids_mapping,
                        )

                        # Update depth map, occupancy map
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
                    logging.info(f"Goal object mapping: {goal_obj_ids_mapping}")
                    timer.stop("Observe the surroundings")
                    
                    timer.start("Update Memory")
                    # (2) Update memory with hierarchical clustering
                    # Choose all the newly added objects as well as the objects nearby as the cluster targets
                    all_added_obj_ids = [
                        obj_id
                        for obj_id in all_added_obj_ids
                        if obj_id in scene.objects
                    ]
                    for obj_id, obj in scene.objects.items():
                        if (
                            np.linalg.norm(obj["bbox"].center[[0, 2]] - pts[[0, 2]])
                            < cfg.scene_graph.obj_include_dist + 0.5
                        ):
                            all_added_obj_ids.append(obj_id)
                        
                    scene.del_unused_scene_graph_edges()
                    timer.stop("Update Memory")

                    timer.start("Update Frontier")
                    # (3) Update frontier map
                    update_success = tsdf_planner.update_frontier_map(
                        pts=pts,
                        cfg=cfg.planner,
                        scene=scene,
                        cnt_step=cnt_step,
                        save_frontier_image=cfg.save_visualization,
                        eps_frontier_dir=eps_frontier_dir,
                        prompt_img_size=(cfg.prompt_h, cfg.prompt_w),
                    )
                    if not update_success:
                        logging.info("Warning! Update frontier map failed!")
                        step_log["events"].append({"event": "frontier_update_failed"})
                    step_log.update(
                        {
                            "update_frontier_success": update_success,
                            "frontier_count": len(tsdf_planner.frontiers),
                            "frontiers": _summarize_frontiers(tsdf_planner),
                        }
                    )
                    timer.stop("Update Frontier")


                    timer.start("Querying the VLM")
                    # (4) Choose the next navigation point by querying the VLM
                    if cfg.choose_every_step:
                        # if we choose to query vlm every step, we clear the target point every step
                        if (
                            tsdf_planner.max_point is not None
                            and type(tsdf_planner.max_point) == Frontier
                        ):
                            # reset target point to allow the model to choose again
                            tsdf_planner.max_point = None
                            tsdf_planner.target_point = None

                    # use the most common id in the mapped ids as the detected target object id
                    target_obj_ids_estimate = []
                    for obj_id, det_ids in goal_obj_ids_mapping.items():
                        if len(det_ids) == 0:
                            continue
                        target_obj_ids_estimate.append(
                            max(set(det_ids), key=det_ids.count)
                        )
                        #####For each obj_id, pick the detection ID that appears most often in its list of detections,
                        #####and append that as the estimated target ID.
                    step_log.update(
                        {
                            "object_count": len(scene.objects),
                            "objects": _summarize_objects(scene),
                            "added_object_ids": all_added_obj_ids,
                            "goal_obj_ids_mapping": goal_obj_ids_mapping,
                            "target_obj_ids_estimate": target_obj_ids_estimate,
                        }
                    )

                    if (
                        tsdf_planner.max_point is None
                        and tsdf_planner.target_point is None
                    ):
                        # query the VLM for the next navigation point, and the reason for the choice
                        step_log["vlm_called"] = True

                        # Build cross-subtask memory hints (soft prior) for the
                        # VLM prompt BEFORE the query -- the VLM decision precedes
                        # episode_memory.query() below, so hints must be prepared here.
                        subtask_metadata['MEMORY_HINTS'] = episode_memory.annotate_nodes(
                            scene=scene,
                            task_type=subtask_metadata.get("task_type"),
                            target_class=subtask_metadata.get("class"),
                            description=subtask_metadata.get("question"),
                            image_b64=subtask_image_b64,
                            subtask_idx=subtask_idx,
                        )
                        step_log["memory_hints"] = {
                            str(k): v for k, v in subtask_metadata['MEMORY_HINTS'].items()
                        }

                        subtask_metadata['MEMORY_HINT_POLICY'] = {
                            "annotate_positive": episode_memory.pp_annotate_positive,
                            "surface_negatives": episode_memory.surface_negatives_in_prompt,
                            "memory_block": episode_memory.pp_memory_block,
                        }
                        step_log["memory_hint_policy"] = subtask_metadata['MEMORY_HINT_POLICY']

                        vlm_response = query_vlm_for_response(
                            subtask_metadata=subtask_metadata,
                            scene=scene,
                            tsdf_planner=tsdf_planner,
                            rgb_egocentric_views=rgb_egocentric_views,
                            cfg=cfg,
                            pts = pts,
                            verbose=True,
                        )
                        if vlm_response is None:
                            n_filtered_frames = 0
                            logging.info(
                                f"Subtask id {subtask_id} invalid: query_vlm_for_response failed!"
                            )
                            step_log["events"].append({"event": "vlm_response_none"})
                            _finish_step_trajectory(
                                logger,
                                step_log,
                                pts,
                                angle,
                                scene,
                                tsdf_planner,
                                subtask_metadata,
                                target_arrived=False,
                            )
                            break

                        target_type, max_point_choice, n_filtered_frames, target_index = vlm_response
                        decision['target_type'] = target_type
                        decision['max_point_choice'] = target_index
                        last_decision_type = target_type
                        if target_type in ("object", "image"):
                            last_target_index = target_index
                            last_target_type = target_type
                        # query episode memory after the VLM choice. By default this
                        # is log-only; live-center fallback is applied only when an
                        # explicit conservative config gate allows it.
                        memory_result = None
                        memory_match = None
                        memory_fallback_applied = False
                        if target_type in ("object", "image") and target_index is not None:
                            # geometry is resolved live inside query(); we still keep
                            # last_target_center for the record() attribution below.
                            last_target_center = None
                            try:
                                if isinstance(target_index, int) and target_index in scene.objects:
                                    last_target_center = np.array(scene.objects[target_index]["bbox"].center)
                                elif max_point_choice is not None:
                                    last_target_center = np.array(max_point_choice, dtype=float)
                            except Exception as e:
                                logging.warning(
                                    f"Failed to compute target_center for target {target_index} "
                                    f"(type={target_type}): {e}"
                                )
                            memory_result = episode_memory.query(
                                scene=scene,
                                target_obj_id=target_index,
                                target_class=subtask_metadata.get("class"),
                                task_type=subtask_metadata.get("task_type"),
                                description=subtask_metadata.get("question"),
                                image_b64=subtask_image_b64,
                                subtask_idx=subtask_idx,
                            )
                            positive = memory_result.get("positive") if memory_result else None
                            memory_fallback_applied = False
                            if positive is not None:
                                memory_match = positive["match"]
                                n_memory_positive_candidates += 1
                                n_memory_positive_hits += 1
                                if episode_memory._should_apply_live_center_fallback(
                                    positive, subtask_metadata.get("task_type")
                                ):
                                    max_point_choice = np.array(positive["live_center"], dtype=float).copy()
                                    memory_fallback_applied = True
                                    n_memory_fallback_applied += 1
                                    n_live_center_overrides += 1
                                    logging.info(
                                        f"EpisodeMemory fallback applied [{memory_match}]: target {target_index} "
                                        f"reached in subtask {positive['from_subtask']} "
                                        f"(dist={positive['final_dist']:.3f}m, "
                                        f"n_prior={positive.get('n_prior_positive_subtasks', 0)}), using live target center"
                                    )
                                else:
                                    logging.info(
                                        f"EpisodeMemory candidate [{memory_match}]: target {target_index} "
                                        f"reached in subtask {positive['from_subtask']} "
                                        f"(dist={positive['final_dist']:.3f}m, "
                                        f"n_prior={positive.get('n_prior_positive_subtasks', 0)}), log-only"
                                    )
                        _mem_neg = memory_result.get("negatives") if memory_result else []
                        # Instrumentation (data_analysis §1.2): tell "abandoned a reached
                        # GT" apart from "passed by GT while committed to a wrong instance".
                        # target_obj_ids_estimate = scene-graph detections mapped to any GT
                        # obj id this step; committed target is target_index (int obj id for
                        # object/description; a "<n>-view_*.png" string for image).
                        gt_perceived_now = len(target_obj_ids_estimate) > 0
                        if not gt_perceived_now:
                            committed_target_is_gt = None  # GT not in scene graph yet
                        elif isinstance(target_index, int):
                            committed_target_is_gt = target_index in target_obj_ids_estimate
                        else:
                            # image target: no obj id to compare -> proximity of the chosen
                            # nav point to any GT-mapped detection center (success_distance bar)
                            committed_target_is_gt = None
                            try:
                                if max_point_choice is not None:
                                    mpc = np.array(max_point_choice, dtype=float)
                                    committed_target_is_gt = any(
                                        det_id in scene.objects
                                        and np.linalg.norm(
                                            np.array(scene.objects[det_id]["bbox"].center, dtype=float) - mpc
                                        ) <= cfg.success_distance
                                        for det_id in target_obj_ids_estimate
                                    )
                            except Exception:
                                pass
                        step_log.update(
                            {
                                "target_type": target_type,
                                "target_index": target_index,
                                "n_filtered_frames": n_filtered_frames,
                                "max_point_choice": max_point_choice,
                                "gt_perceived_now": gt_perceived_now,
                                "committed_target_is_gt": committed_target_is_gt,
                                "decision": decision.copy(),
                                "memory_hit": ({"match": memory_match,
                                                "from_subtask": memory_result["positive"]["from_subtask"],
                                                "from_dist": memory_result["positive"]["final_dist"],
                                                "n_prior_positive_subtasks": memory_result["positive"].get("n_prior_positive_subtasks"),
                                                "fallback_applied": memory_fallback_applied}
                                               if (memory_result and memory_result.get("positive")) else None),
                                "memory_negatives": [
                                    {"node_id": n["node_id"], "neg_reason": n["neg_reason"],
                                     "from_subtask": n["from_subtask"], "match": n["match"]}
                                    for n in (_mem_neg or [])
                                ],
                            }
                        )
                        # set the vlm choice as the navigation target
                        update_success = tsdf_planner.set_next_navigation_point(
                            target_type = target_type, 
                            choice=max_point_choice,
                            pts=pts,
                            objects=scene.objects,
                            obs_points = scene.all_obs_point,
                            cfg=cfg.planner,
                            pathfinder=scene.pathfinder,
                        )
                        step_log.update(
                            {
                                "set_next_navigation_success": update_success,
                                "planner_after_set_target": _planner_point_summary(tsdf_planner),
                            }
                        )
                        if not update_success:
                            logging.info(
                                f"Subtask id {subtask_id} invalid: set_next_navigation_point failed!"
                            )
                            step_log["events"].append(
                                {"event": "set_next_navigation_failed"}
                            )
                            _finish_step_trajectory(
                                logger,
                                step_log,
                                pts,
                                angle,
                                scene,
                                tsdf_planner,
                                subtask_metadata,
                                target_arrived=False,
                            )
                            break
                    else:
                        step_log.update(
                            {
                                "target_type": getattr(tsdf_planner, "target_type", None),
                                "target_index": None,
                                "n_filtered_frames": n_filtered_frames,
                                "decision": decision.copy(),
                                "set_next_navigation_success": None,
                            }
                        )
                    timer.stop("Querying the VLM")

                    timer.start("Planner navigate to the target point")
                    # (5) Agent navigate to the target point for one step

                    mpc_for_agent_step = None
                    if target_type == "image" or target_type == "object":############# draw max_point_choice only when target is image or object
                        mpc_for_agent_step = max_point_choice

                    return_values = tsdf_planner.agent_step(
                        pts=pts,
                        angle=angle,
                        pathfinder=scene.pathfinder,
                        cfg=cfg.planner,
                        path_points=None,
                        save_visualization=cfg.save_visualization,
                        max_point_choice=mpc_for_agent_step,###################### VVD demonstration
                    )
                    if return_values[0] is None:
                        logging.info(
                            f"Subtask id {subtask_id} invalid: agent_step failed!"
                        )
                        step_log["agent_step_success"] = False
                        step_log["agent_step_failure_reason"] = "agent_step_returned_none"
                        step_log["events"].append({"event": "agent_step_failed"})
                        _finish_step_trajectory(
                            logger,
                            step_log,
                            pts,
                            angle,
                            scene,
                            tsdf_planner,
                            subtask_metadata,
                            target_arrived=False,
                        )
                        break

                    # update agent's position and rotation
                    pts, angle, pts_voxel, fig, _, target_arrived = return_values
                    step_log.update(
                        {
                            "agent_step_success": True,
                            "pts_voxel_after": pts_voxel,
                        }
                    )

                    logger.log_step(pts_voxel=pts_voxel)
                    logging.info(
                        f"Current position: {pts}, {logger.subtask_explore_dist:.3f}"
                    )

                    # sanity check about objects, scene graph, ...
                    scene.sanity_check(cfg=cfg)

                    if cfg.save_visualization:
                        # save the top-down visualization
                        logger.save_topdown_visualization(
                            global_step=global_step,
                            subtask_id=subtask_id,
                            subtask_metadata=subtask_metadata,
                            goal_obj_ids_mapping=goal_obj_ids_mapping,
                            fig=fig,
                        )
                        # save the visualization of vlm's choice at each step
                        logger.save_frontier_visualization(
                            global_step=global_step,
                            subtask_id=subtask_id,
                            tsdf_planner=tsdf_planner,
                            max_point_choice=max_point_choice,
                            global_caption=f"{subtask_metadata['question']}\n{subtask_metadata['task_type']}\n{subtask_metadata['class']}",
                        )
                    timer.stop("Planner navigate to the target point")

                
                    timer.print_summary(logging)
                    # (6) Check if the agent has arrived at the target to finish the question
                    # Final verification: use VLM to confirm the reached target

                    stop_target_type = target_type
                    if target_type != "frontier" and target_arrived:
                        back_frames = min(cnt_step + 1, cfg.frames_to_check)
                        task_check_obs_frames = {}
                        for back_step in range(global_step, global_step - back_frames, -1):
                            task_check_obs = []
                            for view_idx in range(7):
                                obs_file_name = f"{back_step}-view_{view_idx}.png"
                                if obs_file_name not in scene.all_observations.keys():
                                    break
                                task_check_obs.append(
                                    resize_image(scene.all_observations[obs_file_name], cfg.prompt_h, cfg.prompt_w)
                                )
                            task_check_obs_frames[global_step - back_step + 1] = task_check_obs.copy()
                        step_log["stop_check_frames"] = len(task_check_obs_frames)
                        step_log["stop_candidate"] = {
                            "target_type": target_type,
                            "target_index": target_index,
                            "planner": _planner_point_summary(tsdf_planner),
                        }
                        vlm_response = query_vlm_for_response_end(
                            subtask_metadata=subtask_metadata,
                            rgb_egocentric_views=task_check_obs_frames,
                            cfg=cfg,
                            verbose=True,
                        )
                        if (vlm_response == 'yes'):
                            step_log["end_check_called"] = True
                            step_log["end_check_response"] = vlm_response
                            step_log["events"].append({"event": "end_check_yes"})
                            strict_reasons = _strict_stop_reasons(
                                cfg,
                                subtask_metadata,
                                stop_target_type,
                                cnt_step,
                                num_step,
                            )
                            step_log["strict_stop_check_required"] = bool(strict_reasons)
                            step_log["strict_stop_check_reasons"] = strict_reasons
                            step_log["strict_stop_check_called"] = False
                            step_log["strict_stop_check_response"] = None
                            step_log["strict_stop_check_reason"] = ""
                            step_log["stop_validation_final"] = "accepted_normal"
                            if strict_reasons:
                                strict_response, strict_reason = query_vlm_for_response_end_strict(
                                    subtask_metadata=subtask_metadata,
                                    rgb_egocentric_views=task_check_obs_frames,
                                    cfg=cfg,
                                    verbose=True,
                                )
                                step_log["strict_stop_check_called"] = True
                                step_log["strict_stop_check_response"] = strict_response
                                step_log["strict_stop_check_reason"] = strict_reason
                                if strict_response == "yes":
                                    step_log["events"].append(
                                        {
                                            "event": "strict_end_check_yes",
                                            "reasons": strict_reasons,
                                            "reason": strict_reason,
                                        }
                                    )
                                    step_log["stop_validation_final"] = "accepted_strict"
                                else:
                                    decision['object_judge'] = "no"
                                    decision["stop_validation"] = "rejected_strict"
                                    decision["stop_validation_reasons"] = strict_reasons
                                    decision["strict_stop_check_response"] = strict_response
                                    step_log["decision"] = decision.copy()
                                    step_log["events"].append(
                                        {
                                            "event": "strict_end_check_no",
                                            "reasons": strict_reasons,
                                            "response": strict_response,
                                            "reason": strict_reason,
                                        }
                                    )
                                    step_log["stop_validation_final"] = "rejected_strict"
                                    his_decision[cnt_step] = decision
                                    step_log["his_decision_step_keys"] = list(his_decision.keys())
                                    # cross-subtask negative memory: this node was
                                    # strict-rejected as the current target. Recorded
                                    # without a distance bar (negatives don't need one).
                                    if episode_memory.record_strict_rejections and last_target_index is not None:
                                        episode_memory.record(
                                            scene=scene,
                                            node_id=last_target_index,
                                            subtask_idx=subtask_idx,
                                            task_type=subtask_metadata.get("task_type", ""),
                                            target_class=subtask_metadata.get("class", ""),
                                            success=False,
                                            final_dist=None,
                                            view_position=pts,
                                            view_angle=angle,
                                            outcome="negative",
                                            neg_reason="rejected_strict",
                                            description=subtask_metadata.get("question"),
                                            image_b64=subtask_image_b64,
                                        )
                                    _finish_step_trajectory(
                                        logger,
                                        step_log,
                                        pts,
                                        angle,
                                        scene,
                                        tsdf_planner,
                                        subtask_metadata,
                                        target_arrived=target_arrived,
                                    )
                                    continue
                            his_decision[cnt_step] = decision
                            _finish_step_trajectory(
                                logger,
                                step_log,
                                pts,
                                angle,
                                scene,
                                tsdf_planner,
                                subtask_metadata,
                                target_arrived=target_arrived,
                            )
                            break
                        else:
                            decision['object_judge'] = "no"
                            decision["stop_validation"] = "rejected_normal"
                            step_log["end_check_called"] = True
                            step_log["end_check_response"] = vlm_response
                            step_log["strict_stop_check_required"] = False
                            step_log["strict_stop_check_reasons"] = []
                            step_log["strict_stop_check_called"] = False
                            step_log["stop_validation_final"] = "rejected_normal"
                            step_log["decision"] = decision.copy()
                            step_log["events"].append({"event": "end_check_no"})
                    else:
                        step_log["end_check_called"] = False

                    his_decision[cnt_step] = decision
                    step_log["decision"] = decision.copy()
                    step_log["his_decision_step_keys"] = list(his_decision.keys())
                    _finish_step_trajectory(
                        logger,
                        step_log,
                        pts,
                        angle,
                        scene,
                        tsdf_planner,
                        subtask_metadata,
                        target_arrived=target_arrived,
                    )

                # calculate the distance to the nearest GT target object center
                agent_subtask_distance = calc_agent_subtask_distance(
                    pts, subtask_metadata["goal_positions"], scene.pathfinder
                )
                if agent_subtask_distance < cfg.success_distance:
                    success_by_distance = True
                    logging.info(
                        f"Success: agent reached the target object center at distance {agent_subtask_distance}!"
                    )
                else:
                    success_by_distance = False
                    logging.info(
                        f"Fail: agent failed to reach the target object center at distance {agent_subtask_distance}!"
                    )

                # record memory anchor for this subtask.
                # If the last vlm decision before loop exit was a frontier pick
                # (i.e. the object/image target was superseded/abandoned and the
                # agent went back to exploring), last_target_index/last_target_center
                # refer to a stale target unrelated to the final pts -- do not
                # attribute this subtask's outcome to that target.
                if last_decision_type in ("object", "image"):
                    record_target_id = last_target_index
                else:
                    record_target_id = None
                episode_memory.record(
                    scene=scene,
                    node_id=record_target_id,
                    subtask_idx=subtask_idx,
                    task_type=subtask_metadata.get("task_type", ""),
                    target_class=subtask_metadata.get("class", ""),
                    success=success_by_distance,
                    final_dist=agent_subtask_distance,
                    view_position=pts,
                    view_angle=angle,
                    neg_reason=(None if success_by_distance else "distance_fail"),
                    description=subtask_metadata.get("question"),
                    image_b64=subtask_image_b64,
                )

                logger.save_trajectory_jsonl(
                    subtask_id,
                    final_summary={
                        "success_by_distance": success_by_distance,
                        "final_distance_to_goal": agent_subtask_distance,
                        "success_distance": cfg.success_distance,
                        "n_filtered_frames": n_filtered_frames,
                        "n_total_frames": len(scene.img_to_edge),
                        "n_steps": cnt_step + 1,
                        "explore_dist": logger.subtask_explore_dist,
                        # EpisodeMemory instrumentation
                        "n_memory_positive_candidates": n_memory_positive_candidates,
                        "n_memory_fallback_applied": n_memory_fallback_applied,
                        "n_memory_positive_hits": n_memory_positive_hits,
                        "n_live_center_overrides": n_live_center_overrides,
                        "n_vlm_identity_calls": len(episode_memory._identity_cache) - n_vlm_identity_calls_start,
                    },
                )

                logger.log_subtask_result(
                    success_by_distance=success_by_distance,
                    subtask_id=subtask_id,
                    gt_subtask_explore_dist=subtask_metadata["gt_subtask_explore_dist"],
                    goal_type=goal_type,
                    n_filtered_frames=n_filtered_frames,
                    n_total_frames=len(scene.img_to_edge),
                    question=subtask_metadata.get("question", ""),
                    gt_answer=subtask_metadata.get("class", ""),
                    final_distance_to_goal=agent_subtask_distance,
                    success_distance_thresh=cfg.success_distance,
                    n_steps=cnt_step + 1,
                    goal_obj_ids=subtask_metadata.get("goal_obj_ids", None),
                )

                logging.info(f"Scene graph of question {subtask_id}:")
                logging.info(f"Question: {subtask_metadata['question']}")
                logging.info(f"Task type: {subtask_metadata['task_type']}")
                logging.info(f"Answer: {subtask_metadata['class']}")
                #scene.print_scene_graph()

                if not cfg.save_visualization:
                    # clear up the stored images to save memory
                    os.system(
                        f"rm -r {os.path.join(str(cfg.output_dir), f'{subtask_id}')}"
                    )

            # save the results at the end of each episode
            logger.save_results()

            logging.info(f"Episode {episode_id} finish")
            if not cfg.save_visualization:
                os.system(f"rm -r {episode_dir}")

    logger.save_results()
    # aggregate the results from different splits into a single file
    logger.aggregate_results()

    logging.info(f"All scenes finish")


if __name__ == "__main__":
    # Get config path
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file path", default="", type=str)
    parser.add_argument("--start_ratio", help="start ratio", default=0.0, type=float)
    parser.add_argument("--end_ratio", help="end ratio", default=1.0, type=float)
    parser.add_argument("--split", help="which episode", default=1, type=int)
    parser.add_argument("--specific", help="specific scene when multi-gpu evaluation", default=None, type=int)
    parser.add_argument("-ag", "--aggregate", action='store_true', default=False)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(cfg)
    # append the active model name to exp_name so results of different models are kept separate
    from src.const import API_MODE, GPT_MODEL, Qwen_MODEL
    _active_model = GPT_MODEL if API_MODE == "gpt" else Qwen_MODEL
    _model_tag = str(_active_model).replace("/", "_").replace(":", "-")
    cfg.output_dir = os.path.join(cfg.output_parent_dir, f"{cfg.exp_name}_{_model_tag}")
    
    # Set up logging
    
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir, exist_ok=True)  # recursive
    if args.specific is not None:
        logging_path = os.path.join(
            str(cfg.output_dir),
            f"log_{args.start_ratio:.2f}_{args.end_ratio:.2f}_{args.split}_{args.specific}.log",
        )
    else:
        logging_path = os.path.join(
            str(cfg.output_dir),
            f"log_{args.start_ratio:.2f}_{args.end_ratio:.2f}_{args.split}.log",
        )
    os.system(f"cp {args.cfg_file} {cfg.output_dir}")

    class ElapsedTimeFormatter(logging.Formatter):
        def __init__(self, fmt=None, datefmt=None):
            super().__init__(fmt, datefmt)
            self.start_time = time.time()

        def formatTime(self, record, datefmt=None):
            elapsed_seconds = record.created - self.start_time
            hours, remainder = divmod(elapsed_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

    # Set up the logging format
    formatter = ElapsedTimeFormatter(fmt="%(asctime)s - %(message)s")

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="w"),
            logging.StreamHandler(),
        ],
    )

    # Set the custom formatter
    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)

    # run
    logging.info(f"***** Running {cfg.exp_name} *****")
    if args.aggregate:
        args.specific = -1
        logger = Logger(
            cfg.output_dir, args.start_ratio, args.end_ratio, args.split, voxel_size=cfg.tsdf_grid_size, specific=args.specific
        )
        logger.aggregate_results()
        exit(0)
    main(cfg, start_ratio=args.start_ratio, end_ratio=args.end_ratio, split=args.split, specific=args.specific)
