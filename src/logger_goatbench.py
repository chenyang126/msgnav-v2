import os
import json
import pickle
from collections import defaultdict
import logging
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image
from typing import Union
import math
import habitat_sim

from src.multimodal_3d_scene_graph import Scene
from src.tsdf_planner import TSDFPlanner, Frontier


class Logger:
    def __init__(
        self,
        output_dir,
        start_ratio,
        end_ratio,
        split,
        voxel_size,  # used for calculating the moving distance
        specific=None,  # used for specific scene
    ):
        self.output_dir = output_dir
        self.voxel_size = voxel_size
        if specific is None:
            self.success_by_distance_path = success_by_distance_path = os.path.join(
                output_dir, f"success_by_distance_{start_ratio}_{end_ratio}_{split}.pkl"
            )
            self.spl_by_distance_path = spl_by_distance_path = os.path.join(
                output_dir, f"spl_by_distance_{start_ratio}_{end_ratio}_{split}.pkl"
            )
            self.success_by_task_path = success_by_task_path = os.path.join(
                output_dir, f"success_by_task_{start_ratio}_{end_ratio}_{split}.pkl"
            )
            self.spl_by_task_path = spl_by_task_path = os.path.join(
                output_dir, f"spl_by_task_{start_ratio}_{end_ratio}_{split}.pkl"
            )
            self.n_filtered_frames_path = n_filtered_frames_path = os.path.join(
                output_dir,
                f"n_filtered_frames_{start_ratio}_{end_ratio}_{split}.json",
            )
            self.n_total_frames_path = n_total_frames_path = os.path.join(
                output_dir, f"n_total_frames_{start_ratio}_{end_ratio}_{split}.json"
            )       
        else:
            self.success_by_distance_path = success_by_distance_path = os.path.join(
                output_dir, f"success_by_distance_{start_ratio}_{end_ratio}_{split}_{specific}.pkl"
            )   
            self.spl_by_distance_path = spl_by_distance_path = os.path.join(
                output_dir, f"spl_by_distance_{start_ratio}_{end_ratio}_{split}_{specific}.pkl"
            )
            self.success_by_task_path = success_by_task_path = os.path.join(
                output_dir, f"success_by_task_{start_ratio}_{end_ratio}_{split}_{specific}.pkl"
            )
            self.spl_by_task_path = spl_by_task_path = os.path.join(
                output_dir, f"spl_by_task_{start_ratio}_{end_ratio}_{split}_{specific}.pkl"
            )
            self.n_filtered_frames_path = n_filtered_frames_path = os.path.join(
                output_dir,
                f"n_filtered_frames_{start_ratio}_{end_ratio}_{split}_{specific}.json",
            )
            self.n_total_frames_path = n_total_frames_path = os.path.join(
                output_dir, f"n_total_frames_{start_ratio}_{end_ratio}_{split}_{specific}.json"
            )
        if os.path.exists(
            success_by_distance_path
        ):
            self.success_by_distance = pickle.load(
                open(
                    success_by_distance_path,
                    "rb",
                )
            )
        else:
            self.success_by_distance = {}  # subtask id -> success
        
        if os.path.exists(
            spl_by_distance_path
        ):
            self.spl_by_distance = pickle.load(
                open(
                    spl_by_distance_path,
                    "rb",
                )
            )
        else:
            self.spl_by_distance = {}  # subtask id -> spl
        if os.path.exists(
            success_by_task_path
        ):
            self.success_by_task = pickle.load(
                open(
                    success_by_task_path,
                    "rb",
                )
            )
        else:
            # success_by_task = {}  # task type -> success
            self.success_by_task = defaultdict(list)
        if os.path.exists(
            spl_by_task_path
        ):
            self.spl_by_task = pickle.load(
                open(
                    spl_by_task_path,
                    "rb",
                )
            )
        else:
            # spl_by_task = {}  # task type -> spl
            self.spl_by_task = defaultdict(list)
        
        if os.path.exists(
            n_filtered_frames_path
        ):
            with open(
                n_filtered_frames_path,
                "r",
            ) as f:
                self.n_filtered_frames_list = json.load(f)
        else:
            self.n_filtered_frames_list = {}
        
        if os.path.exists(
            n_total_frames_path
        ):
            with open(
                n_total_frames_path,
                "r",
            ) as f:
                self.n_total_frames_list = json.load(f)
        else:
            self.n_total_frames_list = {}

        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.split = split

        # ---- per-split live debug records (JSONL detail + CSV summary) ----
        # each parallel worker (identified by split/specific) writes its own files to
        # avoid write contention; summarize_results.py merges them into a global view.
        _tag = f"{start_ratio}_{end_ratio}_{split}"
        if specific is not None:
            _tag = f"{_tag}_{specific}"
        try:
            from src.const import API_MODE, GPT_MODEL, Qwen_MODEL
            self.model_name = GPT_MODEL if API_MODE == "gpt" else Qwen_MODEL
        except Exception:
            self.model_name = "unknown"
        self.records_jsonl_path = os.path.join(output_dir, f"records_{_tag}.jsonl")
        self.records_csv_path = os.path.join(output_dir, f"records_{_tag}.csv")
        self.trajectory_jsonl_path = os.path.join(output_dir, f"trajectory_{_tag}.jsonl")
        self.trajectory_records = []
        self.current_trajectory_meta = {}
        # accumulate record dicts of this worker so we can rewrite the csv each time.
        # Keep one latest row per subtask_id; JSONL remains one-line-per-subtask.
        self._records = []
        self._records_by_id = {}
        self._csv_fields = [
            "timestamp", "model", "subtask_id", "scene", "episode", "subtask_idx",
            "task_type", "question", "gt_answer", "success", "spl",
            "agent_explore_dist", "gt_explore_dist", "success_distance_thresh",
            "final_distance_to_goal", "n_steps", "n_filtered_frames", "n_total_frames",
            "goal_obj_ids", "note",
        ]
        if os.path.exists(self.records_jsonl_path):
            try:
                with open(self.records_jsonl_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        record = json.loads(line)
                        subtask_id = record.get("subtask_id")
                        if subtask_id:
                            self._records_by_id[subtask_id] = record
                self._records = list(self._records_by_id.values())
            except Exception:
                logging.warning("Failed to preload existing live records; continuing with an empty live record cache")

        # some sanity check
        assert (
            len(self.success_by_distance)
            == len(self.spl_by_distance)
        ), f"{len(self.success_by_distance)} != {len(self.spl_by_distance)}"
        assert (
            sum([len(task_res) for task_res in self.success_by_task.values()])
            == sum([len(task_res) for task_res in self.spl_by_task.values()])
        ), f"{sum([len(task_res) for task_res in self.success_by_task.values()])} != {sum([len(task_res) for task_res in self.spl_by_task.values()])}"

        # logging for episode
        self.episode_dir = None

        # logging for subtask
        self.subtask_object_observe_dir = None
        self.pts_voxels = np.empty((0, 2))
        self.subtask_explore_dist = 0.0

    def save_results(self):
        # some sanity check
        assert (
            len(self.success_by_distance)
            == len(self.spl_by_distance)
        ), f"{len(self.success_by_distance)} != {len(self.spl_by_distance)}"
        assert (
            sum([len(task_res) for task_res in self.success_by_task.values()])
            == sum([len(task_res) for task_res in self.spl_by_task.values()])
        ), f"{sum([len(task_res) for task_res in self.success_by_task.values()])} != {sum([len(task_res) for task_res in self.spl_by_task.values()])}"

        
        with open(
            self.success_by_distance_path,
            "wb",
        ) as f:
            pickle.dump(self.success_by_distance, f)
    
        with open(
            self.spl_by_distance_path,
            "wb",
        ) as f:
            pickle.dump(self.spl_by_distance, f)
        
        with open(
            self.success_by_task_path,
            "wb",
        ) as f:
            pickle.dump(self.success_by_task, f)
        with open(
            self.spl_by_task_path,
            "wb",
        ) as f:
            pickle.dump(self.spl_by_task, f)
        with open(
            self.n_filtered_frames_path,
            "w",
        ) as f:
            json.dump(self.n_filtered_frames_list, f, indent=4)
        
        with open(
            self.n_total_frames_path,
            "w",
        ) as f:
            json.dump(self.n_total_frames_list, f, indent=4)

    def aggregate_results(self):
        # aggregate the results into a single file
        filenames_to_merge = [
            "success_by_distance",
            "spl_by_distance",
        ]
        for filename in filenames_to_merge:
            all_results = {}
            all_results_paths = glob.glob(
                os.path.join(self.output_dir, f"{filename}_*.pkl")
            )
            for results_path in all_results_paths:
                with open(results_path, "rb") as f:
                    all_results.update(pickle.load(f))
            logging.info(
                f"Total {filename} results: {100 * np.mean(list(all_results.values())):.2f}, len: {len(all_results)}"
            )
            with open(os.path.join(self.output_dir, f"{filename}.pkl"), "wb") as f:
                pickle.dump(all_results, f)
        filenames_to_merge = ["success_by_task", "spl_by_task"]
        for filename in filenames_to_merge:
            all_results = {}
            all_results_paths = glob.glob(
                os.path.join(self.output_dir, f"{filename}_*.pkl")
            )
            for results_path in all_results_paths:
                with open(results_path, "rb") as f:
                    separate_stat = pickle.load(f)
                    for task_name, task_res in separate_stat.items():
                        if task_name not in all_results:
                            all_results[task_name] = []
                        all_results[task_name] += task_res
            for task_name, task_res in all_results.items():
                logging.info(
                    f"Total {filename} results for {task_name}: {100 * np.mean(task_res):.2f}, len: {len(task_res)}"
                )
            with open(os.path.join(self.output_dir, f"{filename}.pkl"), "wb") as f:
                pickle.dump(all_results, f)

        n_filtered_frames_list = {}
        all_n_filtered_frames_list_paths = glob.glob(
            os.path.join(self.output_dir, "n_filtered_frames_*.json")
        )
        for n_filtered_frames_list_path in all_n_filtered_frames_list_paths:
            with open(n_filtered_frames_list_path, "r") as f:
                n_filtered_frames_list.update(json.load(f))

        with open(os.path.join(self.output_dir, "n_filtered_frames.json"), "w") as f:
            json.dump(n_filtered_frames_list, f, indent=4)
        logging.info(
            f"Average number of filtered frames: {np.mean(list(n_filtered_frames_list.values()))}"
        )

        n_total_frames_list = {}
        all_n_total_frames_list_paths = glob.glob(
            os.path.join(self.output_dir, "n_total_frames_*.json")
        )
        for n_total_frames_list_path in all_n_total_frames_list_paths:
            with open(n_total_frames_list_path, "r") as f:
                n_total_frames_list.update(json.load(f))

        with open(os.path.join(self.output_dir, "n_total_frames.json"), "w") as f:
            json.dump(n_total_frames_list, f, indent=4)
        logging.info(
            f"Average number of total frames: {np.mean(list(n_total_frames_list.values()))}"
        )

    def log_subtask_result(
        self,
        success_by_distance: bool,
        subtask_id: str,
        gt_subtask_explore_dist: float,
        goal_type: str,
        n_filtered_frames,
        n_total_frames,
        question: str = "",
        gt_answer: str = "",
        final_distance_to_goal: float = None,
        success_distance_thresh: float = None,
        n_steps: int = None,
        goal_obj_ids=None,
        note: str = "",
    ):
        if success_by_distance:
            self.success_by_distance[subtask_id] = 1.0
        else:
            self.success_by_distance[subtask_id] = 0.0


        self.spl_by_distance[subtask_id] = (
            self.success_by_distance[subtask_id]
            * gt_subtask_explore_dist
            / max(gt_subtask_explore_dist, self.subtask_explore_dist)
        )

        if math.isnan(self.spl_by_distance[subtask_id]):
            self.spl_by_distance[subtask_id] = 0
        self.success_by_task[goal_type].append(self.success_by_distance[subtask_id])
        self.spl_by_task[goal_type].append(self.spl_by_distance[subtask_id])

        logging.info(
            f"Subtask {subtask_id} finished, {self.subtask_explore_dist} length"
        )
        logging.info(
            f"Success rate by distance: {100 * np.mean(np.asarray(list(self.success_by_distance.values()))):.2f}"
        )
        logging.info(
            f"SPL by distance: {100 * np.mean(np.asarray(list(self.spl_by_distance.values()))):.2f}"
        )

        for task_name, success_list in self.success_by_task.items():
            logging.info(
                f"Success rate for {task_name}: {100 * np.mean(np.asarray(success_list)):.2f}"
            )
        for task_name, spl_list in self.spl_by_task.items():
            logging.info(
                f"SPL for {task_name}: {100 * np.mean(np.asarray(spl_list)):.2f}"
            )

        logging.info(
            f"Filtered frames/Total frames: {n_filtered_frames}/{n_total_frames}"
        )
        # save the number of frames
        self.n_filtered_frames_list[subtask_id] = n_filtered_frames
        self.n_total_frames_list[subtask_id] = n_total_frames

        # ---- write live debug record (JSONL append + CSV rewrite) ----
        self._write_live_record(
            subtask_id=subtask_id,
            goal_type=goal_type,
            gt_subtask_explore_dist=gt_subtask_explore_dist,
            n_filtered_frames=n_filtered_frames,
            n_total_frames=n_total_frames,
            question=question,
            gt_answer=gt_answer,
            final_distance_to_goal=final_distance_to_goal,
            success_distance_thresh=success_distance_thresh,
            n_steps=n_steps,
            goal_obj_ids=goal_obj_ids,
            note=note,
        )

        # clear the subtask logging
        self.subtask_object_observe_dir = None
        self.pts_voxels = np.empty((0, 2))
        self.subtask_explore_dist = 0.0

    def _write_live_record(
        self,
        subtask_id,
        goal_type,
        gt_subtask_explore_dist,
        n_filtered_frames,
        n_total_frames,
        question="",
        gt_answer="",
        final_distance_to_goal=None,
        success_distance_thresh=None,
        n_steps=None,
        goal_obj_ids=None,
        note="",
    ):
        """Append one subtask record to JSONL and rewrite the summary CSV.

        Robust to any error so it never breaks the evaluation loop.
        """
        import csv
        import datetime
        import traceback

        try:
            parts = subtask_id.split("_")
            scene = parts[0] if len(parts) > 0 else ""
            episode = parts[1] if len(parts) > 1 else ""
            subtask_idx = parts[2] if len(parts) > 2 else ""

            def _num(x):
                try:
                    if x is None:
                        return None
                    v = float(x)
                    return None if math.isnan(v) else round(v, 6)
                except Exception:
                    return None

            record = {
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "model": self.model_name,
                "subtask_id": subtask_id,
                "scene": scene,
                "episode": episode,
                "subtask_idx": subtask_idx,
                "task_type": goal_type,
                "question": question,
                "gt_answer": gt_answer,
                "success": self.success_by_distance.get(subtask_id),
                "spl": _num(self.spl_by_distance.get(subtask_id)),
                "agent_explore_dist": _num(self.subtask_explore_dist),
                "gt_explore_dist": _num(gt_subtask_explore_dist),
                "success_distance_thresh": _num(success_distance_thresh),
                "final_distance_to_goal": _num(final_distance_to_goal),
                "n_steps": n_steps,
                "n_filtered_frames": n_filtered_frames,
                "n_total_frames": n_total_frames,
                "goal_obj_ids": goal_obj_ids,
                "note": note,
            }

            # JSONL should be one line per subtask_id. If this process is resuming
            # from an existing records_*.jsonl, _records_by_id was preloaded above.
            is_new_record = subtask_id not in self._records_by_id
            self._records_by_id[subtask_id] = record
            if is_new_record:
                with open(self.records_jsonl_path, "a") as f:
                    f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

            # rewrite the flat CSV with the latest row for each subtask
            self._records = list(self._records_by_id.values())
            with open(self.records_csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self._csv_fields, extrasaction="ignore")
                w.writeheader()
                for r in sorted(self._records, key=lambda x: x.get("subtask_id", "")):
                    row = dict(r)
                    if isinstance(row.get("goal_obj_ids"), (list, tuple)):
                        row["goal_obj_ids"] = ";".join(str(x) for x in row["goal_obj_ids"])
                    w.writerow(row)
        except Exception:
            logging.warning(
                "Failed to write live record for %s:\n%s",
                subtask_id,
                traceback.format_exc(),
            )
    def _json_safe(self, value):
        """Convert common numpy/map objects into JSON-serializable values."""
        if value is None or isinstance(value, (str, int, float, bool)):
            if isinstance(value, float) and math.isnan(value):
                return None
            return value
        if isinstance(value, np.generic):
            return self._json_safe(value.item())
        if isinstance(value, np.ndarray):
            return self._json_safe(value.tolist())
        if isinstance(value, (list, tuple)):
            return [self._json_safe(v) for v in value]
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, Frontier):
            return {
                "position": self._json_safe(getattr(value, "position", None)),
                "orientation": self._json_safe(getattr(value, "orientation", None)),
                "image": getattr(value, "image", None),
            }
        if hasattr(value, "center"):
            return self._json_safe(value.center)
        return str(value)

    def init_trajectory_logging(self, subtask_id, metadata=None):
        """Initialize lightweight per-step trajectory logging for one subtask."""
        self.trajectory_records = []
        self.current_trajectory_meta = {
            "subtask_id": subtask_id,
            **self._json_safe(metadata or {}),
        }

    def log_step_trajectory(self, step_record):
        """Append one JSON-safe step record for later failure analysis."""
        try:
            self.trajectory_records.append(self._json_safe(step_record))
        except Exception:
            logging.exception("Failed to append trajectory step record")

    def log_event(self, event_type, step=None, payload=None):
        """Attach an event to the current step, or append a standalone event step."""
        event = {
            "event": event_type,
            "step": step,
            "payload": self._json_safe(payload or {}),
        }
        try:
            if self.trajectory_records and (
                step is None or self.trajectory_records[-1].get("step") == step
            ):
                self.trajectory_records[-1].setdefault("events", []).append(event)
            else:
                self.trajectory_records.append({"step": step, "events": [event]})
        except Exception:
            logging.exception("Failed to append trajectory event")

    def save_trajectory_jsonl(self, subtask_id, final_summary=None):
        """Append one subtask trajectory record to JSONL.

        Robust to serialization errors so trajectory logging never breaks evaluation.
        """
        import datetime
        import traceback

        try:
            record = {
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "subtask_id": subtask_id,
                "metadata": self._json_safe(self.current_trajectory_meta),
                "total_steps": len(self.trajectory_records),
                "trajectory": self._json_safe(self.trajectory_records),
                "final_summary": self._json_safe(final_summary or {}),
            }
            with open(self.trajectory_jsonl_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception:
            logging.warning(
                "Failed to write trajectory record for %s:\n%s",
                subtask_id,
                traceback.format_exc(),
            )
        finally:
            self.trajectory_records = []
            self.current_trajectory_meta = {}


    def init_episode(
        self,
        episode_id,
    ):
        self.episode_dir = os.path.join(self.output_dir, episode_id)
        eps_frontier_dir = os.path.join(self.episode_dir, "frontier")

        os.makedirs(self.episode_dir, exist_ok=True)
        os.makedirs(eps_frontier_dir, exist_ok=True)

        return self.episode_dir, eps_frontier_dir

    def init_subtask(
        self,
        subtask_id,
        goal_type,
        subtask_goal,
        pts,
        scene: Scene,  # used to get image goal observation and use its pathfinder
        tsdf_planner: TSDFPlanner,
    ):
        # determine the navigation goals
        goal_category = subtask_goal[0]["object_category"]
        goal_obj_ids = [x["object_id"] for x in subtask_goal]
        goal_obj_ids = [int(x.split("_")[-1]) for x in goal_obj_ids]##### slight modification for ids
        if goal_type != "object":
            assert len(goal_obj_ids) == 1, f"{len(goal_obj_ids)} != 1"

        goal_positions = [x["position"] for x in subtask_goal]##### also from subtask_goal, corresponding to goal_obj_ids
        goal_positions_voxel = [tsdf_planner.habitat2voxel(p) for p in goal_positions]##### same goal_positions, but transformed from habitat to voxel

        viewpoints = [
            view_point["agent_state"]["position"]
            for goal in subtask_goal
            for view_point in goal["view_points"]
        ]
        # get the shortest distance from current position to the viewpoints
        all_distances = []
        for viewpoint in viewpoints:
            path = habitat_sim.ShortestPath()
            path.requested_start = pts
            path.requested_end = viewpoint
            found_path = scene.pathfinder.find_path(path)
            if not found_path:
                all_distances.append(np.inf)
            else:
                all_distances.append(path.geodesic_distance)
        gt_subtask_explore_dist = min(all_distances) + 1e-6

        self.subtask_object_observe_dir = os.path.join(
            self.output_dir, subtask_id, "object_observations"
        )
        if os.path.exists(self.subtask_object_observe_dir):
            os.system(f"rm -r {self.subtask_object_observe_dir}")
        os.makedirs(self.subtask_object_observe_dir, exist_ok=False)

        # Prepare metadata for the subtask
        subtask_metadata = {
            "question_id": subtask_id,
            "question": None,
            "image": None,
            "answer": goal_category,
            "goal_obj_ids": goal_obj_ids,  # this is a list of obj id, since for object class type, there will be multiple target objects
            "class": goal_category,
            "goal_positions_voxel": goal_positions_voxel,  # also a list of positions for possible multiple objects
            "task_type": goal_type,
            "viewpoints": viewpoints,
            "gt_subtask_explore_dist": gt_subtask_explore_dist,
        }
        # format question, according to the goal type
        if goal_type == "object":
            subtask_metadata["question"] = f"Can you find the {goal_category}?"
        elif goal_type == "description":
            subtask_metadata["question"] = (
                f"Could you find the object exactly described as the '{subtask_goal[0]['lang_desc']}'?"
            )
        else:
            subtask_metadata["question"] = (
                "Identify the target object shown near the center of the reference image. "
                "Use scene context to locate the same object in the environment."
            )
            view_pos_dict = subtask_goal[0]["view_points"][0]["agent_state"]
            obs, _ = scene.get_observation(
                pts=view_pos_dict["position"], rotation=view_pos_dict["rotation"]
            )
            plt.imsave(
                os.path.join(self.output_dir, subtask_id, "image_goal.png"),
                obs["color_sensor"],
            )
            subtask_metadata["image"] = f"{self.output_dir}/{subtask_id}/image_goal.png"

        self.pts_voxels = np.empty((0, 2))
        self.pts_voxels = np.vstack(
            [self.pts_voxels, tsdf_planner.habitat2voxel(pts)[:2]]
        )
        self.subtask_explore_dist = 0.0

        return subtask_metadata

    def log_step(self, pts_voxel):
        self.pts_voxels = np.vstack([self.pts_voxels, pts_voxel])
        self.subtask_explore_dist += (
            np.linalg.norm(self.pts_voxels[-1] - self.pts_voxels[-2]) * self.voxel_size
        )

    def save_topdown_visualization(
        self, global_step, subtask_id, subtask_metadata, goal_obj_ids_mapping, fig
    ):
        assert self.episode_dir is not None
        visualization_path = os.path.join(self.episode_dir, "visualization")
        os.makedirs(visualization_path, exist_ok=True)

        ax1 = fig.axes[0]
        ax1.plot(
            self.pts_voxels[:-1, 1], self.pts_voxels[:-1, 0], linewidth=1, color="white"
        )
        ax1.scatter(self.pts_voxels[0, 1], self.pts_voxels[0, 0], c="white", s=50)

        # add target object bbox(?) in topdown visualization
        for goal_id, goal_pos_voxel in zip(
            subtask_metadata["goal_obj_ids"], subtask_metadata["goal_positions_voxel"]
        ):
            color = "green" if len(goal_obj_ids_mapping[goal_id]) > 0 else "red"
            ###### if the list "goal_obj_ids_mapping[goal_id]" is non-empty(got detected), then mark green. Else red.
            ###### color = "yellow" # test: where are these dots?(yellow)
            ax1.scatter(goal_pos_voxel[1], goal_pos_voxel[0], c=color, s=120)

        fig.tight_layout()
        plt.savefig(os.path.join(visualization_path, f"{global_step}_{subtask_id}.png"))
        plt.close()

    def save_frontier_visualization(
        self,
        global_step,
        subtask_id,
        tsdf_planner: TSDFPlanner,
        max_point_choice,
        global_caption,
    ):
        assert self.episode_dir is not None
        frontier_video_path = os.path.join(self.episode_dir, "frontier_video")
        episode_frontier_dir = os.path.join(self.episode_dir, "frontier")
        os.makedirs(frontier_video_path, exist_ok=True)
        num_images = len(tsdf_planner.frontiers)
        side_length = int(np.sqrt(num_images)) + 1
        side_length = max(2, side_length)
        fig, axs = plt.subplots(side_length, side_length, figsize=(20, 20))
        for h_idx in range(side_length):
            for w_idx in range(side_length):
                axs[h_idx, w_idx].axis("off")
                i = h_idx * side_length + w_idx
                if (i < num_images - 1) or (
                    i < num_images and type(max_point_choice) == Frontier
                ):
                    img_path = os.path.join(
                        episode_frontier_dir, tsdf_planner.frontiers[i].image
                    )
                    img = matplotlib.image.imread(img_path)
                    axs[h_idx, w_idx].imshow(img)
                    if (
                        type(max_point_choice) == Frontier
                        and max_point_choice.image == tsdf_planner.frontiers[i].image
                    ):
                        axs[h_idx, w_idx].set_title("Chosen")

        fig.suptitle(global_caption, fontsize=16)
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        plt.savefig(
            os.path.join(frontier_video_path, f"{global_step}_{subtask_id}.png")
        )
        plt.close()
