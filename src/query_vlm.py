import logging
from typing import Tuple, Optional, Union
import random
import numpy as np

from src.explore_utils import task_check, strict_task_check, explore_two_step, same_target_check
from src.tsdf_planner import TSDFPlanner, Frontier
from src.multimodal_3d_scene_graph import Scene
from src.conceptgraph.slam.utils import (
    get_bounding_box,
    init_process_pcd,
    detections_to_obj_pcd_and_bbox,
)
def random_frontier_choice(tsdf_planner: TSDFPlanner, n_filtered_snapshots):
    """
    Choose a random frontier from the TSDF planner.
    """
    if len(tsdf_planner.frontiers) == 0:
        logging.error("No frontiers available, returning None.")
        return None
    idx = random.randint(0, len(tsdf_planner.frontiers)-1)
    random_frontier = tsdf_planner.frontiers[idx]
    logging.info(f"Randomly chosen frontier at {random_frontier.position}")
    return "frontier", random_frontier, n_filtered_snapshots, idx


def query_vlm_for_response(
    subtask_metadata: dict,
    scene: Scene,
    tsdf_planner: TSDFPlanner,
    rgb_egocentric_views: list,
    cfg,
    pts = None,
    verbose: bool = False,
):
    # prepare input for vlm
    step_dict = {}

    # prepare object and image context
    object_id_to_name = {
        obj_id: obj["class_name"] for obj_id, obj in scene.objects.items()
    }
    object_id_to_room = {
        obj_id: [obj["room_label"],obj["room_conf"]] for obj_id, obj in scene.objects.items()
    }
    step_dict["obj_map"] = object_id_to_name

    step_dict["objects"] = scene.objects
    step_dict["all_imgs"] = scene.all_observations
    step_dict["edges"] = scene.edges
    step_dict["prompt_h"] = cfg.prompt_h
    step_dict["prompt_w"] = cfg.prompt_w
    step_dict["use_full_obj_list"] = cfg.use_full_obj_list

    # prepare frontier
    step_dict["frontier_imgs"] = [
        frontier.feature for frontier in tsdf_planner.frontiers
    ]

    # prepare egocentric views
    if cfg.egocentric_views:
        step_dict["egocentric_views"] = rgb_egocentric_views
        step_dict["use_egocentric_views"] = True

    # prepare other metadata
    step_dict["question"] = subtask_metadata["question"]
    step_dict["task_type"] = subtask_metadata["task_type"]
    step_dict["class"] = subtask_metadata["class"]
    step_dict["image"] = subtask_metadata["image"]
    step_dict["CLR"] = subtask_metadata['CLR']
    step_dict["MEMORY_HINTS"] = subtask_metadata.get("MEMORY_HINTS")
    step_dict["MEMORY_HINT_POLICY"] = subtask_metadata.get("MEMORY_HINT_POLICY")
    step_dict["object_id_to_room"] = object_id_to_room
    step_dict['image_to_edges'] = scene.img_to_edge
    # query vlm
    (
        outputs,
        image_map_reverse,
        reason,
        n_filtered_snapshots,
    ) = explore_two_step(step_dict, cfg, verbose=verbose)
    if outputs is None:
        logging.error(f"explore_step failed and returned None, Choose a random frontier instead!")
        return random_frontier_choice(tsdf_planner, n_filtered_snapshots)
    logging.info(f"Response: [{outputs}]\nReason: [{reason}]")
    
    # parse returned results
    try:
        target_type, target_index = outputs.split(",")[0].strip().split(" ")
        logging.info(f"Prediction: {target_type}, {target_index}")
    except:
        logging.info(f"Wrong output format, Choose a random frontier instead!")
        return random_frontier_choice(tsdf_planner, n_filtered_snapshots)

    if target_type not in ["image", "frontier", "object"]:
        logging.info(f"Wrong target type: {target_type}, Choose a random frontier instead!")
        return random_frontier_choice(tsdf_planner, n_filtered_snapshots)


    if target_type == "image":
        #Implementation of AVU.
        #We update only the words we consider to be the target, 
        #so we perform re-perception directly and select it as the answer.
        if int(target_index) >= 0 and int(target_index) < len(image_map_reverse):
            target_index = image_map_reverse[int(target_index)]
        else:
            view_idx = int(target_index) - len(image_map_reverse)
            global_step = scene.global_step_cnt
            target_index = f"{global_step}-view_{view_idx}.png"
        target_image = scene.all_observations[target_index]
        try:
            object_class = outputs.split(",")[1].strip()
        except:
            logging.info(f"Wrong output format, Choose a random frontier instead!")
            return random_frontier_choice(tsdf_planner, n_filtered_snapshots)
        scene.detection_model.set_classes([object_class])
        results = scene.detection_model.predict(
            target_image, conf=cfg.AVU_conf_threshold, verbose=False
        )
        
        scene.detection_model.set_classes(scene.obj_classes.get_classes_arr())
        if len(results) == 0 or len(results[0].boxes) == 0:
            logging.info(
                f"No objects detected in the predicted image: {target_index}, Choose a random frontier instead!"
            )
            return random_frontier_choice(tsdf_planner, n_filtered_snapshots)
        
        confidences = results[0].boxes.conf.cpu().numpy()
        max_idx = confidences.argmax()
        max_confidence = confidences[max_idx: max_idx+1]
        detection_class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        max_detection_class_ids = detection_class_ids[max_idx: max_idx+1]
        xyxy_tensor = results[0].boxes.xyxy[max_idx: max_idx+1, ...]
        sam_out = scene.sam_predictor.predict(
            target_image, bboxes=xyxy_tensor, verbose=False
        )
        masks_tensor = sam_out[0].masks.data
        masks_np = masks_tensor.cpu().numpy()
        obj_pcds_and_bboxes = detections_to_obj_pcd_and_bbox(
            depth_array=scene.all_depths[target_index],
            masks=masks_np,
            cam_K=scene.intrinsics[:3, :3],  # Camera intrinsics
            image_rgb=target_image,
            trans_pose=scene.all_cam_poses[target_index],
            min_points_threshold=scene.cfg_cg.min_points_threshold,
            spatial_sim_type=scene.cfg_cg.spatial_sim_type,
            obj_pcd_max_points=scene.cfg_cg.obj_pcd_max_points,
            device=scene.device,
        )
        
        for obj in obj_pcds_and_bboxes:
            if obj:
                obj["pcd"] = init_process_pcd(
                    pcd=obj["pcd"],
                    downsample_voxel_size=scene.cfg_cg["downsample_voxel_size"],
                    dbscan_remove_noise=scene.cfg_cg["dbscan_remove_noise"],
                    dbscan_eps=scene.cfg_cg["dbscan_eps"],
                    dbscan_min_points=scene.cfg_cg["dbscan_min_points"],
                )
                obj["bbox"] = get_bounding_box(
                    spatial_sim_type=scene.cfg_cg["spatial_sim_type"],
                    pcd=obj["pcd"],
                )
        try:
            obj_pos = np.array(obj["bbox"].center)
            logging.info(f"The index of target Image {target_index} : {obj_pos} (Object Center, Confidence : {max_confidence})")
        except:
            logging.info(
                f"No Object Point Cloud in the predicted image: {target_index}, Choose a random frontier instead!"
            )
            return random_frontier_choice(tsdf_planner, n_filtered_snapshots)
        # logging.info(f"The index of target Image {target_index} : {obj_pos} (Mask Center, Confidence : {max_confidence})")
        
        
        # a = []
        # for idx in scene.objects.keys():
        #     if idx != target_index:
        #         a.append(scene.objects[idx]["pcd"].points)
        # np.save(f'/hsun/0625/3D-Mem/vis/other_obj_{pts[1]}.npy', np.concatenate(a, axis=0))
        # np.save(f'/hsun/0625/3D-Mem/vis/target_obj_{pts[1]}.npy', np.array(obj["pcd"].points))
        

        return target_type, obj_pos, n_filtered_snapshots, target_index
    elif target_type == "object":
        target_index = int(target_index)
        if target_index not in list(scene.objects.keys()):
            logging.info(
                f"Predicted object index not in list: {target_index}, Choose a random frontier instead!"
            )
            return random_frontier_choice(tsdf_planner, n_filtered_snapshots)
        # a = []
        # for idx in scene.objects.keys():
        #     if idx != target_index:
        #         a.append(scene.objects[idx]["pcd"].points)
        # np.save(f'/hsun/0625/3D-Mem/vis/other_obj_{pts[1]}.npy', np.concatenate(a, axis=0))
        # np.save(f'/hsun/0625/3D-Mem/vis/target_obj_{pts[1]}.npy', np.array(scene.objects[target_index]["pcd"].points))
        
        
        
        
        target_point = np.array(scene.objects[target_index]["bbox"].center)
        logging.info(f"Next choice: Object {target_point} (Object Center)")

        return target_type, target_point, n_filtered_snapshots, target_index
    else:  # target_type == "frontier"
        target_index = int(target_index)
        if target_index < 0 or target_index >= len(tsdf_planner.frontiers):
            logging.info(
                f"Predicted frontier target index out of range: {target_index}, Choose a random frontier instead!"
            )
            return random_frontier_choice(tsdf_planner, n_filtered_snapshots)
        target_point = tsdf_planner.frontiers[target_index].position
        logging.info(f"Next choice: Frontier at {target_point}")
        pred_target_frontier = tsdf_planner.frontiers[target_index]

        return target_type, pred_target_frontier, n_filtered_snapshots, target_index
    

def query_vlm_for_response_end(
    subtask_metadata: dict,
    rgb_egocentric_views: dict,
    cfg,
    verbose: bool = False,
):
    # prepare input for vlm
    step_dict = {}
    # prepare egocentric views
    step_dict["egocentric_views"] = rgb_egocentric_views
    step_dict["use_egocentric_views"] = True

    # prepare other metadata
    step_dict["question"] = subtask_metadata["question"]
    step_dict["task_type"] = subtask_metadata["task_type"]
    step_dict["class"] = subtask_metadata["class"]
    step_dict["image"] = subtask_metadata["image"]
    # query vlm
    (
        outputs,
        reason,
    ) = task_check(step_dict, verbose=verbose)

    logging.info(f"Response: [{outputs}]\nReason: [{reason}]")

    return outputs


def query_vlm_for_response_end_strict(
    subtask_metadata: dict,
    rgb_egocentric_views: dict,
    cfg,
    verbose: bool = False,
):
    # prepare input for vlm
    step_dict = {}
    # prepare egocentric views
    step_dict["egocentric_views"] = rgb_egocentric_views
    step_dict["use_egocentric_views"] = True

    # prepare other metadata
    step_dict["question"] = subtask_metadata["question"]
    step_dict["task_type"] = subtask_metadata["task_type"]
    step_dict["class"] = subtask_metadata["class"]
    step_dict["image"] = subtask_metadata["image"]
    # query vlm
    (
        outputs,
        reason,
    ) = strict_task_check(step_dict, verbose=verbose)

    logging.info(f"Strict response: [{outputs}]\nStrict reason: [{reason}]")

    return outputs, reason


def query_vlm_same_target_check(
    target_class,
    old_description,
    new_description,
    old_image=None,
    new_image=None,
    verbose: bool = False,
):
    """Ask the VLM whether an EpisodeMemory anchor and the current subtask are
    the same physical object instance, given each side's own description (and,
    for image-type subtasks, reference image encoded as base64).

    Returns False (treat as a different instance, don't reuse the anchor) if
    the VLM's answer is inconclusive after retries -- reusing a memorized
    target center for the wrong physical object is worse than a fresh
    detection, so an unclear signal should not authorize a reuse.
    """
    response, reason = same_target_check(
        target_class,
        old_description,
        new_description,
        old_image=old_image,
        new_image=new_image,
        verbose=verbose,
    )

    logging.info(f"EpisodeMemory identity check: [{response}]\nReason: [{reason}]")

    return response == "yes"