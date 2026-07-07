import numpy as np
from PIL import Image
import habitat_sim
from habitat_sim.utils.common import quat_to_angle_axis, quat_from_coeffs
import quaternion
import logging
import open3d as o3d

def resize_image(image, target_h, target_w):
    # image: np.array, h, w, c
    image = Image.fromarray(image)
    image = image.resize((target_w, target_h))
    return np.array(image)


def find_center_in_room(centers, confidences, xyxy, class_ids, rooms):
    if len(confidences) > 0:
        sorted_indices = np.argsort(confidences)[::-1]
        class_ids = class_ids[sorted_indices]
        confidences = confidences[sorted_indices]
        xyxy = xyxy[sorted_indices]
    room_label = []
    room_conf = []
    for center in centers:   
        find_room = False
        x, y = center
        for idx in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[idx]
            if x1 <= x <= x2 and y1 <= y <= y2:
                room_label.append(rooms[class_ids[idx]])
                room_conf.append(confidences[idx])
                find_room = True
                break
        if not find_room:
            room_label.append('unknown')
    return room_label, room_conf

def rgba2rgb(rgba, background=(255, 255, 255)):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, "RGBA image has 4 channels."

    rgb = np.zeros((row, col, 3), dtype="float32")
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

    a = np.asarray(a, dtype="float32") / 255.0

    R, G, B = background

    rgb[:, :, 0] = r * a + (1.0 - a) * R
    rgb[:, :, 1] = g * a + (1.0 - a) * G
    rgb[:, :, 2] = b * a + (1.0 - a) * B

    return np.asarray(rgb, dtype="uint8")


def get_pts_angle_aeqa(init_pts, init_quat):
    pts = np.asarray(init_pts)

    init_quat = quaternion.quaternion(*init_quat)
    angle, axis = quat_to_angle_axis(init_quat)
    angle = angle * axis[1] / np.abs(axis[1])

    return pts, angle


def get_pts_angle_goatbench(init_pos, init_rot):
    pts = np.asarray(init_pos)

    init_quat = quat_from_coeffs(init_rot)
    angle, axis = quat_to_angle_axis(init_quat)
    angle = angle * axis[1] / np.abs(axis[1])

    return pts, angle

def calc_agent_subtask_distance(curr_pts, viewpoints, pathfinder):
    # calculate the distance to the nearest view point
    path = habitat_sim.MultiGoalShortestPath()
    path.requested_start = curr_pts
    path.requested_ends = viewpoints
    # np.save(f'/hsun/0625/3D-Mem/vis/pos_{curr_pts[1]}.npy', np.array(viewpoints))
    found_path = pathfinder.find_path(path)
    if found_path:
        distance = path.geodesic_distance
    else:
        distance = 10.0
    return distance