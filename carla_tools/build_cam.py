"""
    Copyright (C) 2022 Martin Ahrnbom
    This work is released under the MIT License. 
    See the file LICENSE for details

    
    A module for loading camera matrices from json files 
"""

from scipy.linalg import null_space
from pathlib import Path 
import json 
import numpy as np 

from tools.visualization.geometry import projective_flatten as pflat

def build_camera_matrices(folder:Path, output_K=False):
    txt_path = folder / 'cameras.json'
    text = txt_path.read_text()
    cams_obj = json.loads(text)
    f = cams_obj['instrinsics']['f']
    Cx = cams_obj['instrinsics']['Cx']
    Cy = cams_obj['instrinsics']['Cy']

    cameras = dict()
    for cam in cams_obj['cams']:
        values = {'f':f, 'Cx':Cx, 'Cy':Cy}
        for key in ('x', 'y', 'z', 'pitch', 'roll', 'yaw'):
            values[key] = cam[key]
        
        cam_id = int(cam['id'])
        P, K = build_cam(values)
        cameras[cam_id] = P
    
    if output_K:
        return cameras, K

    return cameras
def convert_to_opencv(P_orig, K, flip_axes=True):
    """
    Convert an existing 4x4 projection matrix to an OpenCV-compatible matrix.
    P_orig: Original 4x4 homogeneous projection matrix.
    K: OpenCV 3x3 camera intrinsic matrix.
    flip_axes: Whether to flip the Y axis.
    """
    if flip_axes:
        # Coordinate transformation matrix T.
        # Assume the original Z axis points up and X/Y differ from OpenCV.
        T = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],  # Flip the Y axis.
            [0, 0, 1, 0],   # Keep the Z axis unchanged.
            [0, 0, 0, 1]
        ], dtype=np.float32)
    else:
        T = np.eye(4, dtype=np.float32)

    # Use only the first three rows of the projection matrix.
    P_cv = K @ (T @ P_orig)[:3, :]
    return P_cv

def build_cam(values, lidar_offset=None):
    # Build the camera intrinsic matrix.
    x, y, z = values['x'], values['y'], values['z']
    pitch, roll, yaw = values['pitch'], values['roll'], values['yaw']
    fov, width, height = values['fov'], values['width'], values['height']
    
    f = (width / 2) / np.tan(np.radians(fov / 2))
    Cx, Cy = width / 2, height / 2
    K = np.array([[f, 0, Cx],
                  [0, f, Cy],
                  [0, 0, 1]], dtype=np.float64)
    
    # Camera rotation matrix.
    c_y, s_y = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    c_r, s_r = np.cos(np.radians(roll)), np.sin(np.radians(roll))
    c_p, s_p = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))

    matrix = np.identity(4)
    matrix[0, 3], matrix[1, 3], matrix[2, 3] = x, y, z
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r

    # World-to-camera transformation.
    matrix = np.linalg.inv(matrix)

    # OpenCV coordinate system.
    flip = np.array([[ 0, 1, 0 ],
                     [ 0, 0, -1 ],
                     [ 1, 0, 0 ]], dtype=np.float32)

    P = K @ flip @ matrix[:3, :]
    
    # Homogeneous representation.
    P_homo = np.vstack([P, np.array([0,0,0,1])])

    # Validate the camera center.
    cen = np.array([x, y, z, 1]).reshape((4,1))
    C = pflat(null_space(P))
    assert(np.allclose(C, cen, atol=1e-4)), "Camera center validation failed"

    # Apply the LiDAR offset if provided.
    if lidar_offset is not None:
        tx, ty, tz = lidar_offset['x'], lidar_offset['y'], lidar_offset['z']
        T_lidar_to_cam = np.eye(4)
        T_lidar_to_cam[:3, 3] = [tx, ty, tz]
        return P_homo, K, T_lidar_to_cam

    return P_homo, K


def euler_angles(phi, theta, psi):
    sin = np.sin
    cos = np.cos
    R = [[cos(theta)*cos(psi),  -cos(phi)*sin(psi)+sin(phi)*sin(theta)*cos(psi),  sin(phi)*sin(psi)+cos(phi)*sin(theta)*cos(psi)],
         [cos(theta)*sin(psi), cos(phi)*cos(psi)+sin(phi)*sin(theta)*sin(psi), -sin(phi)*cos(psi)+cos(phi)*sin(theta)*sin(psi)],
         [-sin(theta),            sin(phi)*cos(theta),             cos(phi)*cos(theta)]]
    return np.array(R, dtype=np.float32)

def is_visible(x, y, z, P, im_h=720, im_w=1280):
    point = np.array([x, y, z, 1.0], dtype=np.float32).reshape((4,1))
    proj = (P @ point).flatten()
    # Check if in front of the camera
    if proj[-1] > 0.0:
        px, py = pflat(proj)[0:2]
        if px >= 0 and px <= im_w and py >= 0 and py <= im_h:
            return True 
        
    return False 

def is_obj_visible(x, y, z, height, P, im_h=720, im_w=1280, min_height=10.0):
    point = np.array([x, y, z, 1.0], dtype=np.float32).reshape((4,1))
    proj = (P @ point).flatten()

    # Check if behind the camera
    if proj[-1] < 0.0:
        return False 

    px, py = pflat(proj)[0:2]
    if px >= 0 and px <= im_w and py >= 0 and py <= im_h:
        delta_height = np.array([0,0,height,0], dtype=np.float32).reshape((4,1))
        new_point = point + delta_height
        proj2 = pflat(P @ new_point)
        py2 = proj2[1]

        pixel_height = abs(py2 - py)
        if pixel_height >= min_height:
            return True 
    
    return False 
