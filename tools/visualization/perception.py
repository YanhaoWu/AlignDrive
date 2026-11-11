import json
import gzip
import os
import numpy as np
import cv2
import pathlib
import random
import matplotlib.cm as cm
from tqdm import tqdm
from tools.visualization.utils import (
    get_image_point,
    point_in_canvas_wh,
    edges,
    world_to_ego,
    get_forward_vector,
    calculate_cube_vertices,
    draw_dashed_line,
    vector_angle,
)


def visualize_data(
    file_path,
    map_path,
    vis_bbox=True,
    vis_top_down=True,
    vis_road=True,
    vis_lidar_bev=True,
    vis_lidar_to_back_image=True,
    vis_lidar_to_front_image=True,
    vis_lidar_to_front_left_image=True,
    output_dir=None,
):
    """Render Bench2Drive annotations, HD maps, camera boxes, and LiDAR projections for debugging."""
    if any(
        (
            vis_lidar_bev,
            vis_lidar_to_back_image,
            vis_lidar_to_front_image,
            vis_lidar_to_front_left_image,
        )
    ):
        import laspy

    print(f'file_path={file_path}')
    print(f'map_path={map_path}')

    source_path = pathlib.Path(file_path).expanduser().resolve()
    file_path = str(source_path)
    save_path = (
        pathlib.Path(output_dir).expanduser().resolve()
        if output_dir
        else source_path.with_name(source_path.name + '_vis')
    )
    if save_path == source_path:
        raise ValueError('The visualization output directory must differ from the input directory')
    (save_path / 'camera' / 'rgb_front_3d_bbox').mkdir(parents=True, exist_ok=True)
    (save_path / 'camera' / 'rgb_front_landmark').mkdir(parents=True, exist_ok=True)
    (save_path / 'camera' / 'rgb_front_left_3d_bbox').mkdir(parents=True, exist_ok=True)
    (save_path / 'camera' / 'rgb_front_right_3d_bbox').mkdir(parents=True, exist_ok=True)
    (save_path / 'camera' / 'rgb_back_3d_bbox').mkdir(parents=True, exist_ok=True)
    (save_path / 'camera' / 'rgb_back_left_3d_bbox').mkdir(parents=True, exist_ok=True)
    (save_path / 'camera' / 'rgb_back_right_3d_bbox').mkdir(parents=True, exist_ok=True)
    (save_path / 'camera' / 'rgb_top_down_3d_bbox').mkdir(parents=True, exist_ok=True)
    (save_path / 'lidar' / 'bev').mkdir(parents=True, exist_ok=True)
    (save_path / 'lidar' / 'front').mkdir(parents=True, exist_ok=True)
    (save_path / 'lidar' / 'front_left').mkdir(parents=True, exist_ok=True)
    (save_path / 'lidar' / 'back').mkdir(parents=True, exist_ok=True)

    cam_map = {
        'CAM_FRONT': 'rgb_front',
        'CAM_FRONT_LEFT': 'rgb_front_left',
        'CAM_FRONT_RIGHT': 'rgb_front_right',
        'CAM_BACK': 'rgb_back',
        'CAM_BACK_LEFT': 'rgb_back_left',
        'CAM_BACK_RIGHT': 'rgb_back_right',
        'TOP_DOWN': 'rgb_top_down',
    }

    folder_path = os.path.join(file_path, 'anno')
    annotation_paths = sorted(pathlib.Path(folder_path).glob('*.json.gz'))
    if not annotation_paths:
        raise ValueError(f'No annotations found in {folder_path}')
    map_info = dict(np.load(map_path, allow_pickle=True)['arr'])

    for annotation_path in tqdm(annotation_paths, desc='Rendering perception'):
        step = int(annotation_path.name.split('.')[0])
        with gzip.open(annotation_path, 'rt', encoding='utf-8') as gz_file:
            anno = json.load(gz_file)
        bounding_boxes = anno['bounding_boxes']
        sensors_anno = anno['sensors']

        if vis_bbox:
            for key in [
                'CAM_FRONT',
                'CAM_FRONT_LEFT',
                'CAM_FRONT_RIGHT',
                'CAM_BACK',
                'CAM_BACK_LEFT',
                'CAM_BACK_RIGHT',
            ]:
                K = sensors_anno[key]['intrinsic']
                world2cam = sensors_anno[key]['world2cam']
                visulize_img = cv2.imread(
                    os.path.join(file_path, f'camera/{cam_map[key]}/{step:05}.jpg')
                )
                for npc in bounding_boxes:
                    if npc['class'] == 'ego_vehicle':
                        continue
                    if npc['distance'] > 75:
                        continue
                    if abs(npc['location'][2] - anno['bounding_boxes'][0]['location'][2]) > 10:
                        continue
                    if 'vehicle' in npc['class']:
                        forward_vec = get_forward_vector(sensors_anno[key]['rotation'][2])
                        ray = np.array(npc['location']) - np.array(sensors_anno[key]['location'])
                        color = (
                            random.randint(0, 255),
                            random.randint(0, 255),
                            random.randint(0, 255),
                        )
                        if forward_vec.dot(ray) > 1 and vector_angle(forward_vec, ray) < 45:
                            verts = np.array(npc['world_cord'])
                            for edge in edges:
                                p1, p1_depth = get_image_point(verts[edge[0]], K, world2cam)
                                p2, p2_depth = get_image_point(verts[edge[1]], K, world2cam)
                                draw_dashed_line(
                                    visulize_img,
                                    (int(p1[0]), int(p1[1])),
                                    (int(p2[0]), int(p2[1])),
                                    color,
                                    2,
                                )
                            cv2.putText(
                                visulize_img,
                                npc['class'] + npc['id'],
                                (int(p1[0]) + 2, int(p1[1]) + 2),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.5,
                                color,
                                1,
                            )
                    else:
                        if npc['class'] == 'traffic_sign':
                            npc['extent'][1] = 0.5

                        forward_vec = get_forward_vector(sensors_anno[key]['rotation'][2])
                        ray = np.array(npc['location']) - np.array(sensors_anno[key]['location'])
                        if 'affects_ego' in npc.keys() and str(npc['affects_ego']) == 'True':
                            color = (0, 0, 255)
                        else:
                            color = (255, 255, 255)
                        if forward_vec.dot(ray) > 1 and vector_angle(forward_vec, ray) < 45:
                            if 'world_cord' in npc.keys():
                                if 'dirtdebris' in npc['type_id']:
                                    local_verts = calculate_cube_vertices(
                                        npc['bbx_loc'],
                                        [npc['extent'][1], npc['extent'][0], npc['extent'][2]],
                                    )
                                    verts = []
                                    for l_v in local_verts:
                                        g_v = np.dot(
                                            np.matrix(npc['world2sign']).I,
                                            [l_v[0], l_v[1], l_v[2], 1],
                                        )
                                        verts.append(g_v.tolist()[0][:-1])
                                else:
                                    verts = np.array(npc['world_cord'])
                            else:
                                verts = calculate_cube_vertices(npc['center'], npc['extent'])
                            for edge in edges:
                                p1, p1_depth = get_image_point(verts[edge[0]], K, world2cam)
                                p2, p2_depth = get_image_point(verts[edge[1]], K, world2cam)
                                draw_dashed_line(
                                    visulize_img,
                                    (int(p1[0]), int(p1[1])),
                                    (int(p2[0]), int(p2[1])),
                                    color,
                                    2,
                                )
                            if 'affects_ego' in npc.keys():
                                cv2.putText(
                                    visulize_img,
                                    npc['class'],
                                    (int(p1[0]) + 2, int(p1[1]) + 2),
                                    cv2.FONT_HERSHEY_COMPLEX,
                                    0.5,
                                    color,
                                    1,
                                )
                            else:
                                cv2.putText(
                                    visulize_img,
                                    npc['class'],
                                    (int(p1[0]) + 2, int(p1[1]) + 2),
                                    cv2.FONT_HERSHEY_COMPLEX,
                                    0.5,
                                    color,
                                    1,
                                )
                cv2.imwrite(
                    os.path.join(save_path, f'camera/{cam_map[key]}_3d_bbox/{step:05}.jpg'),
                    visulize_img,
                )

        if vis_top_down:
            for key in ['TOP_DOWN']:
                K = sensors_anno[key]['intrinsic']
                world2cam = sensors_anno[key]['world2cam']
                visulize_img = cv2.imread(
                    os.path.join(file_path, f'camera/{cam_map[key]}/{step:05}.jpg')
                )
                road_points = map_info[anno['bounding_boxes'][0]['road_id']]

                for r_p in road_points[anno['bounding_boxes'][0]['lane_id']]:
                    road_point = r_p['Points']
                    road_type = r_p['Type']
                    road_color = r_p['Color']
                    road_topology = r_p['Topology']
                    for point in road_point:
                        point = np.array([point[0][0], point[0][1], point[0][2], 1])
                        point_camera = np.dot(world2cam, point)
                        point_camera = [point_camera[1], -point_camera[2], point_camera[0]]
                        depth = point_camera[2]
                        point_img = np.dot(K, point_camera)
                        if depth > 0:
                            point_img[0] /= point_img[2]
                            point_img[1] /= point_img[2]
                            point_img = point_img[0:2]
                            if point_in_canvas_wh(point_img):
                                if road_color == 'White':
                                    cv2.circle(
                                        visulize_img,
                                        (int(point_img[0]), int(point_img[1])),
                                        radius=1,
                                        color=(255, 255, 255),
                                        thickness=-1,
                                    )
                                    if road_type == 'Center':
                                        cv2.circle(
                                            visulize_img,
                                            (int(point_img[0]), int(point_img[1])),
                                            radius=1,
                                            color=(0, 255, 0),
                                            thickness=-1,
                                        )
                                else:
                                    cv2.circle(
                                        visulize_img,
                                        (int(point_img[0]), int(point_img[1])),
                                        radius=1,
                                        color=(0, 255, 255),
                                        thickness=-1,
                                    )

                for npc in bounding_boxes:
                    if 'vehicle' in npc['class']:
                        if abs(npc['location'][2] - anno['bounding_boxes'][0]['location'][2]) > 10:
                            continue
                        if npc['class'] == 'ego_vehicle':
                            color = (255, 255, 255, 255)
                        else:
                            color = (255, 0, 0, 255)
                        verts = np.array(npc['world_cord'])
                        p1, p1_depth = get_image_point(verts[0], K, world2cam)
                        p2, p2_depth = get_image_point(verts[2], K, world2cam)
                        p3, p3_depth = get_image_point(verts[4], K, world2cam)
                        p4, p4_depth = get_image_point(verts[6], K, world2cam)
                        points = np.array([p1, p2, p4, p3])
                        height, width = visulize_img.shape[:2]
                        blk = np.zeros((height, width, 4), np.uint8)
                        cv2.fillConvexPoly(blk, np.round(points).astype(np.int32), color)
                        if npc['class'] == 'ego_vehicle':
                            visulize_img = cv2.addWeighted(visulize_img, 1.0, blk[:, :, :3], 1, 1)
                        else:
                            visulize_img = cv2.addWeighted(
                                visulize_img, 1.0, blk[:, :, :3], 0.25, 1
                            )
                        if npc['class'] == 'ego_vehicle':
                            cv2.putText(
                                visulize_img,
                                npc['class'],
                                (int(p1[0]) + 2, int(p1[1]) + 2),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.5,
                                (0, 0, 0),
                                1,
                            )
                        else:
                            cv2.putText(
                                visulize_img,
                                npc['class'],
                                (int(p1[0]) + 2, int(p1[1]) + 2),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.5,
                                color,
                                1,
                            )

                for npc in bounding_boxes:
                    if abs(npc['location'][2] - anno['bounding_boxes'][0]['location'][2]) > 10:
                        continue

                    if 'traffic_sign' in npc['class']:
                        color = (0, 0, 255, 255)
                        if 'world_cord' in npc.keys():
                            verts = np.array(npc['world_cord'])
                        else:
                            verts = calculate_cube_vertices(npc['center'], npc['extent'])
                        p1, p1_depth = get_image_point(verts[0], K, world2cam)
                        p2, p2_depth = get_image_point(verts[2], K, world2cam)
                        p3, p3_depth = get_image_point(verts[4], K, world2cam)
                        p4, p4_depth = get_image_point(verts[6], K, world2cam)
                        points = np.array([p1, p2, p4, p3])
                        height, width = visulize_img.shape[:2]
                        blk = np.zeros((height, width, 4), np.uint8)
                        cv2.fillConvexPoly(blk, np.round(points).astype(np.int32), color)
                        visulize_img = cv2.addWeighted(visulize_img, 1.0, blk[:, :, :3], 0.25, 1)
                        cv2.putText(
                            visulize_img,
                            npc['class'],
                            (int(p1[0]) + 2, int(p1[1]) + 2),
                            cv2.FONT_HERSHEY_COMPLEX,
                            0.5,
                            color,
                            1,
                        )

                    if 'traffic_light' in npc['class']:
                        color = (255, 0, 0)
                        verts = calculate_cube_vertices(npc['center'], npc['extent'])
                        for edge in edges:
                            p1, p1_depth = get_image_point(verts[edge[0]], K, world2cam)
                            p2, p2_depth = get_image_point(verts[edge[1]], K, world2cam)
                            cv2.line(
                                visulize_img,
                                (int(p1[0]), int(p1[1])),
                                (int(p2[0]), int(p2[1])),
                                color,
                                2,
                            )
                        cv2.putText(
                            visulize_img,
                            'traffic_light',
                            (int(p1[0]) + 2, int(p1[1]) + 2),
                            cv2.FONT_HERSHEY_COMPLEX,
                            0.5,
                            color,
                            1,
                        )
                cv2.imwrite(
                    os.path.join(save_path, f'camera/{cam_map[key]}_3d_bbox/{step:05}.jpg'),
                    visulize_img,
                )

        if vis_road:
            key = 'CAM_FRONT'
            K = sensors_anno[key]['intrinsic']
            world2cam = sensors_anno[key]['world2cam']
            road_points = map_info[anno['bounding_boxes'][0]['road_id']]
            road_seg = np.zeros((900, 1600, 3), dtype=np.uint8)
            all_road_topology = set()

            for r_p in road_points[anno['bounding_boxes'][0]['lane_id']]:
                road_point = r_p['Points']
                road_type = r_p['Type']
                road_color = r_p['Color']
                road_topology = r_p['Topology']
                for r_t in road_topology:
                    all_road_topology.add(r_t)
                for point in road_point:
                    point = np.array([point[0][0], point[0][1], point[0][2], 1])
                    point_camera = np.dot(world2cam, point)
                    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]
                    depth = point_camera[2]
                    point_img = np.dot(K, point_camera)
                    if depth > 0:
                        point_img[0] /= point_img[2]
                        point_img[1] /= point_img[2]
                        point_img = point_img[0:2]
                        if point_in_canvas_wh(point_img):
                            if road_color == 'White':
                                cv2.circle(
                                    road_seg,
                                    (int(point_img[0]), int(point_img[1])),
                                    radius=1,
                                    color=(255, 255, 255),
                                    thickness=-1,
                                )
                                if road_type == 'Center':
                                    cv2.circle(
                                        road_seg,
                                        (int(point_img[0]), int(point_img[1])),
                                        radius=1,
                                        color=(0, 255, 0),
                                        thickness=-1,
                                    )
                            else:
                                cv2.circle(
                                    road_seg,
                                    (int(point_img[0]), int(point_img[1])),
                                    radius=1,
                                    color=(0, 255, 255),
                                    thickness=-1,
                                )

            for r_t in all_road_topology:
                road_points = map_info[r_t[0]][r_t[1]]
                for r_p in road_points:
                    road_point = r_p['Points']
                    road_type = r_p['Type']
                    road_color = r_p['Color']
                    road_topology = r_p['Topology']
                    for point in road_point:
                        point = np.array([point[0][0], point[0][1], point[0][2], 1])
                        point_camera = np.dot(world2cam, point)
                        point_camera = [point_camera[1], -point_camera[2], point_camera[0]]
                        depth = point_camera[2]
                        point_img = np.dot(K, point_camera)
                        if depth > 0:
                            point_img[0] /= point_img[2]
                            point_img[1] /= point_img[2]
                            point_img = point_img[0:2]
                            if point_in_canvas_wh(point_img):
                                if road_color == 'White':
                                    cv2.circle(
                                        road_seg,
                                        (int(point_img[0]), int(point_img[1])),
                                        radius=1,
                                        color=(255, 255, 255),
                                        thickness=-1,
                                    )
                                    if road_type == 'Center':
                                        cv2.circle(
                                            road_seg,
                                            (int(point_img[0]), int(point_img[1])),
                                            radius=1,
                                            color=(0, 255, 0),
                                            thickness=-1,
                                        )
                                else:
                                    cv2.circle(
                                        road_seg,
                                        (int(point_img[0]), int(point_img[1])),
                                        radius=1,
                                        color=(0, 255, 255),
                                        thickness=-1,
                                    )
            cv2.imwrite(
                os.path.join(save_path, f'camera/{cam_map[key]}_landmark/{step:05}.png'), road_seg
            )

        if vis_lidar_bev:
            lidar_path = os.path.join(file_path, f'lidar/{step:05}.laz')
            lidars = laspy.read(lidar_path).xyz

            lidar_image = np.zeros((900, 1600, 3), dtype=np.uint8)
            header = laspy.LasHeader(point_format=0)
            header.offsets = np.min(lidars, axis=0)
            point_precision = 0.001
            header.scales = np.array([point_precision, point_precision, point_precision])
            point_record = laspy.ScaleAwarePointRecord.zeros(lidars.shape[0], header=header)

            point_record.x = lidars[:, 0]
            point_record.y = lidars[:, 1]
            point_record.z = lidars[:, 2]

            point_record.x = lidars[:, 1]
            point_record.y = -lidars[:, 0]
            point_record.z = lidars[:, 2]

            range_x = 85.0
            range_y = 85.0
            range_z = 85.0

            x_normalized = (point_record.x + range_x) / (2 * range_x)
            y_normalized = (point_record.y + range_y) / (2 * range_y)
            z_normalized = (point_record.z + range_z) / (2 * range_z)

            x_pixels = x_normalized * (lidar_image.shape[1] - 1)
            y_pixels = y_normalized * (lidar_image.shape[0] - 1)

            for x, y, z in zip(x_pixels, y_pixels, z_normalized):
                rgb_color = (255, 255, 255)
                cv2.circle(
                    lidar_image,
                    (int(x), int(y)),
                    radius=2,
                    color=(int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2])),
                    thickness=-1,
                )

            cv2.imwrite(os.path.join(save_path, f'lidar_bev/{step:05}.png'), lidar_image)
            for npc in bounding_boxes:
                if npc['class'] not in ['vehicle', 'ego_vehicle']:
                    continue
                if abs(npc['location'][2] - anno['bounding_boxes'][0]['location'][2]) > 10:
                    continue
                verts = calculate_cube_vertices(npc['center'], npc['extent'])
                verts = np.array(npc['world_cord'])

                if npc['class'] == 'ego_vehicle':
                    color = (0, 255, 0)
                else:
                    color = (0, 128, 255)
                for edge in edges:
                    p1 = verts[edge[0]]
                    p1 = world_to_ego(p1, anno['bounding_boxes'][0]['world2ego'])
                    p1_x = (p1[0] + range_x) / (2 * range_x) * (lidar_image.shape[1] - 1)
                    p1_y = (p1[1] + range_y) / (2 * range_y) * (lidar_image.shape[0] - 1)

                    p2 = verts[edge[1]]
                    p2 = world_to_ego(p2, anno['bounding_boxes'][0]['world2ego'])
                    p2_x = (p2[0] + range_x) / (2 * range_x) * (lidar_image.shape[1] - 1)
                    p2_y = (p2[1] + range_y) / (2 * range_y) * (lidar_image.shape[0] - 1)
                    cv2.line(lidar_image, (int(p1_x), int(p1_y)), (int(p2_x), int(p2_y)), color, 2)

            cv2.imwrite(os.path.join(save_path, f'lidar/bev/{step:05}.png'), lidar_image)

        if vis_lidar_to_front_image:
            key = 'CAM_FRONT'
            K = sensors_anno[key]['intrinsic']
            visulize_img = cv2.imread(
                os.path.join(file_path, f'camera/{cam_map[key]}/{step:05}.jpg')
            )
            ego2cam = np.matrix(sensors_anno[key]['cam2ego']).I.tolist()

            lidar_path = os.path.join(file_path, f'lidar/{step:05}.laz')
            lidars = laspy.read(lidar_path).xyz

            for lidar in lidars:
                lidar = np.array([lidar[0], lidar[1], lidar[2], 1])
                point_camera = np.dot(ego2cam, lidar)
                point_camera = [point_camera[1], -point_camera[2], point_camera[0]]
                depth = point_camera[2]
                point_img = np.dot(K, point_camera)
                if depth > 0:
                    point_img[0] /= point_img[2]
                    point_img[1] /= point_img[2]
                    point_img = point_img[0:2]
                    if point_in_canvas_wh(point_img):
                        color_scale = min(depth / 80, 1)
                        color = cm.rainbow(color_scale)
                        color = tuple([int(x * 255) for x in color[:3]])
                        cv2.circle(
                            visulize_img,
                            (int(point_img[0]), int(point_img[1])),
                            radius=2,
                            color=color,
                            thickness=-1,
                        )
            cv2.imwrite(os.path.join(save_path, f'lidar/front/{step:05}_front.png'), visulize_img)

        if vis_lidar_to_back_image:
            key = 'CAM_BACK'
            K = sensors_anno[key]['intrinsic']
            visulize_img = cv2.imread(
                os.path.join(file_path, f'camera/{cam_map[key]}/{step:05}.jpg')
            )
            ego2cam = np.matrix(sensors_anno[key]['cam2ego']).I.tolist()

            lidar_path = os.path.join(file_path, f'lidar/{step:05}.laz')
            lidars = laspy.read(lidar_path).xyz

            for lidar in lidars:
                lidar = np.array([lidar[0], lidar[1], lidar[2], 1])
                point_camera = np.dot(ego2cam, lidar)
                point_camera = [point_camera[1], -point_camera[2], point_camera[0]]
                depth = point_camera[2]
                point_img = np.dot(K, point_camera)
                if depth > 0:
                    point_img[0] /= point_img[2]
                    point_img[1] /= point_img[2]
                    point_img = point_img[0:2]
                    if point_in_canvas_wh(point_img):
                        color_scale = min(depth / 80, 1)
                        color = cm.rainbow(color_scale)
                        color = tuple([int(x * 255) for x in color[:3]])
                        cv2.circle(
                            visulize_img,
                            (int(point_img[0]), int(point_img[1])),
                            radius=2,
                            color=color,
                            thickness=-1,
                        )
            cv2.imwrite(os.path.join(save_path, f'lidar/back/{step:05}_back.png'), visulize_img)

        if vis_lidar_to_front_left_image:
            key = 'CAM_FRONT_LEFT'
            K = sensors_anno[key]['intrinsic']
            visulize_img = cv2.imread(
                os.path.join(file_path, f'camera/{cam_map[key]}/{step:05}.jpg')
            )
            ego2cam = np.matrix(sensors_anno[key]['cam2ego']).I.tolist()

            lidar_path = os.path.join(file_path, f'lidar/{step:05}.laz')
            lidars = laspy.read(lidar_path).xyz

            for lidar in lidars:
                lidar = np.array([lidar[0], lidar[1], lidar[2], 1])
                point_camera = np.dot(ego2cam, lidar)
                point_camera = [point_camera[1], -point_camera[2], point_camera[0]]
                depth = point_camera[2]
                point_img = np.dot(K, point_camera)
                if depth > 0:
                    point_img[0] /= point_img[2]
                    point_img[1] /= point_img[2]
                    point_img = point_img[0:2]
                    if point_in_canvas_wh(point_img):
                        color_scale = min(depth / 80, 1)
                        color = cm.rainbow(color_scale)
                        color = tuple([int(x * 255) for x in color[:3]])
                        cv2.circle(
                            visulize_img,
                            (int(point_img[0]), int(point_img[1])),
                            radius=2,
                            color=color,
                            thickness=-1,
                        )
            cv2.imwrite(
                os.path.join(save_path, f'lidar/front_left/{step:05}_front_left.png'), visulize_img
            )

    return str(save_path)


def render_predictions(dataset, results, save_dir=None, pipeline=None):
    """Render dataset model outputs as six-camera overlays and a BEV video."""
    from mmcv.utils import print_log
    from mmdet.datasets.pipelines import Compose
    from projects_sp.mmdet3d_plugin.datasets.utils import (
        draw_lidar_bbox3d_on_img,
        draw_lidar_bbox3d_on_bev,
    )

    save_dir = "./" if save_dir is None else save_dir
    save_dir = os.path.join(save_dir, "visual")
    print_log(os.path.abspath(save_dir))
    pipeline = dataset.pipeline if pipeline is None else Compose(pipeline)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    videoWriter = None

    for i, result in enumerate(results):
        if "img_bbox" in result.keys():
            result = result["img_bbox"]
        data_info = pipeline(dataset.get_data_info(i))
        imgs = []

        raw_imgs = data_info["img"]
        lidar2img = data_info["img_metas"].data["lidar2img"]
        visible = result['scores_3d'] > dataset.vis_score_threshold
        pred_bboxes_3d = result['boxes_3d'][visible]
        if "instance_ids" in result and getattr(dataset, 'tracking', False):
            color = []
            for id in result["instance_ids"][visible].cpu().numpy().tolist():
                color.append(dataset.ID_COLOR_MAP[int(id % len(dataset.ID_COLOR_MAP))])
        elif "labels_3d" in result:
            color = []
            for id in result["labels_3d"][visible].cpu().numpy().tolist():
                color.append(dataset.ID_COLOR_MAP[id])
        else:
            color = (255, 0, 0)

        for j, img_origin in enumerate(raw_imgs):
            img = img_origin.copy()
            if len(pred_bboxes_3d) != 0:
                img = draw_lidar_bbox3d_on_img(
                    pred_bboxes_3d,
                    img,
                    lidar2img[j],
                    img_metas=None,
                    color=color,
                    thickness=3,
                )
            imgs.append(img)

        bev = draw_lidar_bbox3d_on_bev(
            pred_bboxes_3d,
            bev_size=img.shape[0] * 2,
            color=color,
        )

        for j, name in enumerate(
            [
                "front",
                "front right",
                "front left",
                "rear",
                "rear left",
                "rear right",
            ]
        ):
            imgs[j] = cv2.rectangle(
                imgs[j],
                (0, 0),
                (440, 80),
                color=(255, 255, 255),
                thickness=-1,
            )
            w, h = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)[0]
            text_x = int(220 - w / 2)
            text_y = int(40 + h / 2)

            imgs[j] = cv2.putText(
                imgs[j],
                name,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
        image = np.concatenate(
            [
                np.concatenate([imgs[2], imgs[0], imgs[1]], axis=1),
                np.concatenate([imgs[5], imgs[3], imgs[4]], axis=1),
            ],
            axis=0,
        )
        image = np.concatenate([image, bev], axis=1)

        if videoWriter is None:
            videoWriter = cv2.VideoWriter(
                os.path.join(save_dir, "video.avi"),
                fourcc,
                7,
                image.shape[:2][::-1],
            )
        cv2.imwrite(os.path.join(save_dir, f"{i}.jpg"), image)
        videoWriter.write(image)
    if videoWriter is not None:
        videoWriter.release()
