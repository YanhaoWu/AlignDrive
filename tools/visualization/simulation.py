"""Render saved CARLA inference frames using the shared visualization backends."""

import copy
import os
import pickle
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.interpolate import splprep, splev
from tqdm import tqdm

from carla_tools.build_cam import build_cam
from tools.visualization.constants import B2D_DETECTION_CLASS, COLORMAP, COLOR_VECTORS
from tools.visualization.geometry import box3d_to_corners, to_numpy
from tools.visualization.video import write_video

COLOR_VECTORS_BGR = {
    'cornflowerblue': (100, 149, 237),
    'royalblue': (65, 105, 225),
    'slategrey': (112, 128, 144),
    'green': (0, 255, 0),
    'red': (0, 0, 255),
    'brown': (165, 42, 42),
}


class SimulationBEVRenderer:
    """Project aligned trajectories and optional perception onto a saved simulator image."""

    def __init__(self, out_dir, render_version='full', bev_view='topdown'):
        self.pred_dir = os.path.join(out_dir, "bev_pred")
        self.text_color = (255, 255, 255)
        self.text_position = (50, 50)
        self.bev_view = bev_view
        if self.bev_view == 'topdown':
            self.coor2topdown = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 50.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            topdown_intrinsics = np.array(
                [
                    [548.993771650447, 0.0, 256.0, 0],
                    [0.0, 548.993771650447, 256.0, 0],
                    [0.0, 0.0, 1.0, 0],
                    [0, 0, 0, 1.0],
                ]
            )
            self.coor2topdown = topdown_intrinsics @ self.coor2topdown
            self.map_line_width = 1
            self.text_width = 2
            self.line_height = 15
            self.font_scale = 1
            self.traj_thickness = 3
            self.traj_dot_size = 3
        else:
            new_cam_value = {
                'type': 'sensor.camera.rgb',
                'x': -6.0,
                'y': 0.0,
                'z': 4.0,
                'roll': 0.0,
                'pitch': -20.0,
                'yaw': 0.0,
                'width': 1920,
                'height': 1080,
                'fov': 90.0,
                'id': 'bev',
            }
            lidar_offset = {
                'x': -0.39,
                'y': 0.0,
                'z': 1.84,
            }

            self.coor2topdown, topdown_intrinsics, _ = build_cam(new_cam_value, lidar_offset)
            self.map_line_width = 5
            self.text_width = 5
            self.line_height = 40
            self.font_scale = 2
            self.traj_thickness = 5
            self.traj_dot_size = 8

        self.set_grand()
        self.set_mode(mode=render_version)

    def set_mode(self, mode):

        self.mode = mode

    def set_grand(self, sptial_grand=None, temporal_grand=None, control_version='2m5hz'):
        self.grand_name = [
            'spatial_interval_2_traj',
            'spatial_interval_5_traj',
            'temporal_interval_2_traj',
            'temporal_interval_5_traj',
        ]
        if control_version is None:
            if sptial_grand is None:
                sptial_grand = 'spatial_interval_2_traj'
            if temporal_grand is None:
                temporal_grand = 'temporal_interval_2_traj'
            self.sptial_grand = sptial_grand
            self.temporal_grand = temporal_grand
        elif control_version == '2m5hz':
            self.sptial_grand = 'spatial_interval_2_traj'
            self.temporal_grand = 'temporal_interval_2_traj'
        elif control_version == '5m2hz':
            self.sptial_grand = 'spatial_interval_5_traj'
            self.temporal_grand = 'temporal_interval_5_traj'
        else:
            raise NotImplementedError

    def draw_traj_bev(
        self,
        traj,
        raw_img,
        canvas_size=(1080, 1920),
        thickness=3,
        dot_size=5,
        is_ego=False,
        hue_start=120,
        hue_end=80,
        render_version='trajs',
        color=(0, 255, 0),
        markerType=cv2.MARKER_STAR,
        dashed=False,
    ):
        if is_ego:
            traj = np.concatenate([np.zeros((1, 2)), traj], axis=0)
            if self.bev_view != 'topdown':
                traj = traj[:, :10]

        line = traj.copy()
        if self.bev_view != 'topdown':
            line[:, 0] = traj[:, 1]
            line[:, 1] = traj[:, 0]

        img = raw_img.copy()
        pts_4d = np.stack(
            [line[:, 0], line[:, 1], np.zeros((line.shape[0])), np.ones((line.shape[0]))]
        )
        pts_2d = (self.coor2topdown @ pts_4d).T
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        mask = (
            (pts_2d[:, 0] > 0)
            & (pts_2d[:, 0] < canvas_size[1])
            & (pts_2d[:, 1] > 0)
            & (pts_2d[:, 1] < canvas_size[0])
        )
        if not mask.any():
            return img
        if render_version == 'points':

            pts_2d = pts_2d[mask, 0:2]

            for point in pts_2d:
                cv2.drawMarker(
                    img,
                    tuple(point.astype(int)),
                    color,
                    markerType=markerType,
                    markerSize=5,
                    thickness=5,
                )
        elif render_version == 'points_line':

            pts_2d = pts_2d[mask, 0:2]
            for i in range(len(pts_2d) - 1):
                if dashed:
                    cv2.line(
                        img,
                        tuple(pts_2d[i].astype(int)),
                        tuple(pts_2d[i + 1].astype(int)),
                        color,
                        self.traj_thickness,
                        cv2.LINE_AA,
                        5,
                    )
                else:
                    cv2.line(
                        img,
                        tuple(pts_2d[i].astype(int)),
                        tuple(pts_2d[i + 1].astype(int)),
                        color,
                        self.traj_thickness,
                    )

                cv2.circle(img, tuple(pts_2d[i].astype(int)), self.traj_dot_size, color, -1)

        elif render_version == 'trajs':
            pts_2d = pts_2d[mask, 0:2]
            try:
                tck, u = splprep([pts_2d[:, 0], pts_2d[:, 1]], s=0)
            except:
                return img
            unew = np.linspace(0, 1, 100)
            smoothed_pts = np.stack(splev(unew, tck)).astype(int).T

            num_points = len(smoothed_pts)
            for i in range(num_points - 1):
                hue = hue_start + (hue_end - hue_start) * (i / num_points)
                hsv_color = np.array([hue, 255, 255], dtype=np.uint8)
                rgb_color = cv2.cvtColor(
                    hsv_color[np.newaxis, np.newaxis, :], cv2.COLOR_HSV2RGB
                ).reshape(-1)
                rgb_color_tuple = (float(rgb_color[0]), float(rgb_color[1]), float(rgb_color[2]))
                if (
                    smoothed_pts[i, 0] > 0
                    and smoothed_pts[i, 0] < canvas_size[1]
                    and smoothed_pts[i, 1] > 0
                    and smoothed_pts[i, 1] < canvas_size[0]
                ):
                    cv2.line(
                        img,
                        (smoothed_pts[i, 0], smoothed_pts[i, 1]),
                        (smoothed_pts[i + 1, 0], smoothed_pts[i + 1, 1]),
                        color=rgb_color_tuple,
                        thickness=thickness,
                    )
                elif i == 0:
                    break
        return img

    def draw_lidar_bbox3d_on_img(
        self,
        bboxes3d,
        raw_img,
        lidar2img_rt,
        canvas_size=(900, 1600),
        img_metas=None,
        scores=None,
        labels=None,
        trajs=None,
        color=(0, 255, 0),
        thickness=1,
    ):
        img = raw_img.copy()
        bboxes3d_numpy = to_numpy(bboxes3d)
        if len(bboxes3d_numpy) == 0:
            return img

        corners_3d = box3d_to_corners(bboxes3d)

        if self.bev_view != 'topdown':
            new_corners_3d = np.zeros_like(corners_3d)
            new_corners_3d[:, :, 0] = corners_3d[:, :, 1]
            new_corners_3d[:, :, 1] = corners_3d[:, :, 0]
            new_corners_3d[:, :, 2] = corners_3d[:, :, 2]
            corners_3d = new_corners_3d

        num_bbox = corners_3d.shape[0]
        pts_4d = np.concatenate([corners_3d.reshape(-1, 3), np.ones((num_bbox * 8, 1))], axis=-1)
        lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
        if isinstance(lidar2img_rt, torch.Tensor):
            lidar2img_rt = lidar2img_rt.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        pts_2d = (lidar2img_rt @ pts_4d.T).T
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)
        depth = pts_2d[..., 2].reshape(num_bbox, 8)
        mask1 = (
            (imgfov_pts_2d[:, :, 0] > -1e5)
            & (imgfov_pts_2d[:, :, 0] < 1e5)
            & (imgfov_pts_2d[:, :, 1] > -1e5)
            & (imgfov_pts_2d[:, :, 1] < 1e5)
            & (depth > -1)
        ).all(-1)
        mask2 = (
            imgfov_pts_2d.reshape(num_bbox, 16).max(axis=-1)
            - imgfov_pts_2d.reshape(num_bbox, 16).min(axis=-1)
        ) < 2000
        mask = mask1 & mask2
        if scores is not None:

            mask = mask & (scores >= 0.3)
        if not mask.any():
            return img

        scores = scores[mask] if scores is not None else None
        labels = labels[mask] if labels is not None else None

        imgfov_pts_2d = imgfov_pts_2d[mask]

        num_bbox = mask.sum()
        if trajs is not None:

            trajs = trajs[mask]
            agent_boxes = bboxes3d_numpy[mask]
            for traj, agent_box, label in zip(trajs, agent_boxes, labels):
                if label in [0, 1, 2, 3, 7]:

                    traj1 = np.concatenate([agent_box[None, 0:2], traj.reshape(6, 2)], axis=0)
                    if canvas_size == (900, 1600):
                        img = self.draw_traj_bev(
                            traj1,
                            img,
                            hue_start=0,
                            hue_end=20,
                            render_version='points_line',
                            color=(255, 0, 0),
                        )
                    else:
                        img = self.draw_traj_bev(
                            traj1,
                            img,
                            hue_start=0,
                            hue_end=20,
                            render_version='trajs',
                            color=(255, 0, 0),
                        )

        return self.plot_rect3d_on_img(
            img,
            num_bbox,
            imgfov_pts_2d,
            scores,
            labels,
            color,
            thickness,
            bev=(canvas_size != (900, 1600)),
        )

    def plot_rect3d_on_img(
        self,
        img,
        num_rects,
        rect_corners,
        scores=None,
        labels=None,
        color=(0, 255, 0),
        thickness=1,
        bev=False,
    ):
        line_indices = (
            (0, 1),
            (0, 3),
            (0, 4),
            (1, 2),
            (1, 5),
            (3, 2),
            (3, 7),
            (4, 5),
            (4, 7),
            (2, 6),
            (5, 6),
            (6, 7),
        )
        if bev and self.bev_view == 'topdown':
            line_indices = ((0, 3), (3, 7), (4, 7), (0, 4))
        for i in range(num_rects):
            c = COLORMAP[labels[i]]
            thinck = 2
            corners = rect_corners[i].astype(np.int32)

            if scores[i] < 0.3:
                continue

            for start, end in line_indices:
                cv2.line(
                    img,
                    (corners[start, 0], corners[start, 1]),
                    (corners[end, 0], corners[end, 1]),
                    c,
                    thinck,
                    cv2.LINE_AA,
                )

            if self.mode == 'full':
                text = "{}".format(B2D_DETECTION_CLASS[labels[i]])
                cv2.putText(
                    img,
                    text,
                    (corners[0, 0], corners[0, 1]),
                    cv2.FONT_HERSHEY_PLAIN,
                    0.3,
                    (0, 0, 0),
                    1,
                )

        return img.astype(np.uint8)

    def draw_map_pred(self, img, result):
        render_num = min(10, len(result['vectors']))

        for i in range(render_num):

            color = COLOR_VECTORS[result['labels'][i]]
            color = COLOR_VECTORS_BGR[color]
            pts = result['vectors'][i]

            line = pts.copy()

            if self.bev_view != 'topdown':
                line[:, 0] = pts[:, 1]
                line[:, 1] = pts[:, 0]

            pts_4d = np.stack(
                [line[:, 0], line[:, 1], np.zeros((line.shape[0])), np.ones((line.shape[0]))]
            )
            pts_2d = (self.coor2topdown @ pts_4d).T
            pts_2d[:, 0] /= pts_2d[:, 2]
            pts_2d[:, 1] /= pts_2d[:, 2]

            x = pts_2d[:, 0]
            y = pts_2d[:, 1]

            for j in range(len(x) - 1):
                cv2.line(
                    img,
                    (int(x[j]), int(y[j])),
                    (int(x[j + 1]), int(y[j + 1])),
                    color,
                    self.map_line_width,
                )
        return img

    def put_information(self, image, meta, frame_index=None):
        """Overlay controller and model information."""
        commands = [
            'LEFT',
            'RIGHT',
            'STRAIGHT',
            'LANE FOLLOW',
            'CHANGE LANE LEFT',
            'CHANGE LANE RIGHT',
        ]
        command = int(meta.get('command', 3))
        command_text = commands[command] if 0 <= command < len(commands) else str(command)
        lines = [command_text]
        if self.mode == 'full':
            lines = [
                f"speed: {float(meta.get('speed', 0)):+.2f}, steer: {float(meta.get('steer', 0)):+.2f}",
                f"throttle: {float(meta.get('throttle', 0)):+.2f}, brake: {float(meta.get('brake', 0)):+.2f}",
                f"command: {command_text}",
                f"desired_speed: {float(meta.get('desired_speed', 0)):+.2f}",
            ]
            if meta.get('light_state') is not None:
                lines.append(f"light_state: {meta['light_state']}")
            if frame_index is not None:
                lines.append(f'frame: {frame_index}')
        for row_index, text in enumerate(lines):
            position = (self.text_position[0], self.text_position[1] + row_index * self.line_height)
            cv2.putText(
                image,
                text,
                position,
                cv2.FONT_HERSHEY_PLAIN,
                self.font_scale,
                self.text_color,
                self.text_width,
                cv2.LINE_AA,
            )
        return image

    def render_frames(
        self, frame_index, meta, image, images_folder, view_name='rgb_front', sp_waypoints_show=True
    ):
        """Render one saved image; the red path and blue timing sequence share coordinates."""
        canvas = cv2.imread(str(Path(images_folder) / view_name / image))
        if canvas is None:
            raise ValueError(f'Cannot read image: {Path(images_folder) / view_name / image}')
        result = meta.get('result', {})
        if self.mode != 'clean':
            canvas = self.put_information(canvas, meta, frame_index)
            if self.mode == 'full':
                for key, color in [('tp_near', (255, 0, 0)), ('tp_far', (0, 0, 255))]:
                    if key in meta.get('data', {}):
                        points = np.asarray(meta['data'][key]).reshape(-1, 2)
                        canvas = self.draw_traj_bev(
                            points, canvas, render_version='points', color=color
                        )
            if self.mode == 'full' or self.bev_view == 'topdown':
                if 'vectors' in result:
                    canvas = self.draw_map_pred(canvas, result)
                if 'boxes_3d' in result:
                    canvas = self.draw_lidar_bbox3d_on_img(
                        result['boxes_3d'],
                        canvas,
                        self.coor2topdown,
                        scores=result.get('scores_3d'),
                        labels=result.get('labels_3d'),
                        trajs=result.get('trajs_3d', result.get('traj')),
                        canvas_size=canvas.shape[:2],
                    )
        path = result.get('waypoints_drivepath')
        if sp_waypoints_show and path is not None:
            canvas = self.draw_traj_bev(
                np.asarray(path).reshape(-1, 2),
                canvas,
                is_ego=True,
                render_version='points_line',
                color=(0, 0, 255),
                canvas_size=canvas.shape[:2],
            )
            canvas = self.draw_traj_bev(
                np.asarray(meta['plan']).reshape(-1, 2),
                canvas,
                is_ego=True,
                render_version='points_line',
                color=(255, 0, 0),
                canvas_size=canvas.shape[:2],
            )
        else:
            canvas = self.draw_traj_bev(
                np.asarray(meta['plan']).reshape(-1, 2),
                canvas,
                is_ego=True,
                render_version='trajs',
                canvas_size=canvas.shape[:2],
            )
        return canvas

    def create_video(
        self,
        images_folder,
        fps,
        font_scale=1,
        text_color=(255, 255, 255),
        text_position=(50, 50),
        scene_name=None,
        view_name='rgb_front',
        sp_waypoints_show=True,
    ):
        """Render camera or BEV overlays through the common scene-to-video pipeline."""
        self.font_scale, self.text_color, self.text_position = font_scale, text_color, text_position
        return render_scene(
            images_folder,
            self.pred_dir,
            scene_name,
            fps,
            self.mode,
            self.bev_view,
            view_name,
            False,
            sp_waypoints_show,
            renderer=self,
        )


class SimulationRenderer:
    """Compose simulator BEV overlays and optional multi-camera perception panels."""

    def __init__(self, out_dir, render_version='full', bev_view='topdown'):
        self.out_dir = str(out_dir)
        self.render_version = render_version
        self.bev_view = bev_view
        self.pred_dir = str(Path(out_dir) / 'multimodal_pred')

    def create_video(
        self,
        images_folder,
        fps,
        font_scale=1,
        text_color=(255, 255, 255),
        text_position=(50, 50),
        scene_name=None,
        sp_waypoints_show=True,
        render_img=False,
        bev_view=None,
    ):
        return render_scene(
            images_folder,
            self.pred_dir,
            scene_name,
            fps,
            self.render_version,
            bev_view or self.bev_view,
            'bev',
            render_img,
            sp_waypoints_show,
            text_options=(font_scale, text_color, text_position),
        )

    def create_video_parallel(
        self,
        images_folder,
        fps,
        font_scale=1,
        text_color=(255, 255, 255),
        text_position=(50, 50),
        scene_name=None,
        sp_waypoints_show=True,
        render_img=False,
        bev_view=None,
        max_workers=8,
    ):
        return render_scene(
            images_folder,
            self.pred_dir,
            scene_name,
            fps,
            self.render_version,
            bev_view or self.bev_view,
            'bev',
            render_img,
            sp_waypoints_show,
            workers=max_workers,
            text_options=(font_scale, text_color, text_position),
        )


def collect_frames(scene_dir, view_name):
    """Match images to metadata by frame ID rather than their position in a sorted list."""
    scene_dir = Path(scene_dir)
    frames = []
    image_paths = (scene_dir / view_name).iterdir()
    ordered_paths = sorted(
        image_paths, key=lambda path: (0, int(path.stem)) if path.stem.isdigit() else (1, path.stem)
    )
    for image_path in ordered_paths:
        if image_path.suffix.lower() not in {'.png', '.jpg', '.jpeg'}:
            continue
        candidates = [scene_dir / 'meta' / f'{image_path.stem}.pkl']
        if image_path.stem.isdigit():
            candidates.append(scene_dir / 'meta' / f'{int(image_path.stem):04d}.pkl')
        meta_path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if meta_path is None:
            raise FileNotFoundError(f'Missing inference metadata for image: {image_path}')
        frames.append((image_path, meta_path))
    if not frames:
        raise ValueError(f'No images found in {scene_dir / view_name}')
    return frames


def render_frame(frame, scene_name, renderer, camera_renderer, sp_waypoints_show):
    """Render one matched frame with the same implementation for serial and parallel jobs."""
    image_path, meta_path = frame
    with meta_path.open('rb') as stream:
        meta = pickle.load(stream)
    meta = dict(meta, scene_id=scene_name)
    frame_index = int(image_path.stem) if image_path.stem.isdigit() else image_path.stem
    image = renderer.render_frames(
        frame_index,
        meta,
        image_path.name,
        str(image_path.parent.parent),
        view_name=image_path.parent.name,
        sp_waypoints_show=sp_waypoints_show,
    )
    if camera_renderer is not None:
        camera_data = dict(meta['data'])
        scene_dir = image_path.parent.parent
        camera_paths = []
        for saved_path in camera_data['img_filename']:
            saved_path = Path(saved_path)
            local_path = scene_dir / saved_path.parent.name / saved_path.name
            camera_paths.append(str(local_path if local_path.is_file() else saved_path))
        camera_data['img_filename'] = camera_paths
        camera_image = camera_renderer.render_frames(
            frame_index,
            camera_data,
            meta['result'],
            str(image_path.parent.parent),
            view_name='rgb',
            sp_waypoints_show=sp_waypoints_show,
        )
        if camera_image is None:
            raise ValueError(f'Missing camera images for frame: {image_path.stem}')
        camera_height = camera_image.shape[0]
        image = cv2.resize(
            image, (int(image.shape[1] * camera_height / image.shape[0]), camera_height)
        )
        image = np.concatenate((camera_image, image), axis=1)
    return image


def make_renderers(output_dir, version, bev_view, render_rgb, text_options):
    renderer = SimulationBEVRenderer(str(output_dir), version, bev_view)
    if text_options is not None:
        renderer.font_scale, renderer.text_color, renderer.text_position = text_options
    camera_renderer = None
    if render_rgb:
        from tools.visualization.cam_render import CamRenderWYH

        choices = dict(draw_pred=True, det=True, track=False, motion=True, map=False, planning=True)
        camera_renderer = CamRenderWYH(choices, str(output_dir), dataset_type='b2d', quiet=True)
    return renderer, camera_renderer


def render_worker(
    frame, scene_name, output_dir, version, bev_view, render_rgb, sp_waypoints_show, text_options
):
    renderer, camera_renderer = make_renderers(
        output_dir, version, bev_view, render_rgb, text_options
    )
    return render_frame(frame, scene_name, renderer, camera_renderer, sp_waypoints_show)


def render_scene(
    scene_dir,
    output_dir,
    scene_name,
    fps,
    version,
    bev_view,
    view_name,
    render_rgb,
    sp_waypoints_show,
    workers=1,
    renderer=None,
    text_options=None,
):
    """Stream ordered frame renders into one video without accumulating the whole scene in RAM."""
    if fps < 1 or workers < 1:
        raise ValueError('fps and workers must be positive')
    frames = collect_frames(scene_dir, view_name)
    scene_name = scene_name or Path(scene_dir).name
    destination = Path(output_dir) / f'video_{scene_name}.mp4'
    if workers == 1:
        default_renderer, camera_renderer = make_renderers(
            output_dir, version, bev_view, render_rgb, text_options
        )
        render = partial(
            render_frame,
            scene_name=scene_name,
            renderer=renderer or default_renderer,
            camera_renderer=camera_renderer,
            sp_waypoints_show=sp_waypoints_show,
        )
        return write_video(
            tqdm(map(render, frames), total=len(frames), desc='Rendering frames'), destination, fps
        )
    render = partial(
        render_worker,
        scene_name=scene_name,
        output_dir=output_dir,
        version=version,
        bev_view=bev_view,
        render_rgb=render_rgb,
        sp_waypoints_show=sp_waypoints_show,
        text_options=text_options,
    )
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context('spawn')
    ) as executor:

        def ordered_images():
            batch_size = workers * 2
            for start in range(0, len(frames), batch_size):
                yield from executor.map(render, frames[start : start + batch_size], chunksize=1)

        images = ordered_images()
        return write_video(
            tqdm(images, total=len(frames), desc='Rendering frames'), destination, fps
        )
