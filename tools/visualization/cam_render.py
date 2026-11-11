from tools.visualization.constants import prediction_color
from tools.visualization.geometry import prediction_modes
import os
import numpy as np
import cv2
from PIL import Image


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box as NuScenesBox
from nuscenes.utils.geometry_utils import view_points, box_in_image
import torch
from tools.visualization.constants import color_mapping, SCORE_THRESH
from tools.visualization.constants import B2D_DETECTION_CLASS

CAM_NAMES_NUSC = [
    'CAM_FRONT_LEFT',
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT',
    'CAM_BACK',
    'CAM_BACK_RIGHT',
]
CAM_NAMES_NUSC_converter = [
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_FRONT_LEFT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
]


class CamRenderWYH:
    def __init__(
        self,
        plot_choices,
        out_dir,
        **kwargs,
    ):
        self.plot_choices = plot_choices
        self.pred_dir = os.path.join(out_dir, "cam_pred")
        self.video_dir = os.path.join(out_dir, "video")
        self.out_dir = out_dir
        os.makedirs(self.pred_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)
        self.dataset_type = kwargs.get('dataset_type', 'nuscenes')
        if self.dataset_type == 'b2d':
            self.CMD_LIST = [
                'LEFT',
                'RIGHT',
                'STRAIGHT',
                'LANE_FOLLOW',
                'CHANGE_LANE_LEFT',
                'CHANGE_LANE_RIGHT',
            ]
            self.detect_class = B2D_DETECTION_CLASS
        else:
            self.CMD_LIST = ['right', 'left', 'straight']
            self.detect_class = None
        if 'image_wh' in kwargs:
            self.image_wh = kwargs['image_wh']
        else:
            self.image_wh = (1600, 900)

        self.light_state = ['unseen', 'red', 'unknown', 'green']

        quiet = kwargs.get('quiet', False)
        if not quiet:
            print("CamRender Save to: ", self.pred_dir)

    def set_pred_dir(self, pred_dir):
        self.pred_dir = os.path.join(self.out_dir, pred_dir)
        if not os.path.exists(self.pred_dir):
            os.makedirs(self.pred_dir, exist_ok=True)
            print("Make dir: ", self.pred_dir)

    def reset_pred_dir(self):
        self.pred_dir = os.path.join(self.out_dir, "cam_pred")

    def get_fig_as_array(self):

        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)

        self.fig.canvas.draw()
        buf = self.fig.canvas.tostring_rgb()
        width, height = self.fig.canvas.get_width_height()

        rgb_array = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

        return bgr_array

    def to_video(self, file_path=None):
        """Encode debug images with the shared browser-compatible video writer."""
        from tools.visualization.video import images_to_video

        return images_to_video(
            file_path or self.pred_dir, os.path.join(self.video_dir, 'video.mp4'), fps=5
        )

    def render_anchor(
        self,
        data,
        result,
        index,
        only_front=True,
        draw_all=False,
        stage_select=None,
    ):

        if only_front:

            anchors = result['anchors']
            scores = result['mm_score']
            set_cam = ["CAM_FRONT"] * len(anchors)

            self.reset_canvas()
            self.render_image_data(data, index, set_cam=set_cam)
            save_path = os.path.join(self.pred_dir, str(index).zfill(4) + '.jpg')

            self.draw_anchors_mutil_stage_front(data, anchors, scores, draw_all=draw_all)
            if 'stage_output' in result:
                for i in range(len(result['stage_output'])):
                    traj = result['stage_output'][i][0]['waypoints']
                    self.draw_planning_pred(data, traj, color=(1, 0.5, 0), j=i)

            self.draw_planning_pred(data, result['gt_traj'], color=(0, 0, 1), j=i)

            self.render_text(data, result, index)

            self.save_fig(save_path)

        else:
            if stage_select is None:
                stage_select = range(len(result['stage_output']))
            for i in stage_select:

                anchors = result['anchors']
                scores = result['mm_score']
                set_cam = [
                    "CAM_FRONT_LEFT",
                    "CAM_FRONT",
                    "CAM_FRONT_RIGHT",
                    "CAM_BACK_RIGHT",
                    "CAM_BACK",
                    "CAM_BACK_LEFT",
                ]
                self.reset_canvas()
                self.render_image_data(data, index, set_cam=None)
                save_path = os.path.join(
                    self.pred_dir, str(index).zfill(4) + '_stage_' + str(i) + '.jpg'
                )

                self.draw_anchors_mutil_cam(data, anchors[i], scores, draw_all=draw_all)
                traj = result['stage_output'][i][0]['waypoints']
                self.draw_planning_pred(data, traj, color=(1, 0.5, 0), j=1)
                self.render_text(data, result, index)
                self.save_fig(save_path)

        return save_path

    def project_points(self, key_points, projection_mat, image_wh=None):
        bs, num_anchor, num_pts = key_points.shape[:3]

        pts_extend = torch.cat([key_points, torch.ones_like(key_points[..., :1])], dim=-1)
        points_2d = torch.matmul(
            projection_mat[:, :, None, None], pts_extend[:, None, ..., None]
        ).squeeze(-1)
        points_2d = points_2d[..., :2] / torch.clamp(points_2d[..., 2:3], min=1e-5)
        if image_wh is not None:
            points_2d = points_2d / image_wh[:, :, None, None]
        return points_2d

    def draw_anchors_mutil_cam(self, data, anchors, scores, color=(0.5, 1, 0), draw_all=False):
        winner = np.argmax(scores)

        if draw_all:
            traj_points = anchors
        else:
            traj_points = anchors[winner][None, ...]
        if len(traj_points.shape) == 2:
            traj_points = traj_points[None, ...]

        num_anchor = traj_points.shape[0]
        num_pts = traj_points.shape[1]
        num_cam = 6
        projection_mat = torch.tensor(data['lidar2img'], dtype=torch.float32).unsqueeze(0).cpu()
        traj_points_3d = torch.tensor(traj_points, dtype=torch.float32).unsqueeze(0).cpu()
        traj_points_all = (
            self.project_points(traj_points_3d, projection_mat)
            .permute(0, 2, 3, 1, 4)
            .reshape(1, num_anchor, num_pts, num_cam, 2)[0]
            .numpy()
        )

        for j, cam in enumerate(CAM_NAMES_NUSC):

            idx = CAM_NAMES_NUSC_converter.index(cam)
            traj_points = traj_points_all[:, :, idx].reshape(-1, 2)

            valid_mask = (
                (traj_points[:, 0] > 0)
                & (traj_points[:, 0] < 1600)
                & (traj_points[:, 1] > 0)
                & (traj_points[:, 1] < 900)
            )
            traj_points = traj_points[valid_mask]
            if len(traj_points) > 0:

                original_xlim = self.axes[j // self.col_row[1], j % self.col_row[1]].get_xlim()
                original_ylim = self.axes[j // self.col_row[1], j % self.col_row[1]].get_ylim()
                self.axes[j // self.col_row[1], j % self.col_row[1]].scatter(
                    traj_points[:, 0], traj_points[:, 1], color=color, s=150
                )
                self.axes[j // self.col_row[1], j % self.col_row[1]].set_xlim(original_xlim)
                self.axes[j // self.col_row[1], j % self.col_row[1]].set_ylim(original_ylim)
                self.axes[j // self.col_row[1], j % self.col_row[1]].set_aspect('equal')

    def draw_anchors_mutil_stage_front(
        self, data, anchors, scores, color=(0.5, 1, 0), draw_all=False
    ):

        idx = 0
        cam_intrinsic = data['cam_intrinsic'][idx]
        lidar2cam = data['lidar2cam']
        extrinsic = lidar2cam[idx]
        trans = extrinsic[3, :3]

        winner = np.argmax(scores)
        for j in range(len(anchors)):

            if draw_all:
                traj_points = anchors[j].reshape(-1, 3)
            else:
                traj_points = anchors[j][winner].reshape(-1, 3)

            traj_points = traj_points @ extrinsic[:3, :3] + trans
            traj_points = view_points(traj_points.T, cam_intrinsic, normalize=True)[:2, :]

            valid_mask = (
                (traj_points[0] > 0)
                & (traj_points[0] < 1600)
                & (traj_points[1] > 0)
                & (traj_points[1] < 900)
            )
            traj_points = traj_points[:, valid_mask]

            original_xlim = self.axes[j // self.col_row[1], j % self.col_row[1]].get_xlim()
            original_ylim = self.axes[j // self.col_row[1], j % self.col_row[1]].get_ylim()
            self.axes[j // self.col_row[1], j % self.col_row[1]].scatter(
                traj_points[0], traj_points[1], color=color, s=150
            )
            self.axes[j // self.col_row[1], j % self.col_row[1]].set_xlim(original_xlim)
            self.axes[j // self.col_row[1], j % self.col_row[1]].set_ylim(original_ylim)
            self.axes[j // self.col_row[1], j % self.col_row[1]].set_aspect('equal')

    def reset_canvas(self, size=None):
        plt.close()
        plt.gca().set_axis_off()
        plt.axis('off')
        if size is None:
            self.fig, self.axes = plt.subplots(2, 3, figsize=(160 / 3, 20))
            self.col_row = (2, 3)
        else:
            self.fig, self.axes = plt.subplots(
                size[0], size[1], figsize=(16, 12), constrained_layout=True
            )
            self.col_row = size

        plt.tight_layout()

    def render(
        self,
        data,
        result={},
        index=0,
        only_plan=False,
    ):
        self.reset_canvas()
        self.render_image_data(data, index)
        save_path = os.path.join(self.pred_dir, str(index).zfill(4) + '.jpg')

        self.draw_detection_pred(data, result)
        self.draw_motion_pred(data, result)
        if 'pred_traj' in result:
            pred_traj = result['pred_traj']
            self.draw_planning_pred(data, pred_traj, color=(0.5, 1, 0))
        if 'gt_traj' in result:
            gt_traj = result['gt_traj']
            self.draw_planning_pred(data, gt_traj, color=(1, 0.5, 0))
        self.render_text(data, result, index)
        self.save_fig(save_path)
        return save_path

    def render_gt(
        self,
        data,
        index=0,
    ):
        self.reset_canvas()
        self.render_image_data(data, index)
        save_path = os.path.join(self.pred_dir, str(index).zfill(4) + '.jpg')
        self.draw_detection_gt(data)

        self.save_fig(save_path)
        return save_path

    def render_text(self, data, result, index):

        cmd = np.argmax(data['gt_ego_fut_cmd'])

        cmd = self.CMD_LIST[cmd]

        self.axes[0, 1].text(50, 150, f'cmd: {cmd}', fontsize=50, color='black')

        if 'light_state' in result:
            light_state = result['light_state']
            if isinstance(light_state, torch.Tensor):
                light_state = light_state.cpu().numpy()
            light_state = self.light_state[light_state[0]]
            self.axes[0, 1].text(50, 250, f'light: {light_state}', fontsize=50, color='black')

        if 'gt_traj' not in result or 'pred_traj' not in result:
            return
        gt_traj = result['gt_traj']
        pred_traj = result['pred_traj']
        gt_traj = gt_traj[:, :2]
        pred_traj = pred_traj[:, :2]

    def load_image(self, data_path, cam):
        """Update the axis of the plot with the provided image."""
        image = np.array(Image.open(data_path))

        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 60)
        fontScale = 2
        color = (0, 0, 0)
        thickness = 4
        return cv2.putText(image, cam, org, font, fontScale, color, thickness, cv2.LINE_AA)

    def update_image(self, image, index, cam):
        """Render image data for each camera."""
        ax = self.get_axis(index)
        ax.imshow(image)
        plt.axis('off')
        ax.axis('off')
        ax.grid(False)

    def get_axis(self, index):
        """Retrieve the corresponding axis based on the index."""
        row = index // self.col_row[1]
        col = index % self.col_row[1]
        return self.axes[row, col]

    def save_fig(self, filename):
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(filename)

    def render_image_data(self, data, index, set_cam=None):
        """Load and annotate image based on the provided path."""
        if set_cam is None:
            set_cam = CAM_NAMES_NUSC
        for i, cam in enumerate(set_cam):
            idx = CAM_NAMES_NUSC_converter.index(cam)
            img_path = data['img_filename'][idx]
            if not os.path.exists(img_path):
                return False
            image = self.load_image(img_path, cam)
            self.update_image(image, i, cam)
        return True

    def draw_detection_pred(self, data, result):

        if not self.plot_choices.get('det', True) or 'boxes_3d' not in result:
            return

        bboxes = result['boxes_3d']
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.cpu().numpy()
        for j, cam in enumerate(CAM_NAMES_NUSC):
            idx = CAM_NAMES_NUSC_converter.index(cam)
            cam_intrinsic = data['cam_intrinsic'][idx]
            lidar2cam = data['lidar2cam']
            extrinsic = lidar2cam[idx]
            trans = extrinsic[3, :3]
            rot = Quaternion(matrix=extrinsic[:3, :3]).inverse
            imsize = (1600, 900)

            for i in range(result['labels_3d'].shape[0]):
                score = result['scores_3d'][i]
                if score < SCORE_THRESH:
                    continue
                color = prediction_color(result, i)
                object_class = result['labels_3d'][i]
                object_class = self.detect_class[object_class]
                center = bboxes[i, 0:3]
                box_dims = bboxes[i, 3:6]
                nusc_dims = box_dims[..., [1, 0, 2]]
                quat = Quaternion(axis=[0, 0, 1], radians=bboxes[i, 6])
                box = NuScenesBox(center, nusc_dims, quat)
                box.rotate(rot)
                box.translate(trans)
                if box_in_image(box, cam_intrinsic, imsize):
                    box.render(
                        self.axes[j // 3, j % 3],
                        view=cam_intrinsic,
                        normalize=True,
                        colors=(color, color, color),
                        linewidth=4,
                    )

                text_pos = view_points(np.array([box.center]).T, cam_intrinsic, normalize=True)[
                    :2, :
                ]
                text_pos = text_pos[:, 0]

                if (
                    text_pos[0] <= 0
                    or text_pos[0] >= imsize[0]
                    or text_pos[1] <= 0
                    or text_pos[1] >= imsize[1]
                ):
                    continue

                text_pos[0] = min(max(text_pos[0], 0), imsize[0])
                text_pos[1] = min(max(text_pos[1], 0), imsize[1])

            self.axes[j // 3, j % 3].set_xlim(0, imsize[0])
            self.axes[j // 3, j % 3].set_ylim(imsize[1], 0)

    def draw_detection_gt(self, data):

        bboxes = data['gt_bboxes_3d']
        for j, cam in enumerate(CAM_NAMES_NUSC):
            idx = CAM_NAMES_NUSC_converter.index(cam)
            cam_intrinsic = data['cam_intrinsic'][idx]
            lidar2cam = data['lidar2cam']
            extrinsic = lidar2cam[idx]
            trans = extrinsic[3, :3]
            rot = Quaternion(matrix=extrinsic[:3, :3]).inverse
            imsize = (1600, 900)

            for i in range(data['gt_labels_3d'].shape[0]):
                color = color_mapping[data['instance_inds'][i] % len(color_mapping)]
                object_class = data['gt_labels_3d'][i]
                object_class = self.detect_class[object_class]
                center = bboxes[i, 0:3]
                box_dims = bboxes[i, 3:6]
                nusc_dims = box_dims[..., [1, 0, 2]]
                quat = Quaternion(axis=[0, 0, 1], radians=bboxes[i, 6])
                box = NuScenesBox(center, nusc_dims, quat)
                box.rotate(rot)
                box.translate(trans)
                if box_in_image(box, cam_intrinsic, imsize):
                    box.render(
                        self.axes[j // 3, j % 3],
                        view=cam_intrinsic,
                        normalize=True,
                        colors=(color, color, color),
                        linewidth=4,
                    )

                text_pos = view_points(np.array([box.center]).T, cam_intrinsic, normalize=True)[
                    :2, :
                ]
                text_pos = text_pos[:, 0]

                if (
                    text_pos[0] <= 0
                    or text_pos[0] >= imsize[0]
                    or text_pos[1] <= 0
                    or text_pos[1] >= imsize[1]
                ):
                    continue

                text_pos[0] = min(max(text_pos[0], 0), imsize[0])
                text_pos[1] = min(max(text_pos[1], 0), imsize[1])

                self.axes[j // 3, j % 3].text(
                    text_pos[0], text_pos[1], object_class, fontsize=20, color=color
                )

            self.axes[j // 3, j % 3].set_xlim(0, imsize[0])
            self.axes[j // 3, j % 3].set_ylim(imsize[1], 0)

    def draw_motion_pred(self, data, result, points_per_step=10):
        if not (
            self.plot_choices['draw_pred'] and self.plot_choices['motion'] and "trajs_3d" in result
        ):
            return

        bboxes = result['boxes_3d']
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.cpu().numpy()

        for j, cam in enumerate(CAM_NAMES_NUSC):
            idx = CAM_NAMES_NUSC_converter.index(cam)
            cam_intrinsic = data['cam_intrinsic'][idx]
            lidar2cam = data['lidar2cam']
            extrinsic = lidar2cam[idx]
            trans = extrinsic[3, :3]
            rot = Quaternion(matrix=extrinsic[:3, :3]).inverse
            imsize = (1600, 900)

            for i in range(result['labels_3d'].shape[0]):
                score = result['scores_3d'][i]
                if score < SCORE_THRESH:
                    continue
                color = prediction_color(result, i)

                modes, mode_scores = prediction_modes(result, i)
                traj = modes[np.argmax(mode_scores)]

                origin = bboxes[i, :2][None]
                traj = np.concatenate([origin, traj], axis=0)
                traj_expand = np.ones((traj.shape[0], 1))
                traj_expand[:] = bboxes[i, 2] - bboxes[i, 5] / 2
                traj = np.concatenate([traj, traj_expand], axis=1)

                center = bboxes[i, 0:3]
                box_dims = bboxes[i, 3:6]
                nusc_dims = box_dims[..., [1, 0, 2]]
                quat = Quaternion(axis=[0, 0, 1], radians=bboxes[i, 6])
                box = NuScenesBox(center, nusc_dims, quat)
                box.rotate(rot)
                box.translate(trans)
                if not box_in_image(box, cam_intrinsic, imsize):
                    continue
                traj_points = traj @ extrinsic[:3, :3] + trans
                self._render_traj(traj_points, cam_intrinsic, j, color=color, s=15)

    def draw_planning_pred(self, data, traj, color=(1, 0.5, 0), j=1):

        idx = 0
        cam_intrinsic = data['cam_intrinsic'][idx]
        lidar2cam = data['lidar2cam']
        extrinsic = lidar2cam[idx]
        trans = extrinsic[3, :3]

        plan_traj = traj
        plan_traj = np.concatenate((np.zeros((1, 2)), plan_traj), axis=0)
        traj_expand = np.ones((plan_traj.shape[0], 1)) * -1.8
        plan_traj = np.concatenate([plan_traj, traj_expand], axis=1)

        traj_points = plan_traj @ extrinsic[:3, :3] + trans
        self._render_traj(traj_points, cam_intrinsic, j=j, color=color)

    def _render_traj(
        self, traj_points, cam_intrinsic, j, color=(1, 0.5, 0), s=150, points_per_step=10
    ):
        total_steps = (len(traj_points) - 1) * points_per_step + 1
        total_xy = np.zeros((total_steps, 3))
        for k in range(total_steps - 1):
            unit_vec = traj_points[k // points_per_step + 1] - traj_points[k // points_per_step]
            total_xy[k] = (k / points_per_step - k // points_per_step) * unit_vec + traj_points[
                k // points_per_step
            ]
        in_range_mask = total_xy[:, 2] > 0.1
        traj_points = view_points(total_xy.T, cam_intrinsic, normalize=True)[:2, :]
        traj_points = traj_points[:2, in_range_mask]

        valid_mask = (
            (traj_points[0] > 0)
            & (traj_points[0] < 1600)
            & (traj_points[1] > 0)
            & (traj_points[1] < 900)
        )
        traj_points = traj_points[:, valid_mask]

        original_xlim = self.axes[j // self.col_row[1], j % self.col_row[1]].get_xlim()
        original_ylim = self.axes[j // self.col_row[1], j % self.col_row[1]].get_ylim()
        self.axes[j // self.col_row[1], j % self.col_row[1]].scatter(
            traj_points[0], traj_points[1], color=color, s=s
        )
        self.axes[j // self.col_row[1], j % self.col_row[1]].set_xlim(original_xlim)
        self.axes[j // self.col_row[1], j % self.col_row[1]].set_ylim(original_ylim)
        self.axes[j // self.col_row[1], j % self.col_row[1]].set_aspect('equal')

    def _render_anchors(self, traj_points, cam_intrinsic, j, color=(1, 0.5, 0), s=150):
        total_xy = traj_points
        in_range_mask = total_xy[:, 2] > 0.1
        traj_points = view_points(total_xy.T, cam_intrinsic, normalize=True)[:2, :]
        traj_points = traj_points[:2, in_range_mask]

        valid_mask = (
            (traj_points[0] > 0)
            & (traj_points[0] < 1600)
            & (traj_points[1] > 0)
            & (traj_points[1] < 900)
        )
        traj_points = traj_points[:, valid_mask]

        original_xlim = self.axes[j // self.col_row[1], j % self.col_row[1]].get_xlim()
        original_ylim = self.axes[j // self.col_row[1], j % self.col_row[1]].get_ylim()
        self.axes[j // self.col_row[1], j % self.col_row[1]].scatter(
            traj_points[0], traj_points[1], color=color, s=s
        )
        self.axes[j // self.col_row[1], j % self.col_row[1]].set_xlim(original_xlim)
        self.axes[j // self.col_row[1], j % self.col_row[1]].set_ylim(original_ylim)
        self.axes[j // self.col_row[1], j % self.col_row[1]].set_aspect('equal')

    def render_frames(
        self, index, data, result, images_folder, view_name='rgb_front', sp_waypoints_show=False
    ):

        self.reset_canvas()
        exist_img = self.render_image_data(data, index)
        if not exist_img:
            return None

        self.draw_detection_pred(data, result)
        self.draw_motion_pred(data, result)
        if 'pred_traj' in result:
            pred_traj = result['pred_traj']
            self.draw_planning_pred(data, pred_traj, color=(0.5, 1, 0))
        if 'waypoints_drivepath' in result:
            waypoints_dirvepath = result['waypoints_drivepath']
            self.draw_planning_pred(data, waypoints_dirvepath, color=(1, 0, 0))
        if 'waypoints' in result:
            waypoints = result['waypoints']
            self.draw_planning_pred(data, waypoints, color=(0, 0, 1))
        if 'gt_traj' in result:
            gt_traj = result['gt_traj']
            self.draw_planning_pred(data, gt_traj, color=(1, 0.5, 0))
        self.render_text(data, result, index)

        img = self.get_fig_as_array()
        return img
