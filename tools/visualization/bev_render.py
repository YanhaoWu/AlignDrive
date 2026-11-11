from tools.visualization.constants import prediction_color
from tools.visualization.geometry import prediction_modes
import os
import numpy as np
import cv2


import matplotlib
import matplotlib.pyplot as plt


from tools.visualization.geometry import box3d_to_corners
from tools.visualization.constants import (
    CMD_LIST,
    CMD_LIST_B2D,
    GRAND_LIST,
    COLOR_VECTORS,
    SCORE_THRESH,
    MAP_SCORE_THRESH,
    B2D_DETECTION_CLASS,
    color_mapping,
)


class BEVRender:
    """Render perception ground truth and predictions on a debug BEV canvas."""

    def __init__(self, plot_choices, out_dir, xlim=40, ylim=40, dataset_type='nuscene'):
        self.plot_choices = plot_choices
        self.xlim = xlim
        self.ylim = ylim
        self.gt_dir = os.path.join(out_dir, "bev_gt")
        self.pred_dir = os.path.join(out_dir, "bev_pred")
        os.makedirs(self.gt_dir, exist_ok=True)
        os.makedirs(self.pred_dir, exist_ok=True)
        self.dataset_type = dataset_type

    def reset_canvas(self, size=(20, 20)):
        plt.close()
        self.fig, self.axes = plt.subplots(1, 1, figsize=(size[0], size[1]))
        self.axes.set_xlim(-self.xlim, self.xlim)
        self.axes.set_ylim(-self.ylim, self.ylim)
        self.axes.axis('off')

    def render(
        self,
        data,
        result,
        index,
    ):
        self.reset_canvas()
        self.draw_detection_gt(data)
        self.draw_motion_gt(data)
        self.draw_map_gt(data)
        self.draw_planning_gt(data)
        self._render_sdc_car()
        self._render_command(data)
        self._render_legend()
        save_path_gt = os.path.join(self.gt_dir, str(index).zfill(4) + '.jpg')
        self.save_fig(save_path_gt)

        self.reset_canvas()
        self.draw_detection_pred(result)
        self.draw_track_pred(result)
        self.draw_motion_pred(result)
        self.draw_map_pred(result)
        self.draw_planning_pred(data, result)
        self._render_sdc_car()
        self._render_command(data)
        self._render_legend()
        save_path_pred = os.path.join(self.pred_dir, str(index).zfill(4) + '.jpg')
        self.save_fig(save_path_pred)

        return save_path_gt, save_path_pred

    def save_fig(self, filename):
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(filename)

    def draw_detection_gt(self, data):
        if not self.plot_choices['det']:
            return

        for i in range(data['gt_labels_3d'].shape[0]):
            label = data['gt_labels_3d'][i]
            if label == -1:
                continue
            color = color_mapping[i % len(color_mapping)]

            corners = box3d_to_corners(data['gt_bboxes_3d'])[i, [0, 3, 7, 4, 0]]
            x = corners[:, 0]
            y = corners[:, 1]
            self.axes.plot(x, y, color=color, linewidth=3, linestyle='-')

            forward_center = np.mean(corners[2:4], axis=0)
            center = np.mean(corners[0:4], axis=0)
            x = [forward_center[0], center[0]]
            y = [forward_center[1], center[1]]
            self.axes.plot(x, y, color=color, linewidth=3, linestyle='-')

    def draw_detection_pred(self, result):
        if not (
            self.plot_choices['draw_pred'] and self.plot_choices['det'] and "boxes_3d" in result
        ):
            return

        bboxes = result['boxes_3d']
        for i in range(result['labels_3d'].shape[0]):
            score = result['scores_3d'][i]
            if score < SCORE_THRESH:
                continue
            color = prediction_color(result, i)

            corners = box3d_to_corners(bboxes)[i, [0, 3, 7, 4, 0]]
            x = corners[:, 0]
            y = corners[:, 1]
            self.axes.plot(x, y, color=color, linewidth=3, linestyle='-')

            forward_center = np.mean(corners[2:4], axis=0)
            center = np.mean(corners[0:4], axis=0)
            x = [forward_center[0], center[0]]
            y = [forward_center[1], center[1]]
            self.axes.plot(x, y, color=color, linewidth=3, linestyle='-')

    def draw_track_pred(self, result):
        if not (
            self.plot_choices['draw_pred']
            and self.plot_choices['track']
            and "anchor_queue" in result
        ):
            return

        temp_bboxes = result["anchor_queue"]
        period = result["period"]
        bboxes = result['boxes_3d']
        for i in range(result['labels_3d'].shape[0]):
            score = result['scores_3d'][i]
            if score < SCORE_THRESH:
                continue
            color = prediction_color(result, i)
            center = bboxes[i, :3]
            centers = [center]
            for j in range(period[i]):

                corners = box3d_to_corners(temp_bboxes[:, -1 - j])[i, [0, 3, 7, 4, 0]]
                x = corners[:, 0]
                y = corners[:, 1]
                self.axes.plot(x, y, color=color, linewidth=2, linestyle='-')

                forward_center = np.mean(corners[2:4], axis=0)
                center = np.mean(corners[0:4], axis=0)
                x = [forward_center[0], center[0]]
                y = [forward_center[1], center[1]]
                self.axes.plot(x, y, color=color, linewidth=2, linestyle='-')
                centers.append(center)

            centers = np.stack(centers)
            xs = centers[:, 0]
            ys = centers[:, 1]
            self.axes.plot(xs, ys, color=color, linewidth=2, linestyle='-')

    def draw_motion_gt(self, data):
        if not self.plot_choices['motion']:
            return

        for i in range(data['gt_labels_3d'].shape[0]):
            label = data['gt_labels_3d'][i]
            if label == -1:
                continue

            vehicle_id_list = [0, 1, 2, 3, 4, 6, 7]
            if label in vehicle_id_list:
                dot_size = 150
            else:
                dot_size = 25

            center = data['gt_bboxes_3d'][i, :2]
            masks = data['gt_agent_fut_masks'][i].astype(bool)
            if masks[0] == 0:
                continue
            trajs = data['gt_agent_fut_trajs'][i][masks]
            trajs = trajs.cumsum(axis=0) + center
            trajs = np.concatenate([center.reshape(1, 2), trajs], axis=0)

            self._render_traj(trajs, traj_score=1.0, colormap='winter', dot_size=dot_size)

    def draw_motion_pred(self, result, top_k=3):
        if not (
            self.plot_choices['draw_pred'] and self.plot_choices['motion'] and "trajs_3d" in result
        ):
            return

        bboxes = result['boxes_3d']
        labels = result['labels_3d']
        for i in range(result['labels_3d'].shape[0]):
            score = result['scores_3d'][i]
            if score < SCORE_THRESH:
                continue
            label = labels[i]
            vehicle_id_list = [0, 1, 2, 3, 4, 6, 7]
            if label in vehicle_id_list:
                dot_size = 150
            else:
                dot_size = 25

            traj, traj_score = prediction_modes(result, i)
            num_modes = len(traj_score)
            center = bboxes[i, :2][None, None].repeat(num_modes, 1, 1).numpy()
            traj = np.concatenate([center, traj], axis=1)

            sorted_ind = np.argsort(traj_score)[::-1]
            sorted_traj = traj[sorted_ind, :, :2]
            sorted_score = traj_score[sorted_ind]
            norm_score = np.exp(sorted_score[0])

            for j in range(min(top_k, len(sorted_score)) - 1, -1, -1):
                viz_traj = sorted_traj[j]
                traj_score = np.exp(sorted_score[j]) / norm_score
                self._render_traj(
                    viz_traj, traj_score=traj_score, colormap='winter', dot_size=dot_size
                )

    def draw_map_gt(self, data):
        if not self.plot_choices['map']:
            return
        vectors = data['map_infos']
        for label, vector_list in vectors.items():
            color = COLOR_VECTORS[label]
            for vector in vector_list:
                pts = vector[:, :2]
                x = np.array([pt[0] for pt in pts])
                y = np.array([pt[1] for pt in pts])
                self.axes.plot(
                    x, y, color=color, linewidth=3, marker='o', linestyle='-', markersize=7
                )

    def draw_map_pred(self, result):
        if not (
            self.plot_choices['draw_pred'] and self.plot_choices['map'] and "vectors" in result
        ):
            return

        for i in range(result['scores'].shape[0]):
            score = result['scores'][i]
            if score < MAP_SCORE_THRESH:
                continue
            color = COLOR_VECTORS[result['labels'][i]]
            pts = result['vectors'][i]
            x = pts[:, 0]
            y = pts[:, 1]
            plt.plot(x, y, color=color, linewidth=3, marker='o', linestyle='-', markersize=7)

    def draw_planning_gt(self, data):
        if not self.plot_choices['planning']:
            return

        masks = data['gt_ego_fut_masks'].astype(bool)
        if masks[0] != 0:
            plan_traj = data['gt_ego_fut_trajs'][masks]

            plan_traj[abs(plan_traj) < 0.01] = 0.0
            plan_traj = plan_traj.cumsum(axis=0)
            plan_traj = np.concatenate((np.zeros((1, plan_traj.shape[1])), plan_traj), axis=0)
            self._render_traj(plan_traj, traj_score=1.0, colormap='autumn', dot_size=50)

    def draw_planning_pred(self, data, result, top_k=3):
        if not (
            self.plot_choices['draw_pred']
            and self.plot_choices['planning']
            and "planning" in result
        ):
            return

        if self.plot_choices['track'] and "ego_anchor_queue" in result:
            ego_temp_bboxes = result["ego_anchor_queue"]
            ego_period = result["ego_period"]
            for j in range(ego_period[0]):

                corners = box3d_to_corners(ego_temp_bboxes[:, -1 - j])[0, [0, 3, 7, 4, 0]]
                x = corners[:, 0]
                y = corners[:, 1]
                self.axes.plot(x, y, color='mediumseagreen', linewidth=2, linestyle='-')

                forward_center = np.mean(corners[2:4], axis=0)
                center = np.mean(corners[0:4], axis=0)
                x = [forward_center[0], center[0]]
                y = [forward_center[1], center[1]]
                self.axes.plot(x, y, color='mediumseagreen', linewidth=2, linestyle='-')

        plan_trajs = result['planning'].cpu().numpy()
        num_cmd = len(CMD_LIST)
        num_mode = plan_trajs.shape[1]
        plan_trajs = np.concatenate((np.zeros((num_cmd, num_mode, 1, 2)), plan_trajs), axis=2)
        plan_score = result['planning_score'].cpu().numpy()

        cmd = data['gt_ego_fut_cmd'].argmax()
        plan_trajs = plan_trajs[cmd]
        plan_score = plan_score[cmd]

        sorted_ind = np.argsort(plan_score)[::-1]
        sorted_traj = plan_trajs[sorted_ind, :, :2]
        sorted_score = plan_score[sorted_ind]
        norm_score = np.exp(sorted_score[0])

        for j in range(min(top_k, len(sorted_score)) - 1, -1, -1):
            viz_traj = sorted_traj[j]
            traj_score = np.exp(sorted_score[j]) / norm_score
            self._render_traj(viz_traj, traj_score=traj_score, colormap='autumn', dot_size=50)

    def _render_traj(
        self, future_traj, traj_score=1, colormap='winter', points_per_step=20, dot_size=25
    ):
        total_steps = (len(future_traj) - 1) * points_per_step + 1
        dot_colors = matplotlib.colormaps[colormap](np.linspace(0, 1, total_steps))[:, :3]
        dot_colors = dot_colors * traj_score + (1 - traj_score) * np.ones_like(dot_colors)
        total_xy = np.zeros((total_steps, 2))
        for i in range(total_steps - 1):
            unit_vec = future_traj[i // points_per_step + 1] - future_traj[i // points_per_step]
            total_xy[i] = (i / points_per_step - i // points_per_step) * unit_vec + future_traj[
                i // points_per_step
            ]
        total_xy[-1] = future_traj[-1]
        self.axes.scatter(total_xy[:, 0], total_xy[:, 1], c=dot_colors, s=dot_size)

    def _render_sdc_car(self):
        sdc_car_png = cv2.imread('resources/sdc_car.png')
        sdc_car_png = cv2.cvtColor(sdc_car_png, cv2.COLOR_BGR2RGB)
        im = self.axes.imshow(sdc_car_png, extent=(-1, 1, -2, 2))
        im.set_zorder(2)

    def _render_legend(self):
        legend = cv2.imread('resources/legend.png')
        legend = cv2.cvtColor(legend, cv2.COLOR_BGR2RGB)
        self.axes.imshow(legend, extent=(15, 40, -40, -30))

    def _render_command(self, data):
        cmd = data['gt_ego_fut_cmd'].argmax()
        self.axes.text(-38, -38, CMD_LIST[cmd], fontsize=60)


class BEVRenderWYH(BEVRender):
    """Extend the perception canvas with multimodal planning diagnostics."""

    def __init__(
        self, plot_choices, out_dir, xlim=40, ylim=40, mutil_mode_num=6, dataset_type='nuscene'
    ):
        self.plot_choices = plot_choices
        self.xlim = xlim
        self.ylim = ylim
        self.out_dir = out_dir
        self.gt_dir = os.path.join(out_dir, "bev_gt")
        self.pred_dir = os.path.join(out_dir, "bev_pred")
        self.merge_dir = os.path.join(out_dir, "bev_merge")
        self.video_dir = os.path.join(out_dir, "video")
        self.tem_dir = os.path.join(out_dir, "bev_temp")
        self.mutil_mode_num = mutil_mode_num
        os.makedirs(self.gt_dir, exist_ok=True)
        os.makedirs(self.pred_dir, exist_ok=True)
        os.makedirs(self.merge_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)
        os.makedirs(self.tem_dir, exist_ok=True)
        self.dataset_type = dataset_type

        self.text_color = (255, 255, 255)
        self.text_position = (50, 50)
        self.font_scale = 0.5
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

        if self.dataset_type == 'b2d':
            self.command_list = CMD_LIST_B2D
            self.detect_class = B2D_DETECTION_CLASS
        else:
            self.command_list = CMD_LIST
            self.detect_class = None

        print("BEV render Path: ", self.pred_dir)

    def reset_pred_dir(self):
        self.pred_dir = os.path.join(self.out_dir, "bev_pred")

    def set_pred_dir(self, pred_dir):
        self.pred_dir = os.path.join(self.out_dir, pred_dir)
        if not os.path.exists(self.pred_dir):
            os.makedirs(self.pred_dir, exist_ok=True)
            print("Make dir: ", self.pred_dir)
        else:
            print("BEV render Path: ", self.pred_dir)

    def to_video(self, file_path=None):
        """Encode debug images with the shared browser-compatible video writer."""
        from tools.visualization.video import images_to_video

        return images_to_video(
            file_path or self.merge_dir, os.path.join(self.video_dir, 'video.mp4'), fps=5
        )

    def _render_text(self, data, result, scene_index, text='GT'):

        if 'pred_traj' in result:
            gt_traj = result['gt_traj']
            pred_traj = result['pred_traj']
            gt_traj = gt_traj[:, :2]
            pred_traj = pred_traj[:, :2]
            distance = np.linalg.norm(gt_traj - pred_traj, axis=1)

            avg_distance = (distance[1] + distance[3] + distance[5]) / 3
            avg_distance = round(avg_distance, 2)
            if 'mm_waypoints' in result and text != 'GT':

                mm_waypoints = result['mm_waypoints']
                score = result['mm_score']
                if 'voca_winner' in result:
                    voca_gt_index = result['voca_winner']
                else:
                    voca_gt_index = None
                mm_waypoints = mm_waypoints[:, :, :2]
                distance = np.linalg.norm(mm_waypoints - gt_traj[None], axis=2)
                mm_avg_disatnce = (distance[:, 1] + distance[:, 3] + distance[:, 5]) / 3

                sort_index = np.argsort(mm_avg_disatnce)
                sorted_mm_avg_disatnce = mm_avg_disatnce[sort_index]

                score = score[sort_index]

                score = (score - np.min(score)) / (np.max(score) - np.min(score))

                min_distance_index = sort_index[0] + 1

                min_score = score[0]
                min_score = round(min_score, 2)

                mm_distance = sorted_mm_avg_disatnce[0]

                if voca_gt_index is not None:
                    mm_distance_voca = mm_avg_disatnce[voca_gt_index]
                else:
                    mm_distance_voca = 0

                self.axes.text(
                    -38, 32, f'min_distance: {mm_distance:.2f}', fontsize=40, color='black'
                )
                self.axes.text(
                    -38, 29, f'corres_score: {min_score:.2f}', fontsize=40, color='black'
                )
                self.axes.text(-38, 26, f'index: {min_distance_index}', fontsize=40, color='black')
                self.axes.text(
                    -38, 23, f'voca_gt_dis: {mm_distance_voca:.2f}', fontsize=40, color='black'
                )
                self.axes.text(
                    -38, 38, f'avg_distance: {avg_distance:.2f}', fontsize=40, color='black'
                )

        if 'ego_status' in result:
            ego_status = result['status'].flatten()
            self.axes.text(
                -38,
                20,
                f'Acc: ({ego_status[0]:.2f}, {ego_status[1]:.2f}, {ego_status[2]:.2f})',
                fontsize=40,
                color='black',
            )
            self.axes.text(
                -38,
                17,
                f'Rota: ({ego_status[3]:.2f}, {ego_status[4]:.2f}, {ego_status[5]:.2f})',
                fontsize=40,
                color='black',
            )
            self.axes.text(
                -38,
                14,
                f'Vel: ({ego_status[6]:.2f}, {ego_status[7]:.2f}, {ego_status[8]:.2f})',
                fontsize=40,
                color='black',
            )
            self.axes.text(-38, 11, f'Dire: {ego_status[-1]:.2f}', fontsize=40, color='black')

        ego_status = data['ego_status']
        self.axes.text(
            -38,
            8,
            f'GT_Acc: ({ego_status[0]:.2f}, {ego_status[1]:.2f}, {ego_status[2]:.2f})',
            fontsize=40,
            color='black',
        )
        self.axes.text(
            -38,
            5,
            f'GT_Rota: ({ego_status[3]:.2f}, {ego_status[4]:.2f}, {ego_status[5]:.2f})',
            fontsize=40,
            color='black',
        )
        self.axes.text(
            -38,
            2,
            f'GT_Vel: ({ego_status[6]:.2f}, {ego_status[7]:.2f}, {ego_status[8]:.2f})',
            fontsize=40,
            color='black',
        )
        self.axes.text(-38, -1, f'GT_Dire: {ego_status[-1]:.2f}', fontsize=40, color='black')

        self.axes.text(-38, -4, f'scene_index: {scene_index}', fontsize=40, color='black')

        self.axes.text(-38, 35, text, fontsize=40, color='black')

    def render(
        self,
        data,
        result={},
        index=0,
    ):

        self.reset_canvas()
        self.draw_detection_gt(data)
        self.draw_motion_gt(data)
        self.draw_map_gt(data)
        pred_traj = result.get('pred_traj', None)
        if pred_traj is not None:
            self.draw_single_planning_pred(result)
        else:
            self.draw_planning_gt(data)
        self._render_sdc_car()
        self._render_command(data)
        self._render_legend()
        if pred_traj is not None:
            self._render_text(data, result, index, text='Pred')
        else:
            self._render_text(data, result, index, text='GT')
        save_path_pred = os.path.join(self.pred_dir, str(index).zfill(4) + '.jpg')

        if 'tp_near' in data:
            self.draw_point_on_axes(data['tp_near'].reshape(-1, 2), color='red')
        if 'tp_far' in data:
            self.draw_point_on_axes(data['tp_far'].reshape(-1, 2), color='blue')
        self.save_fig(save_path_pred)

        self.reset_canvas(size=(40, 20))
        if 'mm_waypoints' in result:
            self.draw_mutil_planning_pred_on_axis(result, data)

        save_path_gt = os.path.join(self.gt_dir, str(index).zfill(4) + '.jpg')
        self.save_fig(save_path_gt)

        if pred_traj is not None:
            self.combine(save_path_pred, save_path_gt, index)
        return save_path_gt

    def combine(self, bev_gt_path, bev_pred_path, index):
        bev_gt = cv2.imread(bev_gt_path)
        bev_image = cv2.imread(bev_pred_path)
        merge_image = cv2.hconcat([bev_gt, bev_image])
        save_path = os.path.join(self.merge_dir, str(index).zfill(4) + '.jpg')
        cv2.imwrite(save_path, merge_image)

    def draw_point_on_axes(self, points, color='red', type='scatter', marker='*', size=500):
        """Draw (N, 2) points as scatter markers or a connected line on the debug canvas."""
        if type == 'scatter':

            self.axes.scatter(points[:, 0], points[:, 1], color=color, marker=marker, s=size)
        elif type == 'line':

            self.axes.plot(
                points[:, 0],
                points[:, 1],
                color=color,
                marker='o',
                markersize=10,
                linestyle='-',
                linewidth=2,
            )
        else:
            self.axes.plot(points[:, 0], points[:, 1], color=color)

    def draw_planning_gt(self, data):
        if not self.plot_choices['planning']:
            return

        masks = data['gt_ego_fut_masks'].astype(bool)
        if masks[0] != 0:
            plan_traj = data['gt_ego_fut_trajs'][masks]

            plan_traj[abs(plan_traj) < 0.01] = 0.0
            plan_traj = plan_traj.cumsum(axis=0)
            plan_traj = np.concatenate((np.zeros((1, plan_traj.shape[1])), plan_traj), axis=0)
            self._render_traj(plan_traj, traj_score=1.0, colormap='cividis', dot_size=50)

    def draw_single_planning_pred(self, data):
        if not self.plot_choices['planning']:
            return

        plan_traj = data['pred_traj']
        plan_traj[abs(plan_traj) < 0.01] = 0.0

        plan_traj = np.concatenate((np.zeros((1, plan_traj.shape[1])), plan_traj), axis=0)
        self._render_traj(plan_traj, traj_score=1.0, colormap='autumn', dot_size=50)

    def draw_mutil_planning_pred_on_axis(
        self, result, data, keys=["mm_waypoints", "mm_score", "voca_winner"]
    ):
        if not self.plot_choices['planning']:
            return

        mutil_plan_traj = result[keys[0]].copy()
        scores = result[keys[1]].copy()

        plan_score = np.exp(scores) / np.sum(np.exp(scores))

        sorted_ind = np.arange(len(mutil_plan_traj))
        max_score_ind = np.argmax(plan_score)
        if keys[2] in result:
            voca_winner = result[keys[2]]
        else:
            voca_winner = max_score_ind

        mutil_plan_traj = mutil_plan_traj[sorted_ind]
        plan_score = plan_score[sorted_ind]

        num_mode_each_cmd = len(mutil_plan_traj) // len(self.command_list)
        n_rows = 3
        ncols = len(mutil_plan_traj) // n_rows + 1
        if ncols <= 1:
            gs = self.fig.add_gridspec(nrows=ncols, ncols=n_rows, hspace=0.1)
        else:
            gs = self.fig.add_gridspec(nrows=n_rows, ncols=ncols, hspace=0.1)

        collision_matrix = None
        if 'collision_matrix' in result:
            collision_matrix = result['collision_matrix']
            if len(collision_matrix.shape) == 3:
                collision_matrix = collision_matrix[0]

        for i, (traj, score) in enumerate(zip(mutil_plan_traj, plan_score)):

            ax = self.fig.add_subplot(gs[i])

            traj[abs(traj) < 0.01] = 0.0
            traj = np.concatenate((np.zeros((1, 2)), traj), axis=0)

            ax.plot(
                traj[:, 0],
                traj[:, 1],
                color='blue',
                linewidth=2,
                linestyle='-',
                zorder=2,
            )

            masks = data['gt_ego_fut_masks'].astype(bool)
            if masks[0] != 0:

                plan_traj = data['gt_ego_fut_trajs'][masks]

                cmd = data['gt_ego_fut_cmd']
                plan_traj[abs(plan_traj) < 0.01] = 0.0
                plan_traj = plan_traj.cumsum(axis=0)
                plan_traj = np.concatenate((np.zeros((1, plan_traj.shape[1])), plan_traj), axis=0)

                ax.plot(
                    plan_traj[:, 0],
                    plan_traj[:, 1],
                    color='red',
                    linewidth=2,
                    linestyle='-',
                    zorder=2,
                )

            index = sorted_ind[i]

            cmd = index // num_mode_each_cmd
            cmd_res = index % num_mode_each_cmd

            if cmd >= len(self.command_list):
                cmd_name = 'STOP'
            else:
                cmd_name = self.command_list[cmd] + '_' + GRAND_LIST[cmd_res]

            ax.text(-10, 35, cmd_name, fontsize=20, color='black')

            bar_len = np.clip(20 * score, 0, 20)
            ax.barh(37, bar_len, height=1, left=-10, color='red')

            if index == max_score_ind:
                ax.text(-10 + bar_len, 35, '*', fontsize=50, color='black')
            elif index == voca_winner:
                ax.text(-10 + bar_len, 35, 'v', fontsize=50, color='black')

            ax.set_xlim(-10, 10)
            ax.set_ylim(0, self.ylim)
            ax.set_title(f"Traj {i+1}  Score: {score:.2f}", fontsize=20, y=0.95)
            ax.tick_params(axis='both', labelsize=8)

            if collision_matrix is not None:
                collision_index = collision_matrix[i]

                for j in range(collision_index.shape[0]):
                    if collision_index[j]:

                        color = 'blue'

                        corners = box3d_to_corners(data['gt_bboxes_3d'])[j, [0, 3, 7, 4, 0]]
                        x = corners[:, 0]
                        y = corners[:, 1]
                        ax.plot(x, y, color=color, linewidth=3, linestyle='-')

    def draw_mutil_planning_pred(self, data):
        if not self.plot_choices['planning']:
            return

        mutil_plan_traj = data['mm_waypoints'].copy()
        scores = data['mm_score'].copy()

        max_score = np.max(scores)
        min_score = np.min(scores)
        plan_score = (scores - min_score) / (max_score - min_score)

        plan_score = plan_score * 0.5 + 0.5
        plan_score = np.clip(plan_score, 0.5, 1)

        sorted_ind = np.argsort(plan_score)[::-1]
        mutil_plan_traj = mutil_plan_traj[sorted_ind]
        plan_score = plan_score[sorted_ind]

        selected_traj = mutil_plan_traj[0]
        selected_score = plan_score[0]
        selected_traj[abs(selected_traj) < 0.01] = 0.0

        selected_traj = np.concatenate(
            (np.zeros((1, selected_traj.shape[1])), selected_traj), axis=0
        )
        self._render_traj(selected_traj, traj_score=selected_score, colormap='autumn', dot_size=50)

        mutil_plan_traj = mutil_plan_traj[: self.mutil_mode_num + 1]
        plan_score = plan_score[: self.mutil_mode_num + 1]

        for i in range(1, mutil_plan_traj.shape[0]):
            plan_traj = mutil_plan_traj[i]
            score = plan_score[i]
            plan_traj[abs(plan_traj) < 0.01] = 0.0

            plan_traj = np.concatenate((np.zeros((1, plan_traj.shape[1])), plan_traj), axis=0)
            self._render_traj_wyh(
                plan_traj, traj_score=score, colormap='autumn', dot_size=50, idx=i
            )

    def _render_traj_wyh(self, traj, traj_score, colormap, dot_size, idx=0):
        cmap = plt.get_cmap(colormap)
        color = cmap(traj_score)

        markers = ['*', 's', 'o', 'D', 'P', 'v']
        marker_idx = idx
        marker = markers[marker_idx % len(markers)]

        line_style = '-' if traj_score > 0.5 else '--'
        line_alpha = 0.6 + 0.4 * traj_score
        line_width = 1 + 2 * traj_score

        plt.plot(
            traj[:, 0],
            traj[:, 1],
            color=color,
            linestyle=line_style,
            linewidth=line_width,
            alpha=line_alpha,
            zorder=10,
        )

        if idx == 0:

            dx = traj[-1, 0] - traj[-2, 0]
            dy = traj[-1, 1] - traj[-2, 1]
            plt.arrow(
                traj[-2, 0],
                traj[-2, 1],
                dx,
                dy,
                color='gold',
                head_width=0.5,
                head_length=0.7,
                width=0.2,
                zorder=12,
            )

        plt.scatter(
            traj[::3, 0],
            traj[::3, 1],
            color='white' if idx == 0 else color,
            marker=marker,
            s=50 + 100 * traj_score if idx == 0 else 30 + 70 * traj_score,
            edgecolors='black' if idx == 0 else color,
            linewidths=1.5 if idx == 0 else 0.5,
            zorder=11,
        )

    def draw_sptial_traj(self, traj, valid_mask, data, route_name, index):

        if data is not None:
            data['gt_ego_fut_trajs'] = traj
            data['gt_ego_fut_masks'] = valid_mask.astype(np.bool_)

            self.reset_canvas(size=(40, 20))
            self.reset_canvas()
            self.draw_detection_gt(data)
            self.draw_motion_gt(data)
            self.draw_map_gt(data)
            self.draw_planning_gt(data)
            self._render_text(data, {}, index, text='GT')
            self._render_sdc_car()
            self._render_command(data)
            self._render_legend()
            save_path = os.path.join(self.pred_dir, route_name + '.jpg')
            self.save_fig(save_path)
        else:
            self.reset_canvas(size=(40, 20))
            save_path = os.path.join(self.pred_dir, route_name + '.jpg')
            self.draw_sptial_traj_on_axis(traj, valid_mask, route_name, save_path)

    def draw_sptial_traj_on_axis(self, traj, valid_mask, route_name, save_path):
        """Render sampled trajectory points and distinguish valid samples from padding."""

        plt.figure(figsize=(8, 6))
        ax = plt.gca()

        ax.plot(traj[:, 0], traj[:, 1], color='blue', linewidth=2, linestyle='-', zorder=1)

        valid_mask = valid_mask.astype(np.bool_)
        valid_points = traj[valid_mask]
        filled_points = traj[~valid_mask]

        if len(valid_points) > 0:
            ax.scatter(
                valid_points[:, 0],
                valid_points[:, 1],
                color='limegreen',
                marker='o',
                s=60,
                edgecolor='black',
                linewidth=1,
                zorder=2,
                label='Valid Points',
            )

        if len(filled_points) > 0:
            ax.scatter(
                filled_points[:, 0],
                filled_points[:, 1],
                color='none',
                marker='o',
                s=60,
                edgecolor='red',
                linewidth=1.5,
                linestyle='--',
                zorder=3,
                label='Filled Points',
            )

        ax.scatter(
            traj[0, 0],
            traj[0, 1],
            color='blue',
            marker='*',
            s=150,
            edgecolor='gold',
            linewidth=1,
            zorder=4,
            label='Start Point',
        )

        ax.axis('on')
        ax.tick_params(axis='both', labelsize=12)
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)

        ax.set_xlim(-5, 5)
        ax.set_ylim(0, 20)

        ax.set_xticks([-5, -2.5, 0, 2.5, 5])
        ax.set_yticks(range(0, int(self.ylim) + 1, 50))

        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

        plt.title(f"Trajectory: {route_name}", fontsize=14)
        plt.tight_layout()

        plt.savefig(save_path)
        plt.close()

    def _render_command(self, data):
        cmd = data['gt_ego_fut_cmd'].argmax()
        if self.dataset_type == 'b2d':
            self.axes.text(-38, -38, CMD_LIST_B2D[cmd], fontsize=60)
        else:
            self.axes.text(-38, -38, CMD_LIST[cmd], fontsize=60)
