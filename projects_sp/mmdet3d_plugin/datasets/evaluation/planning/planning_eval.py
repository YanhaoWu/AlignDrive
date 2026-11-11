from tqdm import tqdm
import warnings
import torch
import torch.nn as nn
import numpy as np
from shapely.geometry import Polygon

from mmcv.utils import print_log
from mmdet.datasets import build_dataset, build_dataloader

from projects_sp.mmdet3d_plugin.datasets.utils import box3d_to_corners


def check_collision(ego_box, boxes):
    '''
        ego_box: tensor with shape [7], [x, y, z, w, l, h, yaw]
        boxes: tensor with shape [N, 7]
    '''
    if  boxes.shape[0] == 0:
        return False

    # follow uniad, add a 0.5m offset
    ego_box[0] += 0.5 * torch.cos(ego_box[6])
    ego_box[1] += 0.5 * torch.sin(ego_box[6])
    ego_corners_box = box3d_to_corners(ego_box.unsqueeze(0))[0, [0, 3, 7, 4], :2]
    corners_box = box3d_to_corners(boxes)[:, [0, 3, 7, 4], :2]
    ego_poly = Polygon([(point[0], point[1]) for point in ego_corners_box])
    for i in range(len(corners_box)):
        box_poly =  Polygon([(point[0], point[1]) for point in corners_box[i]])
        collision = ego_poly.intersects(box_poly)
        if collision:
            return True

    return False

def get_yaw(traj):
    start = traj[0]
    end = traj[-1]
    dist = torch.linalg.norm(end - start, dim=-1)
    if dist < 0.5:
        return traj.new_ones(traj.shape[0]) * np.pi / 2

    zeros = traj.new_zeros((1, 2))
    traj_cat = torch.cat([zeros, traj], dim=0)
    yaw = traj.new_zeros(traj.shape[0]+1)
    yaw[..., 1:-1] = torch.atan2(
        traj_cat[..., 2:, 1] - traj_cat[..., :-2, 1],
        traj_cat[..., 2:, 0] - traj_cat[..., :-2, 0],
    )
    yaw[..., -1] = torch.atan2(
        traj_cat[..., -1, 1] - traj_cat[..., -2, 1],
        traj_cat[..., -1, 0] - traj_cat[..., -2, 0],
    )
    return yaw[1:]

class PlanningMetric():
    def __init__(
        self,
        n_future=6,
        compute_on_step: bool = False,
    ):
        self.W = 1.85
        self.H = 4.084

        self.n_future = n_future
        self.reset()

    def reset(self):
        self.obj_col = torch.zeros(self.n_future)
        self.obj_box_col = torch.zeros(self.n_future)
        self.L2 = torch.zeros(self.n_future)
        self.Consist = torch.zeros(self.n_future)
        self.total = torch.tensor(0)

    def evaluate_single_coll(self, traj, fut_boxes):
        n_future = traj.shape[0]
        yaw = get_yaw(traj)
        ego_box = traj.new_zeros((n_future, 7))
        ego_box[:, :2] = traj
        ego_box[:, 3:6] = ego_box.new_tensor([self.H, self.W, 1.56])
        ego_box[:, 6] = yaw
        collision = torch.zeros(n_future, dtype=torch.bool)

        for t in range(n_future):
            ego_box_t = ego_box[t].clone()
            boxes = fut_boxes[t][0].clone()
            collision[t] = check_collision(ego_box_t, boxes)
        return collision

    def evaluate_coll(self, trajs, gt_trajs, fut_boxes):
        B, n_future, _ = trajs.shape

        obj_coll_sum = torch.zeros(n_future, device=trajs.device)
        obj_box_coll_sum = torch.zeros(n_future, device=trajs.device)

        assert B == 1, 'only supprt bs=1'
        for i in range(B):
            gt_box_coll = self.evaluate_single_coll(gt_trajs[i], fut_boxes)
            box_coll = self.evaluate_single_coll(trajs[i], fut_boxes)
            box_coll = torch.logical_and(box_coll, torch.logical_not(gt_box_coll))
            
            obj_coll_sum += gt_box_coll.long()
            obj_box_coll_sum += box_coll.long()

        return obj_coll_sum, obj_box_coll_sum

    def compute_L2(self, trajs, gt_trajs, gt_trajs_mask):
        '''
        trajs: torch.Tensor (B, n_future, 3)
        gt_trajs: torch.Tensor (B, n_future, 3)
        '''
        return torch.sqrt((((trajs[:, :, :2] - gt_trajs[:, :, :2]) ** 2) * gt_trajs_mask).sum(dim=-1)) 

    def compute_Consist(self, trajs, last_final_planning, gt_trajs, gt_trajs_mask):
        '''
        trajs: torch.Tensor (B, n_future, 3)
        gt_trajs: torch.Tensor (B, n_future, 3)
        '''

        return torch.sqrt(((((trajs[:, :, :2] - last_final_planning[:, :, :2])) ** 2) * gt_trajs_mask).sum(dim=-1)) 
    def update(self, trajs, gt_trajs, gt_trajs_mask, fut_boxes, last_final_planning):
        """
        trajs  pred_final_planning
        gt_trajs  gt_ego_fut_trajs
        """
        assert trajs.shape == gt_trajs.shape
        L2 = self.compute_L2(trajs, gt_trajs, gt_trajs_mask)
        if last_final_planning is not None:
            Consist = self.compute_Consist(trajs, last_final_planning,gt_trajs, gt_trajs_mask)
            self.Consist += Consist.sum(dim=0)

        if fut_boxes is not None:
            obj_coll_sum, obj_box_coll_sum = self.evaluate_coll(trajs[:,:,:2], gt_trajs[:,:,:2], fut_boxes)
        else:
            obj_coll_sum = 0
            obj_box_coll_sum = 0
        self.obj_col += obj_coll_sum
        self.obj_box_col += obj_box_coll_sum
        self.L2 += L2.sum(dim=0)
        self.total +=len(trajs)

    def compute(self):
        return {
            'obj_col': self.obj_col / self.total,
            'obj_box_col': self.obj_box_col / self.total,
            'L2' : self.L2 / self.total,
            'Consist':self.Consist / self.total
        }



def write_to_table(planning_results, prefix='par', metric_dict=None, timestep=0.5,
                   grand_names=None):

    if metric_dict is None:
        metric_dict = {}

    from prettytable import PrettyTable
    planning_tab = PrettyTable()
    field_names = ["metrics"] + [
        f"{step * timestep:.1f}s" for step in range(1, 7)
    ] + ["avg"]
    if grand_names is not None:
        field_names.insert(1, "grand")
    planning_tab.field_names = field_names
    for key in planning_results.keys():
        value = planning_results[key].tolist()
        new_values = []
        for i in range(len(value)):
            new_values.append(np.array(value[:i+1]).mean())
        value = new_values
        avg = [value[1], value[3], value[5]]
        avg = sum(avg) / len(avg)
        value.append(avg)
        metric_name = key if prefix is None else key + '_' + prefix
        metric_dict[metric_name] = avg
        row_value = []
        row_value.append(key)
        if grand_names is not None:
            row_value.append(grand_names[key])
        for i in range(len(value)):
            if 'col' in key:
                row_value.append('%.3f' % float(value[i]*100) + '%')
            else:
                row_value.append('%.4f' % float(value[i]))
        planning_tab.add_row(row_value)
    return planning_tab, metric_dict



def find_pretext(results):
    """Find each decoded single-trajectory branch exactly once."""
    if not results:
        return []
    return [
        key[:-len('_waypoints')]
        for key in results[0]['img_bbox']
        if key.endswith('_waypoints')
        and key != 'mm_waypoints'
        and not key.endswith('_mm_waypoints')
    ]


def resample_planning_trajectory(waypoints, target_times, timestep=None):
    """Interpolate relative positions at GT timestamps, never extrapolating."""
    positions = np.asarray(waypoints, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] < 2 or not np.isfinite(positions).all():
        raise ValueError('Planning waypoints must be a finite (T, 2+) position sequence')
    if timestep is None:
        legacy_timesteps = {6: 0.5, 15: 0.2}
        if len(positions) not in legacy_timesteps:
            raise ValueError('Missing waypoints_timestep for an unknown trajectory layout')
        timestep = legacy_timesteps[len(positions)]
        warnings.warn(
            f'Legacy planning output has no waypoints_timestep; assuming '
            f'{timestep}s for its {len(positions)} points. Regenerate predictions '
            'to record the sampling interval explicitly.',
            RuntimeWarning,
        )
    timestep = float(timestep)
    if not np.isfinite(timestep) or timestep <= 0:
        raise ValueError('waypoints_timestep must be finite and positive')
    target_times = np.asarray(target_times, dtype=np.float64)
    if (target_times < 0).any() or (target_times > len(positions) * timestep + 1e-6).any():
        raise ValueError('Planning prediction does not cover the requested evaluation horizon')
    source_times = np.arange(len(positions) + 1) * timestep
    positions = np.concatenate([np.zeros((1, 2)), positions[:, :2]], axis=0)
    aligned = np.stack([
        np.interp(target_times, source_times, positions[:, axis]) for axis in range(2)
    ], axis=-1)
    return torch.tensor(aligned, dtype=torch.float32).unsqueeze(0)



def planning_eval(results, eval_config, logger):
    dataset = build_dataset(eval_config)
    dataloader = build_dataloader(
            dataset, samples_per_gpu=1, workers_per_gpu=1, shuffle=False, dist=False)
        
    first_output = results[0]['img_bbox'] if results else {}
    selected_head = first_output.get(
        'planning_head', 'longitudinal' if 'longitudinal' in find_pretext(results) else 'base'
    )
    if selected_head not in ('base', 'longitudinal'):
        raise ValueError(f'Unsupported final planning head: {selected_head}')
    prediction_keys = {
        'spatial': 'base_waypoints' if 'base_waypoints' in first_output else 'waypoints'
    }
    if selected_head == 'longitudinal':
        if 'base_waypoints' not in first_output:
            raise ValueError('Longitudinal evaluation requires base_waypoints for Spatial L2')
        prediction_keys['final'] = 'waypoints'
    grand_names = {
        name: first_output.get(key + '_grand', 'unknown (legacy)')
        for name, key in prediction_keys.items()
    }
    all_metric = {name: PlanningMetric() for name in prediction_keys}
    timestep = eval_config.get('sample_interval', 5) / 10.0
    target_times = np.arange(1, 7) * timestep

    for i, data in enumerate(tqdm(dataloader)):
        if i >= len(results):
            print("WARNING: results are less than the dataset size")
            break

        sdc_planning = data['gt_ego_fut_trajs'].cumsum(dim=-2).unsqueeze(1)
        sdc_planning_mask = data['gt_ego_fut_masks'].unsqueeze(-1).repeat(1, 1, 2).unsqueeze(1)
        res = results[i]
        if not sdc_planning_mask.all():
            continue
        for name, prediction_key in prediction_keys.items():
            output = res['img_bbox']
            if output.get('planning_head', selected_head) != selected_head:
                raise ValueError('Cannot mix Spatial and Longitudinal evaluation outputs')
            if output.get(prediction_key + '_grand', grand_names[name]) != grand_names[name]:
                raise ValueError('Cannot mix different planning grands in one metric')
            pred_sdc_traj = resample_planning_trajectory(
                output[prediction_key], target_times,
                timestep=output.get(prediction_key + '_timestep'),
            )
            all_metric[name].update(
                pred_sdc_traj, sdc_planning[0, :, :6, :2],
                sdc_planning_mask[0, :, :6, :2], None, None,
            )

       
    if all_metric['spatial'].total.item() == 0:
        warnings.warn(
            'Planning metrics are unavailable: no sample has complete future '
            'ground truth. Use longer contiguous routes for evaluation.',
            RuntimeWarning,
        )
        return {}

    planning_results = {f'L2_{name}': metric.compute()['L2']
                        for name, metric in all_metric.items()}
    planning_tab, metric_dict = write_to_table(
        planning_results, prefix=None, timestep=timestep,
        grand_names={f'L2_{name}': grand for name, grand in grand_names.items()},
    )
    if logger is not None:
        print_log('\n' + str(planning_tab), logger=logger)
    else:
        print('\n' + str(planning_tab))
    return metric_dict
