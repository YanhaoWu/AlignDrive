import os
import json
import datetime
import pathlib
import time

import math
from scipy.optimize import fsolve
from pyquaternion import Quaternion

from PIL import Image
import cv2
import numpy as np
import torch
import pickle

import carla
from team_code.pid_controller import PIDController
from team_code.planner import RoutePlanner
from leaderboard.autoagents import autonomous_agent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.parallel.collate import collate as  mm_collate_to_batch_form
from mmdet.models import build_detector
from mmdet.datasets.pipelines import Compose


IS_BENCH2DRIVE = os.environ.get('IS_BENCH2DRIVE', None)
CAMERAS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
frame_rate = 10
resize_scale = 1
save_interval = 1


lefthand_ego_to_lidar = np.array([[ 0, 1, 0, 0],
                                  [ 1, 0, 0, 0],
                                  [ 0, 0, 1, 0],
                                  [ 0, 0, 0, 1]])

left2right = np.eye(4)
left2right[1,1] = -1


def get_entry_point():
    return 'AlignDriveAgent'

class Clock():
    def __init__(self):
        self.time =  time.time()
        self.verbose = False

    def count(self, tag):
        if self.verbose:
            prev_time = self.time
            self.time = time.time()
            print(tag, self.time - prev_time)
        else:
            pass
    

class AlignDriveAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file, gpu_rank=[0], save_name='SpareDrive_WYH_Unified_VIS', agent_id=0, scene_id=0, pid_param=None):
        self.save_name = save_name
        self.track = autonomous_agent.Track.SENSORS
        self.steer_step = 0
        self.last_moving_status = 0
        self.last_moving_step = -1
        self.last_steer = 0
        self.pidcontroller = PIDController(pid_param) 
        self.config_path = path_to_conf_file.split('+')[0]
        self.ckpt_path = path_to_conf_file.split('+')[1]
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False
        self.agent_id = agent_id
        self.scene_id = scene_id
        cfg = Config.fromfile(self.config_path)
        self.cfg = cfg
        if hasattr(cfg, "plugin"):
            if cfg.plugin:
                import importlib

                if hasattr(cfg, "plugin_dir"):
                    plugin_dir = cfg.plugin_dir
                    _module_dir = os.path.dirname(plugin_dir)
                    _module_dir = _module_dir.split("/")
                    _module_path = _module_dir[0]

                    for m in _module_dir[1:]:
                        _module_path = _module_path + "." + m
                    print(_module_path)
                    plg_lib = importlib.import_module(_module_path)
                else:
                    # import dir is the dirpath for the config file
                    _module_dir = os.path.dirname(args.config)
                    _module_dir = _module_dir.split("/")
                    _module_path = _module_dir[0]
                    for m in _module_dir[1:]:
                        _module_path = _module_path + "." + m
                    print(_module_path)
                    plg_lib = importlib.import_module(_module_path)
  
        self.gpu_rank = gpu_rank
        control_setting = cfg.get('control_setting', None)
        if control_setting is not None:
            self.control_setting = control_setting
        else:
            self.control_setting = {       
                "spatial": "spatial_interval_5_traj",
                "temporal": "temporal_interval_5_traj",
                "valid_waypoints": 2,# 预测了几秒的有效轨迹
            }
        self.valid_waypoints = self.control_setting['valid_waypoints']
        print("Valid waypoints:", self.valid_waypoints)
        model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        checkpoint = load_checkpoint(model, self.ckpt_path, map_location='cpu', strict=cfg.get('strict_checkpoint', False))
        self.require_longitudinal_output = cfg.get('require_longitudinal_output', False)
        if self.require_longitudinal_output:
            assert model.head.motion_plan_head.final_used_head == 'longitudinal'
            print('LONGITUDINAL_CONTRACT checkpoint=' + self.ckpt_path + ' final_used_head=longitudinal strict_checkpoint=True', flush=True)
        print("SUCCESSFULLY LOAD CHECKPOINT from :", self.ckpt_path)
        self.model = MMDataParallel(model, device_ids=gpu_rank)
        self.device = next(self.model.module.parameters()).device
        self.model.eval()
        self.test_pipeline = []
        for test_pipeline in cfg.test_pipeline:
            if test_pipeline["type"] not in ['LoadMultiViewImageFromFilesInCeph','LoadMultiViewImageFromFiles']:
                self.test_pipeline.append(test_pipeline)
        self.test_pipeline = Compose(self.test_pipeline)
        self.data_aug_conf = cfg.data_aug_conf

        self.takeover = False
        self.stop_time = 0
        self.takeover_time = 0
        self.save_path = None
        self.lat_ref, self.lon_ref = 42.0, 2.0

        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0	
        self.prev_control = control
        self.prev_control_cache = []

        self.save_path = pathlib.Path(f'{save_name}')
        print("The visulization will be saved in :", self.save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        (self.save_path / 'rgb_front' ).mkdir(exist_ok=True)
        (self.save_path / 'rgb_front_right').mkdir(exist_ok=True)
        (self.save_path / 'rgb_front_left').mkdir(exist_ok=True)
        (self.save_path / 'rgb_back').mkdir(exist_ok=True)
        (self.save_path / 'rgb_back_right').mkdir(exist_ok=True)
        (self.save_path / 'rgb_back_left').mkdir(exist_ok=True)
        (self.save_path / 'meta').mkdir(exist_ok=True)
        (self.save_path / 'bev').mkdir(exist_ok=True)
   
        self.lidar2cam = {
        'CAM_FRONT':np.array([[ 1.  ,  0.  ,  0.  ,  0.  ],
                                [ 0.  ,  0.  ,  1.  ,  0.  ],
                                [ 0.  , -1.  ,  0.  ,  0.  ],
                                [ 0.  , -0.24, -1.19,  1.  ]]),
        'CAM_FRONT_RIGHT':np.array([[ 0.57357644,  0.        ,  0.81915204,  0.        ],
                                    [-0.81915204,  0.        ,  0.57357644,  0.        ],
                                    [ 0.        , -1.        ,  0.        ,  0.        ],
                                    [ 0.22517331, -0.24      , -0.82909407,  1.        ]]),
        'CAM_FRONT_LEFT':np.array([[ 0.57357644,  0.        , -0.81915204,  0.        ],
                                    [ 0.81915204,  0.        ,  0.57357644,  0.        ],
                                    [ 0.        , -1.        ,  0.        ,  0.        ],
                                    [-0.22517331, -0.24      , -0.82909407,  1.        ]]),
        'CAM_BACK':np.array([[-1.00000000e+00,  0.00000000e+00,  1.22464680e-16, 0.00000000e+00],
                            [-1.22464680e-16,  0.00000000e+00, -1.00000000e+00, 0.00000000e+00],
                            [ 0.00000000e+00, -1.00000000e+00,  0.00000000e+00, 0.00000000e+00],
                            [-1.97168135e-16, -2.40000000e-01, -1.61000000e+00, 1.00000000e+00]]),
        'CAM_BACK_LEFT':np.array([[-0.34202014,  0.        , -0.93969262,  0.        ],
                                    [ 0.93969262,  0.        , -0.34202014,  0.        ],
                                    [ 0.        , -1.        ,  0.        ,  0.        ],
                                    [-0.25388956, -0.24      , -0.49288953,  1.        ]]),
        'CAM_BACK_RIGHT':np.array([[-0.34202014,  0.        ,  0.93969262,  0.        ],
                                    [-0.93969262,  0.        , -0.34202014,  0.        ],
                                    [ 0.        , -1.        ,  0.        ,  0.        ],
                                    [ 0.25388956, -0.24      , -0.49288953,  1.        ]])
        }
        self.cam_intrinsic = {
        'CAM_FRONT': np.array([[1.14251841e+03, 0.00000000e+00, 8.00000000e+02],
                            [0.00000000e+00, 1.14251841e+03, 4.50000000e+02],
                            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
        'CAM_FRONT_RIGHT': np.array([[1.14251841e+03, 0.00000000e+00, 8.00000000e+02],
                                    [0.00000000e+00, 1.14251841e+03, 4.50000000e+02],
                                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
        'CAM_FRONT_LEFT': np.array([[1.14251841e+03, 0.00000000e+00, 8.00000000e+02],
                                    [0.00000000e+00, 1.14251841e+03, 4.50000000e+02],
                                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
        'CAM_BACK':np.array([[560.16603057,   0.        , 800.        ],
                            [  0.        , 560.16603057, 450.        ],
                            [  0.        ,   0.        ,   1.        ]]),
        'CAM_BACK_LEFT':np.array([[1.14251841e+03, 0.00000000e+00, 8.00000000e+02],
                                    [0.00000000e+00, 1.14251841e+03, 4.50000000e+02],
                                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
        'CAM_BACK_RIGHT':np.array([[1.14251841e+03, 0.00000000e+00, 8.00000000e+02],
                                    [0.00000000e+00, 1.14251841e+03, 4.50000000e+02],
                                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
        }

        self.lidar2img = {}
        for key, value in self.cam_intrinsic.items():
            intrinsic = value * resize_scale
            self.cam_intrinsic[key] = intrinsic

            viewpad = np.eye(4)
            viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
            lidar2cam = self.lidar2cam[key]
            self.lidar2img[key] = viewpad @ lidar2cam.T

        self.lidar2ego = np.array([[ 0. ,  1. ,  0. , -0.39],
                                   [-1. ,  0. ,  0. ,  0.  ],
                                   [ 0. ,  0. ,  1. ,  1.84],
                                   [ 0. ,  0. ,  0. ,  1.  ]])
        
        topdown_extrinsics =  np.array([[0.0, -0.0, -1.0, 50.0], [0.0, 1.0, -0.0, 0.0], [1.0, -0.0, 0.0, -0.0], [0.0, 0.0, 0.0, 1.0]])
        unreal2cam = np.array([[0,1,0,0], [0,0,-1,0], [1,0,0,0], [0,0,0,1]])
        self.coor2topdown = unreal2cam @ topdown_extrinsics
        topdown_intrinsics = np.array([[548.993771650447, 0.0, 256.0, 0], [0.0, 548.993771650447, 256.0, 0], [0.0, 0.0, 1.0, 0], [0, 0, 0, 1.0]])
        self.coor2topdown = topdown_intrinsics @ self.coor2topdown

        self.clock = Clock() 
        self.stuck_detector = 0
        self.forced_move = 0

    def _init(self):
        try:
            locx, locy = self._global_plan_world_coord[0][0].location.x, self._global_plan_world_coord[0][0].location.y
            lon, lat = self._global_plan[0][0]['lon'], self._global_plan[0][0]['lat']
            EARTH_RADIUS_EQUA = 6378137.0
            def equations(vars):
                x, y = vars
                eq1 = lon * math.cos(x * math.pi / 180) - (locx * x * 180) / (math.pi * EARTH_RADIUS_EQUA) - math.cos(x * math.pi / 180) * y
                eq2 = math.log(math.tan((lat + 90) * math.pi / 360)) * EARTH_RADIUS_EQUA * math.cos(x * math.pi / 180) + locy - math.cos(x * math.pi / 180) * EARTH_RADIUS_EQUA * math.log(math.tan((90 + x) * math.pi / 360))
                return [eq1, eq2]
            initial_guess = [0, 0]
            solution = fsolve(equations, initial_guess)
            self.lat_ref, self.lon_ref = solution[0], solution[1]
        except Exception as e:
            print(e, flush=True)
            self.lat_ref, self.lon_ref = 0, 0      
        self._route_planner = RoutePlanner(4.0, 50.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref)
        self._route_planner.set_route(self._global_plan, True)

        self._route_planner_far = RoutePlanner(7.5, 25.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref)
        self._route_planner_far.set_route(self._global_plan, True)

        # import 

        self.initialized = True
        self.metric_info = {}
        self.command_hist = []
        self.fucos_command = [4, 5] # ['LEFT', 'RIGHT', 'STRAIGHT', 'LANE_FOLLOW', 'CHANGE_LANE_LEFT', 'CHANGE_LANE_RIGHT']

        self.stuck_frame_nomal = 150
        self.stuck_frame_light = 450



    def sensors(self):
        W = 1600 * resize_scale
        H = 900 * resize_scale

        sensors =[
                # camera rgb
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.80, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': W, 'height': H, 'fov': 70,
                    'id': 'CAM_FRONT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.27, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0,
                    'width': W, 'height': H, 'fov': 70,
                    'id': 'CAM_FRONT_LEFT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.27, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 55.0,
                    'width': W, 'height': H, 'fov': 70,
                    'id': 'CAM_FRONT_RIGHT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -2.0, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
                    'width': W, 'height': H, 'fov': 110,
                    'id': 'CAM_BACK'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -0.32, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -110.0,
                    'width': W, 'height': H, 'fov': 70,
                    'id': 'CAM_BACK_LEFT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -0.32, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 110.0,
                    'width': W, 'height': H, 'fov': 70,
                    'id': 'CAM_BACK_RIGHT'
                },
                # imu
                {
                    'type': 'sensor.other.imu',
                    'x': -1.4, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.05,
                    'id': 'IMU'
                },
                # gps
                {
                    'type': 'sensor.other.gnss',
                    'x': -1.4, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.01,
                    'id': 'GPS'
                },
                # speed
                {
                    'type': 'sensor.speedometer',
                    'reading_frequency': 20,
                    'id': 'SPEED'
                },
                # lidar
                {   'type': 'sensor.lidar.ray_cast',
                    'x': -0.39, 'y': 0.0, 'z': 1.84,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'range': 85,
                    'rotation_frequency': 10,
                    'channels': 64,
                    'points_per_second': 600000,
                    'dropoff_general_rate': 0.0,
                    'dropoff_intensity_limit': 0.0,
                    'dropoff_zero_intensity': 0.0,
                    'id': 'LIDAR_TOP'
                },
            ]
        if IS_BENCH2DRIVE:
            # 俯视图
            sensors += [
                    {	
                        'type': 'sensor.camera.rgb',
                        'x': 0.0, 'y': 0.0, 'z': 50.0,
                        'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                        'width': 512, 'height': 512, 'fov': 5 * 10.0,
                        'id': 'bev'
                    }]
            # # 为了可视化的贴近俯视图
            # sensors += [
            #         {    
            #             'type': 'sensor.camera.rgb',
            #             'x': -6.0, 'y': 0.0, 'z': 4.0, 
            #             'roll': 0.0, 'pitch': -20.0, 'yaw': 0.0, 
            #             'width': 1920, 'height': 1080, 
            #             'fov': 90.0, 
            #             'id': 'bev'
            #         }]
        return sensors

    def tick(self, input_data, force_two_tp=False):
        self.step += 1
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
        imgs = {}
        for cam in CAMERAS:
            img = input_data[cam][1][:, :, :3]
            _, img = cv2.imencode('.jpg', img, encode_param)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            imgs[cam] = img
        bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        gps = input_data['GPS'][1][:2]
        speed = input_data['SPEED'][1]['speed']
        compass = input_data['IMU'][1][-1]
        acceleration = input_data['IMU'][1][:3]
        angular_velocity = input_data['IMU'][1][3:6]
  
        pos = self.gps_to_location(gps)
        near_info, near_next_info = self._route_planner.run_step(pos, return_two=True)
        near_node, near_command = near_info
        near_node_next, near_command_next = near_next_info
        
        far_node, far_command = self._route_planner_far.run_step(pos)
        # 如果near_node和far node太近，则将far node设置为near node的下一个
        if force_two_tp:
            if np.linalg.norm(near_node - far_node) < 0.5:
                far_node = near_node_next
                far_command = near_command_next

        # import pdb;pdb.set_trace()
        if (math.isnan(compass) == True): #It can happen that the compass sends nan for a few frames
            compass = 0.0
            acceleration = np.zeros(3)
            angular_velocity = np.zeros(3)

        result = {
                'imgs': imgs,
                'gps': gps,
                'pos':pos,
                'speed': speed,
                'compass': compass,
                'bev': bev,
                'acceleration':acceleration,
                'angular_velocity':angular_velocity,
                'command_near':near_command,
                'command_near_xy':near_node,
                'command_far': far_command,
                'command_far_xy': far_node,
                }
        
        return result
    

    def extend_cmd(self, command):

        if command in self.fucos_command:
            return command

        # 如果历史命令中包含fucos_command，则返回focus_command
        if len(self.command_hist) > 0:
            for cmd in self.fucos_command:
                if cmd in self.command_hist:
                    return cmd
        return command

    def refine_cmd(self, command, tp_near, tp_far):
        # 将一些错误的Command修复
        new_command = command
        if command == 4:    # 4: CHANGE_LANE_LEFT
            if tp_near[0] >= 0:
                new_command = 3
        elif command == 5:  # 5: CHANGE_LANE_RIGHT
            if tp_near[0] <= 0:
                new_command = 3
        elif command == 2:
            if tp_near[0] <= -1.5:        # ['LEFT', 'RIGHT', 'STRAIGHT', 'LANE_FOLLOW', 'CHANGE_LANE_LEFT', 'CHANGE_LANE_RIGHT']
                new_command = 4
            elif tp_near[0] >= 1.5:
                new_command = 5
            else:
                new_command = command
        
        if new_command != command:
            print("refined_cmd: ", new_command, tp_near, tp_far)
        return new_command






    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        self.clock.count("start")
        # import pdb;pdb.set_trace()
        tick_data = self.tick(input_data, force_two_tp=self.pidcontroller.force_two_tp)
        self.clock.count("tick")
        results = {}
        # Route identity required by Collect's inference metadata contract.
        results['scenario_name'] = pathlib.Path(self.save_name).name
        results['timestamp'] = self.step / frame_rate
        results['img'] = []
        results['lidar2img'] = []
  
        for cam in CAMERAS:
            results['img'].append(tick_data['imgs'][cam])
            results['lidar2img'].append(self.lidar2img[cam])
        
        raw_theta = tick_data['compass']   if not np.isnan(tick_data['compass']) else 0
        ego_theta = -raw_theta + np.pi/2
        rotation = list(Quaternion(axis=[0, 0, 1], radians=ego_theta))
        can_bus = np.zeros(18)
        can_bus[0] = tick_data['pos'][0]
        can_bus[1] = -tick_data['pos'][1]
        can_bus[3:7] = rotation
        can_bus[7] = tick_data['speed']
        can_bus[10:13] = tick_data['acceleration']
        can_bus[11] *= -1
        can_bus[13:16] = -tick_data['angular_velocity']
        can_bus[16] = ego_theta
        can_bus[17] = ego_theta / np.pi * 180 
        results['can_bus'] = can_bus
        
        lidar = CarlaDataProvider.get_world().get_actors().filter('*sensor.lidar.ray_cast*')[0]
        world2lidar = lidar.get_transform().get_inverse_matrix()
        world2lidar = lefthand_ego_to_lidar @ world2lidar @ left2right
        lidar2global =  self.invert_pose(world2lidar)
        results['lidar2global'] = lidar2global

        ego_status = np.zeros(10, dtype=np.float32)
        ego_status[:3] = np.array([tick_data['acceleration'][0],-tick_data['acceleration'][1],tick_data['acceleration'][2]])
        ego_status[3:6] = -np.array(tick_data['angular_velocity'])
        ego_status[6:9] = np.array([tick_data['speed'],0,0])
        results["ego_status"] = ego_status
        
        theta_to_lidar = raw_theta
        rotation_matrix = np.array([[np.cos(theta_to_lidar),-np.sin(theta_to_lidar)],[np.sin(theta_to_lidar),np.cos(theta_to_lidar)]])

        command_near_xy = np.array([tick_data['command_near_xy'][0]-can_bus[0],-tick_data['command_near_xy'][1]-can_bus[1]])
        local_command_xy = rotation_matrix @ command_near_xy
        

        command_far_xy = np.array([tick_data['command_far_xy'][0]-can_bus[0],-tick_data['command_far_xy'][1]-can_bus[1]])
        far_command_xy = rotation_matrix @ command_far_xy
        

        if self.cfg.num_cmd > 1:
            # import pdb;pdb.set_trace()
            command = tick_data['command_near']
            if command < 0:
                command = 4
            command -= 1

            if self.pidcontroller.extend_cmd:
                command = self.extend_cmd(command)
            if self.pidcontroller.refine_cmd:
                command = self.refine_cmd(command, local_command_xy, far_command_xy)

            command_onehot = np.zeros(6)
            command_onehot[command] = 1
        else:
            command = 0
            command_onehot = np.array([0, ])

            






        self.command_hist.append(tick_data['command_near'].value - 1)
        if len(self.command_hist) > 30:
            self.command_hist.pop(0)
        # import pdb;pdb.set_trace()
        results['gt_ego_fut_cmd'] = command_onehot
        
        # print(tick_data.keys())

        results['tp_near'] = local_command_xy
        results['tp_far'] = far_command_xy

        stacked_img = np.stack(results['img'], axis=-1)
        results['img_shape'] = stacked_img.shape
        results['ori_shape'] = stacked_img.shape
        results['pad_shape'] = stacked_img.shape

        aug_config = self.get_augmentation()
        results["aug_config"] = aug_config
        results = self.test_pipeline(results)
        input_data_batch = mm_collate_to_batch_form([results], samples_per_gpu=1)

        for key, data in input_data_batch.items():
            if key != 'img_metas':
                if torch.is_tensor(data):
                    data = data.to(self.device)
        self.clock.count("data")
        output_data_batch = self.model(**input_data_batch)
        self.clock.count("model")
        out_truck = output_data_batch[0]['img_bbox']['waypoints']
        if self.require_longitudinal_output:
            longitudinal_waypoints = output_data_batch[0]['img_bbox']['longitudinal_waypoints']
            np.testing.assert_allclose(out_truck, longitudinal_waypoints, rtol=0, atol=0)
            if not np.isfinite(out_truck).all():
                raise RuntimeError('Non-finite LONGITUDINAL waypoints')
            if self.step % 100 == 0:
                print('LONGITUDINAL_OUTPUT_VERIFIED step={} shape={}'.format(self.step, np.shape(out_truck)), flush=True)
        #all_out_truck =  np.cumsum(out_truck,axis=0)
        if 'mm_waypoints' in output_data_batch[0]['img_bbox']:
            mm_output = {}
            mm_output['mm_waypoints'] = output_data_batch[0]['img_bbox']['mm_waypoints']
            mm_output['mm_score'] = output_data_batch[0]['img_bbox']['mm_score']
        ## creep
        is_stuck = False
        # import pdb;pdb.set_trace()
        max_stuck_frame = self.stuck_frame_nomal    # self.stuck_frame_nomal or self.stuck_frame_light
        traffic_state = 0
        if 'traffic_light_state' in output_data_batch[0]['img_bbox']:
            traffic_state = output_data_batch[0]['img_bbox']['traffic_light_state']
            if traffic_state !=0 and traffic_state!=3:  # 不知道 和 绿灯
                max_stuck_frame = self.stuck_frame_light
                print("There is traffic light, and is not none")



        # 先检测速度来更新stuck_detector
        if tick_data['speed'] < 0.3:  # 低速时增加卡住检测器
            self.stuck_detector += 1
        else:  # 速度正常时减少卡住检测器
            self.stuck_detector = max(0, self.stuck_detector - 1)

        # 状态处理
        forced_move_time = 30 
        if traffic_state == 1 or traffic_state == 2:   # 检测到红灯或者黄灯
            forced_move_time = 20
            print("Setting forced move time to 10 frames for RED or YELLOW traffic light.")

        if self.forced_move > 0:  # 已经在人为控制阶段
            if self.forced_move < forced_move_time:  # 继续人为控制
                print(f"Continuing forced move. Frame: {self.forced_move}")
                is_stuck = True
                self.forced_move += 1
                self.command_log = 'man'
            else:  # 人为控制15帧后，完全重置
                print("Forced move completed. Resetting.")
                self.stuck_detector = 0
                self.forced_move = 0
                self.command_log = 'auto'
        elif self.stuck_detector > max_stuck_frame:  # 新检测到卡住，开始人为控制
            print("Detected agent being stuck. Starting forced move.")
            is_stuck = True
            self.forced_move = 1  # 开始计数
            self.command_log = 'man'
        else:  # 正常自动控制状态
            self.command_log = 'auto'
            self.forced_move = 0
        if self.command_log == 'man':
            self.pidcontroller.set_stuck(True)
            if 'fastest_threshold_trajs' in output_data_batch[0]['img_bbox']:
                if output_data_batch[0]['img_bbox']['fastest_threshold_trajs'] is not None:
                    farest_traj = True
                    out_truck = output_data_batch[0]['img_bbox']['fastest_threshold_trajs']
                else:
                    farest_traj = False
            else:
                farest_traj = False
            print("change to drivepath")
            self.model.module.head.motion_plan_head.is_stuck = True
        else:
            self.model.module.head.motion_plan_head.is_stuck = False
            self.pidcontroller.set_stuck(False)




        # out_truck = output_data_batch[0]['img_bbox']['waypoints_drivepath']
        # print("use drivepath")
        out_truck = out_truck[:self.valid_waypoints] #
        steer_traj, throttle_traj, brake_traj, metadata_traj = self.pidcontroller.control_pid(out_truck, tick_data['speed'], local_command_xy, command=command)#, is_stuck)
        if brake_traj < 0.05: brake_traj = 0.0
        if throttle_traj > brake_traj: brake_traj = 0.0




        control = carla.VehicleControl()
        self.pid_metadata = metadata_traj
        self.pid_metadata['agent'] = 'only_traj'

        # import pdb;pdb.set_trace()

        control.steer = float(steer_traj)
        if self.command_log == 'man' and not farest_traj:
            control.throttle = float(0.5)
            print("Force throttle to 0.5")
        # else:
        control.throttle = float(throttle_traj)
        control.brake = np.clip(float(brake_traj), 0, 1)     
  
        self.pid_metadata['steer'] = control.steer
        self.pid_metadata['throttle'] = control.throttle
        self.pid_metadata['brake'] = control.brake
        self.pid_metadata['steer_traj'] = float(steer_traj)
        self.pid_metadata['throttle_traj'] = float(throttle_traj)
        self.pid_metadata['brake_traj'] = float(brake_traj)
        self.pid_metadata['plan'] = out_truck.tolist()
        self.pid_metadata['command'] = command
        self.pid_metadata['local_command_xy'] = local_command_xy
        self.pid_metadata['result'] = output_data_batch[0]['img_bbox']

        
        metric_info = self.get_metric_info()
        self.metric_info[self.step] = metric_info
        tick_data['lidar2global'] = lidar2global
        tick_data['ego_status'] = ego_status
        tick_data['tp_near'] = results['tp_near']
        tick_data['tp_far'] = results['tp_far']
        tick_data['tp_line'] = results['tp_line']
        if self.step % save_interval == 0:
            self.save(tick_data)
        self.prev_control = control
        
        if len(self.prev_control_cache)==10:
            self.prev_control_cache.pop(0)
        self.prev_control_cache.append(control)




        return control

    def save(self, tick_data):
        frame = self.step // save_interval







        # 保存 BEV（假设是 RGB 或单通道）
        Image.fromarray(tick_data['bev']).save(self.save_path / 'bev' / f'{frame:04d}.png')
        data = {}

        cmd = np.zeros(6)
        cmd[self.pid_metadata["command"]] = 1
        data["gt_ego_fut_cmd"] = cmd
        data["tp_near"] = self.pid_metadata["local_command_xy"]
        data["tp_far"] = tick_data['tp_far']
        data["tp_line"] = tick_data['tp_line']

        data["img_filename"] = [
            self.save_path / 'rgb_front' / ('%04d.png' % frame),
            self.save_path / 'rgb_front_right' / ('%04d.png' % frame),
            self.save_path / 'rgb_front_left' / ('%04d.png' % frame),
            self.save_path / 'rgb_back' / ('%04d.png' % frame),
            self.save_path / 'rgb_back_left' / ('%04d.png' % frame),
            self.save_path / 'rgb_back_right' / ('%04d.png' % frame),
        ]
        
        

        data['lidar2cam'] = []
        data['cam_intrinsic'] = []
        for cam in CAMERAS:
            data['lidar2cam'].append(self.lidar2cam[cam])
            data['cam_intrinsic'].append(self.cam_intrinsic[cam])
        
        data['lidar2global'] = tick_data['lidar2global']    # check this
        data['ego_status'] = tick_data['ego_status']    # check this

        self.pid_metadata["data"] = data
        self.pid_metadata["command_log"] = self.command_log
        self.pid_metadata["stuck_num"] = self.stuck_detector

        outfile = open(self.save_path / 'meta' / ('%04d.pkl' % frame), 'wb')
        pickle.dump(self.pid_metadata, outfile)
        outfile.close()

        # metric info
        outfile = open(self.save_path / 'metric_info.json', 'w')
        json.dump(self.metric_info, outfile, indent=4)
        outfile.close()

    def destroy(self):
        del self.model
        torch.cuda.empty_cache()

    def gps_to_location(self, gps):
        EARTH_RADIUS_EQUA = 6378137.0
        # gps content: numpy array: [lat, lon, alt]
        lat, lon = gps
        scale = math.cos(self.lat_ref * math.pi / 180.0)
        my = math.log(math.tan((lat+90) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
        mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
        y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0)) - my
        x = mx - scale * self.lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
        return np.array([x, y])

    def get_augmentation(self):
        H = 900 * resize_scale
        W = 1600 * resize_scale
        fH, fW = self.data_aug_conf["final_dim"]
        resize = max(fH / H, fW / W)
        resize_dims = (int(W * resize), int(H * resize))
        newW, newH = resize_dims
        crop_h = (
            int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH)
            - fH
        )
        crop_w = int(max(0, newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        flip = False
        rotate = 0
        rotate_3d = 0
        aug_config = {
            "resize": resize,
            "resize_dims": resize_dims,
            "crop": crop,
            "flip": flip,
            "rotate": rotate,
            "rotate_3d": rotate_3d,
        }
        return aug_config

    def invert_pose(self, pose):
        inv_pose = np.eye(4)
        inv_pose[:3, :3] = np.transpose(pose[:3, :3])
        inv_pose[:3, -1] = - inv_pose[:3, :3] @ pose[:3, -1]
        return inv_pose

def draw(input, step, cfg):
    img = input['img'].data[0][0, 0]
    projection_mat = input['projection_mat'].data[0, 0]
    key_points = torch.tensor([0, 10, -1.8])
    pts_extend = torch.cat(
        [key_points, torch.ones_like(key_points[..., :1])], dim=-1
    )
    points_2d = torch.matmul(
        projection_mat, pts_extend[..., None]
    ).squeeze(-1)
    points_2d = points_2d[..., :2] / torch.clamp(
        points_2d[..., 2:3], min=1e-5
    )
    print(points_2d[0]/img.shape[2], points_2d[1]/img.shape[1])
    points_2d = points_2d.numpy()
    img = img.numpy().transpose(1, 2, 0).astype(np.uint8)
    img = img.copy()
    cv2.circle(img, (int(points_2d[0]), int(points_2d[1])), 5, (0, 0, 255))
    cv2.imwrite(f'vis3/{resize_scale}_{cfg.data_aug_conf["final_dim"]}{step}.jpg', img)
