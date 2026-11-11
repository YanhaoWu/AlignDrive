from collections import deque
import numpy as np
import json
class PID(object):
	def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
		self._K_P = K_P
		self._K_I = K_I
		self._K_D = K_D

		self._window = deque([0 for _ in range(n)], maxlen=n)
		self._window_empty = deque([0 for _ in range(n)], maxlen=n)
		self._max = 0.0
		self._min = 0.0

	def step(self, error):
		self._window.append(error)
		self._max = max(self._max, abs(error))
		self._min = -abs(self._max)

		if len(self._window) >= 2:
			integral = np.mean(self._window)
			derivative = (self._window[-1] - self._window[-2])
		else:
			integral = 0.0
			derivative = 0.0

		return self._K_P * error + self._K_I * integral + self._K_D * derivative



class PIDController(object):
    
    def __init__(self, pid_param, turn_KP=0.75, turn_KI=0.75, turn_KD=0.3, turn_n=40, speed_KP=5.0, speed_KI=0.5,speed_KD=1.0, speed_n = 40, max_throttle=0.75, brake_speed=0.4,brake_ratio=1.1, clip_delta=0.25, aim_dist=4.0, angle_thresh=0.3, dist_thresh=10 , max_steer=1):
        if pid_param is not None :
            # import pdb;pdb.set_trace()
            with open(pid_param, 'r') as file:
                # 加载JSON数据
                pid_param = json.load(file)
            # 获取 turn 相关的参数，如果不存在则使用默认值
            turn_KP = pid_param.get("turn_KP", 0.75)
            turn_KI = pid_param.get("turn_KI", 0.75)
            turn_KD = pid_param.get("turn_KD", 0.3)
            turn_n = pid_param.get("turn_n", 40)

            # 获取 speed 相关的参数，如果不存在则使用默认值
            speed_KP = pid_param.get("speed_KP", 5.0)
            speed_KI = pid_param.get("speed_KI", 0.5)
            speed_KD = pid_param.get("speed_KD", 1.0)
            speed_n = pid_param.get("speed_n", 40)
            frency = pid_param.get("frency", 2)

            # 获取 其他参数，如果不存在则使用默认值
            max_throttle = pid_param.get("max_throttle", 0.75)
            brake_speed = pid_param.get("brake_speed", 0.4)
            brake_ratio = pid_param.get("brake_ratio", 1.1)
            clip_delta = pid_param.get("clip_delta", 0.25)
            # 後續調參候選
            aim_dist = pid_param.get("aim_dist", 4.0)
            angle_thresh = pid_param.get("angle_thresh", 0.3)
            dist_thresh = pid_param.get("dist_thresh", 10)

            # 死区速度
            min_desired_speed = pid_param.get("min_desired_speed", 0.0)
            max_desired_speed = pid_param.get("max_desired_speed", 100.0)

            # 其他过滤
            fliter_ood_waypoints = pid_param.get("fliter_ood_waypoints", False)
            no_steer_stop = pid_param.get("no_steer_stop", False)
            turn_KP_turn = pid_param.get("turn_KP_turn", turn_KP)
            turn_KI_turn = pid_param.get("turn_KI_turn", turn_KI)

            turn_KP_change_lane = pid_param.get("turn_KP_change_lane", turn_KP)
            turn_KI_change_lane = pid_param.get("turn_KI_change_lane", turn_KI)

            # cmd延长，会在外部被调用
            extend_cmd = pid_param.get("extend_cmd", False)
            force_two_tp = pid_param.get("force_two_tp", False) # 如果两个导航点崩塌为一个，是否强制两个导航点

            refine_cmd = pid_param.get("refine_cmd", False) # 是否使用refine_cmd


            self.control_waypoints_range = pid_param.get("control_waypoints_range", [2, 3, 4])
            self.control_waypoints_range_stuck = pid_param.get("control_waypoints_range_stuck", self.control_waypoints_range)
            self.turn_goal_points = pid_param.get("turn_goal_points", self.control_waypoints_range[0])
            self.turn_goal_points_turn = pid_param.get("turn_goal_points_turn", self.control_waypoints_range[0])

            # 打印结果
            print(f"turn_KP: {turn_KP}")
            print(f"turn_KP_turn {turn_KP_turn}")
            print(f"turn_KI_turn {turn_KI_turn}")
            print(f"turn_KP_change_lane: {turn_KP_change_lane}")
            print(f"turn_KI_change_lane: {turn_KI_change_lane}")
            print(f"turn_KI: {turn_KI}")
            print(f"turn_KD: {turn_KD}")
            print(f"turn_n: {turn_n}")
            print(f"speed_KP: {speed_KP}")
            print(f"speed_KI: {speed_KI}")
            print(f"speed_KD: {speed_KD}")
            print(f"speed_n: {speed_n}")
            print(f"frency: {frency}")
            print(f"max_throttle: {max_throttle}")
            print(f"brake_speed: {brake_speed}")
            print(f"brake_ratio: {brake_ratio}")
            print(f"clip_delta: {clip_delta}")
            print(f"aim_dist: {aim_dist}")
            print(f"angle_thresh: {angle_thresh}")
            print(f"dist_thresh: {dist_thresh}")
            print(f"max_steer: {max_steer}")

            print("control_waypoints_range:", self.control_waypoints_range)
            print("control_waypoints_range_stuck:", self.control_waypoints_range_stuck)
            print("turn_goal_points:", self.turn_goal_points)
            print("turn_goal_points_turn:", self.turn_goal_points_turn)
            print("min_desired_speed:", min_desired_speed)
            print("max_desired_speed:", max_desired_speed)
            print("fliter_ood_waypoints:", fliter_ood_waypoints)
            print("no_steer_stop:", no_steer_stop)
            print("extend_cmd:", extend_cmd)
            print("force_two_tp:", force_two_tp)
            print("refine_cmd:", refine_cmd)


        self.turn_controller = PID(K_P=turn_KP, K_I=turn_KI, K_D=turn_KD, n=turn_n)
        self.turn_KP_turn = turn_KP_turn
        self.turn_KI_turn = turn_KI_turn
        self.turn_KP = turn_KP
        self.turn_KI = turn_KI
        self.turn_KP_change_lane = turn_KP_change_lane
        self.turn_KI_change_lane = turn_KI_change_lane
        self.speed_controller = PID(K_P=speed_KP, K_I=speed_KI, K_D=speed_KD, n=speed_n)
        self.max_throttle = max_throttle
        self.brake_speed = brake_speed
        self.brake_ratio = brake_ratio
        self.clip_delta = clip_delta
        self.aim_dist = aim_dist
        self.angle_thresh = angle_thresh
        self.dist_thresh = dist_thresh
        self.max_steer = max_steer
        self.frency = frency # 2hz or 5hz
        self.min_desired_speed = min_desired_speed
        self.max_desired_speed = max_desired_speed
        self.fliter_ood_waypoints = fliter_ood_waypoints
        self.no_steer_stop = no_steer_stop
        self.extend_cmd = extend_cmd
        self.force_two_tp = force_two_tp
        self.refine_cmd = refine_cmd


        self.prev_angle = 0.0

    def fliter_ood_predicted_waypoints(self, waypoints):
        # 过滤掉横向位移很大而纵向位移很小的点
        # waypoints: N,2
        lateral_displacement = abs(waypoints[-1, 0])
        longitudinal_displacement = abs(waypoints[-1, 1])
        if longitudinal_displacement < 0.2:
            return np.zeros_like(waypoints)
        else:
            return waypoints

    def set_stuck(self, is_stuck):
        if is_stuck:
            self.control_waypoints_range = self.control_waypoints_range_stuck
        else:
            self.control_waypoints_range = self.control_waypoints_range


    def control_pid(self, waypoints, speed, target, command=-1):
        ''' Predicts vehicle control with a PID controller.
        Args:
            waypoints (tensor): output of self.plan()
            speed (tensor): speedometer input
        '''

        # iterate over vectors between predicted waypoints
        # print("Num pair:",num_pairs)
        if self.fliter_ood_waypoints:
            waypoints = self.fliter_ood_predicted_waypoints(waypoints)

        if command == 1:    # 右转, 不换turn_goal_points
            self.turn_controller._K_P = self.turn_KP_turn
            self.turn_controller._K_I = self.turn_KI_turn
            turn_goal_points = self.turn_goal_points
        elif command == 4:  # 左变道，不换PID
            self.turn_controller._K_P = self.turn_KP_change_lane
            self.turn_controller._K_I = self.turn_KI_change_lane
            turn_goal_points = self.turn_goal_points_turn
        else:
            self.turn_controller._K_P = self.turn_KP
            self.turn_controller._K_I = self.turn_KI
            turn_goal_points = self.turn_goal_points
        # print("Current command:", command, "turn_goal_points:", turn_goal_points, "turn_KP:", self.turn_controller._K_P, "turn_KI:", self.turn_controller._K_I)
        print("Current Control Range :", self.control_waypoints_range)
        best_norm = 1e5
        desired_speed = 0
        if turn_goal_points == -1:
            use_dist = True
            aim = waypoints[0]
        else:
            use_dist = False
            aim = waypoints[turn_goal_points]

        num_pairs = len(self.control_waypoints_range)
        for i in self.control_waypoints_range:
            # magnitude of vectors, used for speed
            desired_speed += (np.linalg.norm(waypoints[i+1] - waypoints[i]) * self.frency / num_pairs)

            # norm of vector midpoints, used for steering
            if use_dist:
                norm = np.linalg.norm((waypoints[i+1] + waypoints[i]) / 2.0)
                if abs(self.aim_dist-best_norm) > abs(self.aim_dist-norm):
                    aim = waypoints[i]
                    best_norm = norm

        # print("desired_speed:",desired_speed)

        aim_last = waypoints[-1] - waypoints[-2]

        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        # 如果aim[0]和aim[1]都为极小值，angle设置为0
        if abs(aim[0]) < 1e-2 and abs(aim[1]) < 1e-2:
            angle = np.array(0.0)
        elif aim[1] < 0:
            angle = np.array(0.0)




        angle_last = np.degrees(np.pi / 2 - np.arctan2(aim_last[1], aim_last[0])) / 90
        angle_target = np.degrees(np.pi / 2 - np.arctan2(target[1], target[0])) / 90

        # choice of point to aim for steering, removing outlier predictions
        # use target point if it has a smaller angle or if error is large
        # predicted point otherwise
        # (reduces noise in eg. straight roads, helps with sudden turn commands)
        use_target_to_aim = np.abs(angle_target) < np.abs(angle)
        
        use_target_to_aim = use_target_to_aim or (np.abs(angle_target-angle_last) > self.angle_thresh and target[1] < self.dist_thresh)
        
        use_target_to_aim = False

        if use_target_to_aim:
            angle_final = angle_target
        else:
            angle_final = angle


        # 如果angle角度方向与上一次的方向相反，则清空积分
        if angle_final * self.prev_angle < 0:
            self.turn_controller._window = self.turn_controller._window_empty
            print("Current angle:", angle_final, "clear window")


        steer = self.turn_controller.step(angle_final)



        # steer = np.clip(steer, -1.0, 1.0)
        steer = np.clip(steer, -self.max_steer, self.max_steer)

        if desired_speed < self.min_desired_speed:
            desired_speed = np.array(self.min_desired_speed)
        elif desired_speed > self.max_desired_speed:
            desired_speed =  np.array(self.max_desired_speed)

        brake = desired_speed < self.brake_speed or (speed / desired_speed) > self.brake_ratio 
        brake = brake if speed < np.array(self.max_desired_speed) else True
        
        if desired_speed < self.brake_speed and self.no_steer_stop:
            steer = np.array(0.0)


        delta = np.clip(desired_speed - speed, 0.0, self.clip_delta)
        throttle = self.speed_controller.step(delta)
        # print("throttle",throttle)
        throttle = np.clip(throttle, 0.0, self.max_throttle)
        # print("throttle after clip", throttle)

        throttle = throttle if not brake else 0.0
        self.prev_angle = angle_final
        metadata = {
            'speed': float(speed.astype(np.float64)),
            'steer': float(steer),
            'throttle': float(throttle),
            'brake': float(brake),
            'aim': tuple(aim.astype(np.float64)),
            'target': tuple(target.astype(np.float64)),
            'desired_speed': float(desired_speed.astype(np.float64)),
            'angle': float(angle.astype(np.float64)),
            'angle_last': float(angle_last.astype(np.float64)),
            'angle_target': float(angle_target.astype(np.float64)),
            'angle_final': float(angle_final.astype(np.float64)),
            'delta': float(delta.astype(np.float64)),
        }

        return steer, throttle, brake, metadata