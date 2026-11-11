#!/usr/bin/env python3
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

import os
import subprocess
import signal
import time
import logging
from typing import Dict, List, Optional
import re
import shlex
from carla_tools.split_xml import genereate_xml_from_json, write_route_shards
from carla_tools.paths import REPO_ROOT, repo_path
from time import sleep
import os
import json
import shutil

class ScreenManager:
    def __init__(self, log_dir: str, update_interval: int = 10):
        self.log_dir = log_dir
        self.update_interval = update_interval  # PID update interval in seconds.
        os.makedirs(self.log_dir, exist_ok=True)
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging."""
        logging.basicConfig(
            filename=os.path.join(self.log_dir, "screen_manager.log"),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    def start_command(self, name: str, command: str) -> bool:
        """Launch a simulation shell command in a screen session."""
        try:
            subprocess.run(['screen', '-dmS', name, 'bash', '-c',
                            'set -o pipefail; ' + command], cwd=REPO_ROOT, check=True)
            logging.info(f"Started screen session: {name}, Command='{command}'")
            print("Started screen session: ", name, ", Command='", command, "'")
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to start {name}: {e}")
            print("Failed to start ", name, ": ", e)
            return False

    def _get_screen_pid(self, name: str) -> Optional[int]:
        try:
            sessions = subprocess.check_output(['screen', '-ls'], text=True)
        except subprocess.CalledProcessError as error:
            sessions = error.output or ''
        screen_pids = [int(match.group(1)) for match in
                       re.finditer(r'^\s*(\d+)\.([^\s]+)\s', sessions, re.MULTILINE)
                       if match.group(2) == name]
        if len(screen_pids) > 1:
            raise RuntimeError(f'Ambiguous screen session name: {name}')
        return screen_pids[0] if screen_pids else None

    def get_child_pids(self, name: str) -> List[int]:
        """Get child process IDs belonging to the exact screen session."""
        screen_pid = self._get_screen_pid(name)
        return self._get_descendants(screen_pid) if screen_pid is not None else []

    def _get_descendants(self, screen_pid: int) -> List[int]:
        descendants = []
        pending = [screen_pid]
        while pending:
            parent_pid = pending.pop(0)
            result = subprocess.run(['pgrep', '-P', str(parent_pid)],
                                    capture_output=True, text=True, check=False)
            children = [int(pid) for pid in result.stdout.split()]
            descendants.extend(children)
            pending.extend(children)
        return descendants

    def kill_child_processes(self, name: str, force: bool = True) -> bool:
        """Terminate child processes in a screen session; force-kill by default."""
        screen_pid = self._get_screen_pid(name)
        if screen_pid is None:
            return False
        pids = self._get_descendants(screen_pid)
        killed = False
        for pid in reversed(pids):
            try:
                if force:
                    os.kill(pid, signal.SIGKILL)
                else:
                    os.kill(pid, signal.SIGTERM)
                logging.info(f"Killed PID={pid} (screen session: {name})")
                killed = True
            except ProcessLookupError:
                logging.warning(f"PID={pid} already dead")

        result = subprocess.run(['screen', '-S', f'{screen_pid}.{name}', '-X', 'quit'],
                                capture_output=True, check=False)
        return killed or result.returncode == 0

    def monitor_child_pids(self, name: str, callback: callable):
        """Invoke callback when child PIDs change, polling every update_interval seconds."""
        last_pids = []
        while True:
            current_pids = self.get_child_pids(name)
            if current_pids != last_pids:
                callback(name, current_pids)
                last_pids = current_pids
            time.sleep(self.update_interval)



class SimulatorManager:
    def __init__(
        self,
        route_info_path='./leaderboard/route_data',
        full_route_path='./leaderboard/route_data/bench2drive220.xml',
        base_pid_path='./leaderboard/pid',
        base_command="bash leaderboard/scripts/run_evaluation.sh",
        num_gpu=8,
        used_gpu=[0, 1, 2, 3, 4, 5, 6, 7],
        port_list_0=[30000, 30150, 30300, 30450, 30600, 30750, 30900, 31050, 31200, 31350],
        port_list_1=[50000, 50150, 50300, 50450, 50600, 50750, 50900, 51050, 51200, 51350],
        base_route_path="leaderboard/route_data",
        route_version="CutIn",
        config_path="projects_sp/configs/inference/aligndrive_inference.py",
        weights_path="./ckpt/aligndrive_iter_11004.pth",
        control_version='2m5hz',
        base_save_path="output/AlignDrive",
        save_files = ['leaderboard', 'projects_sp', 'scripts', 'tools'],
        auto_check=True,
        only_evaluate=False,
        args=None,
    ):
        if num_gpu < 1 or num_gpu > min(len(used_gpu), len(port_list_0), len(port_list_1)):
            raise ValueError('num_gpu must fit the supplied GPU and port lists')
        if len(set(used_gpu[:num_gpu])) != num_gpu:
            raise ValueError('Each simulator must use a distinct GPU')
        self.route_info_path = repo_path(route_info_path)
        self.full_route_path = repo_path(full_route_path)
        self.base_command = ('bash ' + shlex.quote(repo_path('leaderboard/scripts/run_evaluation.sh'))
                             if base_command == 'bash leaderboard/scripts/run_evaluation.sh' else base_command)
        self.num_gpu = num_gpu
        self.used_gpu = used_gpu
        self.port_list_0 = port_list_0
        self.port_list_1 = port_list_1
        self.base_route_path = repo_path(base_route_path)
        self.route_version = route_version
        self.config_path = repo_path(config_path)
        self.weights_path = repo_path(weights_path)
        self.control_version = control_version
        self.auto_check = auto_check
        self.args = args
        self.base_save_path = repo_path(base_save_path)
        self.base_pid_path = repo_path(base_pid_path)
        self.only_evaluate = only_evaluate
        self.save_files = save_files
        self.clean_file = repo_path('carla_tools/clean_carla_for_run_scene.sh')
        # Set up paths
        # self.init(base_save_path)

        # Generate route XML files
        if route_version == 'full':
            self.route_version = 'bench2drive220_'
            self.base_route_path = os.path.join(self.base_save_path, 'routes')
            write_route_shards(self.full_route_path, num_gpu, self.base_route_path, 'bench2drive220')
        else:
            self.base_route_path = os.path.join(self.base_save_path, 'routes')
            self.route_version = genereate_xml_from_json(
                os.path.join(self.route_info_path, route_version + '.json'),
                self.full_route_path,
                num_gpu,
                diffculty=['medium', 'difficult'],
                output_dir=self.base_route_path,
            )
        
        self.nick_name_pretext = 'eval_'
        # Set controller parameters based on control version
        if control_version == '5m2hz':
            self.controler_paranter = os.path.join(self.base_pid_path, 'control_5m2hz.json')
        elif control_version == '2m5hz':
            self.controler_paranter = os.path.join(self.base_pid_path, 'control_2m5hz.json')
        else:
            raise ValueError(f"Unknown control version: {control_version}")


    def init(self, base_save_path=None):
        if base_save_path is None:
            base_save_path = self.base_save_path
        base_save_path = repo_path(base_save_path)
        # base_save_path = base_save_path + '_' + self.control_version
        base_result_path = os.path.join(base_save_path, "result")
        base_log_path = os.path.join(base_save_path, "log")
        base_files_save_path = os.path.join(base_save_path, "files")
        visulization_path = os.path.join(base_save_path, "visulization")

        self.base_log_path = base_log_path
        self.base_result_path = base_result_path
        self.base_save_path = base_save_path
        self.visulization_path = visulization_path

        # Create directories
        os.makedirs(base_save_path, exist_ok=True)
        os.makedirs(base_result_path, exist_ok=True)
        os.makedirs(base_log_path, exist_ok=True)
        os.makedirs(visulization_path, exist_ok=True)
        os.makedirs(base_files_save_path, exist_ok=True)
        print("Files:", self.save_files, "have been saved to:", base_files_save_path)
        if self.args is not None and not self.only_evaluate:
            # Save the parsed arguments in the output directory.
            with open(os.path.join(self.base_save_path, "args.json"), "w") as f:
                json.dump(vars(self.args), f, indent=4)
            print("args have been saved to:", os.path.join(self.base_save_path, "args.json"))
        if not self.only_evaluate:
            # Save the files used for this run.
            self.store_files(self.save_files, base_files_save_path)
            # Save the controller parameters in the output directory.
            with open(self.controler_paranter, "r") as src_file:
                data = json.load(src_file)  # Read the controller parameters.
            with open(os.path.join(self.base_save_path, "controler_paranter.json"), "w") as f:
                json.dump(data, f, indent=4)  # Save the actual parameter values.
            print("PID params have beed saved to", self.base_save_path )

    def store_files(self, save_files, save_path):
        """
        Copy the specified files or directories to the destination.
        Args:
        save_files (list): Files or directories to copy.
        save_path (str): Destination directory.
        """
        for file in save_files:
            source_path = repo_path(file)
            destination_path = os.path.join(save_path, file)
            
            # Copy directories recursively.
            if os.path.isdir(source_path):
                os.makedirs(destination_path, exist_ok=True)
                # Keep relative paths when traversing the directory.
                sub_files = [os.path.join(file, f) for f in os.listdir(source_path)]
                self.store_files(sub_files, save_path)  # Keep the original destination root.
            # Copy regular files directly.
            elif os.path.isfile(source_path):
                os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                shutil.copy2(source_path, destination_path)
            else:
                print(f"Warning: {source_path} is neither a file nor a directory.")




    def set_controller_info(self, controler_paranter):
        # Override controller parameters for PID debugging.
        self.controler_paranter = os.path.join(self.base_pid_path, controler_paranter)
        if not '.json' in self.controler_paranter:
            self.controler_paranter = self.controler_paranter + '.json'


    def run_evaluation_pipeline(self):
        command_dict = self.build_command(
            self.base_command,
            self.num_gpu,
            self.used_gpu,
            self.port_list_0,
            self.port_list_1,
            self.base_route_path,
            self.route_version,
            self.config_path,
            self.weights_path,
            self.visulization_path,
            self.nick_name_pretext,
            self.controler_paranter,
            self.base_log_path,
            self.base_result_path
        )


        # Launch the generated commands.
        manager = ScreenManager(self.base_save_path)
        for name, cmd in command_dict.items():
            manager.start_command(name, cmd)
        with open(f"{self.base_save_path}/sessions.json", "w") as f:
            json.dump(list(command_dict.keys()), f)
        print("Simulation tasks started.")
        
        # Extract command metadata, including log paths and screen names.
        commands_info = []
        for i, cmd in enumerate(command_dict):
            cmd = command_dict[self.nick_name_pretext+str(i)]
            screen_name = f"eval_{i}"
            log_path = os.path.join(self.base_log_path, str(i)+'.log')
            result_path = os.path.join(self.base_result_path, str(i)+'.json')
            commands_info.append({
                "raw_command": cmd,
                "screen_name": screen_name,
                "log_path": log_path,
                "result_path": result_path  # Result JSON file.
            })

        self.commands_info = commands_info
        print("colleced simulator info")

        # Start monitoring.
        self.monitor_simulation(commands_info)
    

    def kill_scene(self, info):
    # Terminate the existing screen session.
        kill_args = ["bash", self.clean_file, info["screen_name"]]
        subprocess.run(kill_args,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL)

    def restart_scene(self, info):
        # Restart the command.
        try:
            subprocess.run(
                ["screen", "-dmS", info["screen_name"], "bash", "-c", 'set -o pipefail; ' + info["raw_command"]],
                cwd=REPO_ROOT, check=True
            )
            print(f"🔄 Restarted {info['screen_name']}")
            # Clear the previous log.
            open(info["log_path"], "w").close()
        except subprocess.CalledProcessError as e:
            print(f"🔥 Restart failed {info['screen_name']}: {e}")

    def get_running_status(self, info):
        # Check whether evaluation has completed.
        results_path = info["result_path"]
        # Read the result JSON file.
        if os.path.exists(results_path):
            try:
                with open(results_path, 'r') as f:
                    data = json.load(f)
            except:
                print(f"⚠️ Failed to read result file [{results_path}]")
                return False, None
            # Read the global result record.
            _checkpoint = data.get("_checkpoint", {})
            global_record = _checkpoint.get("global_record", {})
            status = global_record.get("status", False)
            end_flag = False
            if status:
                print(f"✅ {info['screen_name']} Evaluation complete.")
                end_flag = True
            return end_flag, status

        else:
            return False, None





    def monitor_simulation(self, commands_info):
        finished_scenes = []
        print("Starting automatic monitoring...")
        sleep(20)
        finish_flag=False
        # Monitoring loop.
        while True:
            print("\n" + "="*50)
            print(f"Starting a new monitoring round ({time.ctime()})")
            for i in range(len(commands_info)):
                info = commands_info[i]
                if not info["log_path"]:
                    continue
                    
                print(f"Checking {info['screen_name']}...", end=" ")
                # if contains_errors(info["log_path"]):
                if self.auto_check and self.contains_errors(info["log_path"]):
                    print(i , "Error detected; attempting restart...")
                    # Back up the error log.
                    wrong_flag, wrong_info = self.contains_errors(info["log_path"], return_wrong_info=True)
                    if wrong_flag:
                        print("Error details:", wrong_info)
                    error_log_path = os.path.join(self.base_log_path, "error_" + str(i)+'.log')
                    os.rename(info["log_path"], error_log_path)

                    # Terminate the existing screen session.
                    self.kill_scene(info)
                    # Restart the command.
                    self.restart_scene(info)
                # else:
                # print("Running normally")

                # Check whether the evaluation has completed.
                end_flag, status = self.get_running_status(info)

                if end_flag:
                    if i not in finished_scenes:
                        finished_scenes.append(i)
                        print(i, "Scene complete.", "status:", status)
                        self.kill_scene(info) # Avoid leaving residual processes.
                    else:
                        print(i, "Scene already completed.")

            # Check whether every scene has completed.
            if len(finished_scenes) >= len(commands_info):
                print("All scenes completed.")
                finish_flag = True
                break
            
            else:
                print(f"Completed scenes: {len(finished_scenes)}/{len(commands_info)}")


            print("="*50 + "\n")
            time.sleep(60)  # Check once every 60 seconds.

            if finish_flag:
                print("Monitoring stopped.")
                break



    def contains_errors(self, log_path, return_wrong_info=False):
        error_keywords = [
            "Watchdog exception",
            "AttributeError",
            # "Timeout",
            "RuntimeError",
            "fault",
            # "cost time",
            "time-out",
            "60000ms",
            "UnboundLocalError",
        ]
    
        try:
            with open(log_path, "r") as f:
                # Read only the last ten log lines.
                lines = []
                for line in reversed(f.readlines()):
                    lines.append(line)
                    if len(lines) >= 20:
                        break
                content = "".join(reversed(lines))
                if return_wrong_info:
                    for kw in error_keywords:
                        if kw in content:
                            return True, kw
                else:
                    return any(kw in content for kw in error_keywords)
        except Exception as e:
            print(f"⚠️ Failed to inspect log [{log_path}]: {str(e)}")
            return False
    




    # Build commands for each GPU
    def build_command(
        self,
        base_command,
        num_gpu,
        used_gpu,
        port_list_0,
        port_list_1,
        base_route_path,
        route_version,
        config_path,
        weights_path,
        visulization_path,
        nick_name_pretext,
        controler_paranter,
        base_log_path,
        base_result_path,
    ):
        commands = []
        for i in range(num_gpu):
            gpu = str(used_gpu[i])
            arguments = [str(port_list_0[i]), str(port_list_1[i]), 'True',
                         os.path.join(base_route_path, route_version + str(i) + '.xml'),
                         repo_path('leaderboard/team_code/aligndrive_b2d_agent.py'),
                         config_path + '+' + weights_path,
                         os.path.join(base_result_path, str(i) + '.json'),
                         visulization_path, 'traj', gpu, controler_paranter]
            command_i = base_command + ' ' + shlex.join(arguments)
            command_i += ' 2>&1 | tee ' + shlex.quote(os.path.join(base_log_path, str(i) + '.log'))
            commands.append(command_i)
        
        command_dict = {}
        for i in range(num_gpu):
            command_dict[nick_name_pretext+str(i)] = commands[i]
        return command_dict
    
