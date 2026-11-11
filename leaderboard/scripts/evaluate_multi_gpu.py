"""Run the configurable multi-GPU Bench2Drive simulation and result-collection pipeline."""

import sys 
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))
import os
from leaderboard.scripts.process_manager import SimulatorManager
from carla_tools.paths import repo_path
import argparse

PORTList_0=[30000, 32030, 30300, 30450, 30600, 30750, 30900, 31050, 31200, 31350]
PORTList_1=[50000, 52050, 50300, 50450, 50600, 50750, 50900, 51050, 51200, 51350]

def parse_args():
    parser = argparse.ArgumentParser(description='Run the configurable multi-GPU Bench2Drive evaluation pipeline')

    parser.add_argument('--weights_path', type=str,
                        default='./ckpt/aligndrive_iter_11004.pth',
                        help='Path to model weights')
    parser.add_argument('--project_path', default='output/AlignDrive', help='Model output directory')
    parser.add_argument('--project_name', default='AlignDrive', help='Tag for this evaluation run')
    parser.add_argument('--evaluate_output', default='./output/compare', help='Directory for comparing model results')
    parser.add_argument('--base_pid_path', default='./leaderboard/pid', help='Directory containing PID controller presets')
    parser.add_argument('--evaluate_task', nargs='+', choices=['metric', 'visulization'], default=['metric'], help='Metrics and visualization tasks; avoid visualization for the full split')
    parser.add_argument('--no-auto-restart', action='store_true', help='Monitor without restarting failed simulations')
    parser.add_argument('--controler_paranter',
                        nargs='+',  # Accept one or more preset names.
                        default=['control_2m5hz'],  # Default controller preset.
                        help='PID preset names, separated by spaces ')
    parser.add_argument('--route_version', type=str,  # full: complete split; other names select scenario subsets.
                        choices=['full', 'CutIn', 'Pedestrain_Collision', "Ablation", "SelfCutIn", "RightTurn", "LaneChange", "TrafficLight"],
                        default='full') 
    parser.add_argument('--control_version', type=str, 
                        default='2m5hz')
    parser.add_argument('--only_evaluate', action='store_true', default=False,)
    parser.add_argument('--render_version', type=str, 
                        default='full')   
    parser.add_argument('--bev_view', type=str, 
                        default='topdown')
    parser.add_argument('--route_info_path', type=str, 
                        default='./leaderboard/route_data',
                        help='Path to route information directory')
    parser.add_argument('--full_route_path', type=str,
                        default='./leaderboard/route_data/bench2drive220.xml',
                        help='Path to full route XML file')
    parser.add_argument('--base_command', type=str,
                        default="bash leaderboard/scripts/run_evaluation.sh",
                        help='Base command to run evaluation')
    parser.add_argument('--num_gpu', type=int, default=8,
                        help='Number of GPUs to use')
    parser.add_argument('--used_gpu', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7],
                        help='List of GPU indices to use')
    parser.add_argument('--base_route_path', type=str, default="leaderboard/route_data",
                        help='Base path for route files')
    parser.add_argument('--config_path', type=str, default="projects_sp/configs/inference/aligndrive_inference.py",
                        help='Path to model config file')
    return parser.parse_args()

def main():
    args = parse_args()
    from carla_tools.evaluate_utils import Evaluation
    args.project_path = repo_path(args.project_path)
    args.evaluate_output = repo_path(args.evaluate_output)
    args.project_name = args.project_name + '_' + args.route_version
    # Now you can access all parameters as attributes of args
    print(f"Running evaluation with route version: {args.route_version}")
    print(f"Using GPUs: {args.used_gpu}")
    base_save_path  = os.path.join(args.project_path, args.project_name)
    scene_simulator_mannager = SimulatorManager(
        route_info_path=args.route_info_path,
        full_route_path=args.full_route_path,
        base_pid_path=args.base_pid_path,
        base_command=args.base_command,
        num_gpu=args.num_gpu,
        used_gpu=args.used_gpu,
        port_list_0=PORTList_0,
        port_list_1=PORTList_1,
        base_route_path=args.base_route_path,
        route_version=args.route_version,
        config_path=args.config_path,
        weights_path=args.weights_path,
        control_version=args.control_version,
        base_save_path=base_save_path,
        auto_check=not args.no_auto_restart,
        only_evaluate=args.only_evaluate,
        args=args,
    )
    projects_list = [
    ]
    # print("projects_list:", projects_list)
    task = args.evaluate_task
    if args.route_version == 'full':
        scene_task = 'all'
        args.evaluate_output = os.path.join(args.evaluate_output, 'Full_Scene')
    elif args.route_version == 'CutIn':
        scene_task = 'cutin'
        args.evaluate_output = os.path.join(args.evaluate_output, 'CutIn')
    elif args.route_version == 'Pedestrain_Collision':
        scene_task = 'pedestrian_collision'
        args.evaluate_output = os.path.join(args.evaluate_output, 'Pedestrain_Collision')
    elif args.route_version == 'Ablation':
        scene_task = 'ablation'
        args.evaluate_output = os.path.join(args.evaluate_output, 'Ablation')
    elif args.route_version == 'SelfCutIn':
        scene_task = 'selfcutin'
        args.evaluate_output = os.path.join(args.evaluate_output, 'SelfCutIn')
    elif args.route_version == 'RightTurn':
        scene_task = 'rightturn'
        args.evaluate_output = os.path.join(args.evaluate_output, 'RightTurn')
    elif args.route_version == 'LaneChange':
        scene_task = 'lanechange'
        args.evaluate_output = os.path.join(args.evaluate_output, 'LaneChange')
    elif args.route_version == 'TrafficLight':
        scene_task = 'trafficlight'
        args.evaluate_output = os.path.join(args.evaluate_output, 'TrafficLight')


    if len(args.controler_paranter) > 0:
        print("Evaluating multiple PID parameters...")
        for i in range(len(args.controler_paranter)):
            # Update the PID controller preset.
            control_parms_path = args.controler_paranter[i]
            scene_simulator_mannager.set_controller_info(control_parms_path)
            # Update the output directory.
            project_name_with_pid = args.project_name
            if len(args.controler_paranter) > 1:
                project_name_with_pid += '_PID_' + str(args.controler_paranter[i])
            new_save_path = os.path.join(args.project_path, project_name_with_pid)
            scene_simulator_mannager.init(new_save_path)
            # Initialize the evaluator.
            Evaluator = Evaluation(args.project_path)
            Evaluator.init(project_name_with_pid)

            # Override the PID controller preset.
            print("PID controller preset changed to", control_parms_path)
            # Run the simulator.
            if not args.only_evaluate:
                scene_simulator_mannager.run_evaluation_pipeline()
                print("Start collecting results...")

                # Collect results.
                Evaluator.merage_result()
                current_project = os.path.join(args.project_path, project_name_with_pid)
                if current_project not in projects_list:
                    projects_list.append(current_project)

                if scene_task == 'all' or scene_task == 'ablation':
                    Evaluator.collect_fail_scene()
                    Evaluator.anaylize_fail_scene()
                    # Evaluator.visulize_fail_scene()
                else:
                    Evaluator.collect_select_scene(projects_list, scene_task, output_path=args.evaluate_output, task=task, control_version=args.control_version, render_version=args.render_version, bev_view=args.bev_view) 

            else:
                Evaluator.collect_select_scene(projects_list, scene_task, output_path=args.evaluate_output, task=task, control_version=args.control_version, render_version=args.render_version, bev_view=args.bev_view) 

    else:
        
        scene_simulator_mannager.init()
        # Initialize the evaluator.
        Evaluator = Evaluation(args.project_path)
        Evaluator.init(args.project_name)
        # Run the simulator.
        if not args.only_evaluate:
            scene_simulator_mannager.run_evaluation_pipeline()
        # Collect results.
        print("Start collecting results...")
        Evaluator.merage_result()
        current_project = os.path.join(args.project_path, args.project_name)
        if current_project not in projects_list:
            projects_list.append(current_project)

        if scene_task == 'all' or scene_task == 'ablation':
            Evaluator.collect_fail_scene()
            Evaluator.anaylize_fail_scene()
            # Evaluator.visulize_fail_scene()
        else:
            Evaluator.collect_select_scene(projects_list, scene_task, output_path=args.evaluate_output, task=task, control_version=args.control_version) 

if __name__ == "__main__":
    main()
