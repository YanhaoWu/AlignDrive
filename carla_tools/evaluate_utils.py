import json
import glob
import argparse
import os

from tqdm import tqdm
from tools.visualization.simulation import SimulationBEVRenderer, SimulationRenderer
import re
from collections import defaultdict
import shutil
from carla_tools.paths import repo_path

class Evaluation:
    def __init__(self, projects_path, weights_version=None):
        self.projects_path = projects_path
        weights_version = weights_version or repo_path('carla_tools/version_id.json')
        with open(weights_version, 'r') as f:
            self.weights_version = json.load(f)
            print("[INFO] Weight versions: ", self.weights_version)

    def init(self, project_name='Full_Scene'):
        self.project_name = project_name
        self.project_path = os.path.join(self.projects_path, project_name)
        self.result_path = os.path.join(self.project_path, 'result')
        self.visulization = os.path.join(self.project_path,'visulization')
        self.compare_path = os.path.join(self.project_path,'compare')
        self.fail_scene_path = os.path.join(self.result_path, 'fail_scene_info.json')
        self.merged_result_path = os.path.join(self.result_path,'merged.json')

        self.fail_scene_vis_path = os.path.join(self.project_path, 'fail_scene')
        os.makedirs(self.fail_scene_vis_path, exist_ok=True)

        # Initialize scenario subsets for collecting targeted evaluation results.
        cutin_scene_path = repo_path('leaderboard/route_data/CutIn.json')
        pedestrian_collision_scene_path = repo_path('leaderboard/route_data/Pedestrain_Collision.json')
        ablation_scene_path = repo_path('leaderboard/route_data/Ablation.json')
        selfcutin_scene_path = repo_path('leaderboard/route_data/SelfCutIn.json')
        rightturn_scene_poath = repo_path('leaderboard/route_data/RightTurn.json')
        lanechange_scene_path = repo_path('leaderboard/route_data/LaneChange.json')
        trafficlight_scene_path = repo_path('leaderboard/route_data/TrafficLight.json')
        cutin_scene_list = self.collect_scene_id(cutin_scene_path, focus=['collisions_vehicle'])
        pedestrian_collision_scene_list = self.collect_scene_id(pedestrian_collision_scene_path, focus=['collisions_pedestrian'])
        ablation_scene_list = self.collect_scene_id(ablation_scene_path, focus=['collisions_vehicle', 'collisions_pedestrian'])
        selfcutin_scene_list = self.collect_scene_id(selfcutin_scene_path, focus=['collisions_vehicle', 'collisions_pedestrian'])
        rightturn_scene_list = self.collect_scene_id(rightturn_scene_poath, focus=['outside_route_lanes', 'collisions_layout'])
        lanechange_scene_list = self.collect_scene_id(lanechange_scene_path, focus=['route_dev', 'outside_route_lanes'])
        trafficlight_scene_list = self.collect_scene_id(trafficlight_scene_path, focus=['red_light'])

        self.select_scene_list = dict()
        self.select_scene_list['cutin'] = cutin_scene_list
        self.select_scene_list['pedestrian_collision'] = pedestrian_collision_scene_list
        self.select_scene_list['ablation'] = ablation_scene_list
        self.select_scene_list['selfcutin'] = selfcutin_scene_list
        self.select_scene_list['rightturn'] = rightturn_scene_list
        self.select_scene_list['lanechange'] = lanechange_scene_list
        self.select_scene_list['trafficlight'] = trafficlight_scene_list


        # Video format settings.
        self.fps = 10
        self.font_scale = 0.5
        self.text_color = (255, 255, 255)
        self.text_position = (50, 50)


        print("[INFO] Initialization complete...")
        # Print output paths.
        print("[INFO] Project path: ", self.project_path)
        print("[INFO] Results path: ", self.result_path)
        print("[INFO] Visualization path: ", self.visulization)

    def collect_select_scene(self, projects_list, scene_task=None, select_scene_list=None, output_path=None, task=['metric', 'visulization'], focus=None, control_version=None, render_version='full', bev_view='topdown'):
        # Collect results according to the requested task.
        if scene_task is not None:
            select_scene_list = self.select_scene_list[scene_task]['scenes']
            focus = self.select_scene_list[scene_task]['focus']
            print("[INFO] Selected scenarios: ", select_scene_list)
        else:
            print("[INFO] No scenarios specified; collecting all scenes from", select_scene_list)

        # Collect metrics.
        if 'metric' in task:
            self.collect_select_scene_result(projects_list, select_scene_list, output_path, focus)

        # Collect visualization results.
        if'visulization' in task:

            self.collect_select_scene_visulization(projects_list, select_scene_list, output_path, control_version=control_version, render_version=render_version, bev_view=bev_view)

    def check_sucess(self, scene_result, keys=None, focus_test=False):
        rd = scene_result
        if not focus_test:
            if rd['status']=='Completed' or rd['status']=='Perfect':
                success_flag = True
                for k,v in rd['infractions'].items():
                    if keys is None:
                        if len(v)>0 and k != 'min_speed_infractions':
                            success_flag = False
                            break
                    else:
                        # A route succeeds when none of the specified infractions occur.
                        if k in keys and len(v)>0:
                            success_flag = False
                            break
            else:
                success_flag = False
        else:
            success_flag = True
            for k,v in rd['infractions'].items():
                if keys is None:
                    if len(v)>0 and k != 'min_speed_infractions':
                        success_flag = False
                        break
                else:
                    # A route succeeds when none of the specified infractions occur.
                    if k in keys and len(v)>0:
                        success_flag = False
                        break
        return success_flag


    def collect_select_scene_result(self, projects_list, select_scene_list, output_path=None, focus=None):
        # Collect metrics.
        select_scene_result = {}
        for project in projects_list:
            project_name = project.split('/')[-2]
            test_version = project.split('/')[-1]
            project_name = str(self.weights_version[project_name]) + '_' + project_name + '_' + test_version
            result_path = os.path.join(project, 'result')
            meraged_result_json = os.path.join(result_path,'merged.json')
            # Read the JSON file.
            with open(meraged_result_json) as file:
                meraged_result = json.load(file)
            # Collect metrics for the selected scenarios.
            for scene_results in meraged_result['_checkpoint']['records']:
                scene_name = scene_results['route_id']
                for scene in select_scene_list:
                    if scene in scene_name:
                        sucess_flag = self.check_sucess(scene_results, keys=None)
                        focus_sucess_flag = self.check_sucess(scene_results, keys=focus, focus_test=True)

                        # Record this project's status for the current scenario.
                        if scene not in select_scene_result:
                            select_scene_result[scene] = {}
                        select_scene_result[scene][project_name] = {}
                        select_scene_result[scene][project_name]['success'] = sucess_flag
                        select_scene_result[scene][project_name]['focus_success'] = focus_sucess_flag
                        break
        # Compute the mean success rate per scenario.
        for scene, project_results in select_scene_result.items():
            success_num = 0
            focus_success_num = 0
            for project, sucess_flag in project_results.items():
                if sucess_flag['success']:
                    success_num += 1
                if sucess_flag['focus_success']:
                    focus_success_num += 1

            # Compute the success rate.
            total_projects = len(project_results)
            if total_projects > 0:
                select_scene_result[scene]['_success_rate'] = success_num / total_projects
                select_scene_result[scene]['_focus_success_rate'] = focus_success_num / total_projects
            else:
                select_scene_result[scene]['_success_rate'] = 0.0
                select_scene_result[scene]['_focus_success_rate'] = 0.0
        # Compute the mean success rate per project.
        project_success_rates = {}  # Store {project_name: success_rate}.
        
        # First collect success and failure counts for every project.
        project_stats = defaultdict(lambda: {'success': 0, 'focus_sucess':0, 'total': 0})

        for scene_data in select_scene_result.values():
            for project_name, success_flag in scene_data.items():
                if isinstance(success_flag, dict):
                    project_stats[project_name]['total'] += 1
                    if success_flag['success']:
                        project_stats[project_name]['success'] += 1
                    if success_flag['focus_success']:
                        project_stats[project_name]['focus_sucess'] += 1
                    project_success_rates[project_name] = {}


        # Compute each project's success rate.
        for project_name, stats in project_stats.items():
            if stats['total'] > 0:  # Avoid division by zero.
                success_rate = stats['success'] / stats['total']
                focus_success_rate = stats['focus_sucess'] / stats['total']
                project_success_rates[project_name]['success_rate'] = success_rate
                project_success_rates[project_name]['focus_sucess_rate'] = focus_success_rate
            else:
                project_success_rates[project_name] = 0.0  # Default to zero when there is no data.



        # Add the results to the final output.
        select_scene_result['_project_success_rates'] = project_success_rates
        # Write the output JSON file.
        if output_path is None:
            output_path = os.path.join(self.compare_path, 'result')
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, 'select_scene_result.json'), 'w') as file:
            json.dump(select_scene_result, file, indent=4)
        print("[INFO] Collection complete; results saved to: ", os.path.join(output_path,'select_scene_result.json'))
    


    def collect_select_scene_visulization(self, projects_list, select_scene_list, output_path=None, control_version=None, render_version='presentation', bev_view='topdown'):
        # Collect selected scenarios from the requested projects.
        # Iterate over projects.
        for project in projects_list:
            project_name = project.split('/')[-2]
            test_version = project.split('/')[-1]
            project_id = str(self.weights_version[project_name])  + '_' + test_version
            visulization_path = os.path.join(project, 'visulization')
            # Iterate over scenarios.
            for scene in select_scene_list:
                # Build the scenario path.
                print("[INFO] Processing scenario: ", scene, " in Project: ", project_id)
                video_path = None

                scene_obtained = self.serach_scene_path(scene, visulization_path, clean_scene=True)
                scene_obtained = self.serach_scene_path(scene, visulization_path, clean_scene=False)

                if scene_obtained is None:
                    print("[ERROR] Scenario not found: ", scene, " in Project: ", project_name)
                    continue
                visulization_path = os.path.join(project, 'visulization')   # Use the original visualization directory name.
                scene_path = os.path.join(visulization_path, scene_obtained)
                # Find the video.
                video_base_path = os.path.join(scene_path, 'video/bev_pred')
                if video_path is None:
                    # Generate the video.
                    bev_render = SimulationRenderer(out_dir=os.path.join(scene_path, 'video'), render_version=render_version,  bev_view=bev_view)
                    video_path = bev_render.create_video_parallel(scene_path, self.fps, self.font_scale, self.text_color, self.text_position, scene_name=scene_obtained, bev_view=bev_view, sp_waypoints_show=True)

                if scene_path is not None:
                    # Build the output path.
                    scene_obtained_no_prefix = re.sub(r'(_\d+)+$', '', scene_obtained)  # Remove trailing numeric suffixes such as _1_2.
                    print("[INFO] Scenario: ", scene_obtained_no_prefix)
                    if output_path is None:
                        video_output_path = os.path.join(self.compare_path, 'visulization', scene_obtained_no_prefix)
                    else:
                        video_output_path = os.path.join(output_path, scene_obtained_no_prefix)
                # Copy the video to the requested destination.
                    os.makedirs(video_output_path, exist_ok=True)
                    # scene_obtained_no_prefix = scene_obtained.split('_1_')[0]

                    video_output_path = os.path.join(video_output_path, 'ID'+project_id+'_'+scene_obtained_no_prefix+'.mp4')
                    
                    os.system(f'cp "{video_path}" "{video_output_path}"')
                    print("[INFO] Copy complete, from: ", video_path, " to: ", video_output_path)




    def serach_scene_path(self, scene_name, visualization_path, clean_scene=False):
        # Iterate over scenarios.
        contained_scenes = []
        for scene in os.listdir(visualization_path):
            if scene_name in scene:
                contained_scenes.append(scene)
        if len(contained_scenes) == 0:
            return None
        elif len(contained_scenes) == 1:
            return contained_scenes[0]
        elif clean_scene:
            print("[WARNING] Multiple matching scenarios; selecting the newest")
            abs_scene_path = []
            for i in range(len(contained_scenes)):
                scene_path = os.path.join(visualization_path, contained_scenes[i])
                abs_scene_path.append(scene_path)
                
            # Select the most recently created matching scenario.
            latest_scene = max(abs_scene_path, key=lambda x: os.path.getctime(x))
            print("[INFO] Selected newest scenario: ", latest_scene)
            # Optionally delete older matching scenarios.
            for scene in abs_scene_path:
                if scene != latest_scene:
                    try:
                        shutil.rmtree(scene)
                        # print(f"Deleted old scene file: {scene}")
                    except OSError as e:
                        print(f"Error deleting file {scene}: {e}")
            #
            last_scene = latest_scene.split('/')[-1]

            return latest_scene
        else:
            return contained_scenes[0]




    def anaylize_fail_scene(self):
        # Read the JSON file.
        with open(self.fail_scene_path) as file:
            fail_scene_info = json.load(file)
        # Collect all infractions.
        all_infractions = {}
        # Count occurrences of each infraction type.
        for fail_scene in fail_scene_info:
            for k,v in fail_scene.items():
                if k!='route_id' and k!='save_name':
                    if k not in all_infractions and len(v)>0:
                        all_infractions[k] = 1
                    elif len(v)>0:
                        all_infractions[k] += 1
            
        print("[INFO] All infractions: ", all_infractions)
        # Save the results as JSON.
        with open(os.path.join(self.result_path,'all_infractions.json'), 'w') as file:
            json.dump(all_infractions, file, indent=4)


    def visulize_fail_scene(self):
        folder_path = self.visulization
        with open(self.fail_scene_path) as file:
            fail_scene_info = json.load(file)
        bev_render = SimulationBEVRenderer(
            out_dir=os.path.join(self.fail_scene_vis_path),
        )
        print("[INFO] Visualizing failed scenarios: ")
 
        for fail_scene in tqdm(fail_scene_info):
            route_name = fail_scene['save_name']
            images_folder = os.path.join(folder_path, route_name)
            bev_render.create_video(images_folder, self.fps, self.font_scale, self.text_color, self.text_position, scene_name=route_name, view_name='bev', sp_waypoints_show=True)




    def collect_fail_scene(self):
        folder_path = self.result_path
        file_paths = glob.glob(f'{folder_path}/*.json')
        all_fail_scenes = []
        fail_scene = {} 
        success_num = 0

        fliterd_files = []
        for files in file_paths:
            if 'fail_scene_info.json' in files:
                continue
            if 'merged' in files:
                continue
            if 'all' in files:
                continue
            fliterd_files.append(files)
        file_paths = fliterd_files
        total_nums = 0 

        # print("FILE_PATHS :", file_paths)
        for file_path in file_paths:
            with open(file_path) as file:
                data = json.load(file)
                records = data['_checkpoint']['records']
                for rd in records:
                    fail_scene = {}
                    if rd['status']=='Completed' or rd['status']=='Perfect':
                        success_flag = True
                        for k,v in rd['infractions'].items():
                            if len(v)>0 and k != 'min_speed_infractions':
                                success_flag = False
                                break
                        if success_flag:
                            success_num += 1
                        else:
                            fail_scene['route_id'] = rd['route_id']
                            # Save all infractions except min_speed_infractions.
                            for k,v in rd['infractions'].items():
                                if k!='min_speed_infractions':
                                    fail_scene[k] = v
                            fail_scene['save_name'] = rd['save_name']
                            all_fail_scenes.append(fail_scene.copy())
                    else:
                        # Timed-out scenarios.
                        fail_scene['route_id'] = rd['route_id']
                        fail_scene['save_name'] = rd['save_name']
                        # Save all infractions except min_speed_infractions.
                        for k,v in rd['infractions'].items():
                            if k!='min_speed_infractions':
                                fail_scene[k] = v
                        fail_scene[rd['status']] = rd['status']
                        all_fail_scenes.append(fail_scene.copy())



        with open(os.path.join(folder_path, 'fail_scene_info.json'), 'w') as file:
            json.dump(all_fail_scenes, file, indent=4)
        
        print("[INFO] Failed scenario collection complete...")
        print("[INFO] Successful scenarios: ", success_num)
        print("[INFO] Failed scenarios: ", len(all_fail_scenes))
        print("[INFO] Output path: ", self.fail_scene_path)
        # print("[INFO] Result attribute: self.fail_scene_path")

    def merage_result(self):
        # Merge results from multiple scenarios.
        folder_path = self.result_path
        file_paths = glob.glob(f'{folder_path}/*.json')
        merged_records = []
        driving_score = []
        success_num = 0
        for file_path in file_paths:
            if 'merged.json' in file_path: continue
            if 'fail_scene_info.json' in file_path: continue
            if 'infractions.json' in file_path: continue
            with open(file_path) as file:
                data = json.load(file)
                records = data['_checkpoint']['records']
                for rd in records:
                    rd.pop('index')
                    merged_records.append(rd)
                    driving_score.append(rd['scores']['score_composed'])
                    if rd['status']=='Completed' or rd['status']=='Perfect':
                        success_flag = True
                        for k,v in rd['infractions'].items():
                            if len(v)>0 and k != 'min_speed_infractions':
                                success_flag = False
                                break
                        if success_flag:
                            success_num += 1
                            # print(rd['route_id'])
        if len(merged_records) != 220:
            print(f"-----------------------Warning: there are {len(merged_records)} routes in your json, which does not equal to 220. All metrics (Driving Score, Success Rate, Ability) are inaccurate!!!")
        merged_records = sorted(merged_records, key=lambda d: d['route_id'], reverse=True)
        _checkpoint = {
            "records": merged_records
        }

        merged_data = {
            "_checkpoint": _checkpoint,
            "driving score": sum(driving_score) / len(merged_records),
            "success rate": success_num / len(merged_records),
            "eval num": len(driving_score),
        }

        with open(os.path.join(folder_path, 'merged.json'), 'w') as file:
            json.dump(merged_data, file, indent=4)

        print("[INFO] Result merge complete...")
        print("[INFO] Success rate: ", success_num / len(merged_records))
        print("[INFO] Driving Score: ", sum(driving_score) / len(merged_records))
        print("[INFO] Output path: ", self.merged_result_path)
        print("[INFO] Result attribute: self.merged_result_path")



    
    def collect_scene_id(self, route_path, focus):
        # Collect scenario identifiers for subsequent comparison.
        diffculty = ['medium', 'difficult']
        select_scene_list = []
        with open(route_path, 'r') as f:
            routes = json.load(f)
        for keys in diffculty:
            select_scene_list.extend(routes[keys])
        for i in range(len(select_scene_list)):
            select_scene_list[i] = str(select_scene_list[i])+'_'
        print("Collected scenes from ", route_path)

        scene_info = {
           'scenes': select_scene_list,
           'route_path': route_path,
           'focus': focus,
        }

        return scene_info
