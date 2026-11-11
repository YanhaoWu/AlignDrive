import xml.etree.ElementTree as ET
import json
import os
def split_list_into_n_parts(lst, n):
    if n < 1:
        raise ValueError('The number of route shards must be positive')
    k, m = divmod(len(lst), n)
    return (lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def main(base_route, task_num, algo, planner_type):
    tree = ET.parse(f'{base_route}.xml')
    root = tree.getroot()
    case = root.findall('route')
    results = split_list_into_n_parts(case, task_num)
    for index, re in enumerate(results):
        new_root = ET.Element("routes")
        for x in re:
            new_root.append(x)
        new_tree = ET.ElementTree(new_root)
        new_tree.write(f'{base_route}_{index}_{algo}_{planner_type}.xml', encoding='utf-8', xml_declaration=True)


# Select routes using the JSON file, then split them across GPUs.
def write_route_shards(xml_path, gpu_num, output_dir, prefix):
    routes = ET.parse(xml_path).getroot().findall('route')
    if gpu_num > len(routes):
        raise ValueError('There must be at least one route per GPU')
    os.makedirs(output_dir, exist_ok=True)
    for index, shard in enumerate(split_list_into_n_parts(routes, gpu_num)):
        root = ET.Element('routes')
        root.extend(shard)
        ET.ElementTree(root).write(os.path.join(output_dir, f'{prefix}_{index}.xml'),
                                   encoding='utf-8', xml_declaration=True)


def genereate_xml_from_json(json_path, xml_path, gpu_num, diffculty=['medium', 'difficult'], output_dir=None):
    task_name = json_path.split('/')[-1].split('.')[0]
    output_path = output_dir or os.path.dirname(json_path) or '.'
    os.makedirs(output_path, exist_ok=True)
    # Read the JSON file.
    with open(json_path, 'r') as f:
        data = json.load(f)
    # Collect scenes from all requested difficulty levels.
    scene_list = []
    for diff in diffculty:
        scene_list.extend(data[diff])
    # Convert scene identifiers to strings.
    scene_list = [str(x) for x in scene_list]

    tree = ET.parse(xml_path)
    root = tree.getroot()
    case = root.findall('route')

    # Keep only scenes whose identifiers appear in the selected data.
    for i in range(len(case)):
        scene = case[i]
        scene_id = scene.attrib['id']
        if scene_id not in scene_list:
            root.remove(scene)
            i -= 1
            # print("Remove scene: ", scene_id)
    case = root.findall('route')


    if gpu_num > len(case):
        raise ValueError('There must be at least one selected route per GPU')
    results = split_list_into_n_parts(case, gpu_num)
    for index, re in enumerate(results):
        new_root = ET.Element("routes")
        for x in re:
            new_root.append(x)
        new_tree = ET.ElementTree(new_root)
        new_tree.write(os.path.join(output_path, f'{task_name}_{index}.xml'), encoding='utf-8', xml_declaration=True)

    print("Generate xml file successfully, task name:", task_name, "save to:", output_path, "xml file num:", gpu_num, "num route:",len(case) )
    return task_name+"_"



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("base_route", type=str)
    parser.add_argument("task_num", type=int)
    parser.add_argument("algo", type=str)
    parser.add_argument("planner_type", type=str)
    args = parser.parse_args()
    main(args.base_route, args.task_num, args.algo, args.planner_type)
