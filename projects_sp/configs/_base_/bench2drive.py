"""Shared Bench2Drive paths, labels, camera augmentation, and pipeline steps."""

import os

data_root = os.environ.get('ALIGNDRIVE_DATA_ROOT', './data/Bench2Drive')

anno_root = os.environ.get('ALIGNDRIVE_ANNO_ROOT', './data/bench2drive_meta/infos_b2d_full')

class_names = [
    'car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone', 'traffic_light_0',
    'traffic_light_1', 'traffic_light_2', 'pedestrian', 'others', 'traffic.stop'
]

map_class_names = ['Broken', 'Solid', 'SolidSolid', 'Center', 'TrafficLight', 'StopSign']

NameMapping = {
    'vehicle.bh.crossbike': 'bicycle',
    'vehicle.diamondback.century': 'bicycle',
    'vehicle.gazelle.omafiets': 'bicycle',
    'vehicle.chevrolet.impala': 'car',
    'vehicle.dodge.charger_2020': 'car',
    'vehicle.dodge.charger_police': 'car',
    'vehicle.dodge.charger_police_2020': 'car',
    'vehicle.lincoln.mkz_2017': 'car',
    'vehicle.lincoln.mkz_2020': 'car',
    'vehicle.mini.cooper_s_2021': 'car',
    'vehicle.mercedes.coupe_2020': 'car',
    'vehicle.ford.mustang': 'car',
    'vehicle.nissan.patrol_2021': 'car',
    'vehicle.audi.tt': 'car',
    'vehicle.audi.etron': 'car',
    'vehicle.ford.crown': 'car',
    'vehicle.tesla.model3': 'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/FordCrown/SM_FordCrown_parked.SM_FordCrown_parked': 'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Charger/SM_ChargerParked.SM_ChargerParked': 'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Lincoln/SM_LincolnParked.SM_LincolnParked': 'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/MercedesCCC/SM_MercedesCCC_Parked.SM_MercedesCCC_Parked': 'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Mini2021/SM_Mini2021_parked.SM_Mini2021_parked': 'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/NissanPatrol2021/SM_NissanPatrol2021_parked.SM_NissanPatrol2021_parked': 'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/TeslaM3/SM_TeslaM3_parked.SM_TeslaM3_parked': 'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked': 'van',
    'vehicle.ford.ambulance': 'van',
    'vehicle.carlamotors.firetruck': 'truck',
    'traffic.speed_limit.30': 'traffic_sign',
    'traffic.speed_limit.40': 'traffic_sign',
    'traffic.speed_limit.50': 'traffic_sign',
    'traffic.speed_limit.60': 'traffic_sign',
    'traffic.speed_limit.90': 'traffic_sign',
    'traffic.speed_limit.120': 'traffic_sign',
    'traffic.stop': 'traffic.stop',
    'traffic.yield': 'traffic_sign',
    'traffic.traffic_light': 'traffic_light',
    'traffic.traffic_light_0': 'traffic_light_0',
    'traffic.traffic_light_1': 'traffic_light_1',
    'traffic.traffic_light_2': 'traffic_light_2',
    'static.prop.warningconstruction': 'traffic_cone',
    'static.prop.warningaccident': 'traffic_cone',
    'static.prop.trafficwarning': 'traffic_cone',
    'static.prop.constructioncone': 'traffic_cone',
    'walker.pedestrian.0001': 'pedestrian',
    'walker.pedestrian.0004': 'pedestrian',
    'walker.pedestrian.0005': 'pedestrian',
    'walker.pedestrian.0007': 'pedestrian',
    'walker.pedestrian.0013': 'pedestrian',
    'walker.pedestrian.0014': 'pedestrian',
    'walker.pedestrian.0017': 'pedestrian',
    'walker.pedestrian.0018': 'pedestrian',
    'walker.pedestrian.0019': 'pedestrian',
    'walker.pedestrian.0020': 'pedestrian',
    'walker.pedestrian.0022': 'pedestrian',
    'walker.pedestrian.0025': 'pedestrian',
    'walker.pedestrian.0035': 'pedestrian',
    'walker.pedestrian.0041': 'pedestrian',
    'walker.pedestrian.0046': 'pedestrian',
    'walker.pedestrian.0047': 'pedestrian',
    'static.prop.dirtdebris01': 'others',
    'static.prop.dirtdebris02': 'others'
}

input_modality = {
    'use_lidar': False,
    'use_camera': True,
    'use_radar': False,
    'use_map': False,
    'use_external': False
}

data_aug_conf = {
    'resize_lim': (0.4, 0.47),
    'final_dim': (352, 640),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (-5.4, 5.4),
    'H': 900,
    'W': 1600,
    'rand_flip': True,
    'rot3d_range': [0, 0]
}

perception_class_names = [
    'car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone', 'traffic_light', 'pedestrian',
    'others', 'traffic_stop'
]

perception_name_mapping = dict(NameMapping)

perception_name_mapping.update({
    'traffic.stop': 'traffic_stop',
    'traffic.traffic_light_0': 'traffic_light',
    'traffic.traffic_light_1': 'traffic_light',
    'traffic.traffic_light_2': 'traffic_light'
})

train_ann_file = os.path.join(anno_root, 'bench2drive_info_train.pkl')

val_ann_file = os.path.join(anno_root, 'bench2drive_info_val.pkl')

train_waypoints_file = os.path.join(anno_root, 'bench2drive_aligndrive_waypoints_train.pkl')

val_waypoints_file = os.path.join(anno_root, 'bench2drive_aligndrive_waypoints_val.pkl')

pipeline_steps = {
    'load_multi_view_image_from_files': {
        'to_float32': True,
        'type': 'LoadMultiViewImageFromFiles'
    },
    'resize_crop_flip_image': {
        'type': 'ResizeCropFlipImage'
    },
    'normalize_multiview_image': {
        'mean': [123.675, 116.28, 103.53],
        'std': [58.395, 57.12, 57.375],
        'to_rgb': True,
        'type': 'NormalizeMultiviewImage'
    },
    'nu_scenes_sparse4d_adaptor': {
        'type': 'NuScenesSparse4DAdaptor'
    },
    'photo_metric_distortion_multi_view_image': {
        'type': 'PhotoMetricDistortionMultiViewImage'
    }
}
