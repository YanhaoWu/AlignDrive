"""Shared camera ordering, class labels, colors, and display thresholds."""

import numpy as np

CMD_LIST = ['Turn Right', 'Turn Left', 'Go Straight']
CMD_LIST_B2D = ['LEFT', 'RIGHT', 'STRAIGHT', 'LANE_FOLLOW', 'CHANGE_LANE_LEFT', 'CHANGE_LANE_RIGHT']
GRAND_LIST = [
    'spatial_interval_2_traj',
    'spatial_interval_5_traj',
    'temporal_interval_2_traj',
    'temporal_interval_5_traj',
]
COLOR_VECTORS = ['cornflowerblue', 'royalblue', 'slategrey', 'green', 'red', 'brown']
SCORE_THRESH = 0.3
MAP_SCORE_THRESH = 0.3
B2D_DETECTION_CLASS = [
    'car',
    'van',
    'truck',
    'bicycle',
    'traffic_sign',
    'traffic_cone',
    'traffic_light_0',
    'traffic_light_1',
    'traffic_light_2',
    'pedestrian',
    'others',
    'traffic.stop',
]

color_mapping = (
    np.asarray(
        [
            [0, 0, 0],
            [255, 179, 0],
            [128, 62, 117],
            [255, 104, 0],
            [166, 189, 215],
            [193, 0, 32],
            [206, 162, 98],
            [129, 112, 102],
            [0, 125, 52],
            [246, 118, 142],
            [0, 83, 138],
            [255, 122, 92],
            [83, 55, 122],
            [255, 142, 0],
            [179, 40, 81],
            [244, 200, 0],
            [127, 24, 13],
            [147, 170, 0],
            [89, 51, 21],
            [241, 58, 19],
            [35, 44, 22],
            [112, 224, 255],
            [70, 184, 160],
            [153, 0, 255],
            [71, 255, 0],
            [255, 0, 163],
            [255, 204, 0],
            [0, 255, 235],
            [255, 0, 235],
            [255, 0, 122],
            [255, 245, 0],
            [10, 190, 212],
            [214, 255, 0],
            [0, 204, 255],
            [20, 0, 255],
            [255, 255, 0],
            [0, 153, 255],
            [0, 255, 204],
            [41, 255, 0],
            [173, 0, 255],
            [0, 245, 255],
            [71, 0, 255],
            [0, 255, 184],
            [0, 92, 255],
            [184, 255, 0],
            [255, 214, 0],
            [25, 194, 194],
            [92, 0, 255],
            [220, 220, 220],
            [255, 9, 92],
            [112, 9, 255],
            [8, 255, 214],
            [255, 184, 6],
            [10, 255, 71],
            [255, 41, 10],
            [7, 255, 255],
            [224, 255, 8],
            [102, 8, 255],
            [255, 61, 6],
            [255, 194, 7],
            [0, 255, 20],
            [255, 8, 41],
            [255, 5, 153],
            [6, 51, 255],
            [235, 12, 255],
            [160, 150, 20],
            [0, 163, 255],
            [140, 140, 140],
            [250, 10, 15],
            [20, 255, 0],
        ]
    )
    / 255
)

COLORS = [
    (2, 62, 255),
    (255, 124, 0),
    (26, 201, 56),
    (232, 0, 11),
    (139, 43, 226),
    (159, 72, 0),
    (241, 76, 193),
    (163, 163, 163),
    (255, 196, 0),
    (0, 215, 255),
]
COLORMAP = {
    6: COLORS[8],
    4: COLORS[8],
    3: COLORS[0],
    1: COLORS[6],
    0: COLORS[2],
    8: COLORS[7],
    7: COLORS[1],
    5: COLORS[3],
    2: COLORS[5],
    9: COLORS[4],
    10: COLORS[9],
    11: COLORS[0],
}


def prediction_color(result, object_index):
    """Use a track ID when available, otherwise a class ID, for stable object colors."""
    identifiers = result.get('instance_ids', result.get('labels_3d'))
    identifier = object_index if identifiers is None else int(identifiers[object_index])
    return color_mapping[identifier % len(color_mapping)]
