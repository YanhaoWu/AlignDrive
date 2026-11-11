"""Generate detection-query anchors from Bench2Drive object centers."""

import sys
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from tqdm import tqdm

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.kmeans_b2d.common import build_parser, cluster_centers, load_infos, prepare_output_paths


def main(argv=None):
    args = build_parser(__doc__, num_clusters=900).parse_args(argv)
    anchor_path, preview_path = prepare_output_paths(
        args, 'kmeans_det_{}.npy'.format(args.num_clusters),
        'b2d_det_anchor_{}.png'.format(args.num_clusters))
    infos = sorted(load_infos(args.ann_file), key=lambda info: info['timestamp'])
    centers = []
    for info in tqdm(infos):
        boxes = np.asarray(info['gt_boxes'])
        if boxes.size == 0:
            continue
        boxes = boxes[:, :3]
        distance = np.linalg.norm(boxes[:, :2], axis=1)
        centers.extend(boxes[distance < 55])
    print('Clustering detection anchors; this may take a few minutes.')
    clusters = cluster_centers(np.asarray(centers), args.num_clusters, args.seed)
    figure = Figure()
    figure.subplots().scatter(clusters[:, 0], clusters[:, 1])
    figure.savefig(preview_path, bbox_inches='tight')
    attributes = np.tile([1, 1, 1, 1, 0, 0, 0, 0], (args.num_clusters, 1))
    np.save(anchor_path, np.concatenate([clusters, attributes], axis=1))
    print('Saved anchors: {}'.format(anchor_path))
    print('Saved preview: {}'.format(preview_path))


if __name__ == '__main__':
    main()
