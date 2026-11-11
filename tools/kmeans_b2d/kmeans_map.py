"""Generate map-query anchors from Bench2Drive map-element centers."""

import sys
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from tqdm import tqdm

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.kmeans_b2d.common import build_parser, cluster_centers, load_infos, prepare_output_paths


def main(argv=None):
    args = build_parser(__doc__, num_clusters=100).parse_args(argv)
    anchor_path, preview_path = prepare_output_paths(
        args, 'kmeans_map_{}.npy'.format(args.num_clusters),
        'b2d_map_anchor_{}.png'.format(args.num_clusters))
    infos = sorted(load_infos(args.ann_file), key=lambda info: info['timestamp'])
    centers = []
    for info in tqdm(infos):
        for geometries in info['map_annos'].values():
            centers.extend(np.asarray(geometry).mean(axis=0) for geometry in geometries)
    centers = cluster_centers(np.asarray(centers), args.num_clusters, args.seed)
    offsets = np.stack([np.zeros(20), np.linspace(-4, 4, 20)], axis=-1)
    vectors = centers[:, np.newaxis] + offsets[np.newaxis]
    figure = Figure()
    axes = figure.subplots()
    for vector in vectors:
        axes.plot(vector[:, 0], vector[:, 1], linewidth=1, marker='o', linestyle='-', markersize=2)
    figure.savefig(preview_path, bbox_inches='tight')
    np.save(anchor_path, vectors)
    print('Saved anchors: {}'.format(anchor_path))
    print('Saved preview: {}'.format(preview_path))


if __name__ == '__main__':
    main()
