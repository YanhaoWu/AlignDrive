"""Shared paths and options for optional Bench2Drive anchor generation."""

import argparse
import os
import pickle
from pathlib import Path

from sklearn.cluster import KMeans


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NAVIGATION_COMMANDS = ('LEFT', 'RIGHT', 'STRAIGHT', 'LANE_FOLLOW',
                       'CHANGE_LANE_LEFT', 'CHANGE_LANE_RIGHT')
PLANNING_HORIZONS = {
    'spatial_interval_2_traj': 15,
    'spatial_interval_5_traj': 6,
    'temporal_interval_2_traj': 15,
    'temporal_interval_5_traj': 6,
}


def build_parser(description, num_clusters=None):
    annotation_root = Path(os.environ.get(
        'ALIGNDRIVE_ANNO_ROOT', str(PROJECT_ROOT / 'data/bench2drive_meta/infos_b2d_full')))
    anchor_root = Path(os.environ.get('ALIGNDRIVE_ANCHOR_ROOT', str(PROJECT_ROOT / 'kmeans/b2d')))
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--ann-file', type=Path,
                        default=annotation_root / 'bench2drive_info_train.pkl')
    parser.add_argument('--out-dir', type=Path, default=anchor_root)
    parser.add_argument('--vis-dir', type=Path, default=PROJECT_ROOT / 'vis/kmeans')
    if num_clusters is not None:
        parser.add_argument('--num-clusters', type=int, default=num_clusters)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true',
                        help='Replace existing anchor and preview files.')
    return parser


def prepare_output_paths(args, anchor_name, preview_name):
    if args.num_clusters <= 0:
        raise ValueError('--num-clusters must be positive')
    anchor_path = args.out_dir.expanduser().resolve() / anchor_name
    preview_path = args.vis_dir.expanduser().resolve() / preview_name
    for path in (anchor_path, preview_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError('{} already exists; pass --overwrite to replace it'.format(path))
    for path in (anchor_path, preview_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    return anchor_path, preview_path


def load_infos(path):
    with path.expanduser().open('rb') as stream:
        infos = pickle.load(stream)
    if not isinstance(infos, list):
        raise ValueError('Expected a Bench2Drive record list in {}'.format(path))
    return infos


def cluster_centers(samples, num_clusters, seed):
    if len(samples) < num_clusters:
        raise ValueError('Clustering needs at least {} samples; got {}'.format(
            num_clusters, len(samples)))
    return KMeans(n_clusters=num_clusters, random_state=seed, n_init=10).fit(samples).cluster_centers_
