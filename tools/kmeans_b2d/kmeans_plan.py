"""Generate the command-conditioned, multi-granularity AlignDrive planning vocabulary."""

import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from tqdm import tqdm

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.data_converter.common import WAYPOINT_FILENAME, atomic_pickle_dump, file_sha256
from tools.kmeans_b2d.common import (
    NAVIGATION_COMMANDS, PLANNING_HORIZONS, build_parser, cluster_centers,
    load_infos, prepare_output_paths,
)


def validate_waypoint_manifest(annotation_path, waypoint_path):
    manifest_path = waypoint_path.with_suffix('.manifest.json')
    if not manifest_path.is_file():
        print('No waypoint manifest found; checking record counts and navigation commands only.')
        return
    manifest = json.loads(manifest_path.read_text())
    for field, path in (('annotation_sha256', annotation_path), ('waypoint_sha256', waypoint_path)):
        expected = manifest.get(field)
        if not expected or file_sha256(path) != expected:
            raise ValueError('{} does not match {}; regenerate aligned waypoints'.format(field, path))


def collect_trajectories(infos, waypoints):
    """Group valid cumulative trajectories by granularity and navigation command."""
    if len(infos) != len(waypoints):
        raise ValueError('Metadata/waypoint length mismatch: {} != {}'.format(len(infos), len(waypoints)))
    samples = {key: [[] for _ in NAVIGATION_COMMANDS] for key in PLANNING_HORIZONS}
    for index, (info, record) in enumerate(tqdm(zip(infos, waypoints), total=len(infos))):
        raw_command = info['command_near']
        command = int(raw_command)
        if command != raw_command:
            raise ValueError('Expected an integer command_near at frame {}'.format(index))
        if command < 0:
            command = 4
        if not 1 <= command <= len(NAVIGATION_COMMANDS):
            raise ValueError('Expected command_near in 1..6 or a negative sentinel at frame {}'.format(index))
        expected_command = np.eye(len(NAVIGATION_COMMANDS))[command - 1]
        if not np.array_equal(np.asarray(record['command']), expected_command):
            raise ValueError('Metadata/waypoint command mismatch at frame {}'.format(index))
        for key, horizon in PLANNING_HORIZONS.items():
            offsets = np.asarray(record[key])
            mask = np.asarray(record[key.replace('_traj', '_mask')])
            if (offsets.ndim != 2 or offsets.shape[1] != 2 or offsets.shape[0] < horizon
                    or mask.shape != (offsets.shape[0],)):
                raise ValueError('Invalid {} shape at frame {}; need at least ({}, 2)'.format(
                    key, index, horizon))
            if not np.isfinite(offsets).all() or not np.isin(mask, [0, 1]).all():
                raise ValueError('Nonfinite trajectory or invalid mask for {} at frame {}'.format(key, index))
            if not mask[:horizon].all():
                continue
            trajectory = offsets[:horizon].cumsum(axis=0)
            lateral_endpoint = trajectory[-1, 0]
            if (command == 5 and lateral_endpoint > 0) or (command == 6 and lateral_endpoint < 0):
                continue
            samples[key][command - 1].append(trajectory)
    return samples


def main(argv=None):
    parser = build_parser(__doc__, num_clusters=1)
    parser.add_argument('--waypoints-file', type=Path,
                        help='Defaults to bench2drive_aligndrive_waypoints_train.pkl beside --ann-file.')
    args = parser.parse_args(argv)
    filename = 'train_mutil_granular_waypoint_cluster_refine_{}.pkl'.format(args.num_clusters)
    anchor_path, preview_path = prepare_output_paths(
        args, filename, 'b2d_planning_vocabulary_{}.png'.format(args.num_clusters))
    annotation_path = args.ann_file.expanduser().resolve()
    waypoint_path = (args.waypoints_file or annotation_path.parent / WAYPOINT_FILENAME.format(
        split='train')).expanduser().resolve()
    validate_waypoint_manifest(annotation_path, waypoint_path)
    samples = collect_trajectories(load_infos(annotation_path), load_infos(waypoint_path))
    for key, grouped in samples.items():
        for command_name, trajectories in zip(NAVIGATION_COMMANDS, grouped):
            if len(trajectories) < args.num_clusters:
                raise ValueError('{} / {} needs at least {} samples; got {}'.format(
                    key, command_name, args.num_clusters, len(trajectories)))
    vocabulary = {}
    figure = Figure(figsize=(12, 10))
    for axes, (key, horizon) in zip(figure.subplots(2, 2).flat, PLANNING_HORIZONS.items()):
        clusters = []
        for command_name, trajectories in zip(NAVIGATION_COMMANDS, samples[key]):
            print('{} / {}: {} samples'.format(key, command_name, len(trajectories)))
            flattened = np.asarray(trajectories).reshape(-1, horizon * 2)
            anchors = cluster_centers(flattened, args.num_clusters, args.seed).reshape(-1, horizon, 2)
            clusters.append(anchors)
            for cluster_index, anchor in enumerate(anchors):
                axes.plot(anchor[:, 0], anchor[:, 1], label=command_name if cluster_index == 0 else None)
        vocabulary[key] = np.stack(clusters)
        axes.set_title(key)
        axes.set_aspect('equal', adjustable='datalim')
        axes.legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(preview_path, bbox_inches='tight')
    atomic_pickle_dump(vocabulary, anchor_path)
    print('Saved anchors: {}'.format(anchor_path))
    print('Saved preview: {}'.format(preview_path))


if __name__ == '__main__':
    main()
