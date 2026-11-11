"""Prepare all default AlignDrive anchors, reusing existing valid files unless --overwrite is set."""

import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.kmeans_b2d.common import PLANNING_HORIZONS, build_parser


def validate_anchor(path, expected_shape):
    if isinstance(expected_shape, dict):
        with path.open('rb') as stream:
            value = pickle.load(stream)
        if not isinstance(value, dict) or set(value) != set(expected_shape):
            raise ValueError('Invalid planning vocabulary keys in {}'.format(path))
        values = [(value[key], shape) for key, shape in expected_shape.items()]
    else:
        values = [(np.load(path, allow_pickle=False), expected_shape)]
    for value, shape in values:
        array = np.asarray(value)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError('Invalid anchor shape or values in {}; expected {}'.format(path, shape))


def main(argv=None):
    parser = build_parser(__doc__)
    parser.add_argument('--waypoints-file', type=Path,
                        help='Override the planning labels beside --ann-file.')
    args = parser.parse_args(argv)
    output = args.out_dir.expanduser().resolve()
    tools = (
        ('kmeans_det.py', 'kmeans_det_900.npy', (900, 11)),
        ('kmeans_map.py', 'kmeans_map_100.npy', (100, 20, 2)),
        ('kmeans_plan.py', 'train_mutil_granular_waypoint_cluster_refine_1.pkl',
         {key: (6, 1, horizon, 2) for key, horizon in PLANNING_HORIZONS.items()}),
    )
    for script, filename, shape in tools:
        path = output / filename
        if path.exists() and not args.overwrite:
            validate_anchor(path, shape)
            print('Reusing {}'.format(path), flush=True)
            continue
        command = [sys.executable, str(Path(__file__).resolve().parent / script),
                   '--ann-file', str(args.ann_file.expanduser().resolve()),
                   '--out-dir', str(output), '--vis-dir', str(args.vis_dir.expanduser().resolve()),
                   '--seed', str(args.seed)]
        if args.overwrite:
            command.append('--overwrite')
        if script == 'kmeans_plan.py' and args.waypoints_file:
            command.extend(['--waypoints-file', str(args.waypoints_file.expanduser().resolve())])
        subprocess.run(command, check=True)
    print('All default AlignDrive anchors are ready in {}'.format(output), flush=True)


if __name__ == '__main__':
    main()
