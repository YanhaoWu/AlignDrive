"""Unified command line for saved simulation videos and perception debugging."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='task', required=True)
    simulation = commands.add_parser('simulation', help='Render saved closed-loop inference frames')
    simulation.add_argument('--scene-dir', type=Path, required=True)
    simulation.add_argument('--output-dir', type=Path, default=ROOT / 'work_dirs/visualization')
    simulation.add_argument(
        '--scene-name', help='Output name; defaults to the scene directory name'
    )
    simulation.add_argument('--mode', choices=['bev', 'multimodal'], default='multimodal')
    simulation.add_argument(
        '--render-version', choices=['full', 'clean', 'presentation'], default='full'
    )
    simulation.add_argument('--bev-view', choices=['topdown', 'front'], default='topdown')
    simulation.add_argument(
        '--view-name', default='rgb_front', help='Camera directory for BEV overlay mode'
    )
    simulation.add_argument('--fps', type=int, default=10)
    simulation.add_argument('--workers', type=int, default=1)
    simulation.add_argument(
        '--render-rgb', action='store_true', default=False,
        help='Opt in to six-camera overlays in multimodal mode (disabled by default)'
    )
    simulation.add_argument('--trajectory-mode', choices=['aligned', 'plan'], default='aligned')
    perception = commands.add_parser(
        'perception', help='Debug raw Bench2Drive perception annotations'
    )
    perception.add_argument(
        '--scene-dir', '--file_path', '-f', dest='scene_dir', type=Path, required=True
    )
    perception.add_argument(
        '--map-path',
        '--map_path',
        '-m',
        dest='map_path',
        required=True,
        help='HD map .npz path, Town01, or 01',
    )
    perception.add_argument(
        '--output-dir', type=Path, help='Default: a sibling directory named SCENE_vis'
    )
    return parser


def render_simulation(args, parser):
    scene_dir = args.scene_dir.expanduser().resolve()
    if args.fps < 1 or args.workers < 1:
        parser.error('fps and workers must be positive')
    if args.mode == 'bev' and (args.workers != 1 or args.render_rgb):
        parser.error('Parallel rendering and --render-rgb require multimodal mode')
    view_name = 'bev' if args.mode == 'multimodal' else args.view_name
    for folder in ('meta', view_name):
        if not (scene_dir / folder).is_dir():
            parser.error(f'Missing scene directory: {scene_dir / folder}')
    if not any((scene_dir / 'meta').glob('*.pkl')):
        parser.error(
            'Simulation rendering requires inference meta/*.pkl, not dataset anno/*.json.gz'
        )
    from tools.visualization.simulation import SimulationBEVRenderer, SimulationRenderer

    output_dir = args.output_dir.expanduser().resolve()
    options = dict(
        images_folder=str(scene_dir),
        fps=args.fps,
        scene_name=args.scene_name or scene_dir.name,
        sp_waypoints_show=args.trajectory_mode == 'aligned',
    )
    if args.mode == 'bev':
        renderer = SimulationBEVRenderer(str(output_dir), args.render_version, args.bev_view)
        return renderer.create_video(**options, view_name=args.view_name)
    renderer = SimulationRenderer(str(output_dir), args.render_version, args.bev_view)
    options['render_img'] = args.render_rgb
    if args.workers > 1:
        return renderer.create_video_parallel(**options, max_workers=args.workers)
    return renderer.create_video(**options)


def render_perception(args, parser):
    scene_dir = args.scene_dir.expanduser().resolve()
    for folder in ('anno', 'camera'):
        if not (scene_dir / folder).is_dir():
            parser.error(f'Missing scene directory: {scene_dir / folder}')
    map_path = Path(args.map_path).expanduser()
    if map_path.suffix != '.npz':
        town = args.map_path if args.map_path.startswith('Town') else 'Town' + args.map_path
        map_path = ROOT / 'maps' / f'{town}_HD_map.npz'
    if not map_path.is_file():
        parser.error(f'HD map not found: {map_path}')
    from tools.visualization.perception import visualize_data

    return visualize_data(str(scene_dir), str(map_path), output_dir=args.output_dir)


def visualize_mutil_modal_data(file_path, map_path, vis_bbox=True, save_path=None, **options):
    """Compatibility wrapper for the former raw-data debugging function."""
    from tools.visualization.perception import visualize_data

    output_dir = Path(save_path) / Path(file_path).name if save_path else None
    return visualize_data(file_path, map_path, vis_bbox=vis_bbox, output_dir=output_dir, **options)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    result = (
        render_simulation(args, parser)
        if args.task == 'simulation'
        else render_perception(args, parser)
    )
    if result is not None:
        print(result)


if __name__ == '__main__':
    main()
