"""Launch Bench2Drive evaluation with the default AlignDrive configuration."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description='Launch the default AlignDrive closed-loop evaluation preset')
    parser.add_argument('--checkpoint', default=os.environ.get('ALIGNDRIVE_CHECKPOINT'))
    parser.add_argument('--collision-mode', choices=['fast', 'shapely'], default='fast',
                        help='fast: GPU SAT; shapely: CPU geometry reference')
    args, extra_arguments = parser.parse_known_args()
    if not args.checkpoint:
        parser.error('Set ALIGNDRIVE_CHECKPOINT or provide --checkpoint')
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f'Checkpoint not found: {checkpoint}')
    environment = os.environ.copy()
    environment['ALIGNDRIVE_COLLISION_MODE'] = args.collision_mode
    command = [sys.executable, str(ROOT / 'leaderboard/scripts/evaluate_multi_gpu.py'),
               '--weights_path', str(checkpoint), '--project_path', 'output/AlignDrive',
               '--project_name', 'AlignDrive', '--controler_paranter', 'control_2m5hz',
               '--config_path', 'projects_sp/configs/inference/aligndrive_inference.py',
               '--route_version', 'full', *extra_arguments]
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == '__main__':
    main()
