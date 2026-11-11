#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from carla_tools.paths import repo_path
from leaderboard.scripts.process_manager import ScreenManager


def main():
    parser = argparse.ArgumentParser(description='Stop only the selected evaluation screen sessions')
    parser.add_argument('--log-dir', default=repo_path('output/AlignDrive/AlignDrive_full'))
    parser.add_argument('--session', nargs='+', help='Explicit session names; otherwise read sessions.json')
    args = parser.parse_args()
    log_dir = Path(args.log_dir).expanduser().resolve()
    session_names = args.session
    if session_names is None:
        sessions_file = log_dir / 'sessions.json'
        if not sessions_file.is_file():
            parser.error(f'Session metadata not found: {sessions_file}; use --log-dir or --session')
        session_names = json.loads(sessions_file.read_text())
    if not isinstance(session_names, list) or not all(isinstance(name, str) and name and
                                                    not name.startswith('-') for name in session_names):
        parser.error('Session names must be non-empty strings without leading dashes')
    manager = ScreenManager(str(log_dir))
    failed = False
    for name in session_names:
        if manager.kill_child_processes(name):
            print(f'Terminated session: {name}')
        else:
            print(f'Session not running: {name}')
            failed = True
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
