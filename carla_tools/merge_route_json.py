"""Compatibility entry point for the Bench2Drive result merger."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carla_tools.merge import main, merge_route_json


if __name__ == '__main__':
    main()
