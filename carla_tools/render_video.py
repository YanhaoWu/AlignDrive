"""Compatibility entry point; prefer tools/visualization/visualize.py simulation."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from tools.visualization.visualize import main as visualize

    return visualize(['simulation', *sys.argv[1:]])


if __name__ == '__main__':
    main()
