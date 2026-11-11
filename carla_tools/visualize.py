"""Compatibility entry point for raw Bench2Drive perception visualization."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def visualize_data(*args, **kwargs):
    from tools.visualization.perception import visualize_data as render

    return render(*args, **kwargs)


if __name__ == '__main__':
    from tools.visualization.visualize import main

    main(['perception', *sys.argv[1:]])
