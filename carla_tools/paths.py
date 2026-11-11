import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(value):
    path = Path(os.path.expandvars(str(value))).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path.resolve())
