import hashlib
import os
from pathlib import Path
import pickle
import tempfile


INFO_FILENAME = 'bench2drive_info_{split}.pkl'
WAYPOINT_FILENAME = 'bench2drive_aligndrive_waypoints_{split}.pkl'
NUM_POINTS = 15
TRAJECTORY_FIELDS = (
    ('spatial_interval_2', 'distance', 2),
    ('spatial_interval_5', 'distance', 5),
    ('temporal_interval_2', 'time', 2),
    ('temporal_interval_5', 'time', 5),
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_pickle_dump(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=str(path.parent), suffix='.tmp', delete=False) as stream:
            temporary_path = Path(stream.name)
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(str(temporary_path), str(path))
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
