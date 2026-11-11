# Bench2Drive Data Preprocessing

`B2D_converter.py` prepares both the metadata and trajectory labels consumed by
the four AlignDrive configurations in one command. It combines perception and
traffic-light conversion with `StateAnalyzer.obtain_trajs` in the same file,
without a separate waypoint extraction script.

## Dataset Download

Download the **Base** dataset from the official
[Bench2Drive repository](https://github.com/Thinklab-SJTU/Bench2Drive), following
the [upstream data preparation guide](https://github.com/Thinklab-SJTU/Bench2DriveZoo/blob/uniad/vad/docs/DATA_PREP.md).
Obtain the Town map `.npz` files and `bench2drive_base_train_val_split.json` as well.
The CARLA simulator archives do not include the training dataset.

The quick-start layout is:

```text
Bench2Drive/
├── v1/                                      # Scene directories; flat layouts also work
├── maps/                                    # Town*_HD_map.npz
└── bench2drive_base_train_val_split.json
```

If the split JSON is stored separately, place it at
`data/bench2drive_meta/splits/bench2drive_base_train_val_split.json`, or set
`ALIGNDRIVE_SPLIT_FILE` to its actual path. Use `ALIGNDRIVE_MAP_ROOT` if maps
are not under the dataset's `maps/` directory.

## Quick Start

```bash
bash scripts/prepare_data.sh /path/to/Bench2Drive
```

The wrapper creates `data/Bench2Drive` as a symlink to the supplied raw data
without moving it, sets the converter paths, and prepares both splits. If that
path already points elsewhere, it stops rather than replacing your data.
An existing matching symlink or dataset directory is reused. Metadata and
waypoints go to `data/bench2drive_meta/infos_b2d_full/` by default, matching the
training configurations. Full data and checkpoints are not included in the repository.

After preparing the training split, prepare all three model anchor files:

```bash
bash scripts/prepare_anchors.sh
```

The [anchor-generation tools](../kmeans_b2d/README.md) generate detection and map
initialization anchors and the active multi-granularity planning vocabulary.
Valid bundled files under `kmeans/b2d/` are reused; missing files are generated.
Pass `--overwrite` to explicitly regenerate all three. Clustering is a separate
step from `scripts/prepare_data.sh`: validation-only or partial conversion must
not fit new anchors from validation data. The clustering inputs must come from
the training split.

Additional converter options are forwarded, for example:

```bash
bash scripts/prepare_data.sh /path/to/Bench2Drive --split val --stage waypoints
```

For custom output or external data paths, keep `ALIGNDRIVE_DATA_ROOT`,
`ALIGNDRIVE_ANNO_ROOT`, and `ALIGNDRIVE_ANCHOR_ROOT` consistent in the training
environment. Only load trusted pickle and checkpoint files.

## Prerequisites

Run commands from the AlignDrive repository root in the environment described in
the [installation guide](../../doc/INSTALL.md#training-installation). Metadata conversion uses NumPy,
OpenCV, Shapely, tqdm, and pyquaternion. Trajectory generation reuses
`B2D3DDataset` from the project, so the model environment and custom operators must
be installed; no GPU computation or image loading is needed by `obtain_trajs`.

Only load trusted pickle and NumPy object files. Dataset and map downloads are
not included.

The converter expects:

```text
Bench2Drive/
└── [v1/]<scenario>_TownXX_RouteX_WeatherX/
    ├── anno/<frame>.json.gz
    ├── camera/rgb_<view>/<frame>.jpg
    ├── camera/depth_<view>/<frame>.png
    └── expert_assessment/<previous-frame>.npz
```

Both flat and `v1/` scene layouts are supported. Six camera views, depth images,
expert assessments (including `-0001.npz` for frame zero), map `.npz` files, and
the official train/validation split JSON are required. The split JSON must have
a `val` list. Training uses the discovered routes not in that list, as in the
upstream converter. Validation entries may have a `v1/` prefix even for a flat
dataset. Duplicate or missing validation routes are rejected. Relative scene
paths are retained in the metadata; use the same dataset root for training.

Map conversion only reads regular files whose names end in `.npz`; resumable
download sidecars such as `.npz.aria2` are ignored. If using
`carla_tools/download_mini.sh`, incomplete download files are kept in its
separate temporary directory until the archive is complete.

## 1. Prepare Metadata and Waypoints

```bash
export ALIGNDRIVE_DATA_ROOT=/path/to/Bench2Drive
export ALIGNDRIVE_ANNO_ROOT=/path/to/infos_b2d_full

python3 tools/data_converter/B2D_converter.py \
  --data-root "$ALIGNDRIVE_DATA_ROOT" \
  --map-root /path/to/bench2drive/map \
  --split-file /path/to/bench2drive_base_train_val_split.json \
  --out-dir "$ALIGNDRIVE_ANNO_ROOT" \
  --workers 16 --split train val
```

`ALIGNDRIVE_MAP_ROOT` and `ALIGNDRIVE_SPLIT_FILE` also provide default map and split
paths. Use `--split train` or `--split val` to process just one split. The default
`--stage all` converts metadata, then extracts waypoints from that exact metadata
for each split. All selected output paths are checked before conversion starts;
existing metadata or labels are protected unless `--overwrite` is specified.
The map cache is reused if present; use a fresh output directory when changing
source maps.

For partial reruns, use the same file with `--stage metadata` (no model import)
or `--stage waypoints` (no raw-data or map conversion). A waypoint-only run reads
the existing metadata from `--out-dir`. Regenerate waypoints after replacing
metadata to keep frame order aligned.

For waypoint extraction from a single custom annotation file:

```bash
python3 tools/data_converter/B2D_converter.py \
  --stage waypoints --split val \
  --ann-file /path/to/bench2drive_info_val.pkl \
  --out-dir "$ALIGNDRIVE_ANNO_ROOT"
```

Both outputs are written to `--out-dir`, matching `ALIGNDRIVE_ANNO_ROOT` in the
training configurations. There is no separate waypoint output directory.

## 2. Shared Metadata

For each split, the converter produces one metadata list shared by all stages:

| File | Consumer | Traffic-light representation |
| --- | --- | --- |
| `bench2drive_info_<split>.pkl` | Perception, Spatial, Lon, inference | State-suffixed names and a per-frame `traffic_light` list |

The light list retains `id`, `semantic_tags`, `affects_ego`, and `state`. The
Spatial/Lon class mappings distinguish `traffic.traffic_light_0`, `_1`, and
`_2`; Perception maps all three to its existing `traffic_light` class. This
preserves each stage's class vocabulary without duplicating metadata or removing
traffic-light information.
The conversion also writes `b2d_map_infos.pkl`. Worker failures stop conversion
instead of merging stale or incomplete shards.

Coordinate transforms, visibility filtering, and the 10 Hz timestamp convention
follow the upstream conversion recipe. Frames without retained bounding boxes
are omitted, as upstream; temporal sampling advances by metadata records, not by
interpolating missing timestamps.

## 3. Multi-Granularity Trajectory Labels

Trajectory extraction runs automatically after metadata conversion. Labels are
written next to the metadata, exactly where the training configurations expect
them. `--overwrite` explicitly permits replacing existing labels.

The output filenames match all current configurations:

```text
infos_b2d_full/
├── b2d_map_infos.pkl
├── bench2drive_info_train.pkl
├── bench2drive_info_val.pkl
├── bench2drive_aligndrive_waypoints_train.pkl
├── bench2drive_aligndrive_waypoints_val.pkl
├── bench2drive_aligndrive_waypoints_train.manifest.json
└── bench2drive_aligndrive_waypoints_val.manifest.json
```

Each label file is a list in the **unchanged metadata order**, with exactly one
record per metadata frame:

| Keys | Sampling | Shapes |
| --- | --- | --- |
| `spatial_interval_2_traj`, `spatial_interval_2_mask` | 2 m spatial intervals | `(15, 2)`, `(15,)` |
| `spatial_interval_5_traj`, `spatial_interval_5_mask` | 5 m spatial intervals | `(15, 2)`, `(15,)` |
| `temporal_interval_2_traj`, `temporal_interval_2_mask` | Every 2 metadata records, nominally 5 Hz | `(15, 2)`, `(15,)` |
| `temporal_interval_5_traj`, `temporal_interval_5_mask` | Every 5 metadata records, nominally 2 Hz | `(15, 2)`, `(15,)` |
| `command` | Six-way navigation command | `(6,)` |

Trajectories contain **consecutive-point displacement offsets**, not absolute
positions. Use the masks when consuming padded points. Sampling uses the same
dataset helpers as SparseDrive and AlignDrive, including the upstream spatial
interpolation and end-index conventions. Route clipping now covers the final
frame and singleton routes, splits at route changes or timestamp resets/gaps
over three seconds, and masks temporal points beyond the current clip.

## Version Alignment

- The active label files are **`bench2drive_aligndrive_waypoints_<split>.pkl`**,
  with **15 points** per trajectory. The label keys and sampling recipe are unchanged.
- The four sampling calls use distance intervals **2 / 5 m** and temporal strides
  **2 / 5 records**, matching `obtain_trajs` and the current dataset helpers.
- Perception, Spatial, Lon, and inference share
  `bench2drive_info_<split>.pkl`. Only Perception omits the trajectory-label input.
- This is **not** the `with_ds`, `hipad`, agent-path, or collision-score label
  variant. The converter does not generate driving-style labels or clustered anchors.
- `scripts/prepare_anchors.sh` generates or reuses the model anchors separately.
  Its planning tool clusters the first 15 points of each interval-2 trajectory
  and the first 6 points of each interval-5 trajectory, matching the bundled
  vocabulary and current model configuration. The per-frame supervision labels
  remain 15 points for all four granularities.
- Each manifest records shapes, sampling settings, source-metadata and output
  SHA-256 values, and a frame-order fingerprint. Regenerate labels whenever
  metadata content or ordering changes. Matching filenames alone do not prove
  that two independently generated datasets are identical.

For an existing workspace, regenerate metadata and labels using the commands
above. Older metadata without a per-frame `traffic_light` field is not compatible
with the unified format; renaming such a file does not add the missing data.
Metadata already containing that field and state-suffixed traffic-light names
has the same record schema as the unified metadata, but labels must still match
its exact frame order. The converter does not automatically delete older files.

## Validation

Run the focused synthetic conversion and trajectory contract tests with:

```bash
python3 -m pytest scripts/tests/test_data_preprocessing.py -q
```

The migrated converter and trajectory generation were also compared numerically
against the upstream functions on synthetic scenes; the eight dataset sampling
helpers were checked to have identical implementations. A local historical
validation label file was inspected to confirm the same keys, shapes, and dtypes.
It has not been rerun here over the full private dataset, so bit-for-bit identity
with historical released metadata is not claimed.

These checks do not certify every city, scenario, or the complete training and
validation splits.
