# Bench2Drive Anchor Generation

These offline CPU tools generate every anchor file required by the default
AlignDrive configurations. Anchors initialize model queries and planning targets;
they are not the per-frame trajectory labels used for supervision.

## Quick Start

Activate the [training environment](../../doc/INSTALL.md#training-installation),
then prepare the training metadata and waypoints using the
[data converter](../data_converter/README.md). From the repository root, run:

```bash
bash scripts/prepare_anchors.sh
```

This command prepares detection, map, and planning anchors in order. Valid bundled
files are reused; missing files are generated. Existing files are checked for the
default keys, shapes, and finite values. An invalid existing file raises an error
rather than being silently replaced. To regenerate all three explicitly:

```bash
bash scripts/prepare_anchors.sh --overwrite
```

Keep the bundled anchors when reproducing a released checkpoint. Regeneration is
useful when training from scratch or changing the dataset; it is not guaranteed
to reproduce the bundled values or released results bit for bit. Existing files
are checked for structural validity, not whether they were fitted to your dataset.
Clustering is intentionally separate from `scripts/prepare_data.sh` so partial or
validation-only conversions cannot accidentally fit new anchors.

## Tools and Outputs

| Tool | Purpose | Default output | Shape |
| --- | --- | --- | --- |
| `kmeans_det.py` | Cluster object centers within 55 m and append initial box attributes | `kmeans_det_900.npy` | `(900, 11)` |
| `kmeans_map.py` | Cluster map-element centers and initialize 20-point line anchors | `kmeans_map_100.npy` | `(100, 20, 2)` |
| `kmeans_plan.py` | Cluster command-conditioned trajectories at four granularities | `train_mutil_granular_waypoint_cluster_refine_1.pkl` | Dictionary; see below |

All three outputs are used by the default configurations. The planning tool
replaces the unused legacy six-step `kmeans_plan_6.npy` generator; that `.npy` is
not an input to the current model.

### Planning Vocabulary

The implementation follows the command grouping, cumulative-position clustering,
and lane-change filtering of the upstream SparseDrive workspace's
`kmeans_mutil_grand_trajs.py`, with paths and horizons aligned to this repository.

It reads these matching **training-split** files in their original record order:

- `bench2drive_info_train.pkl`: navigation commands and metadata.
- `bench2drive_aligndrive_waypoints_train.pkl`: displacement offsets and masks.

Record counts and six-way navigation commands must agree. If the waypoint
`.manifest.json` exists, both input SHA-256 hashes must also match it. Without a
manifest, command/count checks cannot fully prove frame-order alignment; use
labels generated from the same metadata and only load trusted pickle files.

For each granularity, only trajectories whose selected prefix is fully valid are
used. Offsets are cumulatively summed into positions. As in the original recipe,
left lane changes with a positive lateral endpoint and right lane changes with a
negative lateral endpoint are excluded. KMeans runs independently for commands
`LEFT`, `RIGHT`, `STRAIGHT`, `LANE_FOLLOW`, `CHANGE_LANE_LEFT`, and `CHANGE_LANE_RIGHT`,
in that order.

Negative navigation-command sentinels are treated as `LANE_FOLLOW`, matching the
dataset's `command2hot` behavior and the generated supervision labels.

| Vocabulary key | Selected points | Default shape |
| --- | --- | --- |
| `spatial_interval_2_traj` | First 15 points at 2 m intervals | `(6, 1, 15, 2)` |
| `spatial_interval_5_traj` | First 6 points at 5 m intervals | `(6, 1, 6, 2)` |
| `temporal_interval_2_traj` | First 15 points at stride 2 metadata records | `(6, 1, 15, 2)` |
| `temporal_interval_5_traj` | First 6 points at stride 5 metadata records | `(6, 1, 6, 2)` |

One cluster per command and granularity matches the bundled vocabulary and the
current 24-query planning configuration. All four **supervision label** trajectories
remain 15 points; their model initialization vocabularies have different horizons.
If any command/granularity has too few valid samples, generation fails clearly
instead of filling missing anchors with zeros or borrowing validation data.

### Compatibility Verification

The three bundled anchor files are byte-for-byte identical to the original
training checkout. The planning vocabulary also matches the historical saved
clustering result byte for byte.

The current planning tool was additionally checked by refitting its clustering
step on 234,769 historical waypoint records. Spatial cluster centers differed
from the bundled vocabulary by at most `1.07e-14` m; temporal centers differed by
at most `3.15e-5` m, within the numerical tolerance of the float32 inputs. This
check used navigation commands from the historical one-hot waypoint labels
because the original metadata was not available at its legacy path.

The current train/inference configs agree on the file paths, granularity order,
and 15/6-point encoders. CPU checks of the actual anchor-loading code confirmed
24 flattened planning queries of shape `(24, 30)` and a target vocabulary of
shape `(6, 4, 15, 2)` after padding and conversion back to offsets.

This verifies the existing artifacts and the clustering recipe, not bitwise
reproduction from a fresh raw-data conversion. The converter includes documented
route-boundary/mask corrections, and different inputs or KMeans initializations
can change regenerated values. Full model/CARLA execution was not part of this
check; keep the bundled files when evaluating an existing checkpoint.

## Paths and Options

Defaults are relative to the repository containing these scripts, regardless of
the invoking working directory. No private workspace or bucket path is used.

| Setting | Default | Override |
| --- | --- | --- |
| Metadata | `<AlignDrive>/data/bench2drive_meta/infos_b2d_full/bench2drive_info_train.pkl` | `ALIGNDRIVE_ANNO_ROOT` for the directory, or `--ann-file` |
| Planning labels | `bench2drive_aligndrive_waypoints_train.pkl` beside the metadata | `--waypoints-file` |
| Anchor directory | `<AlignDrive>/kmeans/b2d/` | `ALIGNDRIVE_ANCHOR_ROOT` or `--out-dir` |
| Preview directory | `<AlignDrive>/vis/kmeans/` | `--vis-dir` |

CLI options take precedence over environment defaults. Explicit relative paths
use the invoking working directory. Directories are created automatically.
The wrapper supports `PYTHON` to select the environment's Python executable.

To generate a separate set without replacing the bundled anchors:

```bash
bash scripts/prepare_anchors.sh \
  --ann-file /path/to/bench2drive_info_train.pkl \
  --waypoints-file /path/to/bench2drive_aligndrive_waypoints_train.pkl \
  --out-dir work_dirs/custom_anchors \
  --vis-dir work_dirs/custom_anchors/previews \
  --seed 0
```

Set `ALIGNDRIVE_ANCHOR_ROOT` to the custom directory when training; it must contain
all three active files referenced by
[`_base_/perception.py`](../../projects_sp/configs/_base_/perception.py).

### Individual Tools

Each tool can also run independently:

```bash
python tools/kmeans_b2d/kmeans_det.py --out-dir work_dirs/custom_anchors
python tools/kmeans_b2d/kmeans_map.py --out-dir work_dirs/custom_anchors
python tools/kmeans_b2d/kmeans_plan.py --out-dir work_dirs/custom_anchors
```

Individual tools refuse to replace either an existing anchor or preview unless
`--overwrite` is passed. Their `--num-clusters` defaults are 900 / 100 / 1;
the unified command always uses these model-compatible defaults. Detection/map
need at least that many samples; planning needs that many valid trajectories
**per command and granularity**. Changing counts requires corresponding model and
filename changes; `--num-clusters` does not update the model configuration.

KMeans uses `--seed` (default 0) and `n_init=10`. Previews are
`b2d_det_anchor_900.png`, `b2d_map_anchor_100.png`, and
`b2d_planning_vocabulary_1.png`. The tools require NumPy, scikit-learn, Matplotlib,
and tqdm; they do not import the model, require MMCV, or use a GPU.
