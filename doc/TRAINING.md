# Training

Complete [Training Installation](INSTALL.md#training-installation) and
[Data Preparation](../tools/data_converter/README.md) first.

## Default Training

Run the three stages separately, in order, with the default configurations.

### Stage 1: Perception

```bash
bash scripts/train.sh perception --checkpoint ckpt/resnet50-19c8e357.pth
```

### Stage 2: Spatial

```bash
bash scripts/train.sh spatial --checkpoint work_dirs/perception/last.pth
```

### Stage 3: Longitudinal Planning

```bash
bash scripts/train.sh lon --checkpoint work_dirs/spatial/last.pth
```

Pass the initialization checkpoint explicitly with `--checkpoint`:

- Perception: pretrained ResNet50 backbone weights, not a full AlignDrive checkpoint.
- Spatial: the completed Perception model checkpoint.
- Lon: the completed Spatial model checkpoint.

The configurations and default output locations are:

| Stage | Configuration | Output checkpoint |
| --- | --- | --- |
| Perception | `projects_sp/configs/train/aligndrive_stage_perception.py` | `work_dirs/perception/last.pth` |
| Spatial | `projects_sp/configs/train/aligndrive_stage_spatial.py` | `work_dirs/spatial/last.pth` |
| Lon | `projects_sp/configs/train/aligndrive_stage_lon.py` | `work_dirs/aligndrive/last.pth` |

The three configurations use the shared joint-training model definition.
The stage names select data, heads, checkpoints, and evaluation settings; they
do not enable parameter freezing.

Use the final Lon checkpoint for Bench2Drive evaluation. ResNet50 initialization
is prepared during installation. Wait for each stage to finish before starting
the next; no command starts multiple stages automatically.

`--checkpoint` takes precedence over the `ALIGNDRIVE_*_CHECKPOINT` environment
variables. If omitted, the launcher uses the default paths shown above (or the
environment overrides). Relative checkpoint paths are resolved from the directory
where the command is invoked. Initialization loads weights only; it does not
restore optimizer state or training progress.

The launcher uses two GPUs by default. Set `GPUS` to change the count, for
example `GPUS=8 bash scripts/train.sh lon --checkpoint work_dirs/spatial/last.pth`.
Configured batch sizes, losses,
optimizer, and learning-rate settings are not changed by the wrapper.

Planning evaluation reports only the selected-grand `L2_spatial` for Spatial
training. Lon training reports `L2_spatial` and `L2_final`; best-checkpoint
selection uses `L2_spatial` for Spatial and `L2_final` for Lon.

Additional `train.py` options are forwarded for individual stages:

```bash
bash scripts/train.sh lon --resume-from /path/to/iter_x.pth
```

When overriding `--work-dir`, update the next stage's initialization checkpoint
path accordingly. `last.pth` is selected from periodic checkpoints, not `best_*`
checkpoints.

## Default Configuration

The three configuration files are the standard training settings, with no
additional recipe selection. The final planner uses four trajectory granularities
and longitudinal output. Model architecture and dataset defaults are shared with
inference through `_base_` files; see the
[configuration guide](../projects_sp/configs/README.md#shared-configuration) for
where to edit shared settings and stage overrides.
Checkpoints, annotation paths, and anchors can be overridden with
the `ALIGNDRIVE_*` variables described in the configuration files and
[data preparation guide](../tools/data_converter/README.md).

Return to the [project README](../README.md#training).
