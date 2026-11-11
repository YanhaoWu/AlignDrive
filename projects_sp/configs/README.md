# AlignDrive Configurations

Training and inference use MMCV `_base_` inheritance to share model architecture,
Bench2Drive settings, and runtime defaults. Keep using the four entry points below;
the files in `_base_/` are configuration fragments, not standalone launch recipes.

| Configuration | Purpose | Initialization |
| --- | --- | --- |
| `train/aligndrive_stage_perception.py` | Perception training | ResNet50 backbone |
| `train/aligndrive_stage_spatial.py` | Spatial-planning training | Perception checkpoint, loaded selectively |
| `train/aligndrive_stage_lon.py` | Final longitudinal-planning training | Spatial checkpoint |
| `inference/aligndrive_inference.py` | Bench2Drive inference and open-loop evaluation | Explicit final checkpoint, loaded strictly |

See [Training](../../doc/TRAINING.md) and [Evaluation](../../doc/EVALUATION.md)
for the launch commands and checkpoint arguments.

## Shared Configuration

```text
_base_/perception.py
  ├── train/aligndrive_stage_perception.py
  └── _base_/aligndrive.py
        ├── train/aligndrive_stage_spatial.py
        ├── train/aligndrive_stage_lon.py
        └── inference/aligndrive_inference.py
```

| File | Edit here to change |
| --- | --- |
| `_base_/perception.py` | Shared backbone, neck, detection, map, and anchor paths for all stages |
| `_base_/aligndrive.py` | Complete planning model: query fusion, lateral drive-path planning, and the aligned longitudinal branch |
| `_base_/bench2drive.py` | Dataset paths, class mappings, camera augmentation, and common image transforms |
| `_base_/training.py` | Shared batch size, optimizer, learning-rate schedule, hooks, and automatic training schedule |
| `_base_/runtime.py` | Plugin loading, distributed backend, and mixed precision |
| `train/*.py`, `inference/*.py` | Stage-specific task switches, losses, pipeline composition, checkpoint initialization, and inference controls |

There are only two model files: `perception.py` defines the shared perception
modules, and `aligndrive.py` extends them into the complete AlignDrive planning
model. Lateral and longitudinal planning are configured together in
`aligndrive.py`, not split across separate model inheritance levels.

Perception training inherits `perception.py` directly. Spatial training, final
training, and inference all inherit the same `aligndrive.py`. The spatial entry
disables `longitudinal_head`, selects the base planning output, and explicitly
retains its original loss weights and attention settings. Final training and
inference keep their own loss and control overrides, respectively.

For example, changing `model.img_backbone.depth` in `perception.py` updates all
four entries. Changing the drive-path or longitudinal architecture in
`aligndrive.py` updates every stage that uses that branch. All entries also
inherit the dataset and runtime fragments; only training entries inherit
`training.py`.

Entry files contain only overrides and stage-specific settings. Nested dictionaries
merge recursively, but lists replace the entire inherited list; use `_delete_=True`
only when a whole inherited dictionary must be replaced. The
`{{_base_.pipeline_steps...}}` expressions reuse shared image transforms while
preserving each stage's pipeline order and augmentation differences. These files
must be loaded through MMCV `Config.fromfile`, not imported as Python modules.
Resolved configurations can still be dumped and reloaded normally.

Command-line overrides continue to address the resolved configuration, such as
`model.img_backbone.depth` or `data.samples_per_gpu`. Changing the shared
`training.py` `batch_size` updates `data.samples_per_gpu` in all training entries.
If overriding batch size on the command line, use `data.samples_per_gpu`; the
schedule resolver reads this field and updates the derived batch-size fields.

## Active Settings

- `model` defines the actual model, task switches, decoder operations, and losses.
- `data` defines the datasets and their preprocessing pipelines.
- `optimizer`, `optimizer_config`, and `lr_config` control training optimization.
- `evaluation` defines validation metrics and checkpoint selection.
- Training runtime fields, including `load_from`, `resume_from`, `fp16`,
  `custom_hooks`, and `workflow`, are read by the training entry point and runner;
  they are not unused simply because the configuration does not reference them.

For example, decoder settings belong under `model.head`, not in a separate
top-level copy. Historical shadow variables such as top-level `num_modal`,
`use_longitudinal`, `final_used_head`, and `temporal_map` have been removed because
changing them did not change the model. Duplicate standalone head definitions,
unused visualization pipelines, and `data_basic_config` have also been removed.

The perception configuration does not instantiate a planning head. Unused
`val_debug` datasets have been removed. The inference configuration contains only
the test dataset and evaluation settings, without training datasets or optimizer
settings. Its `test_pipeline`, `data_aug_conf`, `num_cmd`, `control_setting`,
`strict_checkpoint`, and `require_longitudinal_output` are also read by the CARLA
agent and must remain available at the top level. The agent uses six navigation
commands; the planner's seven-entry command embedding additionally includes a
noise command, so these two counts are intentionally different.

## Training Schedule

Training uses the real dataset size and distributed world size:

```text
global_batch = world_size * data.samples_per_gpu
num_iters_per_epoch = dataset_size // global_batch
max_iters = num_epochs * num_iters_per_epoch
```

`tools/training_schedule.py` resolves checkpoint and evaluation intervals from
`checkpoint_epoch_interval` and `eval_epoch_interval`. It also replaces
`auto_half_training` milestones. Leave the automatic runner and hook intervals
unset; do not reintroduce fixed GPU counts, dataset lengths, or iteration counts.

The final training configuration retains the default scenario-specific loss
weights and collision-loss reduction as stage overrides. Shared traffic-light
class weights live in `aligndrive.py`. All are defined in the active
model configuration, rather than overwritten by assignments at the end of a file.

## Paths and Checkpoints

Dataset and anchor paths use `ALIGNDRIVE_DATA_ROOT`, `ALIGNDRIVE_ANNO_ROOT`, and
`ALIGNDRIVE_ANCHOR_ROOT`. Stage initialization uses
`ALIGNDRIVE_RESNET_CHECKPOINT`, `ALIGNDRIVE_PERCEPTION_CHECKPOINT`, and
`ALIGNDRIVE_SPATIAL_CHECKPOINT`.
The shared backbone respects `ALIGNDRIVE_RESNET_CHECKPOINT` in inference as well
as training. Evaluation still loads the explicitly supplied final checkpoint.

Longitudinal-planning keys and outputs use `longitudinal`. Earlier checkpoint
module prefixes are migrated automatically during loading; newly saved
checkpoints use `head.longitudinal_head`. Inference weights are supplied through
the evaluation command, not the training-only `load_from` setting.

## Planning Targets

All configurations use `MultiGranularityPlanningTarget` for spatial and temporal
trajectory assignment. It replaces the legacy `MaplikePlanningTarget` name; no
legacy registry alias is retained. The helper functions are named
`find_closest_trajectory_indices`, `select_speed_granularity`, and
`select_distance_granularity`.
