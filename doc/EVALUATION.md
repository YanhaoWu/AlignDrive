# Evaluation

Complete [CARLA Installation](INSTALL.md#carla-installation) first. Activate the
`aligndrive` environment and set `CARLA_ROOT` in the shell used for evaluation.
Use a trusted final Lon checkpoint, not a Spatial initialization checkpoint.

## Pretrained Model

Download the [released weights (Google Drive)](https://drive.google.com/file/d/1Z91nzvlXuyzKPjxyViKjjfYOpq7RbU6t/view?usp=drive_link)
and save them as `ckpt/AlignDrive_SOTA_weights.pth`. No training is required.
The checkpoint must be paired with
`projects_sp/configs/inference/aligndrive_sota.py`, copied from the resolved
configuration used in the release smoke test. It uses FPN and the released
map-attention settings, which differ from the default training configuration.
Keep strict checkpoint loading enabled; a loading error indicates that the
checkpoint and configuration need to be checked.

Run from the repository root after installation, with `CARLA_ROOT` set:

```bash
GPUS=2 bash scripts/evaluate.sh --checkpoint ckpt/AlignDrive_SOTA_weights.pth \
  --config_path projects_sp/configs/inference/aligndrive_sota.py \
  --project_name AlignDrive_SOTA --collision-mode fast
python3 carla_tools/merge.py output/AlignDrive/AlignDrive_SOTA_full/result
```

This evaluates all 220 routes and writes to `output/AlignDrive/AlignDrive_SOTA_full/`.
Use a different `--project_name` for each repeated run to keep results separate.
`fast` is the collision implementation used for the paper's inference timing.
The alternative `--collision-mode shapely` may change both runtime and planning
decisions, so report the selected mode when comparing results.

For a single-route check using the route exercised in the release smoke test:

```bash
ALIGNDRIVE_CONFIG=projects_sp/configs/inference/aligndrive_sota.py \
ALIGNDRIVE_COLLISION_MODE=fast \
bash scripts/carla_inference.sh ckpt/AlignDrive_SOTA_weights.pth \
  output/AlignDrive/SOTA_single_route --routes-subset 2115
python3 carla_tools/merge.py output/AlignDrive/SOTA_single_route --expected-routes 1
```

Render an individual saved scene directory containing its `meta` and `bev` folders:

```bash
bash scripts/visualize.sh /path/to/saved_scene --render-version full
```

Six-camera rendering is opt-in. See [Visualization](VISUALIZATION.md).
The single-route smoke test checks execution; it is not a full benchmark result.

## 1. Bench2Drive Simulation

```bash
bash scripts/evaluate.sh --checkpoint work_dirs/aligndrive/last.pth
```

Replace `--checkpoint` to evaluate another trained model. The wrapper
uses `projects_sp/configs/inference/aligndrive_inference.py`, the complete
220-route split, longitudinal output, and the `control_2m5hz` controller. It uses two GPUs
and disables automatic restart by default. `GPUS` changes the default GPU count;
other arguments are forwarded through `leaderboard/scripts/launch_evaluation.py`
to the general interface, `leaderboard/scripts/evaluate_multi_gpu.py`.

The default run directory is `output/AlignDrive/AlignDrive_full/`. Route results
are under `result/`; saved inference scenes are under `visulization/`.

## 2. Merge and View Results

After the evaluation finishes:

```bash
python3 carla_tools/merge.py output/AlignDrive/AlignDrive_full/result
```

This writes `result/merged.json` and prints the evaluated route count, Driving
Score, and Success Rate. The default denominator is 220 routes. Incomplete runs
are explicitly warned about and remain normalized to 220; do not report them as
a complete benchmark result. Duplicate route IDs are rejected, and unrelated
summary JSON files are ignored. Do not mix shards from different runs.

For an intentional route subset, pass its expected route count with
`--expected-routes`. The legacy `carla_tools/merge_route_json.py -f RESULT_DIR`
entry point remains available.

## Additional Tools

Open-loop evaluation on prepared validation data:

```bash
bash scripts/test.sh --checkpoint /path/to/aligndrive.pth \
  --out work_dirs/offline/results.pkl --eval bbox
```

Use the prepared validation data and the command above for open-loop planning
metrics; the reported metric names follow the training configuration.

Single-route interface check (requires an already running CARLA server):

```bash
bash scripts/carla_inference.sh /path/to/aligndrive.pth work_dirs/single_route
```

A single-route check does not reproduce the benchmark. GPU selection, screen
sessions, ports, log collection, and cleanup are documented separately in
[Simulation Process Management](SIMULATION.md). See [Visualization](VISUALIZATION.md)
for rendering saved scenes.

Return to the [project README](../README.md#evaluation).
