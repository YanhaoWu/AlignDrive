# Evaluation

Complete [CARLA Installation](INSTALL.md#carla-installation) first. Activate the
`aligndrive` environment and set `CARLA_ROOT` in the shell used for evaluation.
Use a trusted final Lon checkpoint, not a Spatial initialization checkpoint.

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
