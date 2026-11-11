# Simulation Process Management

`leaderboard/scripts/process_manager.py` provides `ScreenManager` for screen sessions and process trees, and `SimulatorManager` for GPU / port allocation, route splitting, logs, and monitoring.

There are two evaluation entry points:

- [`launch_evaluation.py`](../leaderboard/scripts/launch_evaluation.py) selects the
  default AlignDrive model configuration, controller, and full route set, then
  delegates to the general interface. `scripts/evaluate.sh` calls this launcher.
- [`evaluate_multi_gpu.py`](../leaderboard/scripts/evaluate_multi_gpu.py) is the
  general multi-GPU interface for configuring and running the simulation pipeline.

Launch an evaluation through the general interface directly:

```bash
python3 leaderboard/scripts/evaluate_multi_gpu.py \
  --weights_path "$ALIGNDRIVE_CHECKPOINT" \
  --num_gpu 2 --used_gpu 0 1 --route_version full \
  --project_path output/AlignDrive --project_name AlignDrive --no-auto-restart
```

Routes are split for the actual GPU count. This example writes:

```text
output/AlignDrive/AlignDrive_full/
├── args.json                     # Launch arguments
├── sessions.json                 # Screen session names
├── result/                       # Per-GPU JSON and merged.json
├── log/                          # Runtime and error logs
├── routes/                       # Route shards for this run
├── files/                        # Source snapshots
└── visulization/                 # Saved scene frames and inference metadata
```

Inspect or stop sessions:

```bash
screen -ls
screen -r eval_0
# Detach without stopping: press Ctrl-A, then D.

# Stop all sessions recorded for this evaluation.
python3 leaderboard/scripts/process_kill.py --log-dir output/AlignDrive/AlignDrive_full

# Stop only a named session and its descendants.
python3 leaderboard/scripts/process_kill.py --session eval_0
bash carla_tools/clean_carla_for_run_scene.sh eval_0
```

- Default session names are `eval_0`, `eval_1`, etc. Do not launch concurrent batches with the same names under one account.
- RPC / TM ports come from `PORTList_0/1` in the evaluation script; check for conflicts before launch.
- Relative project, configuration, and route paths in the multi-GPU tools are resolved from the repository root.
- Omitting `--no-auto-restart` enables automatic fault recovery. Disabling it may require manually stopping failed jobs.
- Ctrl-C stops the monitor but may leave screen jobs running. Stop them explicitly when needed.
- `clean_carla.sh` also requires a session name; it does not perform global process cleanup.

Completed evaluations are aggregated automatically. Add `--only_evaluate` to the same command to collect existing results. Use `--route_version CutIn` or another supported subset for targeted evaluation; `--evaluate_task metric visulization` also collects subset videos. Existing names such as `visulization` and `mutilmoda_pred` are retained for compatibility.

Keep `scenario_runner/`: closed-loop evaluation imports its `srunner` modules for scenario behavior, timing, data access, and traffic events. It is not required for offline training, but it is required for CARLA evaluation.

Return to the [evaluation guide](EVALUATION.md).
