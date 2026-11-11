# Model Organization

The model code is organized into four task packages. The top-level modules
provide the complete AlignDrive model, multi-task orchestration, and shared
building blocks.

```text
models/
├── aligndrive.py                 # AlignDrive detector and training entry
├── aligndrive_head.py            # Multi-task head
├── unified_head.py               # Unified decoder and query fusion
├── attention.py                  # Shared attention and positional embeddings
├── blocks.py                     # Shared feature aggregation and network blocks
├── base_target.py                # Shared target-assignment interface
├── instance_bank.py              # Instance banks and temporal feature queues
├── grid_mask.py                  # Image augmentation
├── utils.py                      # Shared geometry and planning utilities
├── detection3d/                  # 3D detection and agent-motion refinement
├── drive_path/                   # Lateral driving-path prediction
├── map/                          # Vectorized map prediction
└── longitudinal_planning/        # Longitudinal planning and trajectory supervision
    ├── longitudinal_planning_module.py
    ├── motion_planning_head.py
    ├── motion_blocks.py
    ├── instance_queue.py
    ├── decoder.py
    ├── target.py
    ├── loss.py
    └── score_module/
        └── compute_b2d_score_wyh.py
```

## Task Responsibilities

- **`detection3d`** contains detection heads, box encoders and refinement blocks,
  target assignment, losses, and decoding.
- **`drive_path`** predicts lateral driving paths and provides path-specific
  temporal mixing and feature blocks.
- **`map`** contains vectorized map encoders and refinement blocks, matching
  costs, target assignment, losses, and decoding.
- **`longitudinal_planning`** predicts longitudinal displacements along driving
  paths and contains trajectory targets, losses, decoders, scoring, and temporal
  queues. The existing joint motion/planning head and its supporting components
  remain available in this package without changing their registry names.

## Import Migration and Checkpoints

Use `projects_sp.mmdet3d_plugin.models.longitudinal_planning` for planning
imports. Replace the former `models.motion` prefix in external scripts; no
compatibility package is retained under the old directory name.

This reorganization changes Python module paths, not registered model names,
configuration types, model attributes, or tensor computations. Standard
`state_dict` checkpoints retain their parameter keys. Serialized whole-model
Python objects may still depend on the old module paths; use `state_dict`
checkpoints for model interchange.
