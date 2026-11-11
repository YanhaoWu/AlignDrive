# Visualization

Complete [Visualization Installation](INSTALL.md#visualization-installation) first.
All rendering implementations now live in `tools/visualization/`, with two tasks:

- **Simulation:** overlay model predictions on saved CARLA inference frames and export videos.
- **Perception:** inspect raw Bench2Drive camera, annotation, HD-map, and LiDAR data during debugging.

## Simulation replay

```bash
bash scripts/visualize.sh /path/to/saved_scene
```

Six-camera overlays are disabled by default; enable them explicitly with
`--render-rgb` only when the saved scene contains the six camera image folders.

This is a shortcut for the unified entry point:

```bash
python3 tools/visualization/visualize.py simulation \
  --scene-dir /path/to/saved_scene --output-dir work_dirs/visualization
```

The saved scene must contain matching `meta/<frame>.pkl` and `bev/<frame>.png`
or `.jpg` files. Metadata must include `plan` and may include `result` and
`data`, as written by the AlignDrive inference agent. Only load trusted pkl
files. Frames are matched by filename, including frame zero and nonconsecutive
frame IDs; missing metadata produces an explicit error.

By default, the **red curve** shows `result.waypoints_drivepath` and the
**blue curve** shows the final longitudinally timed trajectory from `plan`.
If the saved result has no drive path, the renderer displays the final plan.

| Option | Description |
| --- | --- |
| `--output-dir` | Output root; default: `work_dirs/visualization` |
| `--scene-name` | Video name; defaults to the input directory name |
| `--workers` | Rendering processes; default: 1; multiple workers require `multimodal` mode |
| `--fps` | Output frame rate; default: 10 |
| `--render-version` | `full`: controller text and perception; `presentation`: compact text and BEV perception; `clean`: trajectories only |
| `--trajectory-mode` | `aligned`: red drive path and blue timed trajectory; `plan`: final trajectory only |
| `--bev-view` | Match the saved BEV camera geometry: `topdown` or `front` |
| `--render-rgb` | Add the six-camera perception panel; requires saved camera images and calibration in `data` |
| `--mode bev --view-name rgb_front` | Draw overlays on the saved front-camera image instead of the BEV panel |

For example:

```bash
bash scripts/visualize.sh /path/to/saved_scene \
  --render-version clean --workers 2 --fps 10
```

Videos are written to `multimodal_pred/video_<scene>.mp4` or
`bev_pred/video_<scene>.mp4` below the output root. Both serial and parallel
rendering use the same frame and video-writing pipeline. Parallel jobs use
bounded batches to avoid retaining an entire scene of rendered images in RAM.

`--render-rgb` needs the six `rgb_*` directories plus camera paths,
`lidar2cam`, `cam_intrinsic`, and the navigation command in each frame's
`data`. Camera paths are resolved against the current scene first, so a saved
scene can be moved without retaining its original absolute paths.

Saved-scene rendering does not require a running CARLA server or compiled
training operators. H.264 output requires `ffmpeg`; camera overlays also use
the rendering dependencies from the training environment.

## Perception debugging

```bash
python3 tools/visualization/visualize.py perception \
  --scene-dir /path/to/Bench2Drive/v1/scene \
  --map-path /path/to/Town01_HD_map.npz \
  --output-dir work_dirs/perception_debug
```

The input is a **raw dataset scene**, not an inference replay: it needs
`camera/`, `anno/*.json.gz`, and `lidar/*.laz`. The command produces camera
bounding-box overlays, a top-down overlay, road annotations, and LiDAR
projections. Output defaults to a sibling directory named `<scene>_vis`.
The output directory must differ from the input directory.

`--map-path` also accepts `Town01` or `01` to use
`maps/Town01_HD_map.npz` under the repository root. To generate a map, connect
to an existing CARLA server; this command loads the specified Town:

```bash
python3 carla_tools/gen_hdmap.py --carla_town Town01 \
  --save_dir maps --host localhost --port 2000
```

For programmatic debugging, import `visualize_data` from
`tools.visualization.perception`; its `vis_*` flags select individual panels.
The camera and BEV prediction renderers remain available in `cam_render.py`
and `bev_render.py`. `B2D3DDataset.show(...)` delegates to the shared
`perception.render_predictions(...)` implementation for model-result debugging.
Pass the image-loading visualization pipeline, rather than a normalized
training pipeline, when using this API.

## RGB-only videos

For scenes with `rgb_front/` images and `meta/<frame>.json` controller logs:

```bash
python3 carla_tools/generate_video.py \
  --scene-dir /path/to/rgb_scene --output work_dirs/rgb.mp4 --fps 15
```

Each JSON file must contain `speed`, `steer`, `throttle`, and `brake`.
This utility does not read inference pkl files.

## Compatibility

The previous `carla_tools/render_video.py` and `carla_tools/visualize.py`
commands remain thin forwarding entry points. `carla_tools/render.py`,
`carla_tools/utils.py`, and `carla_tools/media.py` re-export the shared
implementations; they no longer contain separate rendering logic.

Return to the [project README](../README.md#visualization).
