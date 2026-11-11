# Dependency Reference

Follow the installation order in the [installation guide](../doc/INSTALL.md#training-installation).
The supported reference platform is Linux x86_64 with Python 3.8. Workflows have
only been tested on RTX 4090 and NVIDIA L20; RTX 5090 remains unsupported.

## Files and Provenance

| File | Purpose | Version source |
| --- | --- | --- |
| `../requirement.txt` | Training, preprocessing, and model inference dependencies | Installed versions in the local `sparsedrive` environment |
| `carla.txt` | Additional closed-loop evaluation dependencies | Installed versions in the local `sparsedrive_carla` environment |
| `visualization.txt` | Visualization and LAZ point-cloud tools | Retained pins from the previous project dependency file, not observed in the training environment |
| `dev.txt` | Test dependencies | Local training environment |
| `reference-environment.json` | Project-scoped package/version inventory and provenance | Read directly from local package metadata |

The CARLA additions come from a separate environment, whose NumPy and Shapely
versions differ from the training stack. They are additional package pins, not
a full export of that environment or proof that the combined fresh environment
has been validated end to end.

This is deliberately **not a full `pip freeze`**. The local environment also
contains unrelated projects, internal SDKs, editable installations, and an unused
ROCm torchaudio package. Exporting all of them would not provide a portable
AlignDrive installation. PyTorch and torchvision are installed separately;
torchaudio is not used by AlignDrive and is omitted.

The original K8s training configuration starts from an image with MMCV 1.7.1
already installed. Its extra installation commands add ninja, a prebuilt
FlashAttention 2.3.2 wheel, mpi4py, OmegaConf, and Shapely 2.0.0. The dependency
files retain that layering rather than assuming MMCV exists on a fresh machine.
The K8s script does not pin ninja, mpi4py, or OmegaConf; local installed versions
are used where available. mpi4py is not installed in the inspected training
environment, so no exact K8s mpi4py version is claimed.

## Binary Packages

- Install `torch==1.13.0+cu116` and `torchvision==0.14.0+cu116` before extensions.
- Install `mmcv-full==1.7.1` using the CUDA 11.6 / PyTorch 1.13.0 wheel index in
  the main README. Do not substitute `mmcv`, `mmcv-lite`, or MMCV 2.x. Importing
  `mmcv.ops` verifies that the compiled extension can load, not just its metadata.
- The K8s FlashAttention wheel is named
  `flash_attn-2.3.2+cu116torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl`.
  The wheel itself is not included in this repository. Supply a compatible
  wheel; its Python, PyTorch, CUDA, and C++ ABI tags must match the environment.
- A CUDA development toolkit with `nvcc` and a C++ compiler is needed to build
  the project's deformable aggregation extension. A PyTorch CUDA wheel alone
  does not provide the compiler. Set `CUDA_HOME` and `PATH` to your toolkit.

If the reference FlashAttention wheel is unavailable, an alternative is to build
the same release from source after installing the base dependencies and a
compatible CUDA toolkit:

```bash
FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=4 python3 -m pip install \
  flash-attn==2.3.2 --no-build-isolation --no-deps
```

This alternative is not the prebuilt-wheel path used by the recorded K8s jobs
and has not been validated as a fresh installation here. Do not upgrade PyTorch
to resolve an extension installation problem without rebuilding all extensions.

## Bench2Drive Scope

The documented training, preprocessing, inference, and visualization workflows
use Bench2Drive and do not require `nuscenes-devkit`. The default plugin no
longer registers the legacy nuScenes dataset or constructs its evaluation
configuration. Shared geometry utilities use `pyquaternion` directly, and the
legacy nuScenes box conversion imports the SDK only when explicitly called.

Legacy nuScenes source files remain for reference but are outside the current
supported workflows. No nuScenes SDK installation, Jupyter dependency, or
Shapely compatibility override is needed for Bench2Drive. Shapely 2.0.0 remains
a direct dependency of the Bench2Drive geometry and scoring code.

## Optional Workflows

```bash
python3 -m pip install -r requirements/carla.txt
python3 -m pip install -r requirements/visualization.txt
python3 -m pip install -r requirements/dev.txt
```

The normal single-node training command uses PyTorch's distributed launcher and
does not require MPI. If using the old `mpi_nccl` launcher, install OpenMPI and
its development headers before installing `mpi4py`:

```bash
sudo apt-get install openmpi-bin libopenmpi-dev
python3 -m pip install mpi4py
```

These lists do not install the CARLA simulator, maps, datasets, checkpoints,
ffmpeg, or ROS integrations. Human-controlled and ROS-specific ScenarioRunner
examples may need further dependencies outside the documented AlignDrive path.

## Validation Scope

The version inventory records an existing environment; it is not evidence that a
clean environment has been fully installed or that every optional tool has been
exercised. Run the import checks in the main README, then the project tests and
your intended training/evaluation workflow. No local environments are modified
when generating the inventory.
