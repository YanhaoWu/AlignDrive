# Installation

1. [Training Installation](#training-installation)
2. [CARLA Installation](#carla-installation)
3. [Visualization Installation](#visualization-installation)

Run all commands from the AlignDrive repository root unless stated otherwise.
Replace `/path/to/...` with your own paths.

## Training Installation

**Hardware support:** the workflows have only been tested on **RTX 4090** and
**NVIDIA L20**. **RTX 5090 is not currently supported**; its compatibility issues
remain unresolved. The reference stack uses Linux x86_64, Python 3.8, PyTorch
1.13.0 + CUDA 11.6, MMCV-full 1.7.1, and FlashAttention 2.3.2.

1. **Create the environment and install the base dependencies.**

```bash
conda create -n aligndrive python=3.8 -y
conda activate aligndrive
cd /path/to/AlignDrive

export PYTHON=python3
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python3 -m pip install pip==24.3.1 setuptools==69.5.1 wheel==0.43.0
python3 -m pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 \
  --extra-index-url https://download.pytorch.org/whl/cu116
python3 -m pip install -r requirement.txt
```

2. **Install MMCV-full separately.** The original K8s image already contained
MMCV; a fresh environment does not. Do not substitute MMCV 2.x or `mmcv-lite`.

```bash
python3 -m pip install mmcv-full==1.7.1 --only-binary=mmcv-full \
  -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13.0/index.html
```

3. **Install FlashAttention.** Supply the compatible prebuilt wheel used by the
reference stack; the wheel is not included in this repository.

```bash
export ALIGNDRIVE_FLASH_ATTN_WHEEL=/path/to/flash_attn-2.3.2+cu116torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
python3 -m pip install --no-deps "$ALIGNDRIVE_FLASH_ATTN_WHEEL"
```

For a source-build alternative and the observed local package versions, see
[Dependency Reference](../requirements/README.md) and the
[environment inventory](../requirements/reference-environment.json). A complete
fresh installation has not yet been validated. The documented workflows use
**Bench2Drive only** and do not require `nuscenes-devkit` or torchaudio.

4. **Compile the deformable aggregation operator.** Install a CUDA development
toolkit with `nvcc` and a compatible C++ compiler, then set its path:

```bash
export CUDA_HOME=/path/to/cuda-11.6
export PATH="$CUDA_HOME/bin:$PATH"

(cd projects_sp/mmdet3d_plugin/ops && python3 setup.py build_ext --inplace)
```

The local RTX 4090 smoke test built this operator with **CUDA Toolkit 11.8 and
GCC 11.4**, while keeping PyTorch **1.13.0+cu116** unchanged. The system GCC 4.8
was too old; CUDA 11.6 with GCC 11.4 also failed to compile on that host. Set
`CC` and `CXX` to your compatible compiler before building. This is a tested
host-specific build combination, not a requirement to change the PyTorch wheel.

If importing the operator after OpenCV reports an undefined C++ symbol such as
`exception_ptr::_M_addref`, an older C++ runtime has been loaded first. Select
the runtime belonging to the compiler used for the build, then repeat the
import check and keep the setting for preprocessing/training:

```bash
export LD_PRELOAD="$("${CXX:-g++}" -print-file-name=libstdc++.so.6)${LD_PRELOAD:+:$LD_PRELOAD}"
```

5. **Check the installation.**

```bash
python3 - <<'PYTHON'
import cv2
import torch
import torchvision
import mmcv
from mmcv.ops import MultiScaleDeformableAttention
from flash_attn.flash_attn_interface import flash_attn_varlen_kvpacked_func
import projects_sp.mmdet3d_plugin

assert torch.__version__ == '1.13.0+cu116'
assert torchvision.__version__ == '0.14.0+cu116'
assert mmcv.__version__ == '1.7.1'
print('AlignDrive imports passed; CUDA available:', torch.cuda.is_available())
PYTHON
```

6. **Download the pretrained ResNet50 initialization.**

```bash
mkdir -p ckpt
wget -O ckpt/resnet50-19c8e357.pth https://download.pytorch.org/models/resnet50-19c8e357.pth
```

## CARLA Installation

Complete the [Training Installation](#training-installation) first. CARLA uses
the same `aligndrive` Conda environment; it does not require a separate
environment. This section is required only for closed-loop evaluation and can
be skipped for offline training and evaluation.

1. **Download CARLA 0.9.15 and the additional maps.**
Download CARLA 0.9.15 and the additional maps using the Linux
archives linked in the official
[CARLA 0.9.15 release](https://github.com/carla-simulator/carla/releases/tag/0.9.15).
Run the following commands from the **AlignDrive repository root**:

```bash
export ALIGNDRIVE_ROOT="$PWD"
mkdir -p carla && cd carla
wget -c https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
wget -c https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.15.tar.gz

tar -xvf CARLA_0.9.15.tar.gz
mkdir -p Import
cd Import && tar -xvf ../AdditionalMaps_0.9.15.tar.gz
cd .. && bash ImportAssets.sh

export CARLA_ROOT="$PWD"
cd "$ALIGNDRIVE_ROOT"
```

If CARLA is already installed elsewhere, skip the download and extraction and
set `export CARLA_ROOT=/absolute/path/to/CARLA_0.9.15` instead.

2. **Install the supporting packages** in the activated AlignDrive environment:

```bash
sudo apt-get update
sudo apt-get install screen procps libgl1 libglib2.0-0 vulkan-tools
python3 -m pip install -r requirements/carla.txt

export PYTHONPATH="$CARLA_ROOT/PythonAPI:$CARLA_ROOT/PythonAPI/carla:$PYTHONPATH"
```

3. **Install the CARLA Python API.** Use version **0.9.15 matching Python 3.8**.
If your distribution provides a matching wheel, install it with:

```bash
python3 -m pip install /path/to/carla-0.9.15-cp38-cp38-linux_x86_64.whl
```

The wheel path is a placeholder for your actual matching artifact, not a
download URL. If the archive only provides an API for another Python version,
obtain or build the matching Python 3.8 API using the
[official build instructions](https://carla.readthedocs.io/en/0.9.15/build_linux/).
Do **not** add a `py3.7` egg to this Python 3.8 environment. If using a matching
egg instead of a wheel, register its path in the activated Conda environment:

```bash
export CARLA_EGG="$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.8-linux-x86_64.egg"
test -f "$CARLA_EGG" && printf '%s\n' "$CARLA_EGG" > "$CONDA_PREFIX/lib/python3.8/site-packages/carla.pth"
```

4. **Check the API before evaluation.**

```bash
python3 -c 'import carla; assert hasattr(carla, "TrafficLightState")'
```

## Visualization Installation

Complete [Training Installation](#training-installation), then install the video
and rendering dependencies in the same environment:

```bash
sudo apt-get update
sudo apt-get install ffmpeg
python3 -m pip install -r requirements/visualization.txt
```

See [Visualization](VISUALIZATION.md) for input formats and rendering commands.
CARLA is only needed when connecting to the simulator, not for rendering saved
inference scenes.

Continue with [Data Preparation](../README.md#data-preparation),
[Training](../README.md#training), or [Evaluation](../README.md#evaluation).
