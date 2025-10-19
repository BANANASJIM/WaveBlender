# WaveBlender GPU Sound Rendering Engine

WaveBlender is a (proof-of-concept) finite-difference time-domain (FDTD) acoustic wavesolver for simulating animation sound sources, including vibrating rigid bodies, bubble-based water, animated occluders, and more. The code is based on the paper:
    
> [WaveBlender: Practical Sound-Source Animation in Blended Domains](https://graphics.stanford.edu/papers/waveblender/). Kangrui Xue, Jui-Hsien Wang, Timothy R. Langlois, Doug L. James. *SIGGRAPH Asia 2024 Conference Papers*. 

**9/8 Update:** This codebase has been [gaining some traction recently](https://www.youtube.com/watch?v=1bS7sHyfi58). Keep in mind that this is an early-gen prototype for research. There are plans to release a more comprehensive and useable sound rendering system later -- stay tuned :)

## Build Instructions

**Dependencies:** C++17, CUDA, Eigen 3.4+ (or 5.0+), libigl, Python 3.8+

### Quick Setup (Recommended)

```bash
./setup.sh
```

This script will automatically check dependencies, build the project, setup Python environment, and optionally download additional scenes.

### Manual Build

Building is handled by CMake. For example, to build from source on Linux:

```bash
git clone https://github.com/BANANASJIM/WaveBlender
cd WaveBlender && git submodule update --init --recursive

# Setup Python environment
python -m venv venv
source venv/bin/activate
pip install numpy matplotlib scipy soundfile

# Build
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
```

### Running

Quick run with interactive menu:
```bash
./run.sh
```

Or manually:
```bash
cd build
./WaveBlender ../Scenes/CupPhone/config.json
source ../venv/bin/activate
python ../scripts/write_wav.py CupPhone_out.bin 88200
```

Afterwards, the simulated audio will be written to 'CupPhone_out.wav'. More scenes are available [here](https://graphics.stanford.edu/papers/waveblender/dataset/).

### What's New in This Fork

- ✅ **Eigen 5.0 compatibility** - Removed version constraints, upgraded libigl to v2.6.0
- ✅ **One-click setup script** - `setup.sh` for automated installation
- ✅ **Quick run script** - `run.sh` for easy scene execution
- ✅ **Python venv support** - Isolated dependency management

### Miscellaneous

This code is intended to serve as a reference implementation and to reproduce results from the paper. Some optimizations have been removed in favor of readability. 

Documentation can be built using Doxygen: `doxygen Doxyfile`

### Citation

```bibtex
@article{Xue:2024:WaveBlender,
  author = {Xue, Kangrui and Wang, Jui-Hsien and Langlois, Timothy R. and James, Doug L.},
  title = {WaveBlender: Practical Sound-Source Animation in Blended Domains},
  year = {2024},
  conference = {SIGGRAPH Asia 2024}
}
```
