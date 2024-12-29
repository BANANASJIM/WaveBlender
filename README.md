# WaveBlender GPU Sound Rendering Engine

WaveBlender is a (proof-of-concept) finite-difference time-domain (FDTD) acoustic wavesolver for simulating animation sound sources, including vibrating rigid bodies, bubble-based water, animated occluders, and more. The code is based on the paper:
    
> [WaveBlender: Practical Sound-Source Animation in Blended Domains](https://graphics.stanford.edu/papers/waveblender/). Kangrui Xue, Jui-Hsien Wang, Timothy R. Langlois, Doug L. James. *SIGGRAPH Asia 2024 Conference Papers*. 

## Build Instructions

**Dependencies:** C++17, CUDA, Eigen 3.4, libigl

Building is handled by CMake. For example, to build from source on Linux:

    git clone https://github.com/kangruix/WaveBlender
    cd WaveBlender && git submodule update --init --recursive
    mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release ..
    make -j4

We provide an example scene in Scenes/CupPhone/. To run the code:

    ./WaveBlender ../Scenes/CupPhone/config.json
    python ../scripts/write_wav.py CupPhone_out.bin 88200

Afterwards, the simulated audio will be written to 'CupPhone_out.wav'. More scenes are available [here](https://graphics.stanford.edu/papers/waveblender/dataset/).

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