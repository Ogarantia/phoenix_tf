# Upstride Engine (TensorFlow)
Upstride Engine is a set of deep learning operations with geometric algebras. It is crafted with love and passion.

## Overview
The Engine implements a selected set of geometric algebras within standard neural networks operations following _Geometric Algebra over Matrices_ formalism.

The Engine has three abstraction levels with:

* Its *frontend level* (this repository) implements interaction with a high level framework, i.e., TensorFlow.
  * A keras-like Python API is available to make the use of the Engine easier.
* Its *core level* (the submodule) translates the maths into invocations of scalar (real-valued) operations.
* Its *backend level* (deeply in the submodule) provides interfaces to computational libraries implementing the needed scalar operations efficiently, namely
  * **Intel oneDNN** library optimized for Intel CPUs and also capable of `aarch64`,
  * optional **NVidia cuDNN** library used to run computations on NVidia GPUs.

More details [here](https://upstride.atlassian.net/wiki/spaces/phoenix/pages/405766200/Engine+v2.0+architecture+and+designing+principles).

The Engine is written in CUDA, C++11 and Python. CMake is used as the build system. To compile a TensorFlow-compatible version of the Engine we use a specific build of GCC. Its x64 binary is available in `core/toolchain` folder.

## Quick start

### Usage in a docker container
If you are willing to use the Engine without modifying its code, you can pull a nicely prebuilt docker image from our cloud registry without compiling on your own (make sure you have access to it as explained [here](https://upstride.atlassian.net/wiki/spaces/UPSTRIDE/pages/425525257/How+to+deploy+UpStride+training)).
```bash
  docker run -it --rm --gpus all \
    -v $(pwd):/opt/src \
    eu.gcr.io/fluid-door-230710/upstride:0.5-tf2.3.0-gpu
```

### Compilation in docker
''_Docker is life. Thanks to docker we can create a compilation environment with a simple command._'' (c) Sebastien Iooss

#### Compiling CPU-only version
The simplest option to compile the Engine from sources is using the CPU backend only. You only need git and docker installed.

- Clone this repository and its submodules
- Build the docker image (in the repository root):
```bash
make dev_docker
```

- Power up a docker container from the image just build, mounting the source code to `/opt/src` folder (for example):
```bash
docker run -it --rm \
    -v $(pwd):/opt/src \
    upstride/phoenix:1.0-dev
```

- Compile the engine in the docker container:
```bash
cd /opt/src
make
```

- Check if everything is okay and tests pass:
```bash
python3 test.py
```
  If this fails due to a missing python module, you may need to point its location explicitly first:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src/python
```

#### Compiling with GPU support enabled
If you are lucky to have an NVidia GPU(s), you can get the engine with the GPU support enabled. In addition to git and docker, you need to have NVidia drivers (check if `nvidia-smi` can find your GPU) and [NVidia container runtime](https://github.com/NVIDIA/nvidia-container-runtime#installation) installed.

- Clone this repository and its submodules
- Build the docker image (in the repository root):
```bash
make dev_docker_gpu
```

- Power up a docker container from the image just build, mounting the source code to `/opt/src` folder (for example):
```bash
docker run -it --rm --gpus all \
    -v $(pwd):/opt/src \
    upstride/phoenix:1.0-dev-gpu
```

- Compile the engine in the docker container:
```bash
cd /opt/src
make GPU=ON
```

- Check if everything is okay and tests pass:
```bash
python3 test.py
```

  If this fails due to a missing python module, you may need to point its location explicitly first:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src/python
```

### Compilation locally
Setting up the local compilation environment might be tedious.

- To use the GPU, make sure your GPU is up (run `nvidia-smi`)
- Install TensorFlow
  - Generally, if you follow [the installation instructions](https://www.tensorflow.org/install/gpu#linux_setup) precisely without improvising in the middle of the way, you get things running easily.
- Clone this repository and its submodules
- Install GCC 8.4 contained in the repository as a compiler alternative:
  
```bash
sudo dpkg -i core/toolchain/gcc-8.4_8.4-1_amd64.deb
sudo update-alternatives --install /usr/bin/gcc gcc /usr/local/bin/gcc8.4  100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/local/bin/g++8.4  100
```

- Run `make GPU=ON` (or `make GPU=OFF`) in its root to compile the Engine
- Run tests: `python3 test.py`
  If this fails due to a missing python module, you may need to point its location explicitly first:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src/python
```

## Examples
Train a small quaternion neural network on CIFAR10:
```bash
python3 tests/python_high_level_tests/train_simple_network_quaternions.py
```

Train a small real-valued network on CIFAR10, force training on CPU:
```bash
CUDA_VISIBLE_DEVICES= python3 tests/python_high_level_tests/train_simple_network_scalars.py
```

Enable verbose mode when training the quaternion network:
```bash
UPSTRIDE_VERBOSE=1 python3 tests/python_high_level_tests/train_simple_network_quaternions.py
```