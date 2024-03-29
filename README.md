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

More details [here](https://upstride.atlassian.net/wiki/spaces/phoenix/pages/405766200/Engine+v2.0+architecture+and+designing+principles) and in the core submodule readme.

The Engine is written in CUDA, C++11 and Python. CMake is used as the build system. To compile a TensorFlow-compatible version of the Engine we use a specific build of GCC. Its x64 binary is available in `core/toolchain` folder.

## Quick start

### Usage in a docker container
If you are willing to use the Engine without modifying its code, you can pull a nicely prebuilt docker image from our cloud registry without cloning the code and compiling on your own:
```bash
  docker run -it --rm --gpus all \
    -v $(pwd):/opt/upstride \
    upstride/upstride:$(cat VERSION)-px-tf2.3.0-gpu-x86_64
```
The engine version number is stored in *VERSION* file. You can specify a different version in the docker image reference or remove *-gpu* to get a minimal version without CUDA support (running only on CPU).

### Compilation in docker
''_Docker is life. Thanks to docker we can create a compilation environment with a simple command._'' (c) Sebastien Iooss

#### Compiling CPU-only version
The simplest option to compile the Engine from sources is using the CPU backend only. You only need git and docker installed.

- Clone this repository and its submodules

- Download and install Git Large File Store (https://git-lfs.github.com/), then checkout the toolchain for the appropriate architecture, e.g.:
```
cd core && git lfs fetch && git lfs checkout toolchain/gcc-8.4_8.4-1_x86_64.deb && cd ..
```

- If you have access to the docker the registry, you may pull the prebuilt development docker. (TODO: add instructions.) Otherwise build it (in the repository root):
```bash
make dev_docker
```

- Power up a docker container using the image just build, mounting the source code to `/opt/upstride` folder (for example):
```bash
docker run -it --rm \
    -v $(pwd):/opt/upstride \
    upstride/upstride:$(cat VERSION)-pxdev-tf2.3.0-x86_64
```

- Compile the engine in the docker container:
```bash
cd /opt/upstride
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

The following instructions apply to compile the Engine running on recent GPUs having CUDA compute capability above or equal to 5.3. To compile for an older GPU you need to add `FP16=OFF` option to `make` commands below.

- Clone this repository and its submodules

- Download and install Git Large File Store (https://git-lfs.github.com/), then checkout the toolchain for the appropriate architecture, e.g.:
```
cd core && git lfs fetch && git lfs checkout toolchain/gcc-8.4_8.4-1_x86_64.deb && cd ..
```

- If you have access to the docker the registry, you may pull the prebuilt development docker. (TODO: add instructions.) Otherwise build it (in the repository root):
```bash
make dev_docker GPU=ON
```

- Power up a docker container using the image just build, mounting the source code to `/opt/upstride` folder (for example):
```bash
docker run -it --rm --gpus all \
    -v $(pwd):/opt/upstride \
    upstride/upstride:$(cat VERSION)-pxdev-tf2.3.0-gpu-x86_64
```

- Compile the engine in the docker container:
```bash
cd /opt/upstride
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
Setting up the local compilation environment might be tedious. In what follows we describe steps to get things working on an x64 machine running an Ubuntu-based OS.

- To use the GPU, make sure your GPU is working (run `nvidia-smi`)
- Install TensorFlow
  - Generally, if you follow [the installation instructions](https://www.tensorflow.org/install/gpu#linux_setup) precisely without improvising in the middle of the way, you get things running easily.

- Install CMake 3.18 or a yet newer version. If such a version is not available off the shelf, add [Kitware apt repository](https://apt.kitware.com/) first and then do
```bash
sudo apt install cmake
```

- Clone this repository and its submodules

- Install gcc 8.4 package contained in the repository as a compiler alternative:
```bash
sudo dpkg -i core/toolchain/gcc-8.4_8.4-1_x86_64.deb
sudo update-alternatives --install /usr/bin/gcc gcc /usr/local/bin/gcc8.4  100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/local/bin/g++8.4  100
```

- Run `make GPU=ON` (or `make GPU=OFF`) in its root to compile the Engine

If you have CUDA 10.1, you may run into a cublas-related compilation issue. This is because CUDA 10.1 is shipped with libcublas 10.2 version which is located in a different folder, and even the newest CMake at the moment of writing is unable to find it. In this case you may need to point cublas location explicitly doing something like
```bash
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/usr/local/cuda-10.2/include
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda-10.2/lib64
```

- Run tests: `python3 test.py`
  If this fails due to a missing python module, you may need to point its location explicitly first:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src/python
```


## Examples
Train a small quaternion neural network on CIFAR100:
```bash
python3 tests/python_high_level_tests/train_model.py -d 2 -e 10
```

Train a small real-valued network on CIFAR100, force training on CPU:
```bash
CUDA_VISIBLE_DEVICES= python3 tests/python_high_level_tests/train_model.py -d 0 -e 10
```

Enable verbose mode when training a quaternion network with separable convolutions:
```bash
UPSTRIDE_VERBOSE=1 python3 tests/python_high_level_tests/train_model.py -m model_separable_conv -d 2 -e 10
```

### Tests

- Useful options to try with ‘python3 test.py’:

  - -v : prints pytest verbose information (like names of the tests and progress in percentage)
  - -s : captures stdout and prints it to the terminal
  - -k Conv2D : run only tests with 'Conv2D' in their name
  - -k “Type2 and Pointwise and minimal“ : run only tests with ‘Type2’, ‘Pointwise’ AND ‘minimal’ in their names. Regex expressions using ‘and’, ‘or’, ‘not’ and parentheses are supported.
  - -m slow : run only tests marked with ‘slow’
  - -m “not slow” : run only tests not marked with ‘slow’. Same regex expressions supported as with -k.
  - -m ““ : run all tests
  - -m none : run only tests marked with ‘none’ (likely none)
  - --durations=10 : show 10 slowest tests
  - --durations=10 --durations-min=1.0 : show 10 slowest tests which take at least 1 second
  - --tb=short : show short error messages (only call stack)