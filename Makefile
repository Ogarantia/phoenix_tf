UPSTRIDE_DEBUG ?= ON	# development build
GPU ?= OFF				# enables GPU backend
PYTHON ?= python3
ARCH ?= `arch`           # possible values are local, x86_64 and aarch64

# specifying TensorFlow version (availability depends on the platform)
ifeq ($(shell arch),aarch64)
TF_VERSION?=2.2.0
else
TF_VERSION?=2.3.0
endif

# setting up default docker images references (used if not specified)
ifeq ($(GPU),ON)
DEVELOPMENT_DOCKER_REF?=upstride:`cat VERSION`-pxdev-tf$(TF_VERSION)-gpu-`arch`
PRODUCTION_DOCKER_REF ?=upstride:`cat VERSION`-px-tf$(TF_VERSION)-gpu-`arch`
DOCKERFILE_SUFFIX=gpu-`arch`
else
DEVELOPMENT_DOCKER_REF?=upstride:`cat VERSION`-pxdev-tf$(TF_VERSION)-`arch`
PRODUCTION_DOCKER_REF ?=upstride:`cat VERSION`-px-tf$(TF_VERSION)-`arch`
DOCKERFILE_SUFFIX=`arch`
endif

# paths variables
SOURCE_PATH=src
CORE_SOURCE_PATH=core/src

.PHONY : engine wheel distclean clean install install_wheel dev_docker docker build_nsight run_nsight

# default target building the shared objects and placing them nicely to run tests
engine:
	@make -s build/*.so
	@make install

# building engine and create a wheel file; it will only remain to pip install this wheel file to install it
wheel: engine install
	@cd src/python && $(PYTHON) cythonizer.py bdist_wheel --dist-dir ../../build clean --arch $(ARCH)

# Build libs
# VERBOSE=1 only prints lines during linking; SHELL='sh -x' allows to print all the expanded lines before executing them.
build/*.so : build/Makefile $(shell find $(CORE_SOURCE_PATH) -type f) $(shell find $(SOURCE_PATH) -type f)
	@cd build && make SHELL=$(if $(VERBOSE),'sh -x','') -j`nproc`

# generates Makefile using CMake
build/Makefile: FP16 ?= ON
build/Makefile: CMakeLists.txt
	@mkdir -p build
	@cd build && cmake -DWITH_CUDNN=$(GPU) -DUPSTRIDE_DEBUG=$(UPSTRIDE_DEBUG) -DWITH_FP16=$(FP16) -DARCH=$(ARCH) ..

# removes the build folder
distclean:
	@rm -rf build

# Clean the build folder
clean:
	@touch CMakeLists.txt
	@rm -f build/libs/libupstride.so build/tests

install_wheel: PYTHON_IMPORTLIB_PATH=`$(PYTHON) -c "import importlib; print(importlib.__path__[0])"`
install_wheel: wheel
	@$(PYTHON) -m pip install build/upstride*`arch`.whl
	@ln -s $(PWD)/build/cython/libupstride.so $(PYTHON_IMPORTLIB_PATH)/libupstride.so
	@ln -s $(PWD)/build/cython/libdnnl.so.1   $(PYTHON_IMPORTLIB_PATH)/libdnnl.so.1

# copy shared objects side-by-side with Python code
install:
	@cp build/libs/libupstride.so src/python/upstride/internal
	@cp build/core/thirdparty/onednn/src/libdnnl.so.1 src/python/upstride/internal

# build docker with the compilation environment
dev_docker:
	@docker build --build-arg TF_VERSION=$(TF_VERSION) \
				  -t $(DEVELOPMENT_DOCKER_REF) \
				  -f dockerfiles/dev-$(DOCKERFILE_SUFFIX).dockerfile .

# uses dev docker to build production docker image having the Engine as a Python module
docker: dev_docker
	@docker run --rm --gpus all -v `pwd`:/opt/upstride -w /opt/upstride $(DEVELOPMENT_DOCKER_REF) \
				make clean install_wheel GPU=$(GPU) UPSTRIDE_DEBUG=OFF
	@docker build --build-arg TF_VERSION=$(TF_VERSION) \
				  -t $(PRODUCTION_DOCKER_REF) \
				  -f dockerfiles/prod-$(DOCKERFILE_SUFFIX).dockerfile .

# build docker containing nvidia-nsight
build_nsight:
	@docker build -t upstride/nsight:1.0 -f dockerfiles/nsight.dockerfile .

# run nvidia-nsight. This need to be done on a computer with a gpu and linux should run on X11.
run_nsight:
	@xhost +
	@docker run -it --rm  -e DISPLAY=unix$(DISPLAY)  \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v `pwd`:/opt/src \
    --gpus all \
    --privileged \
    upstride/nsight:1.0 \
    bash
