UPSTRIDE_DEBUG ?= ON	# development build
GPU ?= OFF				# enables GPU backend

# specifying TensorFlow version (availability depends on the platform)
ifeq ($(shell arch),aarch64)
TF_VERSION=2.2.0
else
TF_VERSION=2.3.0
endif

# setting up docker images references
ifeq ($(GPU),ON)
DEVELOPMENT_DOCKER_REF=upstride:`cat VERSION`-pxdev-tf$(TF_VERSION)-gpu-`arch`
PRODUCTION_DOCKER_REF =upstride:`cat VERSION`-px-tf$(TF_VERSION)-gpu-`arch`
DOCKERFILE_SUFFIX=gpu-`arch`
else
DEVELOPMENT_DOCKER_REF=upstride:`cat VERSION`-pxdev-$(TF_VERSION)-`arch`
PRODUCTION_DOCKER_REF =upstride:`cat VERSION`-px-tf$(TF_VERSION)-`arch`
DOCKERFILE_SUFFIX=`arch`
endif

# paths variables
SOURCE_PATH=src
CORE_SOURCE_PATH=core/src

.PHONY : engine distclean clean install dev_docker docker build_nsight run_nsight

# default target building the shared objects and placing them nicely to run tests
engine:
	@make -s build/*.so
	@make install

build/*.so : build/Makefile $(shell find $(CORE_SOURCE_PATH) -type f) $(shell find $(SOURCE_PATH) -type f)
	@cd build && make VERBOSE=$(if $(VERBOSE),1,0) -j`nproc`

# generates Makefile using CMake
build/Makefile: FP16 ?= ON
build/Makefile: CMakeLists.txt
	@mkdir -p build
	@cd build && cmake -DWITH_CUDNN=$(GPU) -DUPSTRIDE_DEBUG=$(UPSTRIDE_DEBUG) -DWITH_FP16=$(FP16) ..

# removes the build folder
distclean:
	@rm -rf build

# Clean the build folder
clean:
	@touch CMakeLists.txt
	@rm -f build/libs/_upstride.so build/tests

# copy shared objects side-by-side with Python code
install:
	@cp build/libs/_upstride.so src/python/upstride/type_generic
	@cp build/core/thirdparty/onednn/src/libdnnl.so.1 src/python/upstride/type_generic

# build docker with the compilation environment
dev_docker:
	@docker build --build-arg TF_VERSION=$(TF_VERSION) \
				  -t $(DEVELOPMENT_DOCKER_REF) \
				  -f dockerfiles/dev-$(DOCKERFILE_SUFFIX).dockerfile .

# uses dev docker to build production docker image having the Engine as a Python module
docker: dev_docker
	@docker run --rm --gpus all -v `pwd`:/opt/upstride -w /opt/upstride $(DEVELOPMENT_DOCKER_REF) \
				make clean engine GPU=$(GPU) UPSTRIDE_DEBUG=OFF
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
