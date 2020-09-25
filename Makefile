ALLOW_VERBOSE ?= ON		# enables verbose messages in the engine when UPSTRIDE_VERBOSE env variable is set to 1 in runtime
GPU ?= OFF				# enables GPU backend

SOURCE_PATH=src
CORE_SOURCE_PATH=core/src
.PHONY : engine pre-build distclean clean dev_docker dev_docker_gpu build_nsight run_nsight install docker

# default target building the shared objects and placing them nicely to run tests
engine:
	@make -s build/*.so
	@make install

build/*.so : build/Makefile $(shell find $(CORE_SOURCE_PATH) -type f) $(shell find $(SOURCE_PATH) -type f)
	@cd build && make VERBOSE=0 -j`nproc`

# generates Makefile using CMake
build/Makefile: FP16 ?= ON
build/Makefile: CMakeLists.txt
	@mkdir -p build
	@cd build && cmake -DWITH_CUDNN=$(GPU) -DALLOW_VERBOSE=$(ALLOW_VERBOSE) -DWITH_FP16=$(FP16) ..

# ensures the options passed by variables are taken into account by cmake
pre-build:
	@touch CMakeLists.txt

# removes the build folder
distclean:
	@rm -rf build

# Clean the build folder
clean:
	@rm build/libs/_upstride.so ; rm build/tests

# build docker with compilation environment
dev_docker:
	@docker build -t upstride/phoenix:1.0-dev -f dockerfiles/dev.dockerfile .

dev_docker_gpu:
	@docker build -t upstride/phoenix:1.0-dev-gpu -f dockerfiles/dev-gpu.dockerfile .

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

# copy shared objects side-by-side with Python code
install:
	@cp build/libs/_upstride.so src/python/upstride/type_generic
	@cp build/core/thirdparty/onednn/src/libdnnl.so.1 src/python/upstride/type_generic

# build production docker image having the Engine as a Python module
docker: ALLOW_VERBOSE=OFF
docker: GPU=ON
docker: pre-build build/Makefile
	@make -s build/*.so install
	@docker build -t eu.gcr.io/fluid-door-230710/upstride:$(shell cat VERSION)-tf2.3.0-gpu -f dockerfiles/production.dockerfile .
