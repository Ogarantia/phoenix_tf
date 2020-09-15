SOURCE_PATH=src
CORE_SOURCE_PATH=core/src
.PHONY : engine distclean clean dev_docker dev_docker_gpu build_nsight run_nsight copy_so docker


# default target building the shared objects and placing them nicely to run tests
engine:
	@make -s build/*.so
	@make copy_so

build/*.so : build/Makefile $(shell find $(CORE_SOURCE_PATH) -type f) $(shell find $(SOURCE_PATH) -type f)
	@cd build && make VERBOSE=0 -j`nproc`

# generates Makefile using CMake
build/Makefile: CMakeLists.txt
	@mkdir -p build
	@cd build && cmake -DWITH_CUDNN=$(if $(GPU),$(GPU),"OFF") ..

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
copy_so:
	@cp build/libs/_upstride.so src/python/upstride/type_generic
	@cp build/core/thirdparty/onednn/src/libdnnl.so.1 src/python/upstride/type_generic

# build "official" docker image having the Engine as a Python module
docker:
	@docker build -t eu.gcr.io/fluid-door-230710/upstride:$(shell cat VERSION)-tf2.3.0-gpu -f dockerfiles/dockerfile .
