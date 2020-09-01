SOURCE_PATH=src
CORE_SOURCE_PATH=core/src
.PHONY : default


# default target building shared objects
default:
	@make -s build/*.so
	@make copy_so

build/*.so : build $(shell find $(CORE_SOURCE_PATH) -type f) $(shell find $(SOURCE_PATH) -type f)
	@cd build && make VERBOSE=0 -j`nproc`

# generates Makefile using CMake
build: CMakeLists.txt
	@mkdir -p build
	@cd build && cmake -DWITH_CUDNN=$(WITH_CUDNN) ..

# removes the build folder
distclean:
	@rm -rf build

# Clean the build folder
clean:
	@rm build/libs/_upstride.so ; rm build/tests

# build docker with compilation environment
build_dev_docker:
	@docker build -t upstride/phoenix:1.0-dev -f dockerfiles/dev.dockerfile .

# build docker containing nvidia-nsight
build_nsight:
	@docker build -t upstride/nsight:1.0 -f dockerfiles/nsight.dockerfile .

# run nvidia-nsight. This need to be done on a computer with a gpu and linux should run on X11.
run_nsight:
	@xhost +
	@docker run -it --rm  -e DISPLAY=unix$DISPLAY  \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd):/opt/src \
    --gpus all \
    --privileged \
    upstride/nsight:1.0 \
    bash

copy_so:
	@cp build/libs/_upstride.so src/python/upstride/type_generic
