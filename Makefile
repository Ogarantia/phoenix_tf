SOURCE_PATH=src

.PHONY : default clean


# default target building shared objects
default:
	@make -s build/*.so

build/*.so : build $(shell find $(SOURCE_PATH) -type f)
	@cd build .. && make


# generates Makefile using CMake
build: CMakeLists.txt
	@mkdir -p build
	@cd build && cmake ..


# removes the build folder
distclean:
	@rm -rf build
