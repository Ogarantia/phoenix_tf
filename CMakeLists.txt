cmake_minimum_required(VERSION 2.8)

project(UpStride)

option(WITH_CUDA "With CUDA" OFF)
message("-- WITH_CUDA: ${WITH_CUDA}")

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(MACOSX TRUE)
endif()

# get tensorflow dirs and flags, see https://www.tensorflow.org/how_tos/adding_an_op/
execute_process(COMMAND python3 -c "import tensorflow; print(tensorflow.sysconfig.get_include(), end='')" OUTPUT_VARIABLE TENSORFLOW_INCLUDE_DIRS)
execute_process(COMMAND python3 -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()), end='')" OUTPUT_VARIABLE TENSORFLOW_COMPILE_FLAGS)
execute_process(COMMAND python3 -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()), end='')" OUTPUT_VARIABLE TENSORFLOW_LINK_FLAGS)

set(COMMON_FLAGS "-fPIC -O2")

# setup compilation flags; C++11 required for tensorflow
set(CMAKE_CXX_FLAGS "-std=c++11 ${TENSORFLOW_COMPILE_FLAGS} ${TENSORFLOW_LINK_FLAGS} -I${TENSORFLOW_INCLUDE_DIRS} ${COMMON_FLAGS} ${CMAKE_CXX_FLAGS}")
if (${MACOSX} MATCHES "Darwin")
  set(CMAKE_CXX_FLAGS "-undefined dynamic_lookup ${CMAKE_CXX_FLAGS}")
endif()

include_directories(${TENSORFLOW_INCLUDE_DIRS})

set(CPP_SOURCES_FOLDER "src/cc")

if (WITH_CUDA)
  set(CMAKE_CXX_FLAGS "-DNDEBUG ${CMAKE_CXX_FLAGS}")
  find_package(CUDA)
  cuda_add_library(_upstride SHARED
    ${CPP_SOURCES_FOLDER}/kernels/upstride_kernels.cu.cc
    ${CPP_SOURCES_FOLDER}/ops/upstride_ops.cc)
else()
  add_library(_upstride SHARED
    ${CPP_SOURCES_FOLDER}/kernels/upstride_kernels.cc
    ${CPP_SOURCES_FOLDER}/ops/upstride_ops.cc)
endif()

SET_TARGET_PROPERTIES(_upstride PROPERTIES
  PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY "../src/python/upstride_ops")
