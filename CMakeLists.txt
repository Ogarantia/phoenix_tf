cmake_minimum_required(VERSION 2.8)

project(UpStride)

option(WITH_CUDA "With CUDA" OFF)
message("-- WITH_CUDA: ${WITH_CUDA}")

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(MACOSX TRUE)
endif()

# get tensorflow dirs and flags, see https://www.tensorflow.org/how_tos/adding_an_op/
execute_process(COMMAND python3 -c "import tensorflow; print(tensorflow.__version__[0], end='')" OUTPUT_VARIABLE TENSORFLOW_VERSION)
execute_process(COMMAND python3 -c "import tensorflow; print(tensorflow.sysconfig.get_include(), end='')" OUTPUT_VARIABLE TENSORFLOW_INCLUDE_DIRS)
execute_process(COMMAND python3 -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()), end='')" OUTPUT_VARIABLE TENSORFLOW_COMPILE_FLAGS)
execute_process(COMMAND python3 -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()), end='')" OUTPUT_VARIABLE TENSORFLOW_LINK_FLAGS)

# setup compilation flags; C++11 required for tensorflow
set(COMMON_FLAGS "-fPIC -O2 -Wfatal-errors")
set(CMAKE_CXX_FLAGS "-std=c++11 ${TENSORFLOW_COMPILE_FLAGS} ${TENSORFLOW_LINK_FLAGS} -I${TENSORFLOW_INCLUDE_DIRS} ${COMMON_FLAGS} ${CMAKE_CXX_FLAGS}")
if (${MACOSX} MATCHES "Darwin")
  set(CMAKE_CXX_FLAGS "-undefined dynamic_lookup ${CMAKE_CXX_FLAGS}")
endif()

# TF 1.x support by a preprocessor directive
if (TENSORFLOW_VERSION STREQUAL "1")
  message("-- Using Tensorflow 1.x")
  set(CMAKE_CXX_FLAGS "-DTENSORFLOW_VERSION_1 ${CMAKE_CXX_FLAGS}")
endif()

# specify paths to include directories
include_directories(${TENSORFLOW_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/cc)

# list source code files
set(CPP_SOURCES_FOLDER "src/cc")
if (WITH_CUDA)
  set(CMAKE_CXX_FLAGS "-DNDEBUG ${CMAKE_CXX_FLAGS}")
  find_package(CUDA)
  set(EXTENSION "cu.cc")
else()
  set(EXTENSION "cc")
endif()

set(SOURCES ${SOURCES} 
    ${CPP_SOURCES_FOLDER}/upstride_kernels.${EXTENSION}
    ${CPP_SOURCES_FOLDER}/upstride_ops.cc
    ${CPP_SOURCES_FOLDER}/utils.cc
)

# add the library
if(WITH_CUDA)
  cuda_add_library(_upstride SHARED ${SOURCES})
else()
  add_library(_upstride SHARED ${SOURCES})
endif()

SET_TARGET_PROPERTIES(_upstride PROPERTIES
  PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY "../src/python/upstride/type2/tf/keras")
