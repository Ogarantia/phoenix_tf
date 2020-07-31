cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0048 NEW)

project(UpStride)

option(WITH_ONEDNN "Enable oneDNN backend (CPU)" ON)
option(WITH_CUDNN "Enable cuDNN backend (GPU)" OFF)

# include core
include(${CMAKE_CURRENT_SOURCE_DIR}/core/CMakeLists.txt)

# list source code files
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/cc)
set(CPP_SOURCES_FOLDER "src/cc")
set(SOURCES ${SOURCES} 
    ${CPP_SOURCES_FOLDER}/kernels.cc
    ${CPP_SOURCES_FOLDER}/ops.cc
)

# add the library
add_library(_upstride SHARED ${SOURCES})

SET_TARGET_PROPERTIES(_upstride PROPERTIES
  PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/python/upstride/type2/tf/keras")

########################
#   TENSORFLOW SETUP   #
########################

message("-- Setting up TensorFlow")

# get tensorflow dirs and flags, see https://www.tensorflow.org/how_tos/adding_an_op/
execute_process(COMMAND python3 -c "import tensorflow; print(tensorflow.__version__[0], end='')"
    OUTPUT_VARIABLE TENSORFLOW_VERSION
    ERROR_QUIET)
execute_process(COMMAND python3 -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()), end='')"
    OUTPUT_VARIABLE TENSORFLOW_COMPILE_FLAGS
    ERROR_QUIET)
execute_process(COMMAND python3 -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()), end='')"
    OUTPUT_VARIABLE TENSORFLOW_LINK_FLAGS
    ERROR_QUIET)

message("     Version: ${TENSORFLOW_VERSION}")
message("     Compile flags: ${TENSORFLOW_COMPILE_FLAGS}")
message("     Link flags:    ${TENSORFLOW_LINK_FLAGS}")

# setup compilation flags; C++11 required for tensorflow
set(COMMON_FLAGS "-g -fPIC -O3 -march=native -mfma -mavx2 -ffast-math -funroll-loops -floop-nest-optimize -fipa-pta -flto -ftree-vectorize  -Wfatal-errors")
set(CMAKE_CXX_FLAGS "-std=c++11 ${TENSORFLOW_COMPILE_FLAGS} ${TENSORFLOW_LINK_FLAGS} ${COMMON_FLAGS} ${CMAKE_CXX_FLAGS}")

# macOS-specific directive
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(CMAKE_CXX_FLAGS "-undefined dynamic_lookup ${CMAKE_CXX_FLAGS}")
endif()

# enable TF 1.x support by a preprocessor directive
if (TENSORFLOW_VERSION STREQUAL "1")
  set(CMAKE_CXX_FLAGS "-DTENSORFLOW_VERSION_1 ${CMAKE_CXX_FLAGS}")
endif()

