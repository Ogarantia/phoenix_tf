cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0048 NEW)

project(UpStride)

option(WITH_ONEDNN "Enable oneDNN backend (CPU)" ON)
option(WITH_CUDNN "Enable cuDNN backend (GPU)" OFF)
option(ARCH "Enable optimization for specific arch (x86_64 / aarch64 / <spe_march>)" OFF)

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
add_library(libupstride SHARED ${SOURCES})

SET_TARGET_PROPERTIES(libupstride PROPERTIES
  PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/libs/")

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

##########################################
# Compilation (optimization) flags setup #
##########################################
# Setup common optimization flags
set(OPTIMIZATION_FLAGS "-O3 -ffast-math -funroll-loops -fipa-pta -flto -ftree-vectorize")

# Setup optimization flags according to the compiler
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(OPTIMIZATION_FLAGS "${OPTIMIZATION_FLAGS} -floop-nest-optimize")
endif()

# Setup optimization flags according to processor vendor
if (ARCH STREQUAL "x86_64")
  set(OPTIMIZATION_FLAGS "${OPTIMIZATION_FLAGS} -march=skylake -mfma -mavx2")
elseif (ARCH STREQUAL "aarch64")
  set(OPTIMIZATION_FLAGS "${OPTIMIZATION_FLAGS} -march=armv8-a+crypto+simd -mcpu=cortex-a57+crypto+simd")
else()
  set(OPTIMIZATION_FLAGS "${OPTIMIZATION_FLAGS} -march=${ARCH}")
endif()

# Enable debug and opitmization flags
set(COMMON_FLAGS "-fPIC ${OPTIMIZATION_FLAGS} -Wfatal-errors")
# C++11 is required by Tensorflow & PyTorch
set(CMAKE_CXX_FLAGS "-std=c++11 ${TENSORFLOW_COMPILE_FLAGS} ${TENSORFLOW_LINK_FLAGS} ${COMMON_FLAGS} ${CMAKE_CXX_FLAGS}")

# MacOS-specific directive
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(CMAKE_CXX_FLAGS "-undefined dynamic_lookup ${CMAKE_CXX_FLAGS}")
endif()

# Enable TF 1.x support by a preprocessor directive
if (TENSORFLOW_VERSION STREQUAL "1")
  set(CMAKE_CXX_FLAGS "-DTENSORFLOW_VERSION_1 ${CMAKE_CXX_FLAGS}")
endif()