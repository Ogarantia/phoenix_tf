cmake_minimum_required(VERSION 2.8)

project(UpStride)

option(WITH_CUDA "With CUDA" OFF)
option(WITH_ONEDNN "With oneDNN" ON)

# include core
include(${CMAKE_CURRENT_SOURCE_DIR}/core/CMakeLists.txt)

# list source code files
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/cc)
set(CPP_SOURCES_FOLDER "src/cc")
set(SOURCES ${SOURCES} 
    ${CPP_SOURCES_FOLDER}/kernels.${EXTENSION}
    ${CPP_SOURCES_FOLDER}/ops.cc
)


# add the library
if(WITH_CUDA)
  cuda_add_library(_upstride SHARED ${SOURCES})
else()
  add_library(_upstride SHARED ${SOURCES})
endif()

SET_TARGET_PROPERTIES(_upstride PROPERTIES
  PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/libs/")

########################
#   TENSORFLOW SETUP   #
########################

# get tensorflow dirs and flags, see https://www.tensorflow.org/how_tos/adding_an_op/
execute_process(COMMAND python3 -c "import tensorflow; print(tensorflow.__version__[0], end='')" OUTPUT_VARIABLE TENSORFLOW_VERSION)
execute_process(COMMAND python3 -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()), end='')" OUTPUT_VARIABLE TENSORFLOW_COMPILE_FLAGS)
execute_process(COMMAND python3 -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()), end='')" OUTPUT_VARIABLE TENSORFLOW_LINK_FLAGS)

message("-- TENSORFLOW_VERSION: ${TENSORFLOW_VERSION}")
message("-- TENSORFLOW_COMPILE_FLAGS: ${TENSORFLOW_COMPILE_FLAGS}")
message("-- TENSORFLOW_LINK_FLAGS: ${TENSORFLOW_LINK_FLAGS}")

# setup compilation flags; C++11 required for tensorflow
set(COMMON_FLAGS "-g -fPIC -O3 -march=native -mfma -mavx2 -ffast-math -funroll-loops -floop-nest-optimize -fipa-pta -flto -ftree-vectorize  -Wfatal-errors")
set(CMAKE_CXX_FLAGS "-std=c++11 ${TENSORFLOW_COMPILE_FLAGS} ${TENSORFLOW_LINK_FLAGS} ${COMMON_FLAGS} ${CMAKE_CXX_FLAGS}")

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(CMAKE_CXX_FLAGS "-undefined dynamic_lookup ${CMAKE_CXX_FLAGS}")
endif()

# enable TF 1.x support by a preprocessor directive
if (TENSORFLOW_VERSION STREQUAL "1")
  message("-- Using Tensorflow 1.x")
  set(CMAKE_CXX_FLAGS "-DTENSORFLOW_VERSION_1 ${CMAKE_CXX_FLAGS}")
endif()
