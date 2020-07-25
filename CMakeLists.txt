cmake_minimum_required(VERSION 2.8)

project(UpStride)

option(BUILD_TESTS "Build tests" ON)
option(ONEDNN_INCLUDE_DIRS "With OneDNN includes" "")
option(ONEDNN_LIB_DIRS "With OneDNN libs" "" )
option(WITH_CUDA "With CUDA" OFF)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(MACOSX TRUE)
endif()

# get tensorflow dirs and flags, see https://www.tensorflow.org/how_tos/adding_an_op/
execute_process(COMMAND python3 -c "import tensorflow; print(tensorflow.__version__[0], end='')" OUTPUT_VARIABLE TENSORFLOW_VERSION)
execute_process(COMMAND python3 -c "import tensorflow; print(tensorflow.sysconfig.get_include(), end='')" OUTPUT_VARIABLE TENSORFLOW_INCLUDE_DIRS)
execute_process(COMMAND python3 -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()), end='')" OUTPUT_VARIABLE TENSORFLOW_COMPILE_FLAGS)
execute_process(COMMAND python3 -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()), end='')" OUTPUT_VARIABLE TENSORFLOW_LINK_FLAGS)

set(ONEDNN_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/modules/dnnl_lnx_1.5.0_cpu_gomp/include")
set(ONEDNN_LIB_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/modules/dnnl_lnx_1.5.0_cpu_gomp/lib")

message("-- BUILD_TESTS: ${BUILD_TESTS}")
message("-- WITH_CUDA: ${WITH_CUDA}")
message("-- TENSORFLOW_VERSION: ${TENSORFLOW_VERSION}")
message("-- TENSORFLOW_INCLUDE_DIRS: ${TENSORFLOW_INCLUDE_DIRS}")
message("-- TENSORFLOW_COMPILE_FLAGS: ${TENSORFLOW_COMPILE_FLAGS}")
message("-- TENSORFLOW_LINK_FLAGS: ${TENSORFLOW_LINK_FLAGS}")
message("-- ONEDNN_INCLUDE_DIRS: ${ONEDNN_INCLUDE_DIRS}")
message("-- ONEDNN_LIB_DIRS: ${ONEDNN_LIB_DIRS}")


# setup compilation flags; C++11 required for tensorflow
set(COMMON_FLAGS "-fPIC -O2 -Wfatal-errors")
set(CMAKE_CXX_FLAGS "-std=c++11 ${TENSORFLOW_COMPILE_FLAGS} ${TENSORFLOW_LINK_FLAGS} ${COMMON_FLAGS} ${CMAKE_CXX_FLAGS}")

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
include_directories(${ONEDNN_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/cc)

link_libraries(${ONEDNN_LIB_DIRS}/libdnnl.so)

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
    ${CMAKE_CURRENT_SOURCE_DIR}/${CPP_SOURCES_FOLDER}/kernels.${EXTENSION}
    ${CMAKE_CURRENT_SOURCE_DIR}/${CPP_SOURCES_FOLDER}/ops.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/${CPP_SOURCES_FOLDER}/utils.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/${CPP_SOURCES_FOLDER}/onednn/conv2d.cc
)

# add the library
if(WITH_CUDA)
  cuda_add_library(_upstride SHARED ${SOURCES})
else()
  add_library(_upstride SHARED ${SOURCES})
endif()

SET_TARGET_PROPERTIES(_upstride PROPERTIES
  PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/python/upstride/type2/tf/keras")


if (BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests/cpp_tests/)
endif()
