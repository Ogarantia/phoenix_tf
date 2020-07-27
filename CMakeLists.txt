cmake_minimum_required(VERSION 2.8)

project(UpStride)

option(BUILD_TESTS "Build tests" ON)
option(WITH_CUDA "With CUDA" OFF)

option(WITH_ONEDNN "With oneDNN" ON)

message("-- BUILD_TESTS: ${BUILD_TESTS}")
message("-- WITH_ONEDNN: ${WITH_ONEDNN}")
message("-- WITH_CUDA: ${WITH_CUDA}")

# CUDA-related stuff
if (WITH_CUDA)
  set(CMAKE_CXX_FLAGS "-DNDEBUG ${CMAKE_CXX_FLAGS}")
  find_package(CUDA)
  set(EXTENSION "cu.cc")
else()
  set(EXTENSION "cc")
endif()

# list source code files
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/cc)
set(CPP_SOURCES_FOLDER "src/cc")
set(SOURCES ${SOURCES} 
    ${CPP_SOURCES_FOLDER}/kernels.${EXTENSION}
    ${CPP_SOURCES_FOLDER}/ops.cc
    ${CPP_SOURCES_FOLDER}/utils.cc
)
if (WITH_ONEDNN)
  set(SOURCES ${SOURCES}
      ${CPP_SOURCES_FOLDER}/onednn/conv2d.cc
  )
endif()

# add the library
if(WITH_CUDA)
  cuda_add_library(_upstride SHARED ${SOURCES})
else()
  add_library(_upstride SHARED ${SOURCES})
endif()

SET_TARGET_PROPERTIES(_upstride PROPERTIES
  PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/python/upstride/type2/tf/keras")

#######################
#    ONEDNN SETUP     #
#######################

# bring up oneDNN
if (WITH_ONEDNN)
  set(ONEDNN_SUBMODULE_PATH "thirdparty/onednn")
  option(DNNL_BUILD_TESTS "Build DNLL tests" OFF)
  option(DNNL_BUILD_EXAMPLES "Build DNLL examples" OFF)
  set( DDNNL_ARCH_OPT_FLAGS "-O3 -ffast-math -funroll-loops -floop-nest-optimize -ftree-vectorize" )
  add_subdirectory(${ONEDNN_SUBMODULE_PATH})
  get_directory_property(DIRECTORY ${ONEDNN_SUBMODULE_PATH} VARIABLES DDNNL_ARCH_OPT_FLAGS)
  #target_include_directories(_upstride PRIVATE "${ONEDNN_SUBMODULE_PATH}/include}")
  target_link_libraries(_upstride dnnl)
endif()


########################
#   TENSORFLOW SETUP   #
########################

# get tensorflow dirs and flags, see https://www.tensorflow.org/how_tos/adding_an_op/
execute_process(COMMAND python3 -c "import tensorflow; print(tensorflow.__version__[0], end='')" OUTPUT_VARIABLE TENSORFLOW_VERSION)
execute_process(COMMAND python3 -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()), end='')" OUTPUT_VARIABLE TENSORFLOW_COMPILE_FLAGS)
execute_process(COMMAND python3 -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()), end='')" OUTPUT_VARIABLE TENSORFLOW_LINK_FLAGS)

message("-- BUILD_TESTS: ${BUILD_TESTS}")
message("-- WITH_ONEDNN: ${WITH_ONEDNN}")
message("-- WITH_CUDA: ${WITH_CUDA}")
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

if (BUILD_TESTS)
  enable_testing()
  #add_subdirectory(tests/cpp_tests/)
  include_directories(tests/cpp_tests/doctest)
  link_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/cc)

  set(TESTSOURCES ${TESTSOURCES} 
    ${CPP_SOURCES_FOLDER}/utils.cc
    tests/cpp_tests/tests.cpp
  )
  if (WITH_ONEDNN)
    set(SOURCES ${SOURCES}
        ${CPP_SOURCES_FOLDER}/onednn/conv2d.cc
    )
  endif()

  add_executable(tests ${TESTSOURCES})
  add_test(tests tests)
  #target_link_libraries(tests _upstride)
endif()
