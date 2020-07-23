cmake_minimum_required(VERSION 2.8)

project(UpStride)


option(BUILD_TESTS "Build tests" ON)
option(WITH_CUDA "With CUDA" OFF)

message("-- BUILD_TESTS: ${BUILD_TESTS}")
message("-- WITH_CUDA: ${WITH_CUDA}")

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(MACOSX TRUE)
endif()

# get tensorflow include dirs, see https://www.tensorflow.org/how_tos/adding_an_op/
execute_process(COMMAND python3 -c "import tensorflow; print(tensorflow.sysconfig.get_include(), end='')" OUTPUT_VARIABLE Tensorflow_INCLUDE_DIRS)
execute_process(COMMAND python3 -c "import tensorflow as tf; print(tf.sysconfig.get_lib()+'/', end='')" OUTPUT_VARIABLE Tensorflow_LIBS_DIRS)
execute_process(COMMAND python3 -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()), end='')" OUTPUT_VARIABLE Tensorflow_LIBS)

# C++11 required for tensorflow
set(CMAKE_CXX_FLAGS "-std=c++11 -shared ${Tensorflow_LIBS} -I${Tensorflow_INCLUDE_DIRS} -fPIC -O3 ${CMAKE_CXX_FLAGS}")

if (${MACOSX} MATCHES "Darwin")
  set(CMAKE_CXX_FLAGS "-undefined dynamic_lookup ${CMAKE_CXX_FLAGS}")
endif()

# if GCC > 5
if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0 OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 5.0)
  set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0 ${CMAKE_CXX_FLAGS}")
endif()

# build the actual operation which can be used directory
include_directories(${Tensorflow_INCLUDE_DIRS})
link_directories(${Tensorflow_LIBS_DIRS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/cc)

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
)

if(WITH_CUDA)
  cuda_add_library(_upstride SHARED ${SOURCES})
else()
  add_library(_upstride SHARED ${SOURCES})
endif()

SET_TARGET_PROPERTIES(_upstride PROPERTIES
  PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY "../src/python/upstride/type2/tf/keras")


if (BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests/cpp_tests/)
endif()
