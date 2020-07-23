include_directories(doctest)

add_executable(tests tests.cpp)
add_executable(tests_tensors tests_tensors.cpp)
add_test(tests tests)
add_test(tests tests_tensors)
