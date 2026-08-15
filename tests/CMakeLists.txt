add_executable(vulkancraft_tests WorldTests.cpp)
target_link_libraries(vulkancraft_tests PRIVATE vulkancraft_core)
target_precompile_headers(vulkancraft_tests PRIVATE ../include/vulkancraft/pch.hpp)
vulkancraft_set_warnings(vulkancraft_tests)

add_test(NAME vulkancraft_tests COMMAND vulkancraft_tests)
