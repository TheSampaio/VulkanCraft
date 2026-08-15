cmake_minimum_required(VERSION 3.25)

project(
    VulkanCraft
    VERSION 1.6.0
    DESCRIPTION "A compact voxel sandbox rendered with modern Vulkan"
    LANGUAGES CXX
)

if(NOT WIN32 OR NOT MSVC)
    message(FATAL_ERROR "VulkanCraft currently supports only Windows with Microsoft Visual C++")
endif()

option(VULKANCRAFT_BUILD_TESTS "Build VulkanCraft tests" ON)
option(VULKANCRAFT_ENABLE_VALIDATION "Enable Vulkan validation layers in debug builds" ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

include(FetchContent)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
    GIT_SHALLOW TRUE
)

set(GLM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLM_BUILD_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 1.0.3
    GIT_SHALLOW TRUE
)

FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG f0569113c93ad095470c54bf34a17b36646bbbb5
)

FetchContent_MakeAvailable(glfw glm stb)
find_package(Vulkan 1.3 REQUIRED COMPONENTS glslc)

include(cmake/CompilerWarnings.cmake)

add_library(vulkancraft_core STATIC
    src/core/Camera.cpp
    src/core/Parallel.cpp
    src/core/PlayerController.cpp
    src/rendering/CascadeShadow.cpp
    src/rendering/CelestialCycle.cpp
    src/rendering/AtlasMipmaps.cpp
    src/rendering/Frustum.cpp
    src/world/World.cpp
    src/world/WaterSimulation.cpp
    src/world/WorldGenerator.cpp
    src/world/WorldMesher.cpp
    src/world/StreamingWorld.cpp
)
target_include_directories(vulkancraft_core PUBLIC include)
target_link_libraries(vulkancraft_core PUBLIC glm::glm)
target_compile_definitions(vulkancraft_core PUBLIC
    GLM_FORCE_DEPTH_ZERO_TO_ONE
    GLM_FORCE_RADIANS
)
target_precompile_headers(vulkancraft_core PRIVATE include/vulkancraft/pch.hpp)
vulkancraft_set_warnings(vulkancraft_core)

set(VULKANCRAFT_SHADER_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY "${VULKANCRAFT_SHADER_OUTPUT_DIRECTORY}")

function(vulkancraft_compile_shader shader_file)
    get_filename_component(shader_name "${shader_file}" NAME)
    set(shader_output "${VULKANCRAFT_SHADER_OUTPUT_DIRECTORY}/${shader_name}.spv")
    add_custom_command(
        OUTPUT "${shader_output}"
        COMMAND Vulkan::glslc "${CMAKE_CURRENT_SOURCE_DIR}/${shader_file}" -o "${shader_output}"
        DEPENDS "${shader_file}"
        COMMENT "Compiling ${shader_file}"
        VERBATIM
    )
    set(VULKANCRAFT_SHADER_OUTPUTS ${VULKANCRAFT_SHADER_OUTPUTS} "${shader_output}" PARENT_SCOPE)
endfunction()

vulkancraft_compile_shader(shaders/world.vert)
vulkancraft_compile_shader(shaders/world.frag)
vulkancraft_compile_shader(shaders/shadow.vert)
vulkancraft_compile_shader(shaders/shadow.frag)
vulkancraft_compile_shader(shaders/depth_prepass.vert)
vulkancraft_compile_shader(shaders/depth_prepass.frag)
vulkancraft_compile_shader(shaders/sky.vert)
vulkancraft_compile_shader(shaders/sky.frag)
vulkancraft_compile_shader(shaders/celestial.vert)
vulkancraft_compile_shader(shaders/celestial.frag)
vulkancraft_compile_shader(shaders/postprocess.vert)
vulkancraft_compile_shader(shaders/postprocess.frag)
add_custom_target(vulkancraft_shaders ALL DEPENDS ${VULKANCRAFT_SHADER_OUTPUTS})

add_executable(vulkancraft
    src/main.cpp
    src/core/Application.cpp
    src/rendering/VulkanRenderer.cpp
)
add_dependencies(vulkancraft vulkancraft_shaders)
target_include_directories(vulkancraft PRIVATE include)
target_include_directories(vulkancraft SYSTEM PRIVATE "${stb_SOURCE_DIR}")
target_link_libraries(vulkancraft PRIVATE vulkancraft_core glfw Vulkan::Vulkan)
target_compile_definitions(vulkancraft PRIVATE
    VULKANCRAFT_SHADER_DIRECTORY="${VULKANCRAFT_SHADER_OUTPUT_DIRECTORY}"
    VULKANCRAFT_ASSET_DIRECTORY="${CMAKE_CURRENT_SOURCE_DIR}/assets"
    VULKANCRAFT_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    VULKANCRAFT_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    VULKANCRAFT_VERSION_PATCH=${PROJECT_VERSION_PATCH}
    $<$<BOOL:${VULKANCRAFT_ENABLE_VALIDATION}>:VULKANCRAFT_ENABLE_VALIDATION>
)
target_precompile_headers(vulkancraft PRIVATE include/vulkancraft/pch.hpp)
vulkancraft_set_warnings(vulkancraft)

add_custom_command(
    TARGET vulkancraft POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/assets"
            "$<TARGET_FILE_DIR:vulkancraft>/assets"
    COMMENT "Copying runtime assets"
)

if(VULKANCRAFT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
