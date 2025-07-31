cmake_minimum_required(VERSION 3.10)
project(test_cuda LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${ROOT_DIR}/bin)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS} ${ROOT_DIR})

# Add exe
add_executable(test_cuda ${ROOT_DIR}/mat_add/main.cu)

target_link_libraries(test_cuda ${CUDA_LIBRARIES})

set_target_properties(test_cuda PROPERTIES
    CUDA_ARCHITECTURES "75;80;89"  # 示例：支持 Turing/Ampere/Lovelace
)
