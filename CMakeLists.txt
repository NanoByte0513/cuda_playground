cmake_minimum_required(VERSION 3.10)
project(test_cuda LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(CUTLASS_INCLUDE_PATH /home/wuyou/cutlass/include)
set(CUTLASS_UTILS_INCLUDE_PATH /home/wuyou/cutlass/tools/util/include)
set(CUTLASS_NVCC_ARCHS 80)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${ROOT_DIR}/bin)

find_package(CUDAToolkit REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS} ${ROOT_DIR} ${CUTLASS_INCLUDE_PATH} ${CUTLASS_UTILS_INCLUDE_PATH})

# Add exe
add_executable(test_cuda ${ROOT_DIR}/try_cutlass/main.cu)

target_link_libraries(test_cuda ${CUDA_LIBRARIES} CUDA::cublas)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(test_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

set_target_properties(test_cuda PROPERTIES
    CUDA_ARCHITECTURES "75;80;89"  # 示例：支持 Turing/Ampere/Lovelace
)
