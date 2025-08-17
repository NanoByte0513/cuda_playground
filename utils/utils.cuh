#pragma once

#include "cuda_fp16.h"
#include <cstdint>
#include <sys/mman.h>
#include <sys/stat.h>  // struct stat, fstat()
#include <fcntl.h>   // open, O_RDONLY
#include <unistd.h>  // close, read

#define CHECK_CUDA_ERROR(expression) do { \
    cudaError_t err = expression; \
    if(err != cudaSuccess) { \
        printf("cuda error occured at line %d, file %s, error name: %s, error string: %s\n", \
        __LINE__, __FILE__, cudaGetErrorName(err), cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define ASSERT(expression) do { \
    if(!(expression)) { \
        printf("Assertion error at line %d, file %s\n", __LINE__, __FILE__); \
        exit(1); \
    } \
} while(0)

#define COORDINATE(x, y, M, N) (N * y + x)

#define HALF_NEG_INF __short_as_half(0xFC00)

#define DEV_CEIL(a, b) ((a + b - 1) / b)
    

namespace utils {
inline void printGPUMsg() {
    int deviceCount;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return;
    }

    cudaDeviceProp prop;
    for (int dev = 0; dev < deviceCount; dev++) {
        cudaGetDeviceProperties(&prop, dev);

        printf("\nDevice %d/%d \n", dev + 1, deviceCount);
        printf("Device %d: %s\n", dev, prop.name);
        printf("  Compute Capability      : %d.%d\n", prop.major, prop.minor);
        printf("  Global Memory           : %.2f GB\n", 
               (float)prop.totalGlobalMem / (1024 * 1024 * 1024));
        printf("  Shared memory per block : %.2f KB\n", 
               (float)prop.sharedMemPerBlock / 1024);  // 字节转KB
        printf("  Regs per block          : %d\n", prop.regsPerBlock);
        printf("  Multiprocessors         : %d\n", prop.multiProcessorCount);
        printf("  Max Threads per Block   : %d\n", prop.maxThreadsPerBlock);
        printf("  Max Block Dimensions    : (%d, %d, %d)\n", 
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Clock Rate              : %.2f GHz\n", prop.clockRate * 1e-6);
    }
}

void printMatrix_fp16(const uint16_t* ptr, int row, int col) {
    bool skip_row = false;
    for(int i = 0; i < row; ++i) {
        if(i > 2 && i < row - 3) {
            if(!skip_row) {
                printf("...,\n");
                skip_row = true;
            } 
            continue;
        }
        bool skip_col = false;
        for(int j = 0; j < col; ++j) {
            if(j > 2 && j < col - 3) {
                if(!skip_col) {
                    skip_col = true;
                    printf(" ..., ");
                }
                continue;
            }
                
            __half_raw raw_half;
            raw_half.x = ptr[i * col + j];
            half fp16_val = raw_half;
            printf("%.8f, ", __half2float(fp16_val));
        }
        printf("\n");
    }
    printf("\n");
}

void* openBin(const char* file_path, int& fd, size_t& file_size) {
    fd = open(file_path, O_RDONLY);
    if (fd == -1) {
        perror("open");
        exit(EXIT_FAILURE);
    }
    
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        perror("fstat");
        close(fd);
        exit(EXIT_FAILURE);
    }
    file_size = sb.st_size;  // 文件大小
    void* ptr = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (ptr == MAP_FAILED) {   // 检查是否失败
        perror("mmap");
        close(fd);
        exit(EXIT_FAILURE);
    }
    return ptr;
    
}

// void genRandomMatrix_fp16(int m, int n, uint16_t* ptr) {
//     curandGenerator_t generator;
//     curandCreateGeneratorHost(&generator, CURAND_RNG_PSEUDO_DEFAULT);
//     curandSetPseudoRandomGeneratorSeed(generator, 1234ULL); // 设置随机种子
// }
} // namespace utils