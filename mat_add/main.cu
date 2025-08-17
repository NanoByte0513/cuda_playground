#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include "cuda_fp16.h"
#include "utils/utils.cuh"
#include <sys/mman.h>
#include <sys/stat.h>  // struct stat, fstat()
#include <cstdint>
#include <math.h>

#define THREADS_PER_BLOCK (128)

#define SEQ_LEN (64)
#define HIDDEN_SIZE (1024)

using float16 = uint16_t;

constexpr char* tensor_path_1 = "/home/wuyou/cuda_playground/tensor1.bin";
constexpr char* tensor_path_2 = "/home/wuyou/cuda_playground/tensor2.bin";

__global__
void mat_add(__half* mat1, __half* mat2, __half* result, int n) {
    int32_t global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    if(global_thread_id < n / 2) {
        half2 h2_1 = __halves2half2(mat1[global_thread_id], mat1[global_thread_id + n / 2]);
        half2 h2_2 = __halves2half2(mat2[global_thread_id], mat2[global_thread_id + n / 2]);
        half2 h2_rslt = __hadd2(h2_1, h2_2);
        result[global_thread_id] = __low2half(h2_rslt);
        result[global_thread_id + n / 2] = __high2half(h2_rslt);
    }
}

int main() {
    // Host malloc
    float16* host_mat1;
    float16* host_mat2;
    float16* host_result;
    // host_mat1
    int file_descriptor_1 = open(tensor_path_1, O_RDONLY);
    size_t file_size1;
    if (file_descriptor_1 == -1) {
        perror("open");
        exit(EXIT_FAILURE);
    }
    {
        struct stat sb;
        if (fstat(file_descriptor_1, &sb) == -1) {
            perror("fstat");
            close(file_descriptor_1);
            exit(EXIT_FAILURE);
        }
        file_size1 = sb.st_size;  // 文件大小
        if(file_size1 != SEQ_LEN * HIDDEN_SIZE * sizeof(float16)) {
            perror("Read file 2 error\n");
            exit(1);
        }
        host_mat1 = static_cast<float16*>(mmap(nullptr, file_size1, PROT_READ, MAP_PRIVATE, file_descriptor_1, 0));
        if (host_mat1 == MAP_FAILED) {   // 检查是否失败
            perror("mmap");
            close(file_descriptor_1);
            exit(EXIT_FAILURE);
        }
    }

    // host_mat2
    int file_descriptor_2 = open(tensor_path_2, O_RDONLY);
    size_t file_size2;
    if (file_descriptor_2 == -1) {
        perror("open");
        exit(EXIT_FAILURE);
    }
    {
        struct stat sb;
        if (fstat(file_descriptor_2, &sb) == -1) {
            perror("fstat");
            close(file_descriptor_2);
            exit(EXIT_FAILURE);
        }
        file_size2 = sb.st_size;  // 文件大小
        if(file_size2 != SEQ_LEN * HIDDEN_SIZE * sizeof(float16)) {
            perror("Read file 2 error\n");
            exit(1);
        }
        host_mat2 = static_cast<float16*>(mmap(nullptr, file_size2, PROT_READ, MAP_PRIVATE, file_descriptor_2, 0));
        if (host_mat2 == MAP_FAILED) {   // 检查是否失败
            perror("mmap");
            close(file_descriptor_1);
            exit(EXIT_FAILURE);
        }
    }

    host_result = (float16*)malloc(SEQ_LEN * HIDDEN_SIZE * sizeof(float16));
    
    // Malloc device memory
    __half* dev_mat1;
    __half* dev_mat2;
    __half* dev_result;
    CHECK_CUDA_ERROR(cudaMalloc(&dev_mat1, SEQ_LEN * HIDDEN_SIZE * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(&dev_mat2, SEQ_LEN * HIDDEN_SIZE * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(&dev_result, SEQ_LEN * HIDDEN_SIZE * sizeof(__half)));
    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(dev_mat1, host_mat1, SEQ_LEN * HIDDEN_SIZE * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(dev_mat2, host_mat2, SEQ_LEN * HIDDEN_SIZE * sizeof(__half), cudaMemcpyHostToDevice));
    
    // Do computations
    int num_blocks = (SEQ_LEN * HIDDEN_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    mat_add<<<num_blocks, THREADS_PER_BLOCK>>>(dev_mat1, dev_mat2, dev_result, SEQ_LEN * HIDDEN_SIZE);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Copy result data from dev to host
    CHECK_CUDA_ERROR(cudaMemcpy(host_result, dev_result, SEQ_LEN * HIDDEN_SIZE * sizeof(__half), cudaMemcpyDeviceToHost));
    
    // Check result
    bool checkNoError = true;
    for(int i = 0; i < SEQ_LEN * HIDDEN_SIZE; ++i) {
        __half_raw raw_half_1;
        raw_half_1.x = host_mat1[i];
        float fp1 = __half2float(raw_half_1);

        __half_raw raw_half_2;
        raw_half_2.x = host_mat2[i];
        float fp2 = __half2float(raw_half_2);

        __half_raw raw_half_result;
        raw_half_result.x = host_result[i];
        float diff = fp1 + fp2 - __half2float(raw_half_result);
        if(fabs(diff) > 1e-2) {
            checkNoError = false;
            printf("result[%d]: %.4f\n", i, diff);
        }
    }
    if(checkNoError)
        printf("No error! \n");

    free(host_result);
    munmap(host_mat1, file_size1);
    close(file_descriptor_1);
    munmap(host_mat2, file_size2);
    close(file_descriptor_2);

    CHECK_CUDA_ERROR(cudaFree(dev_mat1));
    CHECK_CUDA_ERROR(cudaFree(dev_mat2));
    CHECK_CUDA_ERROR(cudaFree(dev_result));
    return 0;
}