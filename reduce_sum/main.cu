#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include "cuda_fp16.h"
#include "utils/utils.cuh"
#include <cstdint>
#include <math.h>

#define THREADS_PER_BLOCK (256)
#define WARP_SIZE (32)

#define SEQ_LEN (64)
#define HIDDEN_SIZE (1024)

using float16 = uint16_t;

constexpr char* tensor_path_1 = "/home/wuyou/cuda_playground/tensor1.bin";
constexpr char* tensor_path_2 = "/home/wuyou/cuda_playground/tensor2.bin";

__global__
void reduceSum(__half* mat, float* result, int n) {
    int32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    extern __shared__ float warpLevSum[]; // size==warps_per_block
    
    // Threads before this one has finished (tid*2) data
    half2 h2_1 = __halves2half2(mat[tid * 2], mat[tid * 2 + 1]);
    half2 h2_2 = __halves2half2(mat[tid * 2 + n / 2], mat[tid * 2 + n / 2 + 1]);
    half2 h2_rslt = __hadd2(h2_1, h2_2);
    float sum = __half2float(h2_rslt.x) + __half2float(h2_rslt.y);

    // Warp level reduce(shuffle)
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);
    // thread0 on each warp write its data to warpLevSum
    int laneIdx = threadIdx.x % warpSize;
    int warpIdx = threadIdx.x / warpSize;
    if(laneIdx == 0)
        warpLevSum[warpIdx] = sum;
    // Wait all threads0(Among different warps) finish its tasks
    __syncthreads();

    // warp0 read all data from warpLevSum
    int warps_per_block = (blockDim.x + warpSize - 1) / warpSize;
    if(warpIdx == 0) {
        sum = laneIdx < warps_per_block? warpLevSum[laneIdx] : 0.0f;
        sum += __shfl_down_sync(0xffffffff, sum, 16);
        sum += __shfl_down_sync(0xffffffff, sum, 8);
        sum += __shfl_down_sync(0xffffffff, sum, 4);
        sum += __shfl_down_sync(0xffffffff, sum, 2);
        sum += __shfl_down_sync(0xffffffff, sum, 1);
    }

    // Now thread0 on each block has its final sum
    if(threadIdx.x == 0) {
        atomicAdd(result, sum);
    }
}

int main() {
    utils::printGPUMsg();
    int num_blocks = (SEQ_LEN * HIDDEN_SIZE / 4 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    // Host malloc
    float16* host_mat;
    float* host_result;
    // host_mat
    int file_descriptor_1;
    size_t file_size1;
    host_mat = static_cast<float16*>(utils::openBin(tensor_path_1, file_descriptor_1, file_size1));

    // host result
    host_result = (float*)malloc(sizeof(float));
    memset(host_result, 0, sizeof(float));

    // Malloc device memory
    __half* dev_mat;
    float* dev_result;
    // dev_mat
    CHECK_CUDA_ERROR(cudaMalloc(&dev_mat, SEQ_LEN * HIDDEN_SIZE * sizeof(__half)));
    // dev_result
    CHECK_CUDA_ERROR(cudaMalloc(&dev_result, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(dev_result, 0, sizeof(float)));
    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(dev_mat, host_mat, SEQ_LEN * HIDDEN_SIZE * sizeof(__half), cudaMemcpyHostToDevice));
    
    // Do computations
    reduceSum<<<num_blocks, THREADS_PER_BLOCK, THREADS_PER_BLOCK / WARP_SIZE * sizeof(float)>>>(dev_mat, dev_result, SEQ_LEN * HIDDEN_SIZE);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Copy result data from dev to host
    CHECK_CUDA_ERROR(cudaMemcpy(host_result, dev_result, sizeof(float), cudaMemcpyDeviceToHost));
    
    // Check result
    float sum = 0.0f;
    for(int i = 0; i < SEQ_LEN * HIDDEN_SIZE; ++i) {
        __half_raw raw_half_1;
        raw_half_1.x = host_mat[i];
        sum += __half2float(raw_half_1);
    }
    float diff = sum - *host_result;
    if(fabs(diff) > 1e-2) {
        printf("diff == %.4f\n", diff);
    } else {
        printf("host_result = %.4f, device_result = %.4f\n", sum, *host_result);
    }

    munmap(host_mat, file_size1);
    close(file_descriptor_1);
    free(host_result);

    CHECK_CUDA_ERROR(cudaFree(dev_mat));
    CHECK_CUDA_ERROR(cudaFree(dev_result));
    return 0;
}