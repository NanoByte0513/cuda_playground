/***
 * SGEMM version 1, read from global memory directly
 * Read 2*K*M*N, write M*N
 */

#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include "cuda_fp16.h"
#include "utils/utils.cuh"
#include <cstdint>
#include <math.h>


#define BLOCK_DIM (16)
#define THREADS_PER_BLOCK (BLOCK_DIM * BLOCK_DIM)
#define WARP_SIZE (32)

#define SEQ_LEN (64)
#define HIDDEN_SIZE (1024)

constexpr char* tensor_path_1 = "/home/wuyou/cuda_playground/sgemm/tensor1_fp32.bin"; // SEQ_LEN * HIDDEN_SIZE
constexpr char* tensor_path_2 = "/home/wuyou/cuda_playground/sgemm/tensor2_fp32.bin"; // HIDDEN_SIZE * SEQ_LEN

/**
 * tensorA(M*K), tensorB(K*N), result(M*N)
 */
__global__
void sgemm(const float* tensorA, const float* tensorB, float* result, int M, int K, int N) {
    auto blk_start_pos_A = blockIdx.y * blockDim.y * K;
    auto blk_start_pos_B = blockIdx.x * blockDim.x * K;

    auto thread_pos_A = blk_start_pos_A + threadIdx.y * K;
    auto thread_pos_B = blk_start_pos_B + threadIdx.x * K;

    float elem = 0.f;
    for(int i = 0; i < K; ++i) {
        elem += tensorA[thread_pos_A + i] * tensorB[thread_pos_B + i];
    }

    auto blk_start_pos_rslt = blockIdx.x * blockDim.y + blockIdx.y * N;
    auto thread_start_pos_rslt = blk_start_pos_rslt + threadIdx.x + threadIdx.y * blockDim.x;
    result[thread_start_pos_rslt] = elem;
}

int main() {
    // Host malloc
    float* host_tensorA;
    int fdA;
    size_t file_sizeA;
    host_tensorA = static_cast<float*>(utils::openBin(tensor_path_1, fdA, file_sizeA));
    ASSERT(file_sizeA == SEQ_LEN * HIDDEN_SIZE * sizeof(float));

    float* host_tensorB;
    int fdB;
    size_t file_sizeB;
    host_tensorB = static_cast<float*>(utils::openBin(tensor_path_2, fdB, file_sizeB));
    ASSERT(file_sizeB == SEQ_LEN * HIDDEN_SIZE * sizeof(float));

    float* host_result = (float*)malloc(SEQ_LEN * SEQ_LEN * sizeof(float));

    // Dev malloc
    float* dev_tensorA;
    CHECK_CUDA_ERROR(cudaMalloc(&dev_tensorA, SEQ_LEN * HIDDEN_SIZE * sizeof(float)));
    float* dev_tensorB;
    CHECK_CUDA_ERROR(cudaMalloc(&dev_tensorB, SEQ_LEN * HIDDEN_SIZE * sizeof(float)));
    float* dev_result;
    CHECK_CUDA_ERROR(cudaMalloc(&dev_result, SEQ_LEN * SEQ_LEN * sizeof(float)));
    // Copy data from host to dev
    CHECK_CUDA_ERROR(cudaMemcpy(dev_tensorA, host_tensorA, SEQ_LEN * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(dev_tensorB, host_tensorB, SEQ_LEN * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    dim3 grid_dim(SEQ_LEN / BLOCK_DIM, SEQ_LEN / BLOCK_DIM, 1);
    dim3 block_dim(BLOCK_DIM, BLOCK_DIM, 1);
    sgemm<<<grid_dim, block_dim>>>(dev_tensorA, dev_tensorB, dev_result, SEQ_LEN, HIDDEN_SIZE, SEQ_LEN);

    // Copy result from dev to host
    CHECK_CUDA_ERROR(cudaMemcpy(host_result, dev_result, SEQ_LEN * SEQ_LEN * sizeof(float), cudaMemcpyDeviceToHost));

    // Check result
    int num_correct = 0;
    int num_wrong = 0;
    for(int i = 0; i < SEQ_LEN; ++i) {
        for(int j = 0; j < SEQ_LEN; ++j) {
            float elem = 0.f;
            for(int k = 0; k < HIDDEN_SIZE; ++k) {
                elem += host_tensorA[k + j * HIDDEN_SIZE] * host_tensorB[k + i * HIDDEN_SIZE];
            }
            float diff = fabs(elem - host_result[i + j * SEQ_LEN]);
            if(diff < 1e-5) {
                num_correct++;
                printf("elem[%d][%d] correct, val=%.5f\n", i, j, host_result[i + j * SEQ_LEN]);

            } else {
                num_wrong++;
                printf("elem[%d][%d] wrong, val=%.5f, truth=%.5f\n", i, j, host_result[i + j * SEQ_LEN], elem);
            }
        }
    }
    printf("correct num: %d, wrong num: %d\n", num_correct, num_wrong);

    // Free dev res
    CHECK_CUDA_ERROR(cudaFree(dev_tensorA));
    CHECK_CUDA_ERROR(cudaFree(dev_tensorB));
    CHECK_CUDA_ERROR(cudaFree(dev_result));
    // Free host res
    munmap(host_tensorA, file_sizeA);
    close(fdA);
    munmap(host_tensorB, file_sizeB);
    close(fdB);
    free(host_result);
    return 0;
}