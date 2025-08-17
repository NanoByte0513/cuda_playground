#include "utils/ptx.cuh"
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include "cuda_fp16.h"
#include "utils/utils.cuh"
#include <mma.h>
#include <cstdint>
#include <math.h>

#define HIDDEN_SIZE 1024
#define Q_PROJ_DIM 2048

using float16 = uint16_t;
using namespace nvcuda;
constexpr char* tensor_path_1 = "/home/wuyou/cuda_playground/hgemm/tensor1_fp16_1024_1024.bin";
constexpr char* tensor_path_2 = "/home/wuyou/cuda_playground/hgemm/tensor2_fp16_1024_2048.bin";

/**
 * C = A.matmul(B)
 * A: M*K
 * B: K*N
 * C: M*N
 * 
 */
template<int M, int K, int N, const int WMMA_M = 16, const int WMMA_K = 16, const int WMMA_N = 16>
__global__ void hgemm_kernel(const __half* ptr_a, const __half* ptr_b, __half* ptr_c) {
    const int blk_pos_a_M = blockIdx.y * WMMA_M;
    const int blk_pos_b_N = blockIdx.x * WMMA_N;

    // Create a 16*16 fragment in matrix C, init with zeros(Layout must be void)
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag;
    wmma::fill_fragment(C_frag, 0.0);
    // Create a 16*16 fragment in matrix A and B(Layout is needed)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> B_frag;

    constexpr int NUM_K_TILES = K / WMMA_K;
    #pragma unroll
    for(int i = 0; i < NUM_K_TILES; ++i) {
        wmma::load_matrix_sync(A_frag, ptr_a + blk_pos_a_M * K + i * WMMA_K, K);
        wmma::load_matrix_sync(B_frag, ptr_b + blk_pos_b_N * K + i * WMMA_K, N);

        wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
        __syncthreads();
    }

    // Write Matrix C to gloab mem
    wmma::store_matrix_sync(ptr_c + blk_pos_a_M * N + blk_pos_b_N, C_frag, N, wmma::mem_row_major);
}

int main() {
    // Host malloc
    float16* host_tensor1;
    int fd1;
    size_t file_size1;
    host_tensor1 = static_cast<float16*>(utils::openBin(tensor_path_1, fd1, file_size1));

    float16* host_tensor2;
    int fd2;
    size_t file_size2;
    host_tensor2 = static_cast<float16*>(utils::openBin(tensor_path_2, fd2, file_size2));

    float16* host_tensor_c = (float16*)malloc(HIDDEN_SIZE * Q_PROJ_DIM * sizeof(__half));

    // Malloc device memory
    __half* dev_tensor1;
    CHECK_CUDA_ERROR(cudaMalloc(&dev_tensor1, HIDDEN_SIZE * HIDDEN_SIZE * sizeof(__half)));
    __half* dev_tensor2;
    CHECK_CUDA_ERROR(cudaMalloc(&dev_tensor2, HIDDEN_SIZE * Q_PROJ_DIM * sizeof(__half)));
    __half* dev_tensor_c;
    CHECK_CUDA_ERROR(cudaMalloc(&dev_tensor_c, HIDDEN_SIZE * Q_PROJ_DIM * sizeof(__half)));

    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(dev_tensor1, host_tensor1, HIDDEN_SIZE * HIDDEN_SIZE * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(dev_tensor2, host_tensor2, HIDDEN_SIZE * Q_PROJ_DIM * sizeof(__half), cudaMemcpyHostToDevice));

    dim3 block_dim(32, 1, 1);
    dim3 grid_dim(Q_PROJ_DIM / 16, HIDDEN_SIZE / 16, 1);

    hgemm_kernel<HIDDEN_SIZE, HIDDEN_SIZE, Q_PROJ_DIM, 16, 16, 16><<<grid_dim, block_dim>>>(dev_tensor1, dev_tensor2, dev_tensor_c);

    CHECK_CUDA_ERROR(cudaMemcpy(host_tensor_c, dev_tensor_c, HIDDEN_SIZE * Q_PROJ_DIM * sizeof(__half), cudaMemcpyDeviceToHost));

    utils::printMatrix_fp16(host_tensor_c, HIDDEN_SIZE, Q_PROJ_DIM);


    CHECK_CUDA_ERROR(cudaFree(dev_tensor1));
    CHECK_CUDA_ERROR(cudaFree(dev_tensor2));
    CHECK_CUDA_ERROR(cudaFree(dev_tensor_c));
    munmap(host_tensor1, file_size1);
    close(fd1);
    munmap(host_tensor2, file_size2);
    close(fd2);
    return 0;
}