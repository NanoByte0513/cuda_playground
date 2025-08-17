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

constexpr char* tensor_path_1 = "/home/wuyou/cuda_playground/online_softmax/tensor1.bin";
constexpr char* softmax_tensor_path = "/home/wuyou/cuda_playground/online_softmax/softmax.bin";


template<int M, int N>
__global__ void online_softmax(__half* mat) {
    __half2* mat_h2 = reinterpret_cast<__half2*>(mat);
    // data pos of thread0 in this block
    int32_t blk_start_pos_data = (blockIdx.x + blockIdx.y * gridDim.y) * blockDim.x * blockDim.y * 2;
    
    // Each thread read 4 elems(half2 * 2) from glob_mem
    // out_of_bounds val filled with -inf
    __half2 h2_elem1 = mat_h2[blk_start_pos_data + threadIdx.x * 2];
    __half2 h2_elem2 = mat_h2[blk_start_pos_data + threadIdx.x * 2 + 1];
    float elem1_1 = __low2float(h2_elem1);
    float elem1_2 = __high2float(h2_elem1);
    float elem2_1 = __low2float(h2_elem2);
    float elem2_2 = __high2float(h2_elem2);
    // Get max_val and exp_sum
    __half2 h2_max = __hmax2(h2_elem1, h2_elem2);
    float max_val = fmaxf(__low2float(h2_max), __high2float(h2_max));
    float exp_sum = __expf(elem1_1 - max_val) + __expf(elem1_2 - max_val) + __expf(elem2_1 - max_val) + __expf(elem2_2 - max_val);

    // Update max_val and exp_sum among warps
    float max_val2;
    float pair_max_val;
    float exp_sum2;

    #pragma unroll
    for(int i = 16; i > 0; i = i / 2) {
        max_val2 = __shfl_down_sync(0xffffffff, max_val, i);
        pair_max_val = fmaxf(max_val, max_val2);
        exp_sum2 = __shfl_down_sync(0xffffffff, exp_sum, i);
        exp_sum = isinf(pair_max_val)? 0 : __expf(max_val - pair_max_val) * exp_sum + __expf(max_val2 - pair_max_val) * exp_sum2;
        max_val = pair_max_val;
    }
    
    // Now thread0 of each warp has max_val and exp_sum of this warp
    extern __shared__ char shared_mem_arr[]; // size == 2 * WARPS_PER_BLOCK + 2
    float* warp_lev_max_val = (float*)shared_mem_arr; // size == WARPS_PER_BLOCK
    float* warp_lev_exp_sum = warp_lev_max_val + blockDim.x / warpSize; // size == WARPS_PER_BLOCK
    float* blk_lev_max_val = warp_lev_exp_sum + 1;
    float* blk_lev_exp_sum = blk_lev_max_val + 1;
    
    
    int warpIdx = threadIdx.x / warpSize;
    int laneIdx = threadIdx.x % warpSize;
    if(laneIdx == 0) { // thread0 of the warp
        warp_lev_max_val[warpIdx] = max_val;
        warp_lev_exp_sum[warpIdx] = exp_sum;
    }

    // Read all data from shared_mem array to warp0 (laneIdx < warps_per_block)
    __syncthreads();
    if(warpIdx == 0) {
        max_val = laneIdx < blockDim.x / warpSize? warp_lev_max_val[laneIdx] : -INFINITY;
        exp_sum = laneIdx < blockDim.x / warpSize? warp_lev_exp_sum[laneIdx] : 0;
        
        // Update max_val and exp_sum among warp0
        #pragma unroll
        for(int i = 16; i > 0; i = i / 2) {
            max_val2 = __shfl_down_sync(0xffffffff, max_val, i);
            pair_max_val = fmaxf(max_val, max_val2);
            exp_sum2 = __expf(max_val2 - pair_max_val) * __shfl_down_sync(0xffffffff, exp_sum, i);
            exp_sum = isinf(pair_max_val)? 0 : __expf(max_val - pair_max_val) * exp_sum + exp_sum2;
            max_val = pair_max_val;
        }
    }

    // Now thread0 of each block has max_val and exp_sum of the block
    // Writes to shared mem
    if(threadIdx.x == 0) {
        blk_lev_max_val[0] = max_val;
        blk_lev_exp_sum[0] = exp_sum;
    }
    // All threads wait for thread0
    __syncthreads();

    // All threads read max_val and exp_sum from shared mem
    max_val = blk_lev_max_val[0];
    exp_sum = blk_lev_exp_sum[0];

    // Write final result
    if(threadIdx.x * 2 < N) { // elem1
        float elem1_final_result = __expf(elem1_1 - max_val) / exp_sum;
        float elem2_final_result = __expf(elem1_2 - max_val) / exp_sum;
        h2_elem1 = {__float2half(elem1_final_result), __float2half(elem2_final_result)};
        mat_h2[blk_start_pos_data + threadIdx.x * 2] = h2_elem1;
    }
    if(threadIdx.x * 2 + 1 < N) { // elem2
        float elem1_final_result = __expf(elem2_1 - max_val) / exp_sum;
        float elem2_final_result = __expf(elem2_2 - max_val) / exp_sum;
        h2_elem2 = {__float2half(elem1_final_result), __float2half(elem2_final_result)};
        mat_h2[blk_start_pos_data + threadIdx.x * 2 + 1] = h2_elem2;
    }
}

int main() {
    // Host malloc
    float16* host_mat;
    int file_descriptor;
    size_t file_size;
    host_mat = static_cast<float16*>(utils::openBin(tensor_path_1, file_descriptor, file_size));
    float16* host_result = (float16*)malloc(SEQ_LEN * HIDDEN_SIZE * sizeof(float16));

    // Malloc device memory
    __half* dev_mat;
    CHECK_CUDA_ERROR(cudaMalloc(&dev_mat, SEQ_LEN * HIDDEN_SIZE * sizeof(__half)));
    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(dev_mat, host_mat, SEQ_LEN * HIDDEN_SIZE * sizeof(__half), cudaMemcpyHostToDevice));

    dim3 gridDim(SEQ_LEN, HIDDEN_SIZE / THREADS_PER_BLOCK / 4, 1); // Each thread deal with 4 elems
    dim3 blockDim(THREADS_PER_BLOCK, 1, 1);
    constexpr size_t shared_mem_size = (THREADS_PER_BLOCK / WARP_SIZE  * 2 + 2) * sizeof(float); // size == warps_per_block * 2 + 2
    online_softmax<SEQ_LEN, HIDDEN_SIZE><<<gridDim, blockDim, shared_mem_size>>>(dev_mat);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Copy result data from dev to host
    CHECK_CUDA_ERROR(cudaMemcpy(host_result, dev_mat, SEQ_LEN * HIDDEN_SIZE * sizeof(__half), cudaMemcpyDeviceToHost));

    // Check result
    float16* softmax_result;
    int softmax_fd;
    size_t softmax_file_size;
    softmax_result = static_cast<float16*>(utils::openBin(softmax_tensor_path, softmax_fd, softmax_file_size));

    int correct_num = 0;
    int wrong_num = 0;
    for(int i = 0; i < SEQ_LEN * HIDDEN_SIZE; ++i) {
        __half_raw halfraw_elem;
        halfraw_elem.x = host_mat[i];

        __half_raw halfraw_ground_truth;
        halfraw_ground_truth.x = softmax_result[i];

        __half_raw halfraw_result;
        halfraw_result.x = host_result[i];

        float diff = __half2float(halfraw_result) - __half2float(halfraw_ground_truth);
        if(fabs(diff) > 1e-7) {
            printf("[%d]: elem=%.8f, result=%.8f, truth=%.8f\n", i, __half2float(halfraw_elem), __half2float(halfraw_result), __half2float(halfraw_ground_truth));
            wrong_num++;
        } else {
            correct_num++;
        }
    }
    printf("\ncorrect_num: %d, wrong_num: %d\n", correct_num, wrong_num);

    munmap(host_mat, file_size);
    close(file_descriptor);
    munmap(softmax_result, softmax_file_size);
    close(softmax_fd);
    free(host_result);
    CHECK_CUDA_ERROR(cudaFree(dev_mat));
    return 0;
}