#include <cutlass/cutlass.h>
#include <cutlass/half.h>
#include <cutlass/aligned_buffer.h>
#include <cute/tensor.hpp>
#include <iostream>
#include "stdio.h"

#define LEN_M 8
#define LEN_N 4
#define LEN_K 4

using namespace cute;

__global__ void kernel(const cutlass::half_t* gmem_A_ptr) {
    // Static Integer Int<32>{} or _32
    auto problem_shape = make_shape(Int<LEN_M>{}, Int<LEN_N>{}, Int<LEN_K>{});
    auto shape_tuple = make_tuple(Int<LEN_M>{}, Int<LEN_N>{}, Int<LEN_K>{});



    // Layout就是Shape和Stride的组合，Shape说明每个维度包含几个元素，Stride说明在把多维坐标转换成线性地址时，每个维度+1会跳过几个元素
    Layout layout_A = make_layout(
        make_shape(LEN_M, LEN_K),
        LayoutRight{} // LayoutRight表示shape的最右边的维度stride是1，其他每个维度的stride是右边维度的乘积（也就是行主序）
        // make_stride(LEN_K, _1{}) // 第0维坐标+1，索引加LEN_K；第1维坐标+1，索引加1；等价于LayoutRight{}
    );
    Tensor gmem_A = make_tensor(
        make_gmem_ptr(gmem_A_ptr),
        layout_A
    );

    int warpIdx = threadIdx.x / warpSize;
    int laneIdx = threadIdx.x % warpSize;
    // Split gmem_A into 2*2 warp tiles
    auto warp_tiler = make_shape(LEN_M / 2, LEN_N / 2, LEN_K / 2); // warp每次处理的块大小，在K维度上滑动
    // Use select<0,2> to use only the M- and K-modes of the tiler and coord
    Tensor warp_tensor_A = local_tile(gmem_A, make_tile(LEN_M / 2, LEN_K / 2), make_coord(warpIdx / 2, warpIdx % 2));

    if(warpIdx == 0 && laneIdx == 0) {
        print_tensor(gmem_A);
        print_tensor(warp_tensor_A);
    }
    __syncthreads();
    if(warpIdx == 1 && laneIdx == 0) {
        // print_tensor(gmem_A);
        print_tensor(warp_tensor_A);
    }
    __syncthreads();
    if(warpIdx == 2 && laneIdx == 0) {
        // print_tensor(gmem_A);
        print_tensor(warp_tensor_A);
    }
    __syncthreads();
    if(warpIdx == 3 && laneIdx == 0) {
        // print_tensor(gmem_A);
        print_tensor(warp_tensor_A);
    }
    __syncthreads();
}

int main() {
    cutlass::half_t* h_A = new cutlass::half_t[LEN_M * LEN_K];
    for(int i = 0; i < LEN_M * LEN_K; ++i) {
        float val = (float)i;
        h_A[i] = static_cast<cutlass::half_t>(val);
        // printf("[%d]: %.2f\n", i, static_cast<float>(h_A[i]));
    }
    cutlass::half_t* d_A;
    cudaMalloc(&d_A, LEN_M * LEN_K * sizeof(cutlass::half_t));
    cudaMemcpy(d_A, h_A, LEN_M * LEN_K * sizeof(cutlass::half_t), cudaMemcpyHostToDevice);
    kernel<<<dim3(1), dim3(128)>>>(d_A);

    delete[] h_A;
    cudaFree(d_A);
    return 0;
}