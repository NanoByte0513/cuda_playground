#include <cutlass/cutlass.h>
#include <cutlass/half.h>
#include <cutlass/aligned_buffer.h>
#include <cute/tensor.hpp>
#include <iostream>
#include "stdio.h"

#define LEN_M 8
#define LEN_N 4
#define LEN_K 2

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
    if(threadIdx.x == 0) {
        for(int i = 0; i < LEN_M; ++i) {
            for(int j = 0; j < LEN_K; ++j) {
                float val = static_cast<float>(gmem_A[make_coord(i, j)]);
                printf("[%d, %d]: %.2f\n", i, j, val);
            }
        }
    }
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
    kernel<<<dim3(1), dim3(32)>>>(d_A);

    delete[] h_A;
    cudaFree(d_A);
    return 0;
}