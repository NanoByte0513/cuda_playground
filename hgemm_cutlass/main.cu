#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cutlass/half.h>
#include <cutlass/cutlass.h>
#include <cutlass/aligned_buffer.h>
#include <cute/tensor.hpp>
#include <cutlass/gemm/warp/default_mma_tensor_op_sm80.h>
#include <cutlass/util/host_tensor.h>
#include "cuda_fp16.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "utils/utils.cuh"
#include <fcntl.h>
#include <unistd.h>
#include <random>

#define LEN_M 32
#define LEN_N 8
#define LEN_K 128

using ElementAccumulator = float;
using ElementComputeEpilogue = ElementAccumulator;
using ElementInputA = cutlass::half_t;
using ElementInputB = cutlass::half_t;
using ElementOutput = float;

using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor; // 这里的列主序并不是指B的数据在内存中是按列主序排列的，而是B要按照列主序访问
using LayoutOutput = cutlass::layout::RowMajor;

using ThreadblockShape = cutlass::gemm::GemmShape<LEN_M, LEN_N, LEN_K>; // 这里指的是一个block沿K维度滑动，每一次要从AB读取的数据大小(From gmem to smem)
using WarpShape = cutlass::gemm::GemmShape<16, 8, 16>; // 这里指的是一个warp每次从smem读取的数据大小(From smem to reg)
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>; // 这里指一次TensorCore指令读取的数据大小
using Mma = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput,
      cutlass::layout::RowMajor>::Type;


using float16 = uint16_t;
/// Test kernel
__global__ void kernel(
    typename Mma::ElementC *output_C, 
    typename Mma::ElementA const *input_A,
    typename Mma::ElementB const *input_B) {
    cute::Tensor gmem_tensorA = cute::make_tensor(cute::make_gmem_ptr(input_A), cute::make_shape(ThreadblockShape::kM, ThreadblockShape::kK), cute::LayoutRight{});
    cute::Tensor gmem_tensorB = cute::make_tensor(cute::make_gmem_ptr(input_B), cute::make_shape(ThreadblockShape::kK, ThreadblockShape::kN), cute::LayoutRight{});
    cute::Tensor gmem_tensorC = cute::make_tensor(cute::make_gmem_ptr(output_C), cute::make_shape(ThreadblockShape::kM, ThreadblockShape::kN), cute::LayoutRight{});

    // Use AlignedBuffer to store trivially copyable objects in unions and __shared__ buffers.
    __shared__ cutlass::AlignedBuffer<
        typename Mma::ElementA, ThreadblockShape::kM * ThreadblockShape::kK> smem_buffer_A;

    __shared__ cutlass::AlignedBuffer<
        typename Mma::ElementB, ThreadblockShape::kN * ThreadblockShape::kK> smem_buffer_B;
    
    // Use the smem_buffer to init cute::tensors
    cute::Tensor smem_tensorA = cute::make_tensor(cute::make_smem_ptr(smem_buffer_A.data()), cute::make_shape(ThreadblockShape::kM, ThreadblockShape::kK), cute::LayoutRight{});
    cute::Tensor smem_tensorB = cute::make_tensor(cute::make_smem_ptr(smem_buffer_B.data()), cute::make_shape(ThreadblockShape::kK, ThreadblockShape::kN), cute::LayoutRight{});

    // Read whole block to smem
    cute::copy(gmem_tensorA, smem_tensorA);
    cute::copy(gmem_tensorB, smem_tensorB);

    __syncthreads();

    // Split block tile to warp tiles
    int warpIdx = threadIdx.x / warpSize;
    cute::Tensor smem_tensorA_warp_tile = cute::local_tile(
        smem_tensorA, 
        cute::make_shape(ThreadblockShape::kM / 2, ThreadblockShape::kK), 
        cute::make_coord(warpIdx)
    );
    cute::Tensor smem_tensorB_warp_tile = cute::local_tile(
        smem_tensorB, 
        cute::make_shape(ThreadblockShape::kK, ThreadblockShape::kN), 
        cute::make_coord(0)
    );
    cute::Tensor gmem_tensorC_warp_tile = cute::local_tile(
        gmem_tensorC, 
        cute::make_shape(ThreadblockShape::kM / 2, ThreadblockShape::kN), 
        cute::make_coord(warpIdx)
    );

    //
    // Construct warp-level matrix product
    //

    using FragmentA = typename Mma::FragmentA;
    using FragmentB = typename Mma::FragmentB;
    using FragmentC = typename Mma::FragmentC;

    typename Mma::LayoutA layout_A = Mma::LayoutA::packed({ThreadblockShape::kM / 2, ThreadblockShape::kK});
    typename Mma::LayoutB layout_B = Mma::LayoutB::packed({ThreadblockShape::kK, ThreadblockShape::kN});
    typename Mma::LayoutC layout_C = Mma::LayoutC::packed({Mma::Shape::kM / 2, Mma::Shape::kN}); // Mma::Shape实际上是WarpShape而不是InstructionShape

    typename Mma::IteratorA iter_A({&(smem_tensorA_warp_tile[0]), layout_A}, cutlass::arch::LaneId());
    typename Mma::IteratorB iter_B({&(smem_tensorB_warp_tile[0]), layout_B}, cutlass::arch::LaneId());

    FragmentA frag_A;
    FragmentB frag_B;
    FragmentC accum;
    accum.clear();

    Mma mma;
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < ThreadblockShape::kK; k += Mma::Policy::MmaShape::kK) { // Mma::Policy::MmaShape is InstructionShape
        iter_A.load(frag_A);
        iter_B.load(frag_B);

        ++iter_A;
        ++iter_B;

        mma(accum, frag_A, frag_B, accum);
    }
  
    typename Mma::IteratorC iter_C({&(gmem_tensorC_warp_tile[0]), layout_C}, cutlass::arch::LaneId());

    iter_C.store(accum);
}



int main() {
    // Create a tuple of problem size for matrix multiplication
    cutlass::gemm::GemmCoord problem_size(LEN_M, LEN_N, LEN_K);

    // Initialize tensors using CUTLASS helper functions
    cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
        problem_size.mk());  // <- Create matrix A with dimensions M x K
    cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
        problem_size.kn());  // <- Create matrix B with dimensions K x N
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(
        problem_size.mn());
    cutlass::reference::host::TensorFill(
        tensor_d.host_view());  // <- fill matrix D on host with zeros


    
    cutlass::half_t *h_A = new cutlass::half_t[LEN_M * LEN_K];
    cutlass::half_t *h_B = new cutlass::half_t[LEN_K * LEN_N];
    cutlass::half_t *h_C = new cutlass::half_t[LEN_M * LEN_N];  // cuBLAS计算结果

    // 初始化输入矩阵（随机值）
    for (int i = 0; i < LEN_M * LEN_K; i++) h_A[i] = static_cast<cutlass::half_t>(static_cast<float>(rand() - RAND_MAX / 2) / RAND_MAX * 2);
    for (int i = 0; i < LEN_K * LEN_N; i++) h_B[i] = static_cast<cutlass::half_t>(static_cast<float>(rand() - RAND_MAX / 2) / RAND_MAX * 2);

    // 设备内存分配（GPU）
    cutlass::half_t *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, LEN_M * LEN_K * sizeof(cutlass::half_t));
    cudaMalloc((void**)&d_B, LEN_K * LEN_N * sizeof(cutlass::half_t));
    cudaMalloc((void**)&d_C, LEN_M * LEN_N * sizeof(cutlass::half_t));

    // 数据从主机复制到设备
    cudaMemcpy(d_A, h_A, LEN_M * LEN_K * sizeof(cutlass::half_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, LEN_K * LEN_N * sizeof(cutlass::half_t), cudaMemcpyHostToDevice);

    // 复制数据到 tensor_a
    size_t num_elements = tensor_a.size();
    size_t size_in_bytes = num_elements * sizeof(ElementInputA);
    std::memcpy(tensor_a.host_data(), h_A, size_in_bytes);

    // 复制数据到 tensor_b
    num_elements = tensor_b.size();
    size_in_bytes = num_elements * sizeof(ElementInputB);
    std::memcpy(tensor_b.host_data(), h_B, size_in_bytes);

    // Copy data from host to GPU
    tensor_a.sync_device();
    tensor_b.sync_device();
    tensor_d.sync_device();

    kernel<<<dim3(1), dim3(64)>>>(tensor_d.device_data(), tensor_a.device_data(), tensor_b.device_data());
    cudaDeviceSynchronize();
    tensor_d.sync_host();


    // 创建cuBLAS句柄
    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    // 定义标量参数（alpha=1, beta=0: C = A*B）
    float alpha = 1.0f;
    float beta = 0.0f;

    // 调用cuBLAS半精度矩阵乘法（cublasGemmEx
    cublasGemmEx(
        handle,
        CUBLAS_OP_N,                     
        CUBLAS_OP_N,                     
        LEN_N, LEN_M, LEN_K,             
        &alpha,                           
        d_B, CUDA_R_16F, LEN_N,              
        d_A, CUDA_R_16F, LEN_K,               
        &beta,                            
        d_C, CUDA_R_16F, LEN_N,              
        CUDA_R_32F,                      
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ); // 这里得到的C是按行存储的，直接C[i]访问

    cudaMemcpy(h_C, d_C, LEN_M * LEN_N * sizeof(cutlass::half_t), cudaMemcpyDeviceToHost);

    int num_error = 0;
    int num_total = 0;
    for(int i = 0; i < LEN_M * LEN_N; ++i) {
        float cub_val = h_C[i];
        float ker_val = tensor_d.host_data()[i];
        float diff = fabs(cub_val - ker_val);
        if(diff > 1e-2) {
            printf("[%d], ", i);
            num_error++;
        }
        num_total++;
    }
    printf("\nnum_error: %d, num_total: %d\n", num_error, num_total);



    // 释放资源
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
    return 0;
}