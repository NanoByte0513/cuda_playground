/**
 * 这一个也跑通了，与16*8*8的区别就是block的k维度变成了16，所以117行的循环会执行两次累加，但结果矩阵C的shape没有变。
 */

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

#define LEN_M 16
#define LEN_N 8
#define LEN_K 16

using ElementAccumulator = float;
using ElementComputeEpilogue = ElementAccumulator;
using ElementInputA = cutlass::half_t;
using ElementInputB = cutlass::half_t;
using ElementOutput = float;

using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor; // 这里的列主序并不是指B的数据在内存中是按列主序排列的，而是B要按照列主序访问
using LayoutOutput = cutlass::layout::RowMajor;

using ThreadblockShape = cutlass::gemm::GemmShape<16, 8, 16>; // 这里指的是一个block沿K维度滑动，每一次要从AB读取的数据大小(From gmem to smem)
using WarpShape = cutlass::gemm::GemmShape<16, 8, 8>; // 这里指的是一个warp每次从smem读取的数据大小(From smem to reg)
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>; // 这里指一次TensorCore指令读取的数据大小
using Mma = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput,
      cutlass::layout::RowMajor>::Type;

constexpr char* tensor_path_1 = "/home/wuyou/cuda_playground/try_cutlass/tensor1_fp16_16_16.bin";
constexpr char* tensor_path_2 = "/home/wuyou/cuda_playground/try_cutlass/tensor2_fp16_8_16.bin";
using float16 = uint16_t;

/// Test kernel
__global__ void kernel(
    typename Mma::ElementC *output_C, 
    typename Mma::ElementA const *input_A,
    typename Mma::ElementB const *input_B) {

    

    // Use AlignedBuffer to store trivially copyable objects in unions and __shared__ buffers.
    __shared__ cutlass::AlignedBuffer<
        typename Mma::ElementA, ThreadblockShape::kM * ThreadblockShape::kK> smem_buffer_A;

    __shared__ cutlass::AlignedBuffer<
        typename Mma::ElementB, ThreadblockShape::kN * ThreadblockShape::kK> smem_buffer_B;

    if (threadIdx.x == 0) {
        typename Mma::ElementA *smem_ptr_A = smem_buffer_A.data();
        #pragma unroll 1
        for (size_t i = 0; i < smem_buffer_A.size(); ++i) {
        cutlass::ReferenceFactory<typename Mma::ElementA>::get(smem_ptr_A, i) =
            cutlass::ReferenceFactory<typename cutlass::platform::remove_const<
                typename Mma::ElementA>::type>::get(input_A, i);
        }

        typename Mma::ElementB *smem_ptr_B = smem_buffer_B.data();
        #pragma unroll 1
        for (size_t i = 0; i < smem_buffer_B.size(); ++i) {
        cutlass::ReferenceFactory<typename Mma::ElementB>::get(smem_ptr_B, i) =
            cutlass::ReferenceFactory<typename cutlass::platform::remove_const<
                typename Mma::ElementB>::type>::get(input_B, i);
        }
    }

    __syncthreads();

    // float tempA[LEN_M * LEN_K];
    // cutlass::half_t* smemA_h = (cutlass::half_t*)(smem_buffer_A.raw_data());
    // for (size_t i = 0; i < LEN_M * LEN_K; ++i) {
    //     tempA[i] = static_cast<float>(smemA_h[i]);
    // }

    // float tempB[LEN_K * LEN_N];
    // cutlass::half_t* smemB_h = (cutlass::half_t*)(smem_buffer_B.raw_data());
    // for (size_t i = 0; i < LEN_K * LEN_N; ++i) {
    //     tempB[i] = static_cast<float>(smemB_h[i]);
    // }

    //
    // Construct warp-level matrix product
    //

    using FragmentA = typename Mma::FragmentA;
    using FragmentB = typename Mma::FragmentB;
    using FragmentC = typename Mma::FragmentC;

    typename Mma::LayoutA layout_A = Mma::LayoutA::packed({ThreadblockShape::kM, ThreadblockShape::kK});
    typename Mma::LayoutB layout_B = Mma::LayoutB::packed({ThreadblockShape::kK, ThreadblockShape::kN});
    typename Mma::LayoutC layout_C = Mma::LayoutC::packed({Mma::Shape::kM, Mma::Shape::kN});

    typename Mma::IteratorA iter_A({smem_buffer_A.data(), layout_A}, cutlass::arch::LaneId());
    typename Mma::IteratorB iter_B({smem_buffer_B.data(), layout_B}, cutlass::arch::LaneId());

    FragmentA frag_A;
    FragmentB frag_B;

    FragmentC accum;

    Mma mma;

    accum.clear();


    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < ThreadblockShape::kK; k += Mma::Policy::MmaShape::kK) {
        iter_A.load(frag_A);
        iter_B.load(frag_B);

        ++iter_A;
        ++iter_B;

        mma(accum, frag_A, frag_B, accum);
    }
  
    typename Mma::IteratorC iter_C({output_C, layout_C}, cutlass::arch::LaneId());

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
        problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from CUTLASS kernel
    
    // // Fill input and output matrices on host using CUTLASS helper functions
    // cutlass::reference::host::TensorFillRandomUniform(
    //     tensor_a.host_view(),
    //     1,
    //     ElementInputA(4),
    //     ElementInputA(-4),
    //     0);  // <- Fill matrix A on host with uniform-distribution random data
    // cutlass::reference::host::TensorFillRandomUniform(
    //     tensor_b.host_view(),
    //     1,
    //     ElementInputB(4),
    //     ElementInputB(-4),
    //     0);  // <- Fill matrix B on host with uniform-distribution random data
    cutlass::reference::host::TensorFill(
        tensor_d.host_view());  // <- fill matrix D on host with zeros

    // Host malloc
    float16* host_tensor1;
    int fd1;
    size_t file_size1;
    host_tensor1 = static_cast<float16*>(utils::openBin(tensor_path_1, fd1, file_size1));

    float16* host_tensor2;
    int fd2;
    size_t file_size2;
    host_tensor2 = static_cast<float16*>(utils::openBin(tensor_path_2, fd2, file_size2));

    // 复制数据到 tensor_a
    size_t num_elements = tensor_a.size();
    size_t size_in_bytes = num_elements * sizeof(ElementInputA);
    std::memcpy(tensor_a.host_data(), host_tensor1, size_in_bytes);

    // 复制数据到 tensor_b
    num_elements = tensor_b.size();
    size_in_bytes = num_elements * sizeof(ElementInputB);
    std::memcpy(tensor_b.host_data(), host_tensor2, size_in_bytes);

    // Copy data from host to GPU
    tensor_a.sync_device();
    tensor_b.sync_device();
    tensor_d.sync_device();

    kernel<<<dim3(1, 1, 1), dim3(32, 1, 1)>>>(tensor_d.device_data(), tensor_a.device_data(), tensor_b.device_data());
    cudaDeviceSynchronize();

    // Copy output data from dev to host
    tensor_d.sync_host();
    for(int i = 0; i < tensor_d.size(); ++i) {
        printf("%9.6f%c", tensor_d.host_data()[i], (i + 1) % LEN_N ? ' ' : '\n');
        // if(i > 64)
        //     break;
    }
    printf("\n");

    
    munmap(host_tensor1, file_size1);
    close(fd1);
    munmap(host_tensor2, file_size2);
    close(fd2);
    return 0;
}