//  file: cute_gemm.cu
#include <cutlass/cutlass.h>
#include <cutlass/aligned_buffer.h>
#include <cute/tensor.hpp>
#include <cutlass/gemm/warp/default_mma_tensor_op_sm80.h>
#include <fcntl.h>
#include <unistd.h>
#include "cuda_fp16.h"
#include "utils/utils.cuh"

#define HIDDEN_SIZE 1024
#define Q_PROJ_DIM 2048
#define BLOCK_DIM 16

/**
 * 在这个问题中，MNK为(1024, 2048, 1024)
 * block负责的规模为(64, 128, 64)，即每个block负责C矩阵上64*128的子矩阵的计算，每次从gmem加载A矩阵的64*64，从B矩阵加载64*128到smem；
 * warp负责的规模为(32, 32, 32)，即每个warp负责C矩阵上32*32的子矩阵计算，每次从smem加载A矩阵的32*32，从B矩阵加载32*32到reg;
 * MMAOp负责的规模为(16, 8, 8)
 */

using float16 = uint16_t;
// using namespace cute;
constexpr char* tensor_path_1 = "/home/wuyou/cuda_playground/hgemm/tensor1_fp16_1024_1024.bin";
constexpr char* tensor_path_2 = "/home/wuyou/cuda_playground/hgemm/tensor2_fp16_1024_2048.bin";

using ElementInputA = cutlass::half_t; // <- data type of elements in input matrix A. CAUTION: __half is invalid
using ElementInputB = cutlass::half_t; // <- data type of elements in input matrix B. CAUTION: __half is invalid
using ElementOutput = float;           // <- data type of elements in output matrix D

using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 64>;
using WarpShape = cutlass::gemm::GemmShape<32, 32, 32>;
using MMAOpShape = cutlass::gemm::GemmShape<16, 8, 8>;
using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementInputA>::value, 64>;
using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementInputB>::value, 64>;

using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
                        WarpShape, MMAOpShape, ElementInputA, LayoutA, 
                        ElementInputB, LayoutB, ElementOutput,
                        cutlass::layout::RowMajor>::Type;

template<int M, int N, int K, int BM, int BN, int BK, int WM, int WN, int WK>
__global__ void hgemm_kernel(const cutlass::half_t* gmem_tensorA_ptr, const cutlass::half_t* gmem_tensorB_ptr, cutlass::half_t* gmem_tensorC_ptr) {
    cute::Tensor gmem_tensorA = cute::make_tensor(cute::make_gmem_ptr(gmem_tensorA_ptr), cute::make_shape(M, K), cute::LayoutRight{});
    cute::Tensor gmem_tensorB = cute::make_tensor(cute::make_gmem_ptr(gmem_tensorB_ptr), cute::make_shape(K, N), cute::LayoutRight{});
    cute::Tensor gmem_tensorC = cute::make_tensor(cute::make_gmem_ptr(gmem_tensorC_ptr), cute::make_shape(M, N), cute::LayoutRight{});

    // Block level tile
    cute::Tensor gmem_tensorA_blk_tile = cute::local_tile(
        gmem_tensorA, 
        cute::make_shape(BM, K), 
        cute::make_coord(blockIdx.y, 0)
    );
    cute::Tensor gmem_tensorB_blk_tile = cute::local_tile(
        gmem_tensorB, 
        cute::make_shape(K, BN), 
        cute::make_coord(0, blockIdx.x)
    );
    cute::Tensor gmem_tensorC_blk_tile = cute::local_tile(
        gmem_tensorC, 
        cute::make_shape(BM, BN), 
        cute::make_coord(blockIdx.y, blockIdx.x)
    );

    // __shared__ cutlass::half_t smemA[BM * BK];
    // __shared__ cutlass::half_t smemB[BK * BN];
    __shared__ cutlass::AlignedBuffer<cutlass::half_t, BM * BK> smemA;
    __shared__ cutlass::AlignedBuffer<cutlass::half_t, BK * BN> smemB;

    cute::Tensor smem_tensorA = cute::make_tensor(cute::make_smem_ptr(smemA.data()), cute::make_shape(BM, BK), cute::LayoutRight{});
    cute::Tensor smem_tensorB = cute::make_tensor(cute::make_smem_ptr(smemB.data()), cute::make_shape(BK, BN), cute::LayoutRight{});


    // Warp level computations
    using FragmentA = typename MmaTensorOp::FragmentA;
    using FragmentB = typename MmaTensorOp::FragmentB;
    using FragmentC = typename MmaTensorOp::FragmentC;

    typename MmaTensorOp::LayoutA layout_A = MmaTensorOp::LayoutA::packed({ThreadblockShape::kM, ThreadblockShape::kK});
    typename MmaTensorOp::LayoutB layout_B = MmaTensorOp::LayoutB::packed({ThreadblockShape::kK, ThreadblockShape::kN});
    typename MmaTensorOp::LayoutC layout_C = MmaTensorOp::LayoutC::packed({MmaTensorOp::Shape::kM, MmaTensorOp::Shape::kN});
    typename MmaTensorOp::IteratorA iter_A({smemA.data(), layout_A}, cutlass::arch::LaneId());
    typename MmaTensorOp::IteratorB iter_B({smemB.data(), layout_B}, cutlass::arch::LaneId());

    FragmentA frag_A;
    FragmentB frag_B;
    FragmentC accum;

    MmaTensorOp mmaOp;
    CUTLASS_PRAGMA_UNROLL
    for(int k = 0; k < K / BK; k++) {
        cute::Tensor gmem_tensorA_blk_tile_at_k = cute::local_tile(gmem_tensorA_blk_tile, cute::make_shape(BM, BK), cute::make_coord(0, k));
        cute::Tensor gmem_tensorB_blk_tile_at_k = cute::local_tile(gmem_tensorB_blk_tile, cute::make_shape(BK, BN), cute::make_coord(k, 0));

        // Copy tensor A and B from gmem to smem
        cute::copy(gmem_tensorA_blk_tile_at_k, smem_tensorA);
        cute::copy(gmem_tensorB_blk_tile_at_k, smem_tensorB);
        __syncthreads();

        // Load from smem to reg, run mma computations
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < ThreadblockShape::kK; k += MmaTensorOp::Policy::MmaShape::kK) {
            iter_A.load(frag_A);
            iter_B.load(frag_B);

            ++iter_A;
            ++iter_B;

            mmaOp(accum, frag_A, frag_B, accum);
        }

        // typename MmaTensorOp::IteratorC iter_C({gmem_tensorC_ptr, layout_C}, cutlass::arch::LaneId());

        // iter_C.store(accum);
        // ++iter_C;

        // break;
    }
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

    float16* host_tensor_c = (float16*)malloc(HIDDEN_SIZE * Q_PROJ_DIM * sizeof(cutlass::half_t));

    // Malloc device memory
    cutlass::half_t* dev_tensor1;
    CHECK_CUDA_ERROR(cudaMalloc(&dev_tensor1, HIDDEN_SIZE * HIDDEN_SIZE * sizeof(cutlass::half_t)));
    cutlass::half_t* dev_tensor2;
    CHECK_CUDA_ERROR(cudaMalloc(&dev_tensor2, HIDDEN_SIZE * Q_PROJ_DIM * sizeof(cutlass::half_t)));
    cutlass::half_t* dev_tensor_c;
    CHECK_CUDA_ERROR(cudaMalloc(&dev_tensor_c, HIDDEN_SIZE * Q_PROJ_DIM * sizeof(cutlass::half_t)));

    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(dev_tensor1, host_tensor1, HIDDEN_SIZE * HIDDEN_SIZE * sizeof(cutlass::half_t), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(dev_tensor2, host_tensor2, HIDDEN_SIZE * Q_PROJ_DIM * sizeof(cutlass::half_t), cudaMemcpyHostToDevice));

    dim3 block_dim(BLOCK_DIM, BLOCK_DIM, 1);
    dim3 grid_dim(Q_PROJ_DIM / BLOCK_DIM / 4, HIDDEN_SIZE / BLOCK_DIM / 4, 1);
    hgemm_kernel<HIDDEN_SIZE, Q_PROJ_DIM, HIDDEN_SIZE, 64, 128, 64, 32, 32, 32>
    <<<grid_dim, block_dim>>>(dev_tensor1, dev_tensor2, dev_tensor_c);

    // CHECK_CUDA_ERROR(cudaMemcpy(host_tensor_c, dev_tensor_c, HIDDEN_SIZE * Q_PROJ_DIM * sizeof(__half), cudaMemcpyDeviceToHost));

    // utils::printMatrix_fp16(host_tensor_c, HIDDEN_SIZE, Q_PROJ_DIM);


    CHECK_CUDA_ERROR(cudaFree(dev_tensor1));
    CHECK_CUDA_ERROR(cudaFree(dev_tensor2));
    CHECK_CUDA_ERROR(cudaFree(dev_tensor_c));
    munmap(host_tensor1, file_size1);
    close(fd1);
    munmap(host_tensor2, file_size2);
    close(fd2);
    return 0;
}