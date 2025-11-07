#include <cublas_v2.h>
#include <cutlass/half.h>
#include "utils/utils.cuh"

#define LEN_M 32
#define LEN_N 16
#define LEN_K 64

constexpr char* tensor1_32_64_path = "/home/wuyou/cuda_playground/test_cublass/tensor1_fp16_32_64.bin";
constexpr char* tensor2_64_16_path = "/home/wuyou/cuda_playground/test_cublass/tensor2_fp16_64_16.bin";
constexpr char* product_32_16_path = "/home/wuyou/cuda_playground/test_cublass/product_fp16_32_16.bin";

bool check_product(const char* path1, const char* path2, const char* path_product, int M, int N, int K) {
    int fd1;
    size_t file_size1;
    cutlass::half_t* h_A = static_cast<cutlass::half_t*>(utils::openBin(path1, fd1, file_size1));

    int fd2;
    size_t file_size2;
    cutlass::half_t* h_B = static_cast<cutlass::half_t*>(utils::openBin(path2, fd2, file_size2));

    int fd3;
    size_t file_size3;
    cutlass::half_t* h_C = static_cast<cutlass::half_t*>(utils::openBin(path_product, fd3, file_size3));
    cutlass::half_t* h_cublass_C = new cutlass::half_t[M * N];

    // 设备内存分配（GPU）
    cutlass::half_t *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M * K * sizeof(cutlass::half_t));
    cudaMalloc((void**)&d_B, K * N * sizeof(cutlass::half_t));
    cudaMalloc((void**)&d_C, M * N * sizeof(cutlass::half_t));

    cudaMemcpy(d_A, h_A, M * K * sizeof(cutlass::half_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(cutlass::half_t), cudaMemcpyHostToDevice);

    // 创建cuBLAS句柄
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);


    float alpha = 1.0f;
    float beta = 0.0f;
    
    /**
     * cublass内部是列主序的，输入的A(M*K)和B(K*N)都是按行主序存储的，因此输入的A会被看作A^T, B会看作B^T，
     * 则为了输出的C能够按行存储，我们希望输出的计算结果是C^T而不是C，
     * 需要把C = A @ B换成 C^T = (B^T @ A^T)，对应的shape为B^T(N*K), A^T(K*M), C^T(N*M)
     * 所以输入是B,A而不是A,B，又因为本身输入的BA都会被视为转置，所以参数12不需要转置。
     * 参数345不应该是MNK而是NMK对应B(N*K), A(M*K)
     * ldA, ldB, ldC可以看成在存储里面，ABC存储的主序是什么（经过多少个元素换行，对于A矩阵每K个元素换行）
     */
    cublasGemmEx(
        handle,
        CUBLAS_OP_N,                     
        CUBLAS_OP_N,                     
        N, M, K,             
        &alpha,                           
        d_B, CUDA_R_16F, N,              
        d_A, CUDA_R_16F, K,               
        &beta,                            
        d_C, CUDA_R_16F, N,              
        CUDA_R_32F,                      
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ); // 这里得到的C是按行存储的，直接C[i]访问
    cudaMemcpy(h_cublass_C, d_C, M * N * sizeof(cutlass::half_t), cudaMemcpyDeviceToHost);

    bool noError = true;
    for(int i = 0; i < M * N; i++) {
        float ground_truth = h_C[i];
        float cublass_val = h_cublass_C[i];
        float diff = fabs(ground_truth - cublass_val);
        if(diff > 1e-3) {
            printf("[%d] ground_truth=%.4f, cublass_val=%.4f\n", i, ground_truth, cublass_val);
            noError = false;
        }
    }

    delete[] h_cublass_C;
    cublasDestroy(handle);
    munmap(h_A, file_size1);
    close(fd1);
    munmap(h_B, file_size2);
    close(fd2);
    munmap(h_C, file_size3);
    close(fd3);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return noError;
}

int main() {
    if(check_product(tensor1_32_64_path, tensor2_64_16_path, product_32_16_path, LEN_M, LEN_N, LEN_K))
        printf("No Error\n");
    return 0;
}