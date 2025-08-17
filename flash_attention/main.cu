#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include "cuda_fp16.h"
#include "utils/utils.cuh"
#include <cstdint>
#include <math.h>

__global__
void flash_attention_fp16_kernel(__half* input, const __half* attn_norm, 
                                 const __half* q_proj, const __half* k_proj, const __half* v_proj, const __half* o_proj, 
                                 const __half* q_norm, const __half* k_norm, const __half* mlp_norm, 
                                 const __half* up_proj, const __half* gate_proj, const __half* down_proj) {

}

int main() {
    return 0;
}