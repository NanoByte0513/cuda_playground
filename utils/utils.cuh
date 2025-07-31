#pragma once

#include "cuda_fp16.h"
#include <cstdint>

#define CHECK_CUDA_ERROR do { \
    \
} \
while(0)

namespace utils {
void printMatrix_fp16(const uint16_t* ptr, int row, int col) {
    for(int i = 0; i < row; ++i) {
        if(i > 2 && i < row - 3)
            continue;
        for(int j = 0; j < col; ++j) {
            if(j > 2 && j < col - 3)
                continue;
            __half_raw raw_half;
            raw_half.x = ptr[i * col + j];
            half fp16_val = raw_half;
            printf("%.4f, ", __half2float(fp16_val));
        }
        printf("\n");
    }
}
} // namespace utils