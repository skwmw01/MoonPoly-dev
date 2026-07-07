#include "moonpoly.cuh"
#include <iostream>

namespace moonpoly {

// INT8 specific GEMM implementations moved from header
extern void run_moonpoly_gemm_int8_row_col_col(int m, int n, int k, const int8_t *A,
                                               const int8_t *B, int32_t *C,
                                               int alpha, int beta,
                                               cudaStream_t stream);

void run_int8_gemm(int m, int n, int k, 
                   const int8_t *A, bool is_a_col_major,
                   const int8_t *B, bool is_b_col_major, 
                   int32_t *C, bool is_c_col_major,
                   int alpha, int beta,
                   cudaStream_t stream) {
    
    if (!is_a_col_major && is_b_col_major && is_c_col_major) {
        run_moonpoly_gemm_int8_row_col_col(m, n, k, A, B, C, static_cast<int>(alpha), static_cast<int>(beta), stream);
    } else {
        throw std::invalid_argument("INT8 GEMM: Unsupported layout combination");
    }
}

} // namespace moonpoly
