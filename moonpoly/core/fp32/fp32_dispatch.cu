#include "moonpoly.cuh"
#include <iostream>

namespace moonpoly {
extern void run_moonpoly_gemm_fp32_row_col_col(int m, int n, int k,
                                               const float* A, const float* B, float* C,
                                               float alpha, float beta,
                                               cudaStream_t stream);

void run_fp32_gemm(int m, int n, int k, 
                   const float* A, bool is_a_col_major,
                   const float* B, bool is_b_col_major,
                   float* C, bool is_c_col_major, 
                   float alpha, float beta,
                   cudaStream_t stream) {
    if (!is_a_col_major && is_b_col_major && is_c_col_major) {
        // A is row-major, B is column-major, C is column-major
        run_moonpoly_gemm_fp32_row_col_col(m, n, k, 
                                           A, B, C, 
                                           alpha, beta,
                                           stream);
    } else {
        throw std::invalid_argument("FP16 GEMM: Unsupported layout combination");
    }
}

} // namespace moonpoly
