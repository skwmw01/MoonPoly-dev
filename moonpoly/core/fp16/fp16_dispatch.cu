#include "cutlass/half.h"
#include "moonpoly.cuh"
#include <iostream>

namespace moonpoly {
extern void run_moonpoly_gemm_fp16_row_col_col(int m, int n, int k, const cutlass::half_t *A,
                                        const cutlass::half_t *B, cutlass::half_t *C,
                                        float alpha, float beta,
                                        cudaStream_t stream);

void run_fp16_gemm(int m, int n, int k, 
                   const cutlass::half_t* A, bool is_a_col_major,
                   const cutlass::half_t* B, bool is_b_col_major,
                   cutlass::half_t* C, bool is_c_col_major, 
                   float alpha, float beta,
                   cudaStream_t stream) {
    if (!is_a_col_major && is_b_col_major && is_c_col_major) {
        // A is row-major, B is column-major, C is column-major
        run_moonpoly_gemm_fp16_row_col_col(m, n, k, 
                                           A, B, C, 
                                           alpha, beta,
                                           stream);
    } else {
        std::cout << "is_a_col_major: " << is_a_col_major << ", is_b_col_major: " << is_b_col_major << ", is_c_col_major: " << is_c_col_major << std::endl;
        throw std::invalid_argument("FP16 GEMM: Unsupported layout combination");
    }
}

} // namespace moonpoly
