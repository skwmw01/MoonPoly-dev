#pragma once

#include "moonpoly.cuh"

namespace moonpoly {
namespace fp16 {
namespace row_col_col {

bool is_explicit_neighborhood_pid(int pid);
bool is_paper_elite_fp16_pid(int pid);
int paper_elite_fp16_kernel_count();
const int *paper_elite_fp16_pids();
int explicit_neighborhood_alignment(int pid);
SharedMem explicit_neighborhood_shared_mem(int pid);
void run_explicit_neighborhood_gemm(
    int pid, int m, int n, int k, cutlass::half_t *A, cutlass::half_t *B,
    cutlass::half_t *C, float alpha, float beta, int split_k_slices,
    cudaStream_t stream);

}  // namespace row_col_col
}  // namespace fp16
}  // namespace moonpoly
