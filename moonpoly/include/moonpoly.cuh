#pragma once
#include "cutlass/cutlass.h"
#include "cutlass/half.h"
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cuda_runtime.h"

#define CUTLASS_CHECK(status)                                                  \
  {                                                                            \
    cutlass::Status error = status;                                            \
    if (error != cutlass::Status::kSuccess) {                                  \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error)      \
                << " at: " << __LINE__ << std::endl;                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

namespace moonpoly {

enum class MoonpolyDataType {
    FP16 = 0,
    INT8 = 1,
    FP32 = 2
};

const int num_SM = 108;
struct SharedMem {
  int pm, pk, pn;
  SharedMem() : pm(0), pn(0), pk(0) {}
  SharedMem(int m, int k, int n) : pm(m), pk(k), pn(n) {}
};

enum class Fp16PolyPattern {
    Single = 1,
    Combine = 2,
    SplitK = 3
};

struct Fp16OnlinePlan {
  Fp16PolyPattern pattern;
  int pid;
  int split_k;
  int split_n;
  float estimated_ms;
};

void moonpoly_gemm(MoonpolyDataType dtype, int m, int n, int k,
                   void* A, bool is_a_col_major,
                   void* B, bool is_b_col_major,
                   void* C, bool is_c_col_major,
                   float alpha, float beta,
                   cudaStream_t stream = nullptr);

void moonpolyLinear(MoonpolyDataType dtype, int m, int n, int k, void* A, void* B, void* C,
                    cudaStream_t stream = nullptr);


void run_fp16_gemm(int m, int n, int k, 
                   const cutlass::half_t* A, bool is_a_col_major,
                   const cutlass::half_t* B, bool is_b_col_major,
                   cutlass::half_t* C, bool is_c_col_major, 
                   float alpha, float beta,
                   cudaStream_t stream = nullptr);

Fp16OnlinePlan select_fp16_online_plan_row_col_col(int m, int n, int k);
void run_fp16_online_gemm_row_col_col(
    int m, int n, int k, const cutlass::half_t *A, const cutlass::half_t *B,
    cutlass::half_t *C, float falpha, float fbeta,
    cudaStream_t stream = nullptr);

void run_int8_gemm(int m, int n, int k, 
                   const int8_t *A, bool is_a_col_major,
                   const int8_t *B, bool is_b_col_major, 
                   int32_t *C, bool is_c_col_major,
                   int ialpha, int ibeta,
                   cudaStream_t stream = nullptr);

void run_fp32_gemm(int m, int n, int k, 
                   const float* A, bool is_a_col_major,
                   const float* B, bool is_b_col_major,
                   float* C, bool is_c_col_major, 
                   float alpha, float beta,
                   cudaStream_t stream = nullptr);

// Used for debugging purposes
void get_gemm_alignment_fp16_row_col_col(int pid, int *alignment);
void get_gemm_shared_mem_fp16_row_col_col(int pid, int *shared_mem);
void run_moonpoly_selected_gemm_fp16_row_col_col(int pid, int m, int n, int k, const cutlass::half_t *A,
                                                 const cutlass::half_t *B, cutlass::half_t *C,
                                                 float falpha, float fbeta,
                                                 cudaStream_t stream = nullptr);

// StreamK-style two-stage API for graph-safe usage.
// For non-streamK kernels, workspace size is 0 and workspace pointer is ignored.
size_t get_gemm_workspace_size_fp16_row_col_col(int pid, int m, int n, int k,
                                                 int split_k_slices = 2);
void run_moonpoly_selected_gemm_fp16_row_col_col_with_workspace(
    int pid, int m, int n, int k, const cutlass::half_t *A,
    const cutlass::half_t *B, cutlass::half_t *C, float falpha, float fbeta,
    void *workspace_ptr, size_t workspace_size_bytes, int split_k_slices = 2,
    cudaStream_t stream = nullptr);

// Explicit C++ API for the validated FP16 StreamK kernel (pid=40).
// This avoids routing through the generic pid-dispatch surface when callers
// want to test or integrate only the dedicated P40 path.
size_t get_fp16_p40_streamk_workspace_size_row_col_col(int m, int n, int k,
                                                       int split_k_slices = 2);
void run_fp16_p40_streamk_singleton_row_col_col(
    int m, int n, int k, const cutlass::half_t *A, const cutlass::half_t *B,
    cutlass::half_t *C, float falpha, float fbeta, int split_k_slices = 2,
    cudaStream_t stream = nullptr);
void run_fp16_p40_streamk_row_col_col(
    int m, int n, int k, const cutlass::half_t *A, const cutlass::half_t *B,
    cutlass::half_t *C, float falpha, float fbeta, void *workspace_ptr,
    size_t workspace_size_bytes, int split_k_slices = 2,
    cudaStream_t stream = nullptr);

// Explicit C++ API for a P40-equivalent validation kernel (pid=66).
// It reuses the same tile configuration as P40 but follows the default
// CUTLASS workspace lifecycle used by the regular kernels.
size_t get_fp16_p66_streamk_workspace_size_row_col_col(int m, int n, int k,
                                                       int split_k_slices = 2);
void run_fp16_p66_streamk_row_col_col(
    int m, int n, int k, const cutlass::half_t *A, const cutlass::half_t *B,
    cutlass::half_t *C, float falpha, float fbeta, int split_k_slices = 2,
    cudaStream_t stream = nullptr);

// Explicit C++ API for a StreamK-form validation kernel (pid=42).
// It reuses P0's tile configuration but uses the StreamK/GemmUniversal form.
size_t get_fp16_p42_streamk_workspace_size_row_col_col(int m, int n, int k,
                                                       int split_k_slices = 2);
void run_fp16_p42_streamk_row_col_col(
    int m, int n, int k, const cutlass::half_t *A, const cutlass::half_t *B,
    cutlass::half_t *C, float falpha, float fbeta, int split_k_slices = 2,
    cudaStream_t stream = nullptr);

// Explicit C++ API for neighboring StreamK validation kernels around P40.
size_t get_fp16_p43_streamk_workspace_size_row_col_col(int m, int n, int k,
                                                       int split_k_slices = 2);
void run_fp16_p43_streamk_row_col_col(
    int m, int n, int k, const cutlass::half_t *A, const cutlass::half_t *B,
    cutlass::half_t *C, float falpha, float fbeta, int split_k_slices = 2,
    cudaStream_t stream = nullptr);
size_t get_fp16_p44_streamk_workspace_size_row_col_col(int m, int n, int k,
                                                       int split_k_slices = 2);
void run_fp16_p44_streamk_row_col_col(
    int m, int n, int k, const cutlass::half_t *A, const cutlass::half_t *B,
    cutlass::half_t *C, float falpha, float fbeta, int split_k_slices = 2,
    cudaStream_t stream = nullptr);

// Explicit Pattern2 API for FP16 row-col-col GEMM.
// It splits the N dimension into two regions and launches two CUTLASS GEMMs
// through the patched TwinGemm path.
void run_fp16_pattern2_twin_gemm_row_col_col(
    int m, int n, int k, int split_n, const cutlass::half_t *A,
    const cutlass::half_t *B, cutlass::half_t *C, float falpha, float fbeta,
    cudaStream_t stream = nullptr);

void get_gemm_alignment_int8_row_col_col(int pid, int *alignment);
void get_gemm_shared_mem_int8_row_col_col(int pid, int *shared_mem);
void run_moonpoly_selected_gemm_int8_row_col_col(int pid, int m, int n, int k, const int8_t *A,
                                                 const int8_t *B, int32_t *C,
                                                 int alpha, int beta,
                                                 cudaStream_t stream = nullptr);

void get_gemm_alignment_fp32_row_col_col(int pid, int *alignment);
void get_gemm_shared_mem_fp32_row_col_col(int pid, int *shared_mem);
void run_moonpoly_selected_gemm_fp32_row_col_col(int pid, int m, int n, int k, 
                                                 const float* A,
                                                 const float* B, float*C,
                                                 float falpha, float fbeta,
                                                 cudaStream_t stream = nullptr);

} // namespace moonpoly
