#include <cmath>
#include <iostream>

#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "moonpoly.cuh"

#define CUDA_CHECK(status)                                                      \
  {                                                                             \
    cudaError_t error = status;                                                 \
    if (error != cudaSuccess) {                                                 \
      std::cerr << "CUDA Error at line " << __LINE__ << ": "                   \
                << cudaGetErrorString(error) << std::endl;                      \
      exit(EXIT_FAILURE);                                                       \
    }                                                                           \
  }

#define CUBLAS_CHECK(status)                                                    \
  {                                                                             \
    cublasStatus_t error = status;                                              \
    if (error != CUBLAS_STATUS_SUCCESS) {                                       \
      std::cerr << "cuBLAS Error at line " << __LINE__ << std::endl;            \
      exit(EXIT_FAILURE);                                                       \
    }                                                                           \
  }

void run_cublas_gemm(cublasHandle_t handle, int m, int n, int k,
                     cutlass::half_t *A, cutlass::half_t *B,
                     cutlass::half_t *C, float alpha, float beta) {
  CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
                            A, CUDA_R_16F, k, B, CUDA_R_16F, k, &beta, C,
                            CUDA_R_16F, m, CUDA_R_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

bool compare_results(cutlass::half_t *actual, cutlass::half_t *expected, int m,
                     int n, float tolerance) {
  using ElementC = cutlass::half_t;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::HostTensor<ElementC, LayoutC> h_actual({m, n});
  cutlass::HostTensor<ElementC, LayoutC> h_expected({m, n});
  CUDA_CHECK(cudaMemcpy(h_actual.host_data(), actual, m * n * sizeof(ElementC),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_expected.host_data(), expected,
                        m * n * sizeof(ElementC), cudaMemcpyDeviceToHost));

  float max_abs = 0.0f;
  int errors = 0;
  int non_finite = 0;
  for (int i = 0; i < m * n; ++i) {
    float a = float(h_actual.host_data()[i]);
    float e = float(h_expected.host_data()[i]);
    if (!std::isfinite(a) || !std::isfinite(e)) {
      ++errors;
      ++non_finite;
      continue;
    }
    float diff = std::abs(a - e);
    max_abs = std::max(max_abs, diff);
    if (diff > tolerance) {
      ++errors;
    }
  }

  std::cout << "max_abs=" << max_abs << ", errors=" << errors
            << ", non_finite=" << non_finite << std::endl;
  return errors == 0;
}

int main(int argc, char **argv) {
  int m = 256;
  int n = 512;
  int k = 256;
  int split_n = 384;
  int warmup = 5;
  int iters = 20;

  if (argc > 1) {
    m = std::stoi(argv[1]);
    n = std::stoi(argv[2]);
    k = std::stoi(argv[3]);
  }
  if (argc > 4) {
    split_n = std::stoi(argv[4]);
  }
  if (argc > 5) {
    warmup = std::stoi(argv[5]);
  }
  if (argc > 6) {
    iters = std::stoi(argv[6]);
  }

  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::HostTensor<ElementA, LayoutA> h_A({m, k});
  cutlass::HostTensor<ElementB, LayoutB> h_B({k, n});
  cutlass::HostTensor<ElementC, LayoutC> h_C({m, n});

  cutlass::reference::host::TensorFillRandomUniform(
      h_A.host_view(), 1, ElementA(2), ElementA(-2), 0);
  cutlass::reference::host::TensorFillRandomUniform(
      h_B.host_view(), 1, ElementB(2), ElementB(-2), 0);
  cutlass::reference::host::TensorFill(h_C.host_view());

  cutlass::device_memory::allocation<ElementA> d_A(h_A.capacity());
  cutlass::device_memory::allocation<ElementB> d_B(h_B.capacity());
  cutlass::device_memory::allocation<ElementC> d_C_pattern2(h_C.capacity());
  cutlass::device_memory::allocation<ElementC> d_C_ref(h_C.capacity());

  d_A.copy_from_host(h_A.host_data());
  d_B.copy_from_host(h_B.host_data());
  d_C_pattern2.copy_from_host(h_C.host_data());
  d_C_ref.copy_from_host(h_C.host_data());

  float alpha = 1.0f;
  float beta = 0.0f;

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  std::cout << "Problem: M=" << m << ", N=" << n << ", K=" << k
            << ", split_n=" << split_n << std::endl;

  moonpoly::run_fp16_pattern2_twin_gemm_row_col_col(
      m, n, k, split_n, d_A.get(), d_B.get(), d_C_pattern2.get(), alpha, beta);
  run_cublas_gemm(handle, m, n, k, d_A.get(), d_B.get(), d_C_ref.get(), alpha,
                  beta);
  CUDA_CHECK(cudaDeviceSynchronize());

  bool ok = compare_results(d_C_pattern2.get(), d_C_ref.get(), m, n, 1.0f);

  for (int i = 0; i < warmup; ++i) {
    moonpoly::run_fp16_pattern2_twin_gemm_row_col_col(
        m, n, k, split_n, d_A.get(), d_B.get(), d_C_pattern2.get(), alpha,
        beta);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    moonpoly::run_fp16_pattern2_twin_gemm_row_col_col(
        m, n, k, split_n, d_A.get(), d_B.get(), d_C_pattern2.get(), alpha,
        beta);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float pattern2_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&pattern2_ms, start, stop));

  for (int i = 0; i < warmup; ++i) {
    run_cublas_gemm(handle, m, n, k, d_A.get(), d_B.get(), d_C_ref.get(),
                    alpha, beta);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    run_cublas_gemm(handle, m, n, k, d_A.get(), d_B.get(), d_C_ref.get(),
                    alpha, beta);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float cublas_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&cublas_ms, start, stop));

  double pattern2_avg = double(pattern2_ms) / double(iters);
  double cublas_avg = double(cublas_ms) / double(iters);
  std::cout << "Pattern2 avg ms: " << pattern2_avg << std::endl;
  std::cout << "cuBLAS avg ms: " << cublas_avg << std::endl;
  std::cout << "speedup: " << (cublas_avg / pattern2_avg) << "x" << std::endl;
  std::cout << (ok ? "PASS" : "FAIL") << std::endl;

  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ok ? 0 : 1;
}
