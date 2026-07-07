#include <cmath>
#include <iostream>

#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "moonpoly.cuh"

#define CUDA_CHECK(status)                                                      \
  {                                                                             \
    cudaError_t error = status;                                                 \
    if (error != cudaSuccess) {                                                 \
      std::cerr << "CUDA Error at line " << __LINE__ << ": "                    \
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
                     cutlass::half_t *ptr_A, cutlass::half_t *ptr_B,
                     cutlass::half_t *ptr_C, float alpha, float beta) {
  CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
                            ptr_A, CUDA_R_16F, k, ptr_B, CUDA_R_16F, k, &beta,
                            ptr_C, CUDA_R_16F, m, CUDA_R_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

bool compare_results(cutlass::half_t *result_a, cutlass::half_t *result_b, int m,
                     int n, float tolerance = 1e-3f) {
  using ElementC = cutlass::half_t;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::HostTensor<ElementC, LayoutC> h_a({m, n});
  cutlass::HostTensor<ElementC, LayoutC> h_b({m, n});
  CUDA_CHECK(cudaMemcpy(h_a.host_data(), result_a, m * n * sizeof(ElementC),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_b.host_data(), result_b, m * n * sizeof(ElementC),
                        cudaMemcpyDeviceToHost));

  float max_diff = 0.0f;
  int error_count = 0;
  int nan_count = 0;
  int inf_count = 0;
  for (int i = 0; i < m * n; ++i) {
    float va = float(h_a.host_data()[i]);
    float vb = float(h_b.host_data()[i]);
    if (std::isnan(va) || std::isnan(vb)) {
      ++nan_count;
      ++error_count;
      continue;
    }
    if (std::isinf(va) || std::isinf(vb)) {
      ++inf_count;
      ++error_count;
      continue;
    }
    float diff = std::abs(va - vb);
    max_diff = std::max(max_diff, diff);
    if (diff > tolerance) {
      ++error_count;
    }
  }
  std::cout << "Result Comparison:" << std::endl;
  std::cout << "  Max absolute difference: " << max_diff << std::endl;
  std::cout << "  NaN values: " << nan_count << std::endl;
  std::cout << "  Inf values: " << inf_count << std::endl;
  std::cout << "  Errors (tolerance=" << tolerance << "): " << error_count
            << "/" << (m * n) << std::endl;
  return error_count == 0;
}

int main(int argc, char **argv) {
  if (argc != 7) {
    std::cerr << "Usage: " << argv[0]
              << " <M> <N> <K> <split_k> <warmup> <iters>" << std::endl;
    return -1;
  }

  int M = std::stoi(argv[1]);
  int N = std::stoi(argv[2]);
  int K = std::stoi(argv[3]);
  int split_k = std::stoi(argv[4]);
  int warmup = std::stoi(argv[5]);
  int iters = std::stoi(argv[6]);
  float alpha = 1.0f;
  float beta = 0.0f;

  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = cutlass::half_t;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::HostTensor<ElementA, LayoutA> h_A({M, K});
  cutlass::HostTensor<ElementB, LayoutB> h_B({K, N});
  cutlass::HostTensor<ElementC, LayoutC> h_C({M, N});
  cutlass::reference::host::TensorFillRandomUniform(
      h_A.host_view(), 1, cutlass::half_t(4), cutlass::half_t(-4), 0);
  cutlass::reference::host::TensorFillRandomUniform(
      h_B.host_view(), 1, cutlass::half_t(4), cutlass::half_t(-4), 0);
  cutlass::reference::host::TensorFillRandomUniform(
      h_C.host_view(), 1, cutlass::half_t(4), cutlass::half_t(-4), 0);

  cutlass::device_memory::allocation<ElementA> d_A(h_A.capacity());
  cutlass::device_memory::allocation<ElementB> d_B(h_B.capacity());
  cutlass::device_memory::allocation<ElementC> d_C_p40(h_C.capacity());
  cutlass::device_memory::allocation<ElementC> d_C_ref(h_C.capacity());
  d_A.copy_from_host(h_A.host_data());
  d_B.copy_from_host(h_B.host_data());
  d_C_p40.copy_from_host(h_C.host_data());
  d_C_ref.copy_from_host(h_C.host_data());

  size_t ws_bytes =
      moonpoly::get_fp16_p40_streamk_workspace_size_row_col_col(M, N, K, split_k);
  cutlass::device_memory::allocation<uint8_t> workspace(ws_bytes);

  std::cout << "Problem: M=" << M << ", N=" << N << ", K=" << K
            << ", split_k=" << split_k << std::endl;
  std::cout << "Workspace bytes: " << ws_bytes << std::endl;

  for (int i = 0; i < warmup; ++i) {
    moonpoly::run_fp16_p40_streamk_row_col_col(
        M, N, K, d_A.get(), d_B.get(), d_C_p40.get(), alpha, beta,
        workspace.get(), ws_bytes, split_k, nullptr);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    moonpoly::run_fp16_p40_streamk_row_col_col(
        M, N, K, d_A.get(), d_B.get(), d_C_p40.get(), alpha, beta,
        workspace.get(), ws_bytes, split_k, nullptr);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float p40_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&p40_ms, start, stop));
  std::cout << "P40 avg ms: " << (p40_ms / iters) << std::endl;

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  for (int i = 0; i < warmup; ++i) {
    run_cublas_gemm(handle, M, N, K, d_A.get(), d_B.get(), d_C_ref.get(), alpha,
                    beta);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    run_cublas_gemm(handle, M, N, K, d_A.get(), d_B.get(), d_C_ref.get(), alpha,
                    beta);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ref_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ref_ms, start, stop));
  std::cout << "cuBLAS avg ms: " << (ref_ms / iters) << std::endl;

  bool ok = compare_results(d_C_p40.get(), d_C_ref.get(), M, N);
  std::cout << (ok ? "PASS" : "FAIL") << std::endl;

  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ok ? 0 : 1;
}
