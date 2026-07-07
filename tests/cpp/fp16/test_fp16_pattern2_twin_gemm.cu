#include <cmath>
#include <cstdlib>
#include <iostream>

#include <cublas_v2.h>
#include "cuda_runtime.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/half.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "moonpoly.cuh"

#define CUDA_CHECK(status)                                                     \
  do {                                                                         \
    cudaError_t error = (status);                                              \
    if (error != cudaSuccess) {                                                \
      std::cerr << "CUDA error: " << cudaGetErrorString(error)                \
                << " at line " << __LINE__ << std::endl;                     \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(status)                                                   \
  do {                                                                         \
    cublasStatus_t error = (status);                                           \
    if (error != CUBLAS_STATUS_SUCCESS) {                                      \
      std::cerr << "cuBLAS error at line " << __LINE__ << std::endl;          \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

using Element = cutlass::half_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::ColumnMajor;

bool allclose(cutlass::HostTensor<Element, LayoutC> const &got,
              cutlass::HostTensor<Element, LayoutC> const &ref) {
  for (int i = 0; i < got.capacity(); ++i) {
    float a = static_cast<float>(got.host_data()[i]);
    float b = static_cast<float>(ref.host_data()[i]);
    float diff = std::abs(a - b);
    if (!std::isfinite(a) || diff > 0.25f + 0.02f * std::abs(b)) {
      std::cerr << "Mismatch at " << i << ": got=" << a << " ref=" << b
                << " diff=" << diff << std::endl;
      return false;
    }
  }
  return true;
}

int main(int argc, char **argv) {
  int m = 128;
  int n = 384;
  int k = 128;
  int split_n = 256;
  if (argc == 5) {
    m = std::stoi(argv[1]);
    n = std::stoi(argv[2]);
    k = std::stoi(argv[3]);
    split_n = std::stoi(argv[4]);
  }

  cutlass::gemm::GemmCoord problem(m, n, k);
  cutlass::HostTensor<Element, LayoutA> a(problem.mk());
  cutlass::HostTensor<Element, LayoutB> b(problem.kn());
  cutlass::HostTensor<Element, LayoutC> got(problem.mn());
  cutlass::HostTensor<Element, LayoutC> ref(problem.mn());

  cutlass::reference::host::TensorFillRandomUniform(
      a.host_view(), 1, Element(2.0f), Element(-2.0f), 0);
  cutlass::reference::host::TensorFillRandomUniform(
      b.host_view(), 1, Element(2.0f), Element(-2.0f), 0);
  cutlass::reference::host::TensorFill(got.host_view());
  cutlass::reference::host::TensorFill(ref.host_view());
  a.sync_device();
  b.sync_device();
  got.sync_device();
  ref.sync_device();

  float alpha = 1.0f;
  float beta = 0.0f;
  moonpoly::run_fp16_pattern2_twin_gemm_row_col_col(
      m, n, k, split_n, a.device_data(), b.device_data(), got.device_data(),
      alpha, beta);
  CUDA_CHECK(cudaDeviceSynchronize());

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasGemmEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
      reinterpret_cast<half *>(a.device_data()), CUDA_R_16F, k,
      reinterpret_cast<half *>(b.device_data()), CUDA_R_16F, k, &beta,
      reinterpret_cast<half *>(ref.device_data()), CUDA_R_16F, m, CUDA_R_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  CUDA_CHECK(cudaDeviceSynchronize());
  CUBLAS_CHECK(cublasDestroy(handle));

  got.sync_host();
  ref.sync_host();
  if (!allclose(got, ref)) {
    return 1;
  }
  std::cout << "PASS fp16 Pattern2 correctness" << std::endl;
  return 0;
}
