#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

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
              cutlass::HostTensor<Element, LayoutC> const &ref,
              float atol = 0.25f, float rtol = 0.02f) {
  for (int i = 0; i < got.capacity(); ++i) {
    float a = static_cast<float>(got.host_data()[i]);
    float b = static_cast<float>(ref.host_data()[i]);
    float diff = std::abs(a - b);
    if (!std::isfinite(a) || diff > atol + rtol * std::abs(b)) {
      std::cerr << "Mismatch at " << i << ": got=" << a << " ref=" << b
                << " diff=" << diff << std::endl;
      return false;
    }
  }
  return true;
}

int run_one(cublasHandle_t handle, int m, int n, int k) {
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
  moonpoly::run_fp16_gemm(m, n, k, a.device_data(), false, b.device_data(),
                          true, got.device_data(), true, alpha, beta);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUBLAS_CHECK(cublasGemmEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
      reinterpret_cast<half *>(a.device_data()), CUDA_R_16F, k,
      reinterpret_cast<half *>(b.device_data()), CUDA_R_16F, k, &beta,
      reinterpret_cast<half *>(ref.device_data()), CUDA_R_16F, m, CUDA_R_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  CUDA_CHECK(cudaDeviceSynchronize());

  got.sync_host();
  ref.sync_host();
  if (!allclose(got, ref)) {
    std::cerr << "FP16 RCC correctness failed for shape (" << m << ", " << n
              << ", " << k << ")" << std::endl;
    return 1;
  }
  return 0;
}

int main(int argc, char **argv) {
  std::vector<cutlass::gemm::GemmCoord> shapes = {
      {16, 32, 64},
      {64, 128, 128},
      {128, 256, 256},
  };
  if (argc == 4) {
    shapes = {{std::stoi(argv[1]), std::stoi(argv[2]), std::stoi(argv[3])}};
  }

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

  int rc = 0;
  for (auto const &shape : shapes) {
    rc |= run_one(handle, shape.m(), shape.n(), shape.k());
  }

  CUBLAS_CHECK(cublasDestroy(handle));
  if (rc == 0) {
    std::cout << "PASS fp16 row-col-col correctness" << std::endl;
  }
  return rc;
}
