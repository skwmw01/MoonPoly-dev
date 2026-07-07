#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <cublas_v2.h>
#include "cuda_runtime.h"
#include "cutlass/gemm/gemm.h"
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

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::ColumnMajor;

bool allclose(cutlass::HostTensor<float, LayoutC> const &got,
              cutlass::HostTensor<float, LayoutC> const &ref) {
  for (int i = 0; i < got.capacity(); ++i) {
    float a = got.host_data()[i];
    float b = ref.host_data()[i];
    float diff = std::abs(a - b);
    if (!std::isfinite(a) || diff > 1.0e-3f + 1.0e-4f * std::abs(b)) {
      std::cerr << "Mismatch at " << i << ": got=" << a << " ref=" << b
                << " diff=" << diff << std::endl;
      return false;
    }
  }
  return true;
}

int run_one(cublasHandle_t handle, int m, int n, int k, bool force_splitk) {
  cutlass::gemm::GemmCoord problem(m, n, k);
  cutlass::HostTensor<float, LayoutA> a(problem.mk());
  cutlass::HostTensor<float, LayoutB> b(problem.kn());
  cutlass::HostTensor<float, LayoutC> got(problem.mn());
  cutlass::HostTensor<float, LayoutC> ref(problem.mn());

  cutlass::reference::host::TensorFillRandomUniform(
      a.host_view(), 1, 2.0f, -2.0f, 0);
  cutlass::reference::host::TensorFillRandomUniform(
      b.host_view(), 1, 2.0f, -2.0f, 0);
  cutlass::reference::host::TensorFill(got.host_view());
  cutlass::reference::host::TensorFill(ref.host_view());
  a.sync_device();
  b.sync_device();
  got.sync_device();
  ref.sync_device();

  float alpha = 1.0f;
  float beta = 0.0f;
  if (force_splitk) {
    setenv("MOONPOLY_FP32_PATTERN3_FORCE_SPLITK", "2", 1);
  }
  moonpoly::run_fp32_gemm(m, n, k, a.device_data(), false, b.device_data(),
                          true, got.device_data(), true, alpha, beta);
  if (force_splitk) {
    unsetenv("MOONPOLY_FP32_PATTERN3_FORCE_SPLITK");
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
                           a.device_data(), k, b.device_data(), k, &beta,
                           ref.device_data(), m));
  CUDA_CHECK(cudaDeviceSynchronize());

  got.sync_host();
  ref.sync_host();
  if (!allclose(got, ref)) {
    std::cerr << "FP32 RCC correctness failed for shape (" << m << ", " << n
              << ", " << k << "), force_splitk=" << force_splitk << std::endl;
    return 1;
  }
  return 0;
}

int main(int argc, char **argv) {
  std::vector<cutlass::gemm::GemmCoord> shapes = {
      {16, 32, 64},
      {64, 128, 128},
  };
  if (argc == 4) {
    shapes = {{std::stoi(argv[1]), std::stoi(argv[2]), std::stoi(argv[3])}};
  }

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  int rc = 0;
  for (auto const &shape : shapes) {
    rc |= run_one(handle, shape.m(), shape.n(), shape.k(), false);
    rc |= run_one(handle, shape.m(), shape.n(), shape.k(), true);
  }
  CUBLAS_CHECK(cublasDestroy(handle));

  if (rc == 0) {
    std::cout << "PASS fp32 row-col-col correctness" << std::endl;
  }
  return rc;
}
