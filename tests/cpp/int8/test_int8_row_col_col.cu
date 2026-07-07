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

bool equal(cutlass::HostTensor<int32_t, LayoutC> const &got,
           cutlass::HostTensor<int32_t, LayoutC> const &ref) {
  for (int i = 0; i < got.capacity(); ++i) {
    int32_t a = got.host_data()[i];
    int32_t b = ref.host_data()[i];
    if (a != b) {
      std::cerr << "Mismatch at " << i << ": got=" << a << " ref=" << b
                << std::endl;
      return false;
    }
  }
  return true;
}

int run_one(cublasHandle_t handle, int m, int n, int k, bool force_splitk) {
  cutlass::gemm::GemmCoord problem(m, n, k);
  cutlass::HostTensor<int8_t, LayoutA> a(problem.mk());
  cutlass::HostTensor<int8_t, LayoutB> b(problem.kn());
  cutlass::HostTensor<int32_t, LayoutC> got(problem.mn());
  cutlass::HostTensor<int32_t, LayoutC> ref(problem.mn());

  cutlass::reference::host::TensorFillRandomUniform(
      a.host_view(), 1, int8_t(2), int8_t(-2), 0);
  cutlass::reference::host::TensorFillRandomUniform(
      b.host_view(), 1, int8_t(2), int8_t(-2), 0);
  cutlass::reference::host::TensorFill(got.host_view());
  cutlass::reference::host::TensorFill(ref.host_view());
  a.sync_device();
  b.sync_device();
  got.sync_device();
  ref.sync_device();

  int alpha = 1;
  int beta = 0;
  if (force_splitk) {
    setenv("MOONPOLY_INT8_PATTERN3_FORCE_SPLITK", "2", 1);
  }
  moonpoly::run_int8_gemm(m, n, k, a.device_data(), false, b.device_data(),
                          true, got.device_data(), true, alpha, beta);
  if (force_splitk) {
    unsetenv("MOONPOLY_INT8_PATTERN3_FORCE_SPLITK");
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
                            a.device_data(), CUDA_R_8I, k, b.device_data(),
                            CUDA_R_8I, k, &beta, ref.device_data(), CUDA_R_32I,
                            m, CUDA_R_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  CUDA_CHECK(cudaDeviceSynchronize());

  got.sync_host();
  ref.sync_host();
  if (!equal(got, ref)) {
    std::cerr << "INT8 RCC correctness failed for shape (" << m << ", " << n
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
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  int rc = 0;
  for (auto const &shape : shapes) {
    rc |= run_one(handle, shape.m(), shape.n(), shape.k(), false);
    rc |= run_one(handle, shape.m(), shape.n(), shape.k(), true);
  }

  CUBLAS_CHECK(cublasDestroy(handle));
  if (rc == 0) {
    std::cout << "PASS int8 row-col-col correctness" << std::endl;
  }
  return rc;
}
