#include <iostream>
#include <iomanip>
#include <sstream>

#include "cuda_runtime.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "moonpoly.cuh"
#include <cublas_v2.h>

#define CUTLASS_CHECK(status)                                                  \
  {                                                                            \
    cutlass::Status error = status;                                            \
    if (error != cutlass::Status::kSuccess) {                                  \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error)      \
                << " at: " << __LINE__ << std::endl;                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#define CUDA_CHECK(status)                                                     \
  {                                                                            \
    cudaError_t error = status;                                                \
    if (error != cudaSuccess) {                                                \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error)        \
                << " at line: " << __LINE__ << std::endl;                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

using ElementInputA = int8_t;
using ElementInputB = int8_t;
using ElementOutput = int32_t;
using ElementComputeEpilogue = int32_t;

using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::ColumnMajor;

int run(cublasHandle_t handle, int length_m, int length_n, int length_k,
        double &cb_time, double &ct_time) {
  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);

  // Initialize tensors using CUTLASS helper functions
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
      problem_size.mk()); // <- Create matrix A with dimensions M x K
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
      problem_size.kn()); // <- Create matrix B with dimensions K x N
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(
      problem_size.mn()); // <- Create matrix C with dimensions M x N
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(
      problem_size.mn()); // <- Create matrix D with dimensions M x N used to
                          // store output from CUTLASS kernel
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(
      problem_size.mn());

  // Fill input and output matrices on host using CUTLASS helper functions
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_a.host_view(), 1, ElementInputA(4), ElementInputA(-4),
      0); // <- Fill matrix A on host with uniform-distribution random data

  cutlass::reference::host::TensorFillRandomUniform(
      tensor_b.host_view(), 1, ElementInputB(4), ElementInputB(-4),
      0); // <- Fill matrix B on host with uniform-distribution random data

  cutlass::reference::host::TensorFillRandomUniform(
      tensor_c.host_view(), 1, ElementOutput(4), ElementOutput(-4),
      0); // <- Fill matrix C on host with uniform-distribution random data
  cutlass::reference::host::TensorFill(
      tensor_d.host_view()); // <- fill matrix D on host with zeros
  cutlass::reference::host::TensorFill(
      tensor_ref_d.host_view()); // <- fill matrix D on host with zeros

  // Copy data from host to GPU
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();
  tensor_d.sync_device();
  tensor_ref_d.sync_device();

  int alpha = 1, beta = 0;
  // std::cout << "m: " << length_m << ", n: " << length_n << ", k: " << length_k
  //           << std::endl;
  // Launch moonpoly gemm kernel
  moonpoly::run_moonpoly_lp_gemm(length_m, length_n, length_k,
                               tensor_a.device_data(), tensor_b.device_data(),
                               tensor_d.device_data(), alpha, beta);
  cudaDeviceSynchronize();
  tensor_d.sync_host();
  // std::cout << "moonpoly lp gemm done" << std::endl;
  
  // Launch cublas gemm kernel
  int8_t *A = reinterpret_cast<int8_t *>(tensor_a.device_data()); // M x K RowMajor
  int8_t *B = reinterpret_cast<int8_t *>(tensor_b.device_data()); // K x N ColumnMajor
  int32_t *C = reinterpret_cast<int32_t *>(tensor_ref_d.device_data()); // M x N ColumnMajor

  cublasMath_t cublas_flags = CUBLAS_TENSOR_OP_MATH;
  cublasSetMathMode(handle, cublas_flags);
  int lda = length_k, ldb = length_k, ldc = length_m;
  cublasStatus_t cu_status = cublasGemmEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, length_n, length_m, length_k, &alpha,
      B, CUDA_R_8I, lda, A, CUDA_R_8I, ldb, &beta, C, CUDA_R_32I, ldc,
      CUDA_R_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  cudaDeviceSynchronize();
  if (cu_status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "Got cublas error: RunTime Error"
              << " at: " << __LINE__ << std::endl;
    exit(EXIT_FAILURE);
  }
  tensor_ref_d.sync_host();
  // std::cout << "cublas gemm done" << std::endl;
  
  // tensor_a.sync_host();
  // tensor_b.sync_host();
  // check result
  // std::cout << "Tensor A" << std::endl;
  // for (int row = 0; row < length_m; ++row) {
  //   for (int col = 0; col < length_k; ++col) {
  //     std::cout << int(tensor_a.at({row, col})) << " ";
  //   }
  //   std::cout << std::endl;
  // }

  // std::cout << "Tensor B" << std::endl;
  // for (int row = 0; row < length_k; ++row) {
  //   for (int col = 0; col < length_n; ++col) {
  //     std::cout << int(tensor_b.at({row, col})) << " ";
  //   }
  //   std::cout << std::endl;
  // }
  
  // std::cout << "Tensor D" << std::endl;
  // for (int row = 0; row < length_m; ++row) {
  //   for (int col = 0; col < length_n; ++col) {
  //     std::cout << tensor_d.at({row, col}) << " ";
  //   }
  //   std::cout << std::endl;
  // }

  // std::cout << "Tensor Ref" << std::endl;
  // for (int row = 0; row < length_m; ++row) {
  //   for (int col = 0; col < length_n; ++col) {
  //       std::cout << (tensor_ref_d.at({row, col}) )<< " ";
  //   }
  //   std::cout << std::endl;
  // }
  bool passed = cutlass::reference::host::TensorEquals(
      tensor_d.host_view(), tensor_ref_d.host_view());
  if (!passed) {
    return -1;
  }

  // test cublas time
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  double total_time = 0.0;
  float milliseconds = 0.0;
  int num_iter = 20;
  for (int i = 0; i < num_iter; ++i) {
    cudaEventRecord(start);
    cublasStatus_t cu_status = cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, length_n, length_m, length_k, &alpha,
        B, CUDA_R_8I, lda, A, CUDA_R_8I, ldb, &beta, C, CUDA_R_32I, ldc,
        CUDA_R_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (cu_status != CUBLAS_STATUS_SUCCESS) {
      std::cerr << "Got cublas error: RunTime Error"
                << " at: " << __LINE__ << std::endl;
      exit(EXIT_FAILURE);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    total_time += milliseconds;
  }
  cudaDeviceSynchronize();
  cb_time = total_time / (num_iter * 1.0);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaDeviceSynchronize();

  // test moonpoly time
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  total_time = 0.0;
  milliseconds = 0.0;
  for (int i = 0; i < num_iter; ++i) {
    cudaEventRecord(start);
    moonpoly::run_moonpoly_lp_gemm(length_m, length_n, length_k,
                                   tensor_a.device_data(), tensor_b.device_data(),
                                   tensor_d.device_data(), alpha, beta);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    total_time += milliseconds;
  }
  cudaDeviceSynchronize();
  ct_time = total_time / (num_iter * 1.0);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}

int main() {
  int num_case = 3;
  int m_arr[] = {365, 5120, 5120};
  int n_arr[] = {1264, 100, 100};
  int k_arr[] = {364, 2560, 6912};

  cublasHandle_t handle;
  cublasStatus_t status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
      std::cerr << "Got cublas error: CUBLAS_STATUS_NOT_INITIALIZED"
                << " at: " << __LINE__ << std::endl;
    }
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < num_case; i++) {
    double cb_time, ct_time;
    int passed = run(handle, m_arr[i], n_arr[i], k_arr[i], cb_time, ct_time);

    if (passed == -1) {
      std::cout << "Failed," << std::flush;
      break;
    } else {
      std::cout << "(m, n, k) = (" << m_arr[i] << ", " << n_arr[i] << ", "
                << k_arr[i] << ")" << std::endl;
      std::cout << "cublas time: " << cb_time << ", moonpoly time: " << ct_time << ", speed_up: " 
                << std::fixed << std::setprecision(3) << cb_time / ct_time << "x "<< std::endl;
    }
  }

  std::cout << std::endl << std::flush; 
  return 0;
}
