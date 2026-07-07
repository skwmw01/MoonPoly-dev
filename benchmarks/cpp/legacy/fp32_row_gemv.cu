#include <iostream>
#include <sstream>

#include "cuda_runtime.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemv.h"
#include "cutlass/gemm/kernel/gemv.h"
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

using ElementInputA = float;
using ElementInputB = float;
using ElementOutput = float;

using LayoutInputA = cutlass::layout::RowMajor;
using LayoutV = cutlass::layout::RowMajor;

int run(cublasHandle_t handle, int length_m, int length_n, int length_k,
        double &cb_time, double &ct_time) {
  assert (length_n == 1);
  
  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);

  // Initialize tensors using CUTLASS helper functions
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a; // <- Create matrix A with dimensions M x K
  cutlass::HostTensor<ElementInputB, LayoutV> tensor_b;     // <- Create matrix B with dimensions K x 1
  cutlass::HostTensor<ElementOutput, LayoutV> tensor_c; // <- Create matrix C with dimensions M x 1
  cutlass::HostTensor<ElementOutput, LayoutV> tensor_d; // <- Create matrix D with dimensions M x 1 used to
                          // store output from CUTLASS kernel
  cutlass::HostTensor<ElementOutput, LayoutV> tensor_ref_d;

  tensor_a.resize({length_m, length_k});
  tensor_b.resize({length_k, 1});
  tensor_c.resize({length_m, 1});
  tensor_d.resize({length_m, 1});
  tensor_ref_d.resize({length_m, 1});

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

  float falpha = 1, fbeta = 0;

  // Launch moonpoly gemv kernel
  moonpoly::run_moonpoly_gemv_fp32_row(length_m, length_k, tensor_a,
                                       tensor_a.device_data(), tensor_b.device_data(),
                                       tensor_d.device_data(), falpha, fbeta);
  cudaDeviceSynchronize();
  tensor_d.sync_host();

  // Launch cublas gemv kernel
  half *A = reinterpret_cast<half *>(tensor_a.device_data());
  half *B = reinterpret_cast<half *>(tensor_b.device_data());
  half *C = reinterpret_cast<half *>(tensor_ref_d.device_data());
  
  cublasMath_t cublas_flags = CUBLAS_DEFAULT_MATH;
  cublasSetMathMode(handle, cublas_flags);
  int lda = length_n, ldb = length_k, ldc = length_n;
  cublasStatus_t cu_status = cublasGemmEx(
                              handle, CUBLAS_OP_N, CUBLAS_OP_N, length_n, length_m, length_k, &falpha,
                              B, CUDA_R_32F, lda, A, CUDA_R_32F, ldb, &fbeta, C, CUDA_R_32F, ldc,
                              CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  cudaDeviceSynchronize();
  if (cu_status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "Got cublas error: RunTime Error"
              << " at: " << __LINE__ << std::endl;
    exit(EXIT_FAILURE);
  }

  // check result
  tensor_ref_d.sync_host();
  std::cout << "cublas gemv done" << std::endl;
  

  // out put result
  // tensor_a.sync_host();
  // tensor_b.sync_host();
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


  // Warm up
  for (int iter = 0; iter < 50; ++iter) {
    moonpoly::run_moonpoly_gemv_fp32_row(length_m, length_k, tensor_a,
                                         tensor_a.device_data(), tensor_b.device_data(),
                                         tensor_d.device_data(), falpha, fbeta);
    cudaDeviceSynchronize();
    cublasStatus_t cu_status = cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, length_n, length_m, length_k, &falpha,
        B, CUDA_R_32F, lda, A, CUDA_R_32F, ldb, &fbeta, C, CUDA_R_32F, ldc,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaDeviceSynchronize();
  }

  // test cublas time
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  double total_time = 0.0;
  float milliseconds = 0.0;
  int num_iter = 1000;
  for (int i = 0; i < num_iter; ++i) {
    cudaEventRecord(start);
    cublasStatus_t cu_status = cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, length_n, length_m, length_k, &falpha,
        B, CUDA_R_32F, lda, A, CUDA_R_32F, ldb, &fbeta, C, CUDA_R_32F, ldc,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
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
    moonpoly::run_moonpoly_gemv_fp32_row(length_m, length_k, tensor_a,
                                         tensor_a.device_data(), tensor_b.device_data(),
                                         tensor_d.device_data(), falpha, fbeta);
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

int main(int argc, char **argv) {

  const int num_case = 1;
  std::cout << "Test gemv input size:" << std::endl;
  int m_arr[]= {4096};
  int k_arr[] = {4096};
  if (argc > 1) {
    m_arr[0] = std::stoi(argv[1]);
    k_arr[0] = std::stoi(argv[2]);
  }

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
    int passed = run(handle, m_arr[i], 1, k_arr[i], cb_time, ct_time);

    if (passed == -1) {
      std::cout << "Failed," << std::flush;
      break;
    } else {
      std::cout << "(m, n, k) = (" << m_arr[i] << ", " << 1 << ", "
                << k_arr[i] << ")" << std::endl;
      std::cout << "cublas time: " << cb_time << ", moonpoly time: " << ct_time << ", speed_up: " 
                << std::fixed << std::setprecision(3) << cb_time / ct_time << "x "<< std::endl;
    }
  }

  std::cout << std::endl << std::flush;
  return 0;
}
