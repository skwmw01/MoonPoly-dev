#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>

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

using ElementInputA = cutlass::half_t;
using ElementInputB = cutlass::half_t;
using ElementOutput = cutlass::half_t;
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::ColumnMajor;


// 计算两个张量之间的余弦相似度差异
double calc_cosine_diff(const cutlass::HostTensor<ElementOutput, LayoutOutput>& x,
                        const cutlass::HostTensor<ElementOutput, LayoutOutput>& y) {
    double x_sum = 0.0, y_sum = 0.0, xy_sum = 0.0;

    for (int i = 0; i < x.capacity(); ++i) {
        double x_val = static_cast<double>(x.host_data()[i]);
        double y_val = static_cast<double>(y.host_data()[i]);

        x_sum += x_val * x_val;
        y_sum += y_val * y_val;
        xy_sum += x_val * y_val;
    }

    double denominator = x_sum + y_sum;
    if (denominator == 0.0) return 1.0; // 完全不相似

    double sim = 2.0 * xy_sum / denominator;
    return 1.0 - sim;
}

int run(cublasHandle_t handle, int pid, int length_m, int length_n, int length_k,
        double &cb_time, double &ct_time) {
  // Create CUDA events for timing
  cudaEvent_t start_moonpoly, stop_moonpoly, start_cublas, stop_cublas;
  CUDA_CHECK(cudaEventCreate(&start_moonpoly));
  CUDA_CHECK(cudaEventCreate(&stop_moonpoly));
  CUDA_CHECK(cudaEventCreate(&start_cublas));
  CUDA_CHECK(cudaEventCreate(&stop_cublas));

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

  float falpha = 1, fbeta = 0;
  const int warmup_iterations = 5;
  const int timing_iterations = 10;

  // Warm up for moonpoly
  for (int i = 0; i < warmup_iterations; i++) {
    moonpoly::run_moonpoly_selected_gemm_fp16_row_col_col(pid, length_m, length_n, length_k,
                                                          tensor_a.device_data(), tensor_b.device_data(),
                                                          tensor_d.device_data(), falpha, fbeta);
  }
  cudaDeviceSynchronize();

  // Launch moonpoly gemm kernel with timing (multiple iterations)
  float total_moonpoly_time_ms = 0;
  CUDA_CHECK(cudaEventRecord(start_moonpoly));
  for (int i = 0; i < timing_iterations; i++) {
    moonpoly::run_moonpoly_selected_gemm_fp16_row_col_col(pid, length_m, length_n, length_k,
                                                          tensor_a.device_data(), tensor_b.device_data(),
                                                          tensor_d.device_data(), falpha, fbeta);
  }
  CUDA_CHECK(cudaEventRecord(stop_moonpoly));
  CUDA_CHECK(cudaEventSynchronize(stop_moonpoly));
  float moonpoly_time_ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&moonpoly_time_ms, start_moonpoly, stop_moonpoly));
  total_moonpoly_time_ms = moonpoly_time_ms;
  ct_time = total_moonpoly_time_ms / timing_iterations;
  tensor_d.sync_host();
  std::cout << "moonpoly gemm done!" << std::endl;

  // Launch cublas gemm kernel
  half *A = reinterpret_cast<half *>(tensor_a.device_data());
  half *B = reinterpret_cast<half *>(tensor_b.device_data());
  half *C = reinterpret_cast<half *>(tensor_ref_d.device_data());
  cublasMath_t cublas_flags = CUBLAS_DEFAULT_MATH;
  cublasSetMathMode(handle, cublas_flags);
  int lda = length_k, ldb = length_k, ldc = length_m;

  // Warm up for cublas
  cublasStatus_t cu_status;
  for (int i = 0; i < warmup_iterations; i++) {
    cu_status = cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, length_m, length_n, length_k, &falpha,
        A, CUDA_R_16F, lda, B, CUDA_R_16F, ldb, &fbeta, C, CUDA_R_16F, ldc,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  }
  cudaDeviceSynchronize();

  // Timing run (multiple iterations)
  float total_cublas_time_ms = 0;
  CUDA_CHECK(cudaEventRecord(start_cublas));
  for (int i = 0; i < timing_iterations; i++) {
    cu_status = cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, length_m, length_n, length_k, &falpha,
        A, CUDA_R_16F, lda, B, CUDA_R_16F, ldb, &fbeta, C, CUDA_R_16F, ldc,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (cu_status != CUBLAS_STATUS_SUCCESS) {
      std::cerr << "Got cublas error: RunTime Error"
                << " at: " << __LINE__ << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  CUDA_CHECK(cudaEventRecord(stop_cublas));
  CUDA_CHECK(cudaEventSynchronize(stop_cublas));

  float cublas_time_ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&cublas_time_ms, start_cublas, stop_cublas));
  total_cublas_time_ms = cublas_time_ms;
  cb_time = total_cublas_time_ms / timing_iterations;

  // check result
  tensor_ref_d.sync_host();

  // 总是打印前10个元素用于人工校对
  std::cout << "\n=== First 10 elements for manual verification ===" << std::endl;
  std::cout << "---------------------------------------------------------------------------------------------" << std::endl;
  std::cout << "Index      | Tensor D (moonpoly) | Tensor Ref D (cublas)   | Abs Diff         | Status" << std::endl;
  std::cout << "---------------------------------------------------------------------------------------------" << std::endl;

  int elements_to_show = std::min(10, (int)tensor_d.capacity());
  for (int i = 0; i < elements_to_show; ++i) {
    float val_d = static_cast<float>(tensor_d.host_data()[i]);
    float val_ref_d = static_cast<float>(tensor_ref_d.host_data()[i]);
    float abs_diff = std::abs(val_d - val_ref_d);

    std::string status = "OK";
    if (std::isnan(val_d)) {
      status = "NaN!";
    } else if (std::isinf(val_d)) {
      status = "Inf!";
    } else if (abs_diff > 1e-3f) {
      status = "Diff!";
    }

    std::cout << std::setw(10) << i << " | "
              << std::fixed << std::setprecision(6) << std::setw(20) << val_d << " | "
              << std::setw(24) << val_ref_d << " | "
              << std::setw(16) << abs_diff << " | "
              << std::setw(10) << status << std::endl;
  }
  std::cout << "---------------------------------------------------------------------------------------------" << std::endl;

  // 计算余弦相似度差异
  double cosine_diff = calc_cosine_diff(tensor_d, tensor_ref_d);
  std::cout << "\nCosine similarity difference: " << std::fixed << std::setprecision(6)
            << cosine_diff << std::endl;

  if (cosine_diff > 1e-3f || std::isnan(cosine_diff)) {
    std::cout << "\nError: Results do not match!" << std::endl;

    // 定义一个小的容差，用于浮点数比较
    const float tolerance = 1e-6f;
    int mismatches_found = 0;
    int nan_count = 0;
    int inf_count = 0;
    const int max_mismatches_to_print = 10;

    // 统计所有差异
    for (int i = 0; i < tensor_d.capacity(); ++i) {
        float val_d = static_cast<float>(tensor_d.host_data()[i]);
        float val_ref_d = static_cast<float>(tensor_ref_d.host_data()[i]);

        if (std::isnan(val_d)) {
            nan_count++;
        } else if (std::isinf(val_d)) {
            inf_count++;
        }

        float abs_diff = std::abs(val_d - val_ref_d);
        float rel_diff = 0.0f;

        // 计算相对差异，避免除以零
        if (std::abs(val_ref_d) > 1e-6) { // 仅当参考值不接近零时计算
            rel_diff = abs_diff / std::abs(val_ref_d);
        }

        // 统计超过容差的差异
        if (abs_diff > tolerance || rel_diff > tolerance || std::isnan(val_d) || std::isinf(val_d)) {
            mismatches_found++;
        }
    }

    std::cout << "\nSummary:" << std::endl;
    std::cout << "Total mismatches: " << mismatches_found << " out of " << tensor_d.capacity() << " elements" << std::endl;
    if (nan_count > 0) {
        std::cout << "NaN values found: " << nan_count << std::endl;
    }
    if (inf_count > 0) {
        std::cout << "Inf values found: " << inf_count << std::endl;
    }

    return -1;
  }

  std::cout << "\nResult: PASSED" << std::endl;

  // Clean up CUDA events
  CUDA_CHECK(cudaEventDestroy(start_moonpoly));
  CUDA_CHECK(cudaEventDestroy(stop_moonpoly));
  CUDA_CHECK(cudaEventDestroy(start_cublas));
  CUDA_CHECK(cudaEventDestroy(stop_cublas));

  return 0;
}

int main(int argc, char **argv) {
  const int num_case = 1;
  int m_arr[]= {5120};
  int n_arr[] = {4};
  int k_arr[] = {5120};
  int pid = 40;
  if (argc > 1) {
    // std::istringstream iss(argv[1]);
    m_arr[0] = std::stoi(argv[1]);
    n_arr[0] = std::stoi(argv[2]);
    k_arr[0] = std::stoi(argv[3]);
    pid = std::stoi(argv[4]);
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
    int passed = run(handle, pid, m_arr[i], n_arr[i], k_arr[i], cb_time, ct_time);

    if (passed == -1) {
      std::cout << "Failed," << std::flush;
      break;
    } else {
      std::cout << "(m, n, k) = (" << m_arr[i] << ", " << n_arr[i] << ", "
                << k_arr[i] << ")" << std::endl;
      std::cout << "cublas time: " << cb_time << " ms, moonpoly time: " << ct_time << " ms, speed_up: "
                << std::fixed << std::setprecision(3) << cb_time / ct_time << "x "<< std::endl;
    }
  }

  std::cout << std::endl << std::flush;
  return 0;
}
