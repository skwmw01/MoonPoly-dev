#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <limits>
#include <exception>

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

struct KernelPerformance {
  int pid;
  double moonpoly_time;
  double speedup;
  bool is_supported;
  bool is_correct;
};

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


int run(cublasHandle_t handle, int length_m, int length_n, int length_k,
        double &cb_time, double &best_moonpoly_time, int &best_pid) {
  // Print CSV header if this is the first run
  static bool header_printed = false;
  if (!header_printed) {
    std::cout << "shape,m,n,k,pid,alignment,supported,correct,moonpoly_time_ms,cublas_time_ms,speedup" << std::endl;
    header_printed = true;
  }
  
  try {
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

    // First, get cuBLAS baseline performance
    int8_t *A = reinterpret_cast<int8_t *>(tensor_a.device_data());
    int8_t *B = reinterpret_cast<int8_t *>(tensor_b.device_data());
    int32_t *C = reinterpret_cast<int32_t *>(tensor_ref_d.device_data());
    
    int alpha = 1, beta = 0;
    cublasMath_t cublas_flags = CUBLAS_TENSOR_OP_MATH;
    cublasSetMathMode(handle, cublas_flags);
    int lda = length_k, ldb = length_k, ldc = length_m;

    // Warmup cuBLAS
    // Since we're doing C = A^T * B, we need to be careful about dimensions
    // A: RowMajor M×K, after transpose becomes K×M for cuBLAS
    // B: ColumnMajor K×N  
    // C: ColumnMajor M×N
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                 length_m, length_n, length_k,
                 &alpha, A, CUDA_R_8I, lda, B, CUDA_R_8I, ldb, 
                 &beta, C, CUDA_R_32I, ldc, CUDA_R_32I, 
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaDeviceSynchronize();
  
    // Benchmark cuBLAS
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int num_iter = 20;
    double total_time = 0.0;
    float milliseconds = 0.0;
    
    for (int i = 0; i < num_iter; ++i) {
      cudaEventRecord(start);

      cublasStatus_t cu_status = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                                              length_m, length_n, length_k,
                                              &alpha, A, CUDA_R_8I, lda, B, CUDA_R_8I, ldb, 
                                              &beta, C, CUDA_R_32I, ldc, CUDA_R_32I, 
                                              CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      if (cu_status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "Got cublas error: RunTime Error at: " << __LINE__ << std::endl;
        exit(EXIT_FAILURE);
      }
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&milliseconds, start, stop);
      total_time += milliseconds;
    }
    cb_time = total_time / (num_iter * 1.0);
    
    tensor_ref_d.sync_host();
  
    // Store cuBLAS result for comparison
    cutlass::HostTensor<ElementOutput, LayoutOutput> cublas_result(problem_size.mn());
    cutlass::reference::host::TensorCopy(cublas_result.host_view(), tensor_ref_d.host_view());
  
    // Test all moonpoly kernels
    std::vector<KernelPerformance> kernel_results;
    best_moonpoly_time = std::numeric_limits<double>::max();
    best_pid = -1;
    
    int alignment[0];
    for (int pid = 0; pid < 46; pid++) {
      KernelPerformance perf;
      perf.pid = pid;
      perf.is_supported = true;
      perf.is_correct = false;
      perf.moonpoly_time = std::numeric_limits<double>::max();
      perf.speedup = 0.0;
      
      moonpoly::get_gemm_alignment_int8_row_col_col(pid, alignment);
      if (length_k % (*alignment) != 0) {
        perf.is_supported = false;
        std::cout << length_m << "x" << length_n << "x" << length_k << "," 
                  << length_m << "," << length_n << "," << length_k << ","
                  << pid << "," << alignment[0] << ",0,0,inf_no_kalign," << cb_time << ",0.0" << std::endl;
        continue;
      }
    
    // Test correctness with exception handling
    double cosine_diff = std::numeric_limits<double>::max(); // Initialize outside try block
    try {
      cutlass::reference::host::TensorFill(tensor_d.host_view());
      tensor_d.sync_device();
      
      moonpoly::run_moonpoly_selected_gemm_int8_row_col_col(pid, length_m, length_n, length_k,
                                                            tensor_a.device_data(), tensor_b.device_data(),
                                                            tensor_d.device_data(), alpha, beta);
      
      // Check for CUDA errors after kernel execution
      cudaError_t cuda_error = cudaGetLastError();
      if (cuda_error != cudaSuccess) {
        std::cerr << "CUDA kernel error for PID " << pid << ": " << cudaGetErrorString(cuda_error) << std::endl;
        throw std::runtime_error("CUDA kernel execution failed: " + std::string(cudaGetErrorString(cuda_error)));
      }
      
      cudaDeviceSynchronize();
      tensor_d.sync_host();
      
      // Check correctness against cuBLAS using cosine similarity
      cosine_diff = calc_cosine_diff(tensor_d, cublas_result);
      double similarity_threshold = 1e-5; // Tolerance for numerical differences
      perf.is_correct = (cosine_diff < similarity_threshold);
      
      if (!perf.is_correct) {
        std::cerr << "Correctness check failed for PID " << pid 
                  << " (Shape: " << length_m << "x" << length_n << "x" << length_k << ")"
                  << " - Cosine diff: " << std::scientific << std::setprecision(6) << cosine_diff 
                  << " (threshold: " << similarity_threshold << ")" << std::endl;
        std::cout << length_m << "x" << length_n << "x" << length_k << "," 
                  << length_m << "," << length_n << "," << length_k << ","
                  << pid << "," << alignment[0] << ",1,0,inf_neq," << cb_time << ",0.0" << std::endl;
        continue;
      }
      
      // Benchmark moonpoly kernel
      total_time = 0.0;
      for (int i = 0; i < num_iter; ++i) {
        cudaEventRecord(start);
        moonpoly::run_moonpoly_selected_gemm_int8_row_col_col(pid, length_m, length_n, length_k,
                                                              tensor_a.device_data(), tensor_b.device_data(),
                                                              tensor_d.device_data(), alpha, beta);
        
        // Check for CUDA errors in benchmark loop
        cudaError_t bench_error = cudaGetLastError();
        if (bench_error != cudaSuccess) {
          std::cerr << "CUDA error during benchmarking PID " << pid << ": " << cudaGetErrorString(bench_error) << std::endl;
          throw std::runtime_error("Benchmark kernel execution failed: " + std::string(cudaGetErrorString(bench_error)));
        }
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_time += milliseconds;
      }
      perf.moonpoly_time = total_time / (num_iter * 1.0);
      perf.speedup = cb_time / perf.moonpoly_time;
    } catch (const std::runtime_error& e) {
      std::cerr << "Runtime error for PID " << pid << " (Shape: " << length_m << "x" << length_n << "x" << length_k << "): " << e.what() << std::endl;
      std::cout << length_m << "x" << length_n << "x" << length_k << "," 
                << length_m << "," << length_n << "," << length_k << ","
                << pid << "," << alignment[0] << ",1,0,inf_runerr," << cb_time << ",0.0" << std::endl;
      continue;
    } catch (const std::exception& e) {
      std::cerr << "Exception caught for PID " << pid << " (Shape: " << length_m << "x" << length_n << "x" << length_k << "): " << e.what() << std::endl;
      std::cout << length_m << "x" << length_n << "x" << length_k << "," 
                << length_m << "," << length_n << "," << length_k << ","
                << pid << "," << alignment[0] << ",1,0,inf_exp," << cb_time << ",0.0" << std::endl;
      continue;
    } catch (...) {
      std::cerr << "Unknown exception caught for PID " << pid << " (Shape: " << length_m << "x" << length_n << "x" << length_k << ")" << std::endl;
      
      // Try to get more CUDA error information
      cudaError_t last_error = cudaGetLastError();
      if (last_error != cudaSuccess) {
        std::cerr << "  Last CUDA error: " << cudaGetErrorString(last_error) << std::endl;
      }
      
      std::cout << length_m << "x" << length_n << "x" << length_k << "," 
                << length_m << "," << length_n << "," << length_k << ","
                << pid << "," << alignment[0] << ",1,0,inf," << cb_time << ",0.0" << std::endl;
      continue;
    }
    
      // Track best performing kernel
      if (perf.moonpoly_time < best_moonpoly_time) {
        best_moonpoly_time = perf.moonpoly_time;
        best_pid = pid;
      }
      
      // Output detailed results in CSV format with cosine similarity info
      std::cerr << "PID " << pid << " passed with cosine diff: " 
                << std::scientific << std::setprecision(3) << cosine_diff << std::endl;
      std::cout << length_m << "x" << length_n << "x" << length_k << "," 
                << length_m << "," << length_n << "," << length_k << ","
                << pid << "," << alignment[0] << ",1,1," 
                << std::fixed << std::setprecision(4) << perf.moonpoly_time << "," 
                << cb_time << "," << std::setprecision(2) << perf.speedup << std::endl;
      
      kernel_results.push_back(perf);
    }
  
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Critical exception in run function: " << e.what() << std::endl;
    // Output at least one failed entry for this shape so Python script can process it
    std::cout << length_m << "x" << length_n << "x" << length_k << "," 
              << length_m << "," << length_n << "," << length_k << ","
              << "-1,1,0,0,inf,inf,0.0" << std::endl;
    cb_time = std::numeric_limits<double>::max();
    best_moonpoly_time = std::numeric_limits<double>::max();
    best_pid = -1;
    return -1;
  } catch (...) {
    std::cerr << "Critical unknown exception in run function" << std::endl;
    // Output at least one failed entry for this shape so Python script can process it
    std::cout << length_m << "x" << length_n << "x" << length_k << "," 
              << length_m << "," << length_n << "," << length_k << ","
              << "-1,1,0,0,inf,inf,0.0" << std::endl;
    cb_time = std::numeric_limits<double>::max();
    best_moonpoly_time = std::numeric_limits<double>::max();
    best_pid = -1;
    return -1;
  }
}

int main(int argc, char **argv) {
  const int num_case = 1;
  int m_arr[]= {4096};
  int n_arr[] = {4096};
  int k_arr[] = {4096};
  if (argc > 1) {
    m_arr[0] = std::stoi(argv[1]);
    n_arr[0] = std::stoi(argv[2]);
    k_arr[0] = std::stoi(argv[3]);
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
    double cb_time, best_moonpoly_time;
    int best_pid;
    int passed = run(handle, m_arr[i], n_arr[i], k_arr[i], cb_time, best_moonpoly_time, best_pid);

    if (passed == -1) {
      std::cerr << "Test failed for shape (" << m_arr[i] << ", " << n_arr[i] << ", " << k_arr[i] << ")" << std::endl;
      break;
    } else {
      // Summary output for easy parsing
      std::cerr << "SUMMARY: Shape (" << m_arr[i] << ", " << n_arr[i] << ", " << k_arr[i] << ") - "
                << "cuBLAS: " << std::fixed << std::setprecision(4) << cb_time << "ms, "
                << "Best Moonpoly (PID " << best_pid << "): " << best_moonpoly_time << "ms, "
                << "Speedup: " << std::setprecision(2) << cb_time / best_moonpoly_time << "x" << std::endl;
    }
  }

  cublasDestroy(handle);
  return 0;
}
