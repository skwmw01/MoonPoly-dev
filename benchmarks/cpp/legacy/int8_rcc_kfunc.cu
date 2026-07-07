#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <limits>
#include <exception>
#include <fstream>
#include <filesystem>

#include "cuda_runtime.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "moonpoly.cuh"

using ElementInputA = int8_t;
using ElementInputB = int8_t;
using ElementOutput = int32_t;

using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::ColumnMajor;

struct KScalingResult {
  int pid;
  int tile_m;
  int tile_n;
  int k_value;
  int alignment;
  bool is_supported;
  bool is_correct;
  double execution_time_ms;
  
  KScalingResult() : pid(-1), tile_m(0), tile_n(0), k_value(0), alignment(0),
                     is_supported(false), is_correct(false), 
                     execution_time_ms(std::numeric_limits<double>::max()) {}
};

KScalingResult test_k_scaling_single_tile(int pid, int tile_m, int tile_n, int k_value) {
  KScalingResult result;
  result.pid = pid;
  result.tile_m = tile_m;
  result.tile_n = tile_n;
  result.k_value = k_value;
  
  try {
    // Get alignment requirements for this PID
    int alignment_val = 0;
    moonpoly::get_gemm_alignment_int8_row_col_col(pid, &alignment_val);
    result.alignment = alignment_val;
    
    // Check if K dimension satisfies alignment requirement
    if (k_value % alignment_val != 0) {
      result.is_supported = false;
      return result;
    }
    
    result.is_supported = true;
    
    // Create problem size
    cutlass::gemm::GemmCoord problem_size(tile_m, tile_n, k_value);
    
    // Initialize tensors
    cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(problem_size.mk());
    cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(problem_size.kn());
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(problem_size.mn());
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref(problem_size.mn());

    // Fill tensors with random data
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_a.host_view(), 1, ElementInputA(4), ElementInputA(-4), 0);
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_b.host_view(), 1, ElementInputB(4), ElementInputB(-4), 0);
    cutlass::reference::host::TensorFill(tensor_d.host_view());
    cutlass::reference::host::TensorFill(tensor_ref.host_view());

    // Copy to device
    tensor_a.sync_device();
    tensor_b.sync_device();
    tensor_d.sync_device();
    tensor_ref.sync_device();

    int alpha = 1, beta = 0;

    // Benchmark performance - mikpoly style K-scaling analysis
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int num_iter = 100;  // More iterations for precise single tile measurement
    double total_time = 0.0;
    float milliseconds = 0.0;

    // Warmup iterations
    for (int i = 0; i < 5; ++i) {
      moonpoly::run_moonpoly_selected_gemm_int8_row_col_col(
          pid, tile_m, tile_n, k_value,
          tensor_a.device_data(), tensor_b.device_data(),
          tensor_d.device_data(), alpha, beta);
    }
    cudaDeviceSynchronize();

    // Actual benchmark iterations for K-scaling analysis
    for (int i = 0; i < num_iter; ++i) {
      cudaEventRecord(start);
      moonpoly::run_moonpoly_selected_gemm_int8_row_col_col(
          pid, tile_m, tile_n, k_value,
          tensor_a.device_data(), tensor_b.device_data(),
          tensor_d.device_data(), alpha, beta);
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&milliseconds, start, stop);
      total_time += milliseconds;
    }
    
    result.execution_time_ms = total_time / num_iter;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

  } catch (const std::exception& e) {
    std::cerr << "Exception in K-scaling test for PID " << pid 
              << ", K=" << k_value << ": " << e.what() << std::endl;
    result.is_supported = false;
    result.is_correct = false;
  } catch (...) {
    std::cerr << "Unknown exception in K-scaling test for PID " << pid 
              << ", K=" << k_value << std::endl;
    result.is_supported = false;
    result.is_correct = false;
  }

  return result;
}

int main(int argc, char **argv) {
  // Default parameters for K-scaling analysis
  int k_start = 1;
  int k_end = 5120;
  int k_step = 1;
  std::string output_dir = "k_scaling_results";
  
  // Parse command line arguments if provided
  if (argc >= 4) {
    k_start = std::stoi(argv[1]);
    k_end = std::stoi(argv[2]);
    k_step = std::stoi(argv[3]);
  }
  if (argc >= 5) {
    output_dir = argv[4];
  }
  
  // Create output directory
  std::filesystem::create_directories(output_dir);
  
  std::cerr << "Starting mikpoly-style K-scaling analysis..." << std::endl;
  std::cerr << "K range: " << k_start << " to " << k_end << " with step " << k_step << std::endl;
  std::cerr << "Output directory: " << output_dir << std::endl;
  
  int total_tests = 0;
  int successful_tests = 0;
  
  // Get shared memory info to determine tile sizes
  int *shared_mem = new int[3];
  
  // Test all available PIDs (moonpoly microkernels)
  for (int pid = 0; pid < 46; ++pid) {
    moonpoly::get_gemm_shared_mem_int8_row_col_col(pid, shared_mem);
    int tile_m = shared_mem[0];
    int tile_n = shared_mem[1];
    int min_k_step = shared_mem[2];
    
    int valid_tests_for_pid = 0;
    double total_time_for_pid = 0.0;
    std::vector<KScalingResult> pid_results;
    
    // K-scaling analysis: test this PID's microkernel with varying K values
    for (int k = k_start; k <= k_end; k += k_step) {
      // Ensure K is aligned with minimum step
      int aligned_k = k * min_k_step;
      
      total_tests++;
      KScalingResult result = test_k_scaling_single_tile(pid, tile_n, tile_m, aligned_k);
      
      pid_results.push_back(result);
      
      // Output to stdout for immediate feedback
      std::cout << result.pid << "," 
                << result.tile_m << "," 
                << result.tile_n << "," 
                << result.k_value << "," 
                << result.alignment << ","
                << (result.is_supported ? "1" : "0") << "," 
                << (result.is_correct ? "1" : "0") << "," 
                << std::fixed << std::setprecision(6) << result.execution_time_ms << std::endl;
      
      if (result.is_supported && result.execution_time_ms < std::numeric_limits<double>::max()) {
        successful_tests++;
        valid_tests_for_pid++;
        total_time_for_pid += result.execution_time_ms;
      }
    }
    
    // Write individual PID results to separate file
    if (!pid_results.empty()) {
      std::string filename = output_dir + "/pid_" + std::to_string(pid) + "_k_scaling.csv";
      std::ofstream pid_file(filename);
      
      if (pid_file.is_open()) {
        // Write header
        pid_file << "k_value,execution_time_ms,supported,tile_m,tile_n,alignment" << std::endl;
        
        // Write data
        for (const auto& result : pid_results) {
          pid_file << result.k_value << ","
                   << std::fixed << std::setprecision(6) << result.execution_time_ms << ","
                   << (result.is_supported ? "1" : "0") << ","
                   << result.tile_m << ","
                   << result.tile_n << ","
                   << result.alignment << std::endl;
        }
        
        pid_file.close();
        std::cerr << "PID " << pid << " results saved to: " << filename << std::endl;
      } else {
        std::cerr << "Failed to open file: " << filename << std::endl;
      }
    }
    
    // Summary for this PID
    if (valid_tests_for_pid > 0) {
      std::cerr << "PID " << pid << " (tile " << tile_m << "x" << tile_n << "): " 
                << valid_tests_for_pid << " valid tests, "
                << "avg time: " << std::fixed << std::setprecision(4) 
                << (total_time_for_pid / valid_tests_for_pid) << "ms" << std::endl;
    } else {
      std::cerr << "PID " << pid << " (tile " << tile_m << "x" << tile_n << "): No valid tests" << std::endl;
    }
  }
  
  delete[] shared_mem;
  
  // Create summary file with all PIDs
  std::string summary_filename = output_dir + "/summary_all_pids.txt";
  std::ofstream summary_file(summary_filename);
  if (summary_file.is_open()) {
    summary_file << "K-scaling Analysis Summary" << std::endl;
    summary_file << "=========================" << std::endl;
    summary_file << "K range: " << k_start << " to " << k_end << " (step: " << k_step << ")" << std::endl;
    summary_file << "Total tests: " << total_tests << std::endl;
    summary_file << "Successful tests: " << successful_tests << std::endl;
    summary_file << "Success rate: " << std::fixed << std::setprecision(1) 
                 << (100.0 * successful_tests / total_tests) << "%" << std::endl;
    summary_file.close();
  }
  
  std::cerr << "K-scaling analysis completed!" << std::endl;
  std::cerr << "Total tests: " << total_tests << ", Successful: " << successful_tests 
            << " (" << std::fixed << std::setprecision(1) 
            << (100.0 * successful_tests / total_tests) << "%)" << std::endl;
  std::cerr << "Results saved in directory: " << output_dir << std::endl;
  
  return 0;
}