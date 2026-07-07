#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <fstream>

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
#include <cmath>

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

struct BenchmarkResult {
    int m, n, k;
    std::string dtype;
    double cublas_time_ms;
    double moonpoly_time_ms;
    double speedup;
    double cosine_similarity;
    bool correctness_passed;
};

// Calculate cosine similarity between two tensors
template<typename T, typename Layout>
double calc_cosine_similarity(const cutlass::HostTensor<T, Layout>& x,
                             const cutlass::HostTensor<T, Layout>& y) {
    double x_sum = 0.0, y_sum = 0.0, xy_sum = 0.0;
    
    for (int i = 0; i < x.capacity(); ++i) {
        double x_val = static_cast<double>(x.host_data()[i]);
        double y_val = static_cast<double>(y.host_data()[i]);
        
        x_sum += x_val * x_val;
        y_sum += y_val * y_val;
        xy_sum += x_val * y_val;
    }
    
    double x_norm = std::sqrt(x_sum);
    double y_norm = std::sqrt(y_sum);
    
    if (x_norm == 0.0 || y_norm == 0.0) return 0.0;
    
    return xy_sum / (x_norm * y_norm);
}

// FP16 benchmark
BenchmarkResult benchmark_fp16(cublasHandle_t handle, int m, int n, int k, int warmup_iter = 5, int bench_iter = 20) {
    using ElementInputA = cutlass::half_t;
    using ElementInputB = cutlass::half_t;
    using ElementOutput = cutlass::half_t;
    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::ColumnMajor;

    cutlass::gemm::GemmCoord problem_size(m, n, k);

    // Initialize tensors
    cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(problem_size.mk());
    cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(problem_size.kn());
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d_moonpoly(problem_size.mn());
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d_cublas(problem_size.mn());

    // Fill tensors with random data
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_a.host_view(), 1, ElementInputA(2), ElementInputA(-2), 0);
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_b.host_view(), 1, ElementInputB(2), ElementInputB(-2), 0);
    cutlass::reference::host::TensorFill(tensor_d_moonpoly.host_view());
    cutlass::reference::host::TensorFill(tensor_d_cublas.host_view());

    // Copy to device
    tensor_a.sync_device();
    tensor_b.sync_device();
    tensor_d_moonpoly.sync_device();
    tensor_d_cublas.sync_device();

    float alpha = 1.0f, beta = 0.0f;

    // Test correctness first
    moonpoly::run_fp16_gemm(m, n, k,
                           tensor_a.device_data(), false,
                           tensor_b.device_data(), true,
                           tensor_d_moonpoly.device_data(), true, alpha, beta);
    cudaDeviceSynchronize();

    // cuBLAS
    half *A = reinterpret_cast<half *>(tensor_a.device_data());
    half *B = reinterpret_cast<half *>(tensor_b.device_data());
    half *C_cublas = reinterpret_cast<half *>(tensor_d_cublas.device_data());
    
    int lda = k, ldb = k, ldc = m;
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
                 A, CUDA_R_16F, lda, B, CUDA_R_16F, ldb, &beta,
                 C_cublas, CUDA_R_16F, ldc, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaDeviceSynchronize();

    // Check correctness
    tensor_d_moonpoly.sync_host();
    tensor_d_cublas.sync_host();
    
    double cosine_sim = calc_cosine_similarity(tensor_d_moonpoly, tensor_d_cublas);
    bool correctness_passed = cosine_sim > 0.999;

    // Warmup
    for (int i = 0; i < warmup_iter; ++i) {
        moonpoly::run_fp16_gemm(m, n, k, tensor_a.device_data(), false,
                               tensor_b.device_data(), true,
                               tensor_d_moonpoly.device_data(), true, alpha, beta);
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
                     A, CUDA_R_16F, lda, B, CUDA_R_16F, ldb, &beta,
                     C_cublas, CUDA_R_16F, ldc, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    cudaDeviceSynchronize();

    // Benchmark cuBLAS
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float total_time = 0.0f;
    for (int i = 0; i < bench_iter; ++i) {
        cudaEventRecord(start);
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
                     A, CUDA_R_16F, lda, B, CUDA_R_16F, ldb, &beta,
                     C_cublas, CUDA_R_16F, ldc, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_time += milliseconds;
    }
    double cublas_time = total_time / bench_iter;

    // Benchmark Moonpoly
    total_time = 0.0f;
    for (int i = 0; i < bench_iter; ++i) {
        cudaEventRecord(start);
        moonpoly::run_fp16_gemm(m, n, k, tensor_a.device_data(), false,
                               tensor_b.device_data(), true,
                               tensor_d_moonpoly.device_data(), true, alpha, beta);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_time += milliseconds;
    }
    double moonpoly_time = total_time / bench_iter;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    BenchmarkResult result;
    result.m = m;
    result.n = n;
    result.k = k;
    result.dtype = "FP16";
    result.cublas_time_ms = cublas_time;
    result.moonpoly_time_ms = moonpoly_time;
    result.speedup = cublas_time / moonpoly_time;
    result.cosine_similarity = cosine_sim;
    result.correctness_passed = correctness_passed;

    return result;
}

// FP32 benchmark
BenchmarkResult benchmark_fp32(cublasHandle_t handle, int m, int n, int k, int warmup_iter = 5, int bench_iter = 20) {
    using ElementInputA = float;
    using ElementInputB = float;
    using ElementOutput = float;
    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::ColumnMajor;

    cutlass::gemm::GemmCoord problem_size(m, n, k);

    cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(problem_size.mk());
    cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(problem_size.kn());
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d_moonpoly(problem_size.mn());
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d_cublas(problem_size.mn());

    cutlass::reference::host::TensorFillRandomUniform(
        tensor_a.host_view(), 1, ElementInputA(2), ElementInputA(-2), 0);
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_b.host_view(), 1, ElementInputB(2), ElementInputB(-2), 0);
    cutlass::reference::host::TensorFill(tensor_d_moonpoly.host_view());
    cutlass::reference::host::TensorFill(tensor_d_cublas.host_view());

    tensor_a.sync_device();
    tensor_b.sync_device();
    tensor_d_moonpoly.sync_device();
    tensor_d_cublas.sync_device();

    float alpha = 1.0f, beta = 0.0f;

    // Test correctness
    moonpoly::run_fp32_gemm(m, n, k,
                           tensor_a.device_data(), false,
                           tensor_b.device_data(), true,
                           tensor_d_moonpoly.device_data(), true, alpha, beta);
    cudaDeviceSynchronize();

    int lda = k, ldb = k, ldc = m;
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
                tensor_a.device_data(), lda, tensor_b.device_data(), ldb,
                &beta, tensor_d_cublas.device_data(), ldc);
    cudaDeviceSynchronize();

    tensor_d_moonpoly.sync_host();
    tensor_d_cublas.sync_host();
    
    double cosine_sim = calc_cosine_similarity(tensor_d_moonpoly, tensor_d_cublas);
    bool correctness_passed = cosine_sim > 0.999;

    // Warmup
    for (int i = 0; i < warmup_iter; ++i) {
        moonpoly::run_fp32_gemm(m, n, k, tensor_a.device_data(), false,
                               tensor_b.device_data(), true,
                               tensor_d_moonpoly.device_data(), true, alpha, beta);
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
                    tensor_a.device_data(), lda, tensor_b.device_data(), ldb,
                    &beta, tensor_d_cublas.device_data(), ldc);
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float total_time = 0.0f;
    for (int i = 0; i < bench_iter; ++i) {
        cudaEventRecord(start);
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
                    tensor_a.device_data(), lda, tensor_b.device_data(), ldb,
                    &beta, tensor_d_cublas.device_data(), ldc);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_time += milliseconds;
    }
    double cublas_time = total_time / bench_iter;

    total_time = 0.0f;
    for (int i = 0; i < bench_iter; ++i) {
        cudaEventRecord(start);
        moonpoly::run_fp32_gemm(m, n, k, tensor_a.device_data(), false,
                               tensor_b.device_data(), true,
                               tensor_d_moonpoly.device_data(), true, alpha, beta);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_time += milliseconds;
    }
    double moonpoly_time = total_time / bench_iter;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    BenchmarkResult result;
    result.m = m;
    result.n = n;
    result.k = k;
    result.dtype = "FP32";
    result.cublas_time_ms = cublas_time;
    result.moonpoly_time_ms = moonpoly_time;
    result.speedup = cublas_time / moonpoly_time;
    result.cosine_similarity = cosine_sim;
    result.correctness_passed = correctness_passed;

    return result;
}

// INT8 benchmark
BenchmarkResult benchmark_int8(cublasHandle_t handle, int m, int n, int k, int warmup_iter = 5, int bench_iter = 20) {
    using ElementInputA = int8_t;
    using ElementInputB = int8_t;
    using ElementOutput = int32_t;
    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::ColumnMajor;

    cutlass::gemm::GemmCoord problem_size(m, n, k);

    cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(problem_size.mk());
    cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(problem_size.kn());
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d_moonpoly(problem_size.mn());
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d_cublas(problem_size.mn());

    // Fill with smaller range for int8
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_a.host_view(), 1, ElementInputA(4), ElementInputA(-4), 0);
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_b.host_view(), 1, ElementInputB(4), ElementInputB(-4), 0);
    cutlass::reference::host::TensorFill(tensor_d_moonpoly.host_view());
    cutlass::reference::host::TensorFill(tensor_d_cublas.host_view());

    tensor_a.sync_device();
    tensor_b.sync_device();
    tensor_d_moonpoly.sync_device();
    tensor_d_cublas.sync_device();

    int alpha = 1, beta = 0;

    // Test correctness
    moonpoly::run_int8_gemm(m, n, k,
                           tensor_a.device_data(), false,
                           tensor_b.device_data(), true,
                           tensor_d_moonpoly.device_data(), true, alpha, beta);
    cudaDeviceSynchronize();

    // cuBLAS INT8 GEMM
    int lda = k, ldb = k, ldc = m;
    int32_t alpha32 = alpha, beta32 = beta;
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha32,
                 tensor_a.device_data(), CUDA_R_8I, lda,
                 tensor_b.device_data(), CUDA_R_8I, ldb,
                 &beta32, tensor_d_cublas.device_data(), CUDA_R_32I, ldc,
                 CUDA_R_32I, CUBLAS_GEMM_DEFAULT);
    cudaDeviceSynchronize();

    tensor_d_moonpoly.sync_host();
    tensor_d_cublas.sync_host();
    
    double cosine_sim = calc_cosine_similarity(tensor_d_moonpoly, tensor_d_cublas);
    bool correctness_passed = cosine_sim > 0.999;

    // Warmup
    for (int i = 0; i < warmup_iter; ++i) {
        moonpoly::run_int8_gemm(m, n, k, tensor_a.device_data(), false,
                               tensor_b.device_data(), true,
                               tensor_d_moonpoly.device_data(), true, alpha, beta);
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha32,
                     tensor_a.device_data(), CUDA_R_8I, lda,
                     tensor_b.device_data(), CUDA_R_8I, ldb,
                     &beta32, tensor_d_cublas.device_data(), CUDA_R_32I, ldc,
                     CUDA_R_32I, CUBLAS_GEMM_DEFAULT);
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float total_time = 0.0f;
    for (int i = 0; i < bench_iter; ++i) {
        cudaEventRecord(start);
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha32,
                     tensor_a.device_data(), CUDA_R_8I, lda,
                     tensor_b.device_data(), CUDA_R_8I, ldb,
                     &beta32, tensor_d_cublas.device_data(), CUDA_R_32I, ldc,
                     CUDA_R_32I, CUBLAS_GEMM_DEFAULT);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_time += milliseconds;
    }
    double cublas_time = total_time / bench_iter;

    total_time = 0.0f;
    for (int i = 0; i < bench_iter; ++i) {
        cudaEventRecord(start);
        moonpoly::run_int8_gemm(m, n, k, tensor_a.device_data(), false,
                               tensor_b.device_data(), true,
                               tensor_d_moonpoly.device_data(), true, alpha, beta);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_time += milliseconds;
    }
    double moonpoly_time = total_time / bench_iter;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    BenchmarkResult result;
    result.m = m;
    result.n = n;
    result.k = k;
    result.dtype = "INT8";
    result.cublas_time_ms = cublas_time;
    result.moonpoly_time_ms = moonpoly_time;
    result.speedup = cublas_time / moonpoly_time;
    result.cosine_similarity = cosine_sim;
    result.correctness_passed = correctness_passed;

    return result;
}

void print_results(const std::vector<BenchmarkResult>& results) {
    std::cout << "\n" << std::string(100, '=') << std::endl;
    std::cout << "                    MOONPOLY vs CUBLAS BENCHMARK RESULTS" << std::endl;
    std::cout << std::string(100, '=') << std::endl;
    
    std::cout << std::left << std::setw(8) << "Type"
              << std::setw(12) << "M x N x K"
              << std::setw(15) << "cuBLAS (ms)"
              << std::setw(15) << "Moonpoly (ms)"
              << std::setw(12) << "Speedup"
              << std::setw(15) << "Cosine Sim"
              << std::setw(12) << "Correctness" << std::endl;
    std::cout << std::string(100, '-') << std::endl;

    for (const auto& result : results) {
        std::string size_str = std::to_string(result.m) + "x" + 
                              std::to_string(result.n) + "x" + 
                              std::to_string(result.k);
        
        std::cout << std::left << std::setw(8) << result.dtype
                  << std::setw(12) << size_str
                  << std::setw(15) << std::fixed << std::setprecision(3) << result.cublas_time_ms
                  << std::setw(15) << std::fixed << std::setprecision(3) << result.moonpoly_time_ms
                  << std::setw(12) << std::fixed << std::setprecision(2) << result.speedup << "x"
                  << std::setw(15) << std::fixed << std::setprecision(6) << result.cosine_similarity
                  << std::setw(12) << (result.correctness_passed ? "PASS" : "FAIL") << std::endl;
    }
    std::cout << std::string(100, '=') << std::endl;
}

void save_results_csv(const std::vector<BenchmarkResult>& results, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << " for writing" << std::endl;
        return;
    }

    file << "DataType,M,N,K,cuBLAS_time_ms,Moonpoly_time_ms,Speedup,Cosine_Similarity,Correctness_Passed\n";
    
    for (const auto& result : results) {
        file << result.dtype << ","
             << result.m << ","
             << result.n << ","
             << result.k << ","
             << std::fixed << std::setprecision(6) << result.cublas_time_ms << ","
             << std::fixed << std::setprecision(6) << result.moonpoly_time_ms << ","
             << std::fixed << std::setprecision(6) << result.speedup << ","
             << std::fixed << std::setprecision(8) << result.cosine_similarity << ","
             << (result.correctness_passed ? "TRUE" : "FALSE") << "\n";
    }
    
    file.close();
    std::cout << "Results saved to " << filename << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "Moonpoly vs cuBLAS Comprehensive Benchmark" << std::endl;
    
    // Initialize cuBLAS
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "Failed to create cuBLAS handle" << std::endl;
        return -1;
    }

    std::vector<BenchmarkResult> results;
    
    // Test configurations (M, N, K)
    std::vector<std::tuple<int, int, int>> test_sizes;
    
    if (argc >= 4) {
        // Single custom size from command line
        int m = std::stoi(argv[1]);
        int n = std::stoi(argv[2]);
        int k = std::stoi(argv[3]);
        test_sizes.push_back(std::make_tuple(m, n, k));
    } else {
        // Default test suite with common matrix sizes
        test_sizes = {
            {1024, 1024, 1024},
            {2048, 2048, 2048},
            {4096, 4096, 4096},
            {8192, 8192, 8192},
            {1024, 4096, 1024},
            {4096, 1024, 4096},
            {2048, 8192, 2048},
            {8192, 2048, 8192},
            {16384, 16384, 16384}
        };
    }

    // Run benchmarks for each data type and size
    for (const auto& size : test_sizes) {
        int m, n, k;
        std::tie(m, n, k) = size;
        
        std::cout << "\nTesting matrix size: " << m << "x" << n << "x" << k << std::endl;
        
        try {
            // FP16 benchmark
            std::cout << "  Running FP16..." << std::flush;
            auto fp16_result = benchmark_fp16(handle, m, n, k);
            results.push_back(fp16_result);
            std::cout << " Done (Speedup: " << std::fixed << std::setprecision(2) 
                      << fp16_result.speedup << "x)" << std::endl;
            
            // FP32 benchmark
            std::cout << "  Running FP32..." << std::flush;
            auto fp32_result = benchmark_fp32(handle, m, n, k);
            results.push_back(fp32_result);
            std::cout << " Done (Speedup: " << std::fixed << std::setprecision(2) 
                      << fp32_result.speedup << "x)" << std::endl;
            
            // INT8 benchmark
            std::cout << "  Running INT8..." << std::flush;
            auto int8_result = benchmark_int8(handle, m, n, k);
            results.push_back(int8_result);
            std::cout << " Done (Speedup: " << std::fixed << std::setprecision(2) 
                      << int8_result.speedup << "x)" << std::endl;
            
        } catch (const std::exception& e) {
            std::cerr << "Error benchmarking size " << m << "x" << n << "x" << k 
                      << ": " << e.what() << std::endl;
        }
    }

    // Print results
    print_results(results);
    
    // Save to CSV
    save_results_csv(results, "moonpoly_vs_cublas_benchmark.csv");
    
    // Summary statistics
    std::cout << "\nSummary Statistics:" << std::endl;
    for (const std::string& dtype : {"FP16", "FP32", "INT8"}) {
        std::vector<double> speedups;
        int passed_count = 0;
        int total_count = 0;
        
        for (const auto& result : results) {
            if (result.dtype == dtype) {
                speedups.push_back(result.speedup);
                if (result.correctness_passed) passed_count++;
                total_count++;
            }
        }
        
        if (!speedups.empty()) {
            double avg_speedup = 0.0;
            double max_speedup = *std::max_element(speedups.begin(), speedups.end());
            double min_speedup = *std::min_element(speedups.begin(), speedups.end());
            
            for (double speedup : speedups) {
                avg_speedup += speedup;
            }
            avg_speedup /= speedups.size();
            
            std::cout << "  " << dtype << ": Avg=" << std::fixed << std::setprecision(2) << avg_speedup
                      << "x, Min=" << min_speedup << "x, Max=" << max_speedup 
                      << "x, Correctness=" << passed_count << "/" << total_count << std::endl;
        }
    }

    cublasDestroy(handle);
    return 0;
}