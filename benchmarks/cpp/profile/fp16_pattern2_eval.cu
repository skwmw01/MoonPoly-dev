#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "cutlass/util/device_memory.h"
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
      std::cerr << "cuBLAS error " << static_cast<int>(error)                 \
                << " at line " << __LINE__ << std::endl;                     \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

using Element = cutlass::half_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::ColumnMajor;

std::vector<int> parse_int_list(const std::string &text) {
  std::vector<int> values;
  std::stringstream ss(text);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (!token.empty()) {
      values.push_back(std::stoi(token));
    }
  }
  return values;
}

void run_cublas(cublasHandle_t handle, int m, int n, int k, Element *a,
                Element *b, Element *c) {
  float alpha = 1.0f;
  float beta = 0.0f;
  CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
                            a, CUDA_R_16F, k, b, CUDA_R_16F, k, &beta, c,
                            CUDA_R_16F, m, CUDA_R_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

template <typename Fn>
double time_ms(Fn fn, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) {
    fn();
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    fn();
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return static_cast<double>(elapsed) / static_cast<double>(iters);
}

bool allclose(Element *got, Element *ref, int m, int n) {
  cutlass::HostTensor<Element, LayoutC> h_got({m, n});
  cutlass::HostTensor<Element, LayoutC> h_ref({m, n});
  CUDA_CHECK(cudaMemcpy(h_got.host_data(), got, m * n * sizeof(Element),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_ref.host_data(), ref, m * n * sizeof(Element),
                        cudaMemcpyDeviceToHost));

  for (int i = 0; i < m * n; ++i) {
    float a = static_cast<float>(h_got.host_data()[i]);
    float b = static_cast<float>(h_ref.host_data()[i]);
    float diff = std::abs(a - b);
    if (!std::isfinite(a) || !std::isfinite(b) ||
        diff > 0.25f + 0.02f * std::abs(b)) {
      return false;
    }
  }
  return true;
}

int main(int argc, char **argv) {
  std::string csv_path =
      "artifacts/benchmarks/fp16_rcc/pattern2_measurements.csv";
  std::string m_list = "128,256";
  std::string n_base_list = "512,1024";
  std::string tail_list = "8,16,32";
  std::string k_list = "512,1024";
  int warmup = 5;
  int iters = 20;
  int max_shapes = 0;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << arg << std::endl;
        std::exit(EXIT_FAILURE);
      }
      return argv[++i];
    };
    if (arg == "--csv") {
      csv_path = next();
    } else if (arg == "--m-list") {
      m_list = next();
    } else if (arg == "--n-base-list") {
      n_base_list = next();
    } else if (arg == "--tail-list") {
      tail_list = next();
    } else if (arg == "--k-list") {
      k_list = next();
    } else if (arg == "--warmup") {
      warmup = std::stoi(next());
    } else if (arg == "--iters") {
      iters = std::stoi(next());
    } else if (arg == "--max-shapes") {
      max_shapes = std::stoi(next());
    } else if (arg == "--help") {
      std::cout
          << "Usage: moonpoly_profile_fp16_pattern2 [options]\n"
          << "  --csv <path>\n"
          << "  --m-list <csv>\n"
          << "  --n-base-list <csv>   Aligned main-body N values\n"
          << "  --tail-list <csv>     Irregular tail N values\n"
          << "  --k-list <csv>\n"
          << "  --warmup <int>\n"
          << "  --iters <int>\n"
          << "  --max-shapes <int>\n";
      return 0;
    } else {
      std::cerr << "Unknown arg: " << arg << std::endl;
      return 1;
    }
  }

  if (warmup < 0 || iters <= 0 || max_shapes < 0) {
    std::cerr << "Invalid warmup/iters/max-shapes" << std::endl;
    return 1;
  }

  std::filesystem::path out_path(csv_path);
  if (!out_path.parent_path().empty()) {
    std::filesystem::create_directories(out_path.parent_path());
  }
  std::ofstream csv(csv_path);
  if (!csv.is_open()) {
    std::cerr << "Cannot open CSV: " << csv_path << std::endl;
    return 1;
  }
  csv << "m,n,k,split_n,tail_n,warmup,iters,pattern2_ms,region0_ms,"
         "region1_ms,critical_ms,overhead_ms,cublas_ms,speedup_vs_cublas,ok\n";

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaSetDevice(device));
  CUDA_CHECK(cudaFree(nullptr));

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  std::vector<int> ms = parse_int_list(m_list);
  std::vector<int> bases = parse_int_list(n_base_list);
  std::vector<int> tails = parse_int_list(tail_list);
  std::vector<int> ks = parse_int_list(k_list);
  int shape_count = 0;

  for (int m : ms) {
    for (int n_base : bases) {
      for (int tail : tails) {
        for (int k : ks) {
          if (max_shapes > 0 && shape_count >= max_shapes) {
            CUBLAS_CHECK(cublasDestroy(handle));
            return 0;
          }
          ++shape_count;
          const int n = n_base + tail;
          const int split_n = n_base;

          cutlass::HostTensor<Element, LayoutA> h_a({m, k});
          cutlass::HostTensor<Element, LayoutB> h_b({k, n});
          cutlass::HostTensor<Element, LayoutC> h_c({m, n});
          cutlass::reference::host::TensorFillRandomUniform(
              h_a.host_view(), 1, Element(2), Element(-2), 0);
          cutlass::reference::host::TensorFillRandomUniform(
              h_b.host_view(), 1, Element(2), Element(-2), 0);
          cutlass::reference::host::TensorFill(h_c.host_view());

          cutlass::device_memory::allocation<Element> d_a(h_a.capacity());
          cutlass::device_memory::allocation<Element> d_b(h_b.capacity());
          cutlass::device_memory::allocation<Element> d_pattern2(h_c.capacity());
          cutlass::device_memory::allocation<Element> d_ref(h_c.capacity());
          d_a.copy_from_host(h_a.host_data());
          d_b.copy_from_host(h_b.host_data());
          d_pattern2.copy_from_host(h_c.host_data());
          d_ref.copy_from_host(h_c.host_data());

          auto pattern2 = [&]() {
            moonpoly::run_fp16_pattern2_twin_gemm_row_col_col(
                m, n, k, split_n, d_a.get(), d_b.get(), d_pattern2.get(),
                1.0f, 0.0f);
          };
          auto full_cublas = [&]() {
            run_cublas(handle, m, n, k, d_a.get(), d_b.get(), d_ref.get());
          };

          pattern2();
          full_cublas();
          CUDA_CHECK(cudaDeviceSynchronize());
          bool ok = allclose(d_pattern2.get(), d_ref.get(), m, n);

          double pattern2_ms = time_ms(pattern2, warmup, iters);
          setenv("MOONPOLY_ENABLE_PATTERN2", "0", 1);
          setenv("MOONPOLY_ENABLE_PATTERN3", "0", 1);
          setenv("MOONPOLY_ENABLE_GENERATED_SELECTOR", "0", 1);

          auto region0 = [&]() {
            moonpoly::run_fp16_online_gemm_row_col_col(
                m, split_n, k, d_a.get(), d_b.get(), d_pattern2.get(), 1.0f,
                0.0f);
          };
          auto region1 = [&]() {
            moonpoly::run_fp16_online_gemm_row_col_col(
                m, tail, k, d_a.get(), d_b.get() + split_n * k,
                d_pattern2.get() + split_n * m, 1.0f, 0.0f);
          };
          double region0_ms = time_ms(region0, warmup, iters);
          double region1_ms = time_ms(region1, warmup, iters);
          double cublas_ms = time_ms(full_cublas, warmup, iters);
          double critical_ms = std::max(region0_ms, region1_ms);
          double overhead_ms = pattern2_ms - critical_ms;

          csv << m << "," << n << "," << k << "," << split_n << "," << tail
              << "," << warmup << "," << iters << "," << std::setprecision(10)
              << pattern2_ms << "," << region0_ms << "," << region1_ms << ","
              << critical_ms << "," << overhead_ms << "," << cublas_ms << ","
              << (cublas_ms / pattern2_ms) << "," << (ok ? 1 : 0) << "\n";
        }
      }
    }
  }

  CUBLAS_CHECK(cublasDestroy(handle));
  return 0;
}
