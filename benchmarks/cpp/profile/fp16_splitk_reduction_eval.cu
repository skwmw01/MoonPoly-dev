#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "cuda_runtime.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "moonpoly.cuh"

#define CUDA_CHECK(status)                                                     \
  {                                                                            \
    cudaError_t error = status;                                                \
    if (error != cudaSuccess) {                                                \
      std::cerr << "CUDA error: " << cudaGetErrorString(error)                 \
                << " at line: " << __LINE__ << std::endl;                     \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  }

using ElementInputA = cutlass::half_t;
using ElementInputB = cutlass::half_t;
using ElementOutput = cutlass::half_t;
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::ColumnMajor;

struct Shape {
  int m;
  int n;
  int k;
};

std::vector<int> parse_int_list(std::string const &s) {
  std::vector<int> vals;
  std::stringstream ss(s);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (!token.empty()) {
      vals.push_back(std::stoi(token));
    }
  }
  return vals;
}

std::vector<Shape> parse_shape_list(std::string const &s) {
  std::vector<Shape> shapes;
  std::stringstream ss(s);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (token.empty()) {
      continue;
    }
    std::replace(token.begin(), token.end(), 'X', 'x');
    std::stringstream ts(token);
    std::string part;
    std::vector<int> dims;
    while (std::getline(ts, part, 'x')) {
      if (!part.empty()) {
        dims.push_back(std::stoi(part));
      }
    }
    if (dims.size() != 3) {
      std::cerr << "Invalid shape: " << token << ", expected MxNxK"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    shapes.push_back({dims[0], dims[1], dims[2]});
  }
  return shapes;
}

int ceil_div(int x, int y) { return (x + y - 1) / y; }

struct DeviceProblem {
  cutlass::HostTensor<ElementInputA, LayoutInputA> a;
  cutlass::HostTensor<ElementInputB, LayoutInputB> b;
  cutlass::HostTensor<ElementOutput, LayoutOutput> c;

  DeviceProblem(int m, int n, int k, int seed)
      : a(cutlass::gemm::GemmCoord(m, n, k).mk()),
        b(cutlass::gemm::GemmCoord(m, n, k).kn()),
        c(cutlass::gemm::GemmCoord(m, n, k).mn()) {
    cutlass::reference::host::TensorFillRandomUniform(
        a.host_view(), 1, ElementInputA(4), ElementInputA(-4), seed);
    cutlass::reference::host::TensorFillRandomUniform(
        b.host_view(), 1, ElementInputB(4), ElementInputB(-4), seed + 1);
    cutlass::reference::host::TensorFill(c.host_view(), ElementOutput(0));
    a.sync_device();
    b.sync_device();
    c.sync_device();
  }
};

double benchmark_pid(int pid, int m, int n, int k, int split_k, int warmup,
                     int iters, DeviceProblem &problem) {
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  for (int i = 0; i < warmup; ++i) {
    moonpoly::run_moonpoly_selected_gemm_fp16_row_col_col_with_workspace(
        pid, m, n, k, problem.a.device_data(), problem.b.device_data(),
        problem.c.device_data(), alpha, beta, nullptr, 0, split_k);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    moonpoly::run_moonpoly_selected_gemm_fp16_row_col_col_with_workspace(
        pid, m, n, k, problem.a.device_data(), problem.b.device_data(),
        problem.c.device_data(), alpha, beta, nullptr, 0, split_k);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return static_cast<double>(ms) / static_cast<double>(iters);
}

int main(int argc, char **argv) {
  std::string csv_path =
      "artifacts/benchmarks/fp16_rcc/splitk_reduction_measurements.csv";
  std::string shape_list = "8x10240x5120,16x5120x13824";
  std::string pid_list = "4,6,7,12,13,15,19,20,23,24,25,29,31,32,33,37,39";
  std::string split_k_list = "2,4";
  int warmup = 5;
  int iters = 20;
  int num_sms = 108;
  int seed = 2026;
  bool measure_full_single = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for arg: " << arg << std::endl;
        std::exit(EXIT_FAILURE);
      }
      return std::string(argv[++i]);
    };
    if (arg == "--csv") {
      csv_path = next();
    } else if (arg == "--shape-list") {
      shape_list = next();
    } else if (arg == "--pid-list") {
      pid_list = next();
    } else if (arg == "--split-k-list") {
      split_k_list = next();
    } else if (arg == "--warmup") {
      warmup = std::stoi(next());
    } else if (arg == "--iters") {
      iters = std::stoi(next());
    } else if (arg == "--num-sms") {
      num_sms = std::stoi(next());
    } else if (arg == "--seed") {
      seed = std::stoi(next());
    } else if (arg == "--measure-full-single") {
      measure_full_single = true;
    } else if (arg == "--help") {
      std::cout
          << "Usage: moonpoly_profile_fp16_splitk_reduction [options]\n"
          << "  --csv <path>\n"
          << "  --shape-list <M1xN1xK1,M2xN2xK2>  Linear shapes\n"
          << "  --pid-list <csv>\n"
          << "  --split-k-list <csv>\n"
          << "  --warmup <int>\n"
          << "  --iters <int>\n"
          << "  --num-sms <int>\n"
          << "  --measure-full-single\n";
      return 0;
    } else {
      std::cerr << "Unknown arg: " << arg << std::endl;
      return 1;
    }
  }

  std::vector<Shape> shapes = parse_shape_list(shape_list);
  std::vector<int> pids = parse_int_list(pid_list);
  std::vector<int> split_ks = parse_int_list(split_k_list);
  if (shapes.empty() || pids.empty() || split_ks.empty()) {
    std::cerr << "shape/pid/split-k lists cannot be empty" << std::endl;
    return 1;
  }

  std::filesystem::path out_path(csv_path);
  if (!out_path.parent_path().empty()) {
    std::filesystem::create_directories(out_path.parent_path());
  }
  std::ofstream ofs(csv_path);
  if (!ofs.is_open()) {
    std::cerr << "Failed to open csv: " << csv_path << std::endl;
    return 1;
  }
  ofs << "M,N,K,pid,split_k,valid,runtime_ok,alignment,tile_m,tile_n,tile_k,"
         "output_tiles,waves_single,waves_split,part_k,full_single_ms,"
         "part_single_ms,measured_ms,pipe_ms,reduction_ms,error\n";
  ofs << std::fixed << std::setprecision(8);

  std::cout << "[splitk reduction eval] shapes=" << shapes.size()
            << " pids=" << pids.size() << " split_ks=" << split_ks.size()
            << " warmup=" << warmup << " iters=" << iters << std::endl;

  for (Shape const &shape : shapes) {
    const int rcc_m = shape.n;
    const int rcc_n = shape.m;
    for (int pid : pids) {
      int alignment = 1;
      int shared[3] = {1, 1, 1};
      moonpoly::get_gemm_alignment_fp16_row_col_col(pid, &alignment);
      moonpoly::get_gemm_shared_mem_fp16_row_col_col(pid, shared);
      const int tile_m = shared[0];
      const int tile_n = shared[1];
      const int tile_k = shared[2];
      const int output_tiles =
          std::max(1, ceil_div(rcc_m, tile_m) * ceil_div(rcc_n, tile_n));
      const int waves_single = std::max(1, ceil_div(output_tiles, num_sms));

      for (int split_k : split_ks) {
        const int part_k = ceil_div(shape.k, split_k);
        const int waves_split =
            std::max(1, ceil_div(output_tiles * split_k, num_sms));
        bool valid = split_k > 1 && shape.k % alignment == 0 &&
                     part_k % alignment == 0;

        double measured_ms = std::numeric_limits<double>::quiet_NaN();
        double part_single_ms = std::numeric_limits<double>::quiet_NaN();
        double full_single_ms = std::numeric_limits<double>::quiet_NaN();
        double pipe_ms = std::numeric_limits<double>::quiet_NaN();
        double reduction_ms = std::numeric_limits<double>::quiet_NaN();
        bool runtime_ok = false;
        std::string error;

        if (valid) {
          DeviceProblem full_problem(rcc_m, rcc_n, shape.k,
                                     seed + pid * 17 + split_k);
          measured_ms = benchmark_pid(pid, rcc_m, rcc_n, shape.k, split_k,
                                      warmup, iters, full_problem);

          DeviceProblem part_problem(rcc_m, rcc_n, part_k,
                                     seed + pid * 31 + split_k);
          part_single_ms = benchmark_pid(pid, rcc_m, rcc_n, part_k, 1, warmup,
                                         iters, part_problem);
          pipe_ms = part_single_ms *
                    (static_cast<double>(waves_split) /
                     static_cast<double>(waves_single));
          reduction_ms = std::max(0.0, measured_ms - pipe_ms);

          if (measure_full_single) {
            full_single_ms = benchmark_pid(pid, rcc_m, rcc_n, shape.k, 1,
                                           warmup, iters, full_problem);
          }
          runtime_ok = true;
        } else {
          error = "unaligned_or_invalid_split";
        }

        ofs << shape.m << ',' << shape.n << ',' << shape.k << ',' << pid << ','
            << split_k << ',' << (valid ? 1 : 0) << ','
            << (runtime_ok ? 1 : 0) << ',' << alignment << ',' << tile_m << ','
            << tile_n << ',' << tile_k << ',' << output_tiles << ','
            << waves_single << ',' << waves_split << ',' << part_k << ',';
        if (std::isnan(full_single_ms)) {
          ofs << ',';
        } else {
          ofs << full_single_ms << ',';
        }
        ofs << part_single_ms << ',' << measured_ms << ',' << pipe_ms << ','
            << reduction_ms << ',' << error << '\n';

        std::cout << "row shape=(" << shape.m << "," << shape.n << ","
                  << shape.k << ") pid=" << pid << " split_k=" << split_k
                  << " measured_ms=" << measured_ms
                  << " reduction_ms=" << reduction_ms << std::endl;
      }
    }
  }

  std::cout << "[splitk reduction eval] wrote CSV: " << csv_path << std::endl;
  return 0;
}
