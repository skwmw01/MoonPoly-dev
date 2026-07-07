#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
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
      std::cerr << "CUDA error: " << cudaGetErrorString(error)                \
                << " at line: " << __LINE__ << std::endl;                    \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  }

namespace moonpoly::fp32::simt::row_col_col {
extern float gemm_cost_model(int ims, int ins, int iks, int pid);
extern float gemm_splitk_cost_model(int ims, int ins, int iks,
                                    int split_k_slices);
extern int cutlass_gemm_predict(int m, int n, int k);
}  // namespace moonpoly::fp32::simt::row_col_col

using ElementInputA = float;
using ElementInputB = float;
using ElementOutput = float;
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::ColumnMajor;

struct Shape {
  int m;
  int n;
  int k;
};

std::vector<int> parse_int_list(std::string const& s) {
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

bool benchmark_pid(int pid, int m, int n, int k, int warmup, int iters,
                   float const* a, float const* b, float* c,
                   double& out_ms) {
  float alpha = 1.0f;
  float beta = 0.0f;
  for (int i = 0; i < warmup; ++i) {
    moonpoly::run_moonpoly_selected_gemm_fp32_row_col_col(
        pid, m, n, k, a, b, c, alpha, beta);
  }
  cudaError_t warmup_error = cudaGetLastError();
  if (warmup_error != cudaSuccess) {
    return false;
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    moonpoly::run_moonpoly_selected_gemm_fp32_row_col_col(
        pid, m, n, k, a, b, c, alpha, beta);
  }
  cudaError_t bench_error = cudaGetLastError();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  if (bench_error != cudaSuccess) {
    return false;
  }
  out_ms = static_cast<double>(ms) / static_cast<double>(iters);
  return true;
}

bool benchmark_online_splitk(int split_k, int m, int n, int k, int warmup,
                             int iters, float const* a, float const* b,
                             float* c, double& out_ms) {
  setenv("MOONPOLY_FP32_PATTERN3_FORCE_SPLITK",
         std::to_string(split_k).c_str(), 1);
  float alpha = 1.0f;
  float beta = 0.0f;
  try {
    for (int i = 0; i < warmup; ++i) {
      moonpoly::run_fp32_gemm(m, n, k, a, false, b, true, c, true, alpha, beta);
    }
  } catch (std::exception const&) {
    unsetenv("MOONPOLY_FP32_PATTERN3_FORCE_SPLITK");
    cudaGetLastError();
    return false;
  }
  cudaError_t warmup_error = cudaGetLastError();
  if (warmup_error != cudaSuccess) {
    unsetenv("MOONPOLY_FP32_PATTERN3_FORCE_SPLITK");
    return false;
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  try {
    for (int i = 0; i < iters; ++i) {
      moonpoly::run_fp32_gemm(m, n, k, a, false, b, true, c, true, alpha, beta);
    }
  } catch (std::exception const&) {
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    unsetenv("MOONPOLY_FP32_PATTERN3_FORCE_SPLITK");
    cudaGetLastError();
    return false;
  }
  cudaError_t bench_error = cudaGetLastError();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  unsetenv("MOONPOLY_FP32_PATTERN3_FORCE_SPLITK");

  if (bench_error != cudaSuccess) {
    return false;
  }
  out_ms = static_cast<double>(ms) / static_cast<double>(iters);
  return true;
}

int legacy_best_only(int m, int n, int k) {
  cutlass::gemm::GemmCoord problem_size(m, n, k);
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(problem_size.mk());
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(problem_size.kn());
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(problem_size.mn());

  cutlass::reference::host::TensorFillRandomUniform(
      tensor_a.host_view(), 1, ElementInputA(4.0f), ElementInputA(-4.0f), 0);
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_b.host_view(), 1, ElementInputB(4.0f), ElementInputB(-4.0f), 0);
  cutlass::reference::host::TensorFill(tensor_c.host_view());
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();

  int best_pid = -1;
  double best_time = std::numeric_limits<double>::max();
  for (int pid = 0; pid < 40; ++pid) {
    int alignment = 1;
    moonpoly::get_gemm_alignment_fp32_row_col_col(pid, &alignment);
    if (k % alignment != 0) {
      continue;
    }
    double measured_ms = std::numeric_limits<double>::quiet_NaN();
    if (benchmark_pid(pid, m, n, k, 5, 20, tensor_a.device_data(),
                      tensor_b.device_data(), tensor_c.device_data(),
                      measured_ms) && measured_ms < best_time) {
      best_time = measured_ms;
      best_pid = pid;
    }
  }

  if (best_pid >= 0) {
    std::cout << best_pid << ',' << best_time << std::endl;
    return 0;
  }
  std::cout << "-1,failed" << std::endl;
  return 1;
}

int main(int argc, char** argv) {
  std::string csv_path;
  std::string splitk_csv_path;
  std::string m_list = "16,64,256";
  std::string n_list = "1024,4096";
  std::string k_list = "1024,4096";
  std::string pid_list;
  std::string split_k_list = "2,4,8";
  int warmup = 5;
  int iters = 20;
  int max_shapes = 0;
  bool no_measure = false;
  bool verbose = false;
  std::vector<std::string> positional;

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
    } else if (arg == "--splitk-csv") {
      splitk_csv_path = next();
    } else if (arg == "--m-list") {
      m_list = next();
    } else if (arg == "--n-list") {
      n_list = next();
    } else if (arg == "--k-list") {
      k_list = next();
    } else if (arg == "--warmup") {
      warmup = std::stoi(next());
    } else if (arg == "--iters") {
      iters = std::stoi(next());
    } else if (arg == "--max-shapes") {
      max_shapes = std::stoi(next());
    } else if (arg == "--pid-list") {
      pid_list = next();
    } else if (arg == "--split-k-list") {
      split_k_list = next();
    } else if (arg == "--no-measure") {
      no_measure = true;
    } else if (arg == "--verbose") {
      verbose = true;
    } else if (arg == "--help") {
      std::cout
          << "Usage: moonpoly_profile_fp32_rcc [m n k] [options]\n"
          << "  --csv <path>        Output per-pid measurement CSV\n"
          << "  --splitk-csv <path> Output Pattern3 split-K measurement CSV\n"
          << "  --m-list <csv>      M list\n"
          << "  --n-list <csv>      N list\n"
          << "  --k-list <csv>      K list\n"
          << "  --warmup <int>      Warmup iterations\n"
          << "  --iters <int>       Measured iterations\n"
          << "  --max-shapes <int>  Stop after N shapes, 0=no limit\n"
          << "  --pid-list <csv>    Evaluate only selected pids\n"
          << "  --split-k-list <csv> Evaluate selected Pattern3 split factors\n"
          << "  --no-measure        Dump predictor metadata only\n";
      return 0;
    } else if (!arg.empty() && arg[0] != '-') {
      positional.push_back(arg);
    } else {
      std::cerr << "Unknown arg: " << arg << std::endl;
      return 1;
    }
  }

  if (csv_path.empty() && splitk_csv_path.empty()) {
    int m = positional.size() >= 3 ? std::stoi(positional[0]) : 4096;
    int n = positional.size() >= 3 ? std::stoi(positional[1]) : 4096;
    int k = positional.size() >= 3 ? std::stoi(positional[2]) : 4096;
    return legacy_best_only(m, n, k);
  }

  std::vector<Shape> shapes;
  for (int m : parse_int_list(m_list)) {
    for (int n : parse_int_list(n_list)) {
      for (int k : parse_int_list(k_list)) {
        shapes.push_back({m, n, k});
      }
    }
  }
  if (max_shapes > 0 && static_cast<int>(shapes.size()) > max_shapes) {
    shapes.resize(max_shapes);
  }

  constexpr int num_pid = 40;
  std::vector<int> eval_pids;
  if (pid_list.empty()) {
    for (int pid = 0; pid < num_pid; ++pid) {
      eval_pids.push_back(pid);
    }
  } else {
    eval_pids = parse_int_list(pid_list);
  }

  if (!splitk_csv_path.empty()) {
    std::vector<int> split_ks = parse_int_list(split_k_list);
    std::filesystem::path out_path(splitk_csv_path);
    if (!out_path.parent_path().empty()) {
      std::filesystem::create_directories(out_path.parent_path());
    }
    std::ofstream ofs(splitk_csv_path);
    if (!ofs.is_open()) {
      std::cerr << "Failed to open csv: " << splitk_csv_path << std::endl;
      return 1;
    }
    ofs << "m,n,k,split_k,valid,runtime_ok,pred_cost,measured_ms\n";
    ofs << std::fixed << std::setprecision(8);
    std::cout << "[fp32 splitk profile] shapes=" << shapes.size()
              << ", split_ks=" << split_ks.size() << ", warmup=" << warmup
              << ", iters=" << iters << std::endl;

    for (Shape shape : shapes) {
      std::unique_ptr<cutlass::HostTensor<ElementInputA, LayoutInputA>> tensor_a;
      std::unique_ptr<cutlass::HostTensor<ElementInputB, LayoutInputB>> tensor_b;
      std::unique_ptr<cutlass::HostTensor<ElementOutput, LayoutOutput>> tensor_c;
      bool allocation_ok = true;
      if (!no_measure) {
        try {
          cutlass::gemm::GemmCoord problem_size(shape.m, shape.n, shape.k);
          tensor_a =
              std::make_unique<cutlass::HostTensor<ElementInputA, LayoutInputA>>(
                  problem_size.mk());
          tensor_b =
              std::make_unique<cutlass::HostTensor<ElementInputB, LayoutInputB>>(
                  problem_size.kn());
          tensor_c =
              std::make_unique<cutlass::HostTensor<ElementOutput, LayoutOutput>>(
                  problem_size.mn());
          cutlass::reference::host::TensorFillRandomUniform(
              tensor_a->host_view(), 1, ElementInputA(4), ElementInputA(-4), 2026);
          cutlass::reference::host::TensorFillRandomUniform(
              tensor_b->host_view(), 1, ElementInputB(4), ElementInputB(-4), 2027);
          cutlass::reference::host::TensorFill(tensor_c->host_view());
          tensor_a->sync_device();
          tensor_b->sync_device();
          tensor_c->sync_device();
        } catch (std::exception const&) {
          allocation_ok = false;
          cudaGetLastError();
        }
      }
      for (int split_k : split_ks) {
        bool runtime_ok = false;
        double measured_ms = std::numeric_limits<double>::quiet_NaN();
        double pred_cost =
            moonpoly::fp32::simt::row_col_col::gemm_splitk_cost_model(
                shape.n, shape.m, shape.k, split_k);
        if (!no_measure && allocation_ok) {
          try {
            runtime_ok = benchmark_online_splitk(
                split_k, shape.m, shape.n, shape.k, warmup, iters,
                tensor_a->device_data(), tensor_b->device_data(),
                tensor_c->device_data(), measured_ms);
          } catch (std::exception const&) {
            runtime_ok = false;
            cudaGetLastError();
          }
        }
        bool valid = no_measure ? std::isfinite(pred_cost) : runtime_ok;
        ofs << shape.m << ',' << shape.n << ',' << shape.k << ','
            << split_k << ',' << (valid ? 1 : 0) << ','
            << (runtime_ok ? 1 : 0) << ',';
        if (std::isnan(pred_cost)) {
          ofs << "nan,";
        } else {
          ofs << pred_cost << ',';
        }
        if (std::isnan(measured_ms)) {
          ofs << "nan\n";
        } else {
          ofs << measured_ms << '\n';
        }
      }
      if (!no_measure) {
        CUDA_CHECK(cudaDeviceSynchronize());
      }
    }
    std::cout << "[fp32 splitk profile] wrote CSV: " << splitk_csv_path
              << std::endl;
    return 0;
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
  ofs << "m,n,k,pid,valid,runtime_ok,alignment,pred_cost,measured_ms,predicted_pid\n";
  ofs << std::fixed << std::setprecision(8);

  std::cout << "[fp32 cost-model profile] shapes=" << shapes.size()
            << ", num_pid=" << num_pid << ", warmup=" << warmup
            << ", iters=" << iters << std::endl;

  for (size_t si = 0; si < shapes.size(); ++si) {
    Shape shape = shapes[si];
    if (verbose) {
      std::cout << "shape " << (si + 1) << '/' << shapes.size() << ": ("
                << shape.m << ',' << shape.n << ',' << shape.k << ')'
                << std::endl;
    }

    std::unique_ptr<cutlass::HostTensor<ElementInputA, LayoutInputA>> tensor_a;
    std::unique_ptr<cutlass::HostTensor<ElementInputB, LayoutInputB>> tensor_b;
    std::unique_ptr<cutlass::HostTensor<ElementOutput, LayoutOutput>> tensor_c;
    float const* a_ptr = nullptr;
    float const* b_ptr = nullptr;
    float* c_ptr = nullptr;
    if (!no_measure) {
      cutlass::gemm::GemmCoord problem_size(shape.m, shape.n, shape.k);
      tensor_a = std::make_unique<cutlass::HostTensor<ElementInputA, LayoutInputA>>(problem_size.mk());
      tensor_b = std::make_unique<cutlass::HostTensor<ElementInputB, LayoutInputB>>(problem_size.kn());
      tensor_c = std::make_unique<cutlass::HostTensor<ElementOutput, LayoutOutput>>(problem_size.mn());
      cutlass::reference::host::TensorFillRandomUniform(
          tensor_a->host_view(), 1, ElementInputA(4), ElementInputA(-4), 2026);
      cutlass::reference::host::TensorFillRandomUniform(
          tensor_b->host_view(), 1, ElementInputB(4), ElementInputB(-4), 2027);
      cutlass::reference::host::TensorFill(tensor_c->host_view());
      tensor_a->sync_device();
      tensor_b->sync_device();
      tensor_c->sync_device();
      a_ptr = tensor_a->device_data();
      b_ptr = tensor_b->device_data();
      c_ptr = tensor_c->device_data();
    }

    int predicted_pid = moonpoly::fp32::simt::row_col_col::cutlass_gemm_predict(
        shape.n, shape.m, shape.k);
    for (int pid : eval_pids) {
      if (pid < 0 || pid >= num_pid) {
        continue;
      }
      int alignment = 1;
      moonpoly::get_gemm_alignment_fp32_row_col_col(pid, &alignment);
      bool alignment_ok = shape.k % alignment == 0;
      bool runtime_ok = false;
      double measured_ms = std::numeric_limits<double>::quiet_NaN();
      double pred_cost = std::numeric_limits<double>::quiet_NaN();
      if (alignment_ok) {
        pred_cost = moonpoly::fp32::simt::row_col_col::gemm_cost_model(
            shape.n, shape.m, shape.k, pid);
        if (!no_measure) {
          runtime_ok = benchmark_pid(pid, shape.m, shape.n, shape.k, warmup,
                                     iters, a_ptr, b_ptr, c_ptr, measured_ms);
        }
      }
      bool valid = no_measure ? alignment_ok : (alignment_ok && runtime_ok);
      ofs << shape.m << ',' << shape.n << ',' << shape.k << ',' << pid << ','
          << (valid ? 1 : 0) << ',' << (runtime_ok ? 1 : 0) << ','
          << alignment << ',';
      if (std::isnan(pred_cost)) {
        ofs << "nan,";
      } else {
        ofs << pred_cost << ',';
      }
      if (std::isnan(measured_ms)) {
        ofs << "nan,";
      } else {
        ofs << measured_ms << ',';
      }
      ofs << predicted_pid << '\n';
    }
    if (!no_measure) {
      CUDA_CHECK(cudaDeviceSynchronize());
    }
  }

  std::cout << "[fp32 cost-model profile] wrote CSV: " << csv_path << std::endl;
  return 0;
}
