#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include <cstdlib>

#include "cuda_runtime.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "moonpoly.cuh"

#define CUDA_CHECK(status)                                                     \
  {                                                                            \
    cudaError_t error = status;                                                \
    if (error != cudaSuccess) {                                                \
      std::cerr << "CUDA error: " << cudaGetErrorString(error)               \
                << " at line: " << __LINE__ << std::endl;                    \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  }

namespace moonpoly::fp16::row_col_col {
extern float gemm_cost_model(int ims, int ins, int iks, int pid);
extern int cutlass_gemm_predict(int m, int n, int k);
}  // namespace moonpoly::fp16::row_col_col

using ElementInputA = cutlass::half_t;
using ElementInputB = cutlass::half_t;
using ElementOutput = cutlass::half_t;
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;

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
    if (token.empty()) {
      continue;
    }
    vals.push_back(std::stoi(token));
  }
  return vals;
}

bool benchmark_pid(int pid,
                   int m,
                   int n,
                   int k,
                   int warmup,
                   int iters,
                   cutlass::half_t const* a,
                   cutlass::half_t const* b,
                   cutlass::half_t* c,
                   double& out_ms) {
  float alpha = 1.0f;
  float beta = 0.0f;

  try {
    for (int i = 0; i < warmup; ++i) {
      moonpoly::run_moonpoly_selected_gemm_fp16_row_col_col(
          pid, m, n, k, a, b, c, alpha, beta);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
      moonpoly::run_moonpoly_selected_gemm_fp16_row_col_col(
          pid, m, n, k, a, b, c, alpha, beta);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    out_ms = static_cast<double>(ms) / static_cast<double>(iters);
    return true;
  } catch (std::exception const& e) {
    (void)e;
    cudaGetLastError();  // clear sticky CUDA error after failed kernel launch
    return false;
  }
}

int main(int argc, char** argv) {
  std::string csv_path = "benchmarks/data/fp16_predictor/fp16_predictor_eval_raw.csv";
  std::string m_list = "16,64,256";
  std::string n_list = "1024,4096";
  std::string k_list = "1024,4096";
  int warmup = 5;
  int iters = 20;
  int max_shapes = 0;
  std::string pid_list;
  bool no_measure = false;
  bool verbose = false;

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
    } else if (arg == "--no-measure") {
      no_measure = true;
    } else if (arg == "--verbose") {
      verbose = true;
    } else if (arg == "--help") {
      std::cout
          << "Usage: moonpoly_profile_fp16_predictor [options]\n"
          << "  --csv <path>        Output CSV path\n"
          << "  --m-list <csv>      M list, e.g. 16,64,256\n"
          << "  --n-list <csv>      N list\n"
          << "  --k-list <csv>      K list\n"
          << "  --warmup <int>      Warmup iterations (default 5)\n"
          << "  --iters <int>       Measured iterations (default 20)\n"
          << "  --max-shapes <int>  Stop after N shapes (0=no limit)\n"
          << "  --pid-list <csv>    Evaluate only these pids, e.g. 0,3,7\n"
          << "  --no-measure        Only dump predictor/cost metadata\n"
          << "  --verbose           Verbose logs\n";
      return 0;
    } else {
      std::cerr << "Unknown arg: " << arg << std::endl;
      return 1;
    }
  }

  if (warmup < 0 || iters <= 0 || max_shapes < 0) {
    std::cerr << "Invalid args: warmup>=0, iters>0, max-shapes>=0 required"
              << std::endl;
    return 1;
  }

  std::vector<int> ms = parse_int_list(m_list);
  std::vector<int> ns = parse_int_list(n_list);
  std::vector<int> ks = parse_int_list(k_list);

  if (ms.empty() || ns.empty() || ks.empty()) {
    std::cerr << "M/N/K list cannot be empty" << std::endl;
    return 1;
  }

  std::vector<Shape> shapes;
  for (int m : ms) {
    for (int n : ns) {
      for (int k : ks) {
        shapes.push_back({m, n, k});
      }
    }
  }

  if (max_shapes > 0 && static_cast<int>(shapes.size()) > max_shapes) {
    shapes.resize(max_shapes);
  }

  constexpr int num_pid = 41;
  std::vector<int> eval_pids;
  if (pid_list.empty()) {
    eval_pids.resize(num_pid);
    for (int i = 0; i < num_pid; ++i) {
      eval_pids[i] = i;
    }
  } else {
    eval_pids = parse_int_list(pid_list);
  }

  std::vector<int> alignments(num_pid, 1);
  for (int pid = 0; pid < num_pid; ++pid) {
    int align = 1;
    moonpoly::get_gemm_alignment_fp16_row_col_col(pid, &align);
    alignments[pid] = align;
  }

  std::filesystem::path out_path(csv_path);
  if (!out_path.parent_path().empty()) {
    std::filesystem::create_directories(out_path.parent_path());
  }
  std::ofstream ofs(csv_path);
  if (!ofs.is_open()) {
    std::cerr << "Failed to open csv for write: " << csv_path << std::endl;
    return 1;
  }

  ofs << "m,n,k,pid,valid,runtime_ok,alignment,pred_cost,measured_ms,predicted_pid\n";
  ofs << std::fixed << std::setprecision(8);

  std::cout << "[fp16 predictor eval] shapes=" << shapes.size()
            << ", num_pid=" << num_pid << ", warmup=" << warmup
            << ", iters=" << iters << std::endl;

  for (size_t si = 0; si < shapes.size(); ++si) {
    const Shape shape = shapes[si];
    if (verbose) {
      std::cout << "shape " << (si + 1) << "/" << shapes.size() << ": ("
                << shape.m << "," << shape.n << "," << shape.k << ")"
                << std::endl;
    }

    cutlass::half_t const* a_ptr = nullptr;
    cutlass::half_t const* b_ptr = nullptr;
    cutlass::half_t* c_ptr = nullptr;
    std::unique_ptr<cutlass::HostTensor<ElementInputA, LayoutInputA>> tensor_a;
    std::unique_ptr<cutlass::HostTensor<ElementInputB, LayoutInputB>> tensor_b;
    std::unique_ptr<cutlass::HostTensor<ElementOutput, cutlass::layout::ColumnMajor>> tensor_c;

    if (!no_measure) {
      cutlass::gemm::GemmCoord problem_size(shape.m, shape.n, shape.k);
      tensor_a = std::make_unique<cutlass::HostTensor<ElementInputA, LayoutInputA>>(
          problem_size.mk());
      tensor_b = std::make_unique<cutlass::HostTensor<ElementInputB, LayoutInputB>>(
          problem_size.kn());
      tensor_c = std::make_unique<cutlass::HostTensor<ElementOutput, cutlass::layout::ColumnMajor>>(
          problem_size.mn());

      cutlass::reference::host::TensorFillRandomUniform(
          tensor_a->host_view(), 1, ElementInputA(4), ElementInputA(-4), 2026);
      cutlass::reference::host::TensorFillRandomUniform(
          tensor_b->host_view(), 1, ElementInputB(4), ElementInputB(-4), 2027);
      cutlass::reference::host::TensorFill(
          tensor_c->host_view(), ElementOutput(0));

      tensor_a->sync_device();
      tensor_b->sync_device();
      tensor_c->sync_device();
      a_ptr = tensor_a->device_data();
      b_ptr = tensor_b->device_data();
      c_ptr = tensor_c->device_data();
    }

    // Keep predictor call consistent with runtime path in fp16_rcc.cu:
    // cutlass_gemm_predict(n, m, k)
    int predicted_pid = moonpoly::fp16::row_col_col::cutlass_gemm_predict(
        shape.n, shape.m, shape.k);

    for (int pid : eval_pids) {
      if (pid < 0 || pid >= num_pid) {
        continue;
      }
      int align = alignments[pid];
      bool alignment_ok = (shape.k % align == 0);

      double pred_cost = std::numeric_limits<double>::quiet_NaN();
      double measured_ms = std::numeric_limits<double>::quiet_NaN();

      bool runtime_ok = false;
      if (alignment_ok) {
        pred_cost = static_cast<double>(moonpoly::fp16::row_col_col::gemm_cost_model(
            shape.n, shape.m, shape.k, pid));
        if (!no_measure) {
          runtime_ok = benchmark_pid(pid, shape.m, shape.n, shape.k, warmup, iters,
                                     a_ptr, b_ptr, c_ptr, measured_ms);
        }
      }
      bool valid = no_measure ? alignment_ok : (alignment_ok && runtime_ok);

      ofs << shape.m << ',' << shape.n << ',' << shape.k << ',' << pid << ','
          << (valid ? 1 : 0) << ',' << (runtime_ok ? 1 : 0) << ','
          << align << ',';
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

  std::cout << "[fp16 predictor eval] wrote raw CSV: " << csv_path << std::endl;
  return 0;
}
