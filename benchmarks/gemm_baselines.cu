#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#define CUDA_CHECK(status)                                                     \
  do {                                                                         \
    cudaError_t error = (status);                                              \
    if (error != cudaSuccess) {                                                \
      std::cerr << "CUDA error: " << cudaGetErrorString(error)                 \
                << " at line " << __LINE__ << std::endl;                      \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

struct Shape {
  int m;
  int n;
  int k;
};

struct Options {
  std::string backend = "both";
  std::string shapes_path = "benchmarks/data/core/gemm.csv";
  int warmup = 5;
  int iters = 20;
};

using CutlassGemm = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    float>;

static bool parse_int(const std::string &s, int &out) {
  char *end = nullptr;
  long value = std::strtol(s.c_str(), &end, 10);
  if (end == s.c_str() || *end != '\0') {
    return false;
  }
  out = static_cast<int>(value);
  return true;
}

static std::vector<std::string> split_csv_line(const std::string &line) {
  std::vector<std::string> fields;
  std::stringstream ss(line);
  std::string field;
  while (std::getline(ss, field, ',')) {
    fields.push_back(field);
  }
  return fields;
}

static std::vector<Shape> load_shapes(const std::string &path) {
  std::ifstream input(path);
  if (!input) {
    std::cerr << "Cannot open shape CSV: " << path << std::endl;
    std::exit(EXIT_FAILURE);
  }

  std::vector<Shape> shapes;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    auto fields = split_csv_line(line);
    if (fields.size() < 3) {
      continue;
    }

    int m = 0, n = 0, k = 0;
    bool ok = false;
    if (fields.size() == 3) {
      ok = parse_int(fields[0], m) && parse_int(fields[1], n) &&
           parse_int(fields[2], k);
    } else {
      ok = parse_int(fields[1], m) && parse_int(fields[2], n) &&
           parse_int(fields[3], k);
    }
    if (ok) {
      shapes.push_back({m, n, k});
    }
  }

  if (shapes.empty()) {
    std::cerr << "No valid shapes found in " << path << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return shapes;
}

static double tflops(const Shape &shape, double ms) {
  if (ms <= 0.0) {
    return 0.0;
  }
  return (2.0 * shape.m * shape.n * shape.k) / (ms / 1000.0) / 1.0e12;
}

static bool benchmark_cublas(cublasHandle_t handle, const Shape &shape,
                             int warmup, int iters, double &avg_ms) {
  half *A = nullptr;
  half *B = nullptr;
  half *C = nullptr;

  int lda = shape.k;
  int ldb = shape.k;
  int ldc = shape.m;
  size_t size_A = static_cast<size_t>(lda) * shape.m * sizeof(half);
  size_t size_B = static_cast<size_t>(ldb) * shape.n * sizeof(half);
  size_t size_C = static_cast<size_t>(ldc) * shape.n * sizeof(half);

  CUDA_CHECK(cudaMalloc(&A, size_A));
  CUDA_CHECK(cudaMalloc(&B, size_B));
  CUDA_CHECK(cudaMalloc(&C, size_C));
  CUDA_CHECK(cudaMemset(A, 0, size_A));
  CUDA_CHECK(cudaMemset(B, 0, size_B));
  CUDA_CHECK(cudaMemset(C, 0, size_C));

  float alpha = 1.0f;
  float beta = 0.0f;
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

  for (int i = 0; i < warmup; ++i) {
    status = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, shape.m, shape.n,
                          shape.k, &alpha, A, CUDA_R_16F, lda, B, CUDA_R_16F,
                          ldb, &beta, C, CUDA_R_16F, ldc, CUDA_R_32F,
                          CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status != CUBLAS_STATUS_SUCCESS) {
      CUDA_CHECK(cudaFree(A));
      CUDA_CHECK(cudaFree(B));
      CUDA_CHECK(cudaFree(C));
      return false;
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  double total_ms = 0.0;
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    status = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, shape.m, shape.n,
                          shape.k, &alpha, A, CUDA_R_16F, lda, B, CUDA_R_16F,
                          ldb, &beta, C, CUDA_R_16F, ldc, CUDA_R_32F,
                          CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    if (status != CUBLAS_STATUS_SUCCESS) {
      CUDA_CHECK(cudaEventDestroy(start));
      CUDA_CHECK(cudaEventDestroy(stop));
      CUDA_CHECK(cudaFree(A));
      CUDA_CHECK(cudaFree(B));
      CUDA_CHECK(cudaFree(C));
      return false;
    }
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    total_ms += ms;
  }

  avg_ms = total_ms / static_cast<double>(iters);
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(A));
  CUDA_CHECK(cudaFree(B));
  CUDA_CHECK(cudaFree(C));
  return true;
}

static bool benchmark_cutlass(const Shape &shape, int warmup, int iters,
                              double &avg_ms) {
  half *A = nullptr;
  half *B = nullptr;
  half *C = nullptr;

  int lda = shape.k;
  int ldb = shape.k;
  int ldc = shape.m;
  size_t size_A = static_cast<size_t>(lda) * shape.m * sizeof(half);
  size_t size_B = static_cast<size_t>(ldb) * shape.n * sizeof(half);
  size_t size_C = static_cast<size_t>(ldc) * shape.n * sizeof(half);

  CUDA_CHECK(cudaMalloc(&A, size_A));
  CUDA_CHECK(cudaMalloc(&B, size_B));
  CUDA_CHECK(cudaMalloc(&C, size_C));
  CUDA_CHECK(cudaMemset(A, 0, size_A));
  CUDA_CHECK(cudaMemset(B, 0, size_B));
  CUDA_CHECK(cudaMemset(C, 0, size_C));

  float alpha = 1.0f;
  float beta = 0.0f;
  CutlassGemm::Arguments args(
      {shape.m, shape.n, shape.k},
      {reinterpret_cast<cutlass::half_t *>(A), lda},
      {reinterpret_cast<cutlass::half_t *>(B), ldb},
      {reinterpret_cast<cutlass::half_t *>(C), ldc},
      {reinterpret_cast<cutlass::half_t *>(C), ldc},
      {alpha, beta});

  CutlassGemm gemm_op;
  cutlass::Status status = gemm_op.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));
    return false;
  }

  for (int i = 0; i < warmup; ++i) {
    status = gemm_op(args);
    if (status != cutlass::Status::kSuccess) {
      CUDA_CHECK(cudaFree(A));
      CUDA_CHECK(cudaFree(B));
      CUDA_CHECK(cudaFree(C));
      return false;
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  double total_ms = 0.0;
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    status = gemm_op(args);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    if (status != cutlass::Status::kSuccess) {
      CUDA_CHECK(cudaEventDestroy(start));
      CUDA_CHECK(cudaEventDestroy(stop));
      CUDA_CHECK(cudaFree(A));
      CUDA_CHECK(cudaFree(B));
      CUDA_CHECK(cudaFree(C));
      return false;
    }
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    total_ms += ms;
  }

  avg_ms = total_ms / static_cast<double>(iters);
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(A));
  CUDA_CHECK(cudaFree(B));
  CUDA_CHECK(cudaFree(C));
  return true;
}

static Options parse_args(int argc, char **argv) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto require_value = [&](const char *name) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << name << std::endl;
        std::exit(EXIT_FAILURE);
      }
      return argv[++i];
    };

    if (arg == "--backend") {
      opts.backend = require_value("--backend");
    } else if (arg == "--shapes") {
      opts.shapes_path = require_value("--shapes");
    } else if (arg == "--warmup") {
      opts.warmup = std::stoi(require_value("--warmup"));
    } else if (arg == "--iters") {
      opts.iters = std::stoi(require_value("--iters"));
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: " << argv[0]
                << " [--backend cublas|cutlass|both] [--shapes CSV]"
                << " [--warmup N] [--iters N]\n";
      std::exit(EXIT_SUCCESS);
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  if (opts.backend != "cublas" && opts.backend != "cutlass" &&
      opts.backend != "both") {
    std::cerr << "Invalid backend: " << opts.backend << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (opts.warmup < 0 || opts.iters <= 0) {
    std::cerr << "Invalid warmup/iters" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return opts;
}

static void print_result(const char *backend, const Shape &shape, bool ok,
                         double avg_ms) {
  double value = ok ? avg_ms : std::numeric_limits<double>::quiet_NaN();
  double perf = ok ? tflops(shape, avg_ms) : 0.0;
  std::cout << backend << ',' << shape.m << ',' << shape.n << ',' << shape.k
            << ',' << (ok ? "ok" : "fail") << ',' << std::fixed
            << std::setprecision(6) << value << ',' << perf << '\n';
}

int main(int argc, char **argv) {
  Options opts = parse_args(argc, argv);
  std::vector<Shape> shapes = load_shapes(opts.shapes_path);

  cublasHandle_t handle = nullptr;
  if (opts.backend == "cublas" || opts.backend == "both") {
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::cerr << "Failed to create cuBLAS handle" << std::endl;
      return EXIT_FAILURE;
    }
  }

  std::cout << "backend,m,n,k,status,avg_ms,tflops\n";
  for (const Shape &shape : shapes) {
    if (opts.backend == "cublas" || opts.backend == "both") {
      double avg_ms = 0.0;
      bool ok = benchmark_cublas(handle, shape, opts.warmup, opts.iters, avg_ms);
      print_result("cublas", shape, ok, avg_ms);
    }
    if (opts.backend == "cutlass" || opts.backend == "both") {
      double avg_ms = 0.0;
      bool ok = benchmark_cutlass(shape, opts.warmup, opts.iters, avg_ms);
      print_result("cutlass", shape, ok, avg_ms);
    }
  }

  if (handle != nullptr) {
    cublasDestroy(handle);
  }
  return EXIT_SUCCESS;
}
