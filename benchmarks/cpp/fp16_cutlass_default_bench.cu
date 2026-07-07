#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "cuda_runtime.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "moonpoly.cuh"

#define CUDA_CHECK(status)                                                       \
  {                                                                              \
    cudaError_t error = status;                                                  \
    if (error != cudaSuccess) {                                                  \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error)          \
                << " at line: " << __LINE__ << std::endl;                        \
      exit(EXIT_FAILURE);                                                        \
    }                                                                            \
  }

using ElementInputA = cutlass::half_t;
using ElementInputB = cutlass::half_t;
using ElementOutput = cutlass::half_t;
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::ColumnMajor;

using DefaultCutlassGemm =
    cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB,
                                LayoutInputB, ElementOutput, LayoutOutput, float>;

double calc_cosine_diff(
    const cutlass::HostTensor<ElementOutput, LayoutOutput> &x,
    const cutlass::HostTensor<ElementOutput, LayoutOutput> &y) {
  double x_sum = 0.0, y_sum = 0.0, xy_sum = 0.0;
  for (int i = 0; i < x.capacity(); ++i) {
    double xv = static_cast<double>(x.host_data()[i]);
    double yv = static_cast<double>(y.host_data()[i]);
    x_sum += xv * xv;
    y_sum += yv * yv;
    xy_sum += xv * yv;
  }
  double denominator = x_sum + y_sum;
  if (denominator == 0.0) {
    return 1.0;
  }
  return 1.0 - (2.0 * xy_sum / denominator);
}

int run(int m, int n, int k, int warmup, int iters, double &cutlass_ms,
        double &moonpoly_ms) {
  cutlass::gemm::GemmCoord problem_size(m, n, k);

  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(problem_size.mk());
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(problem_size.kn());
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(problem_size.mn());
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d_moonpoly(problem_size.mn());
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d_cutlass(problem_size.mn());

  cutlass::reference::host::TensorFillRandomUniform(tensor_a.host_view(), 1,
                                                    ElementInputA(4),
                                                    ElementInputA(-4), 0);
  cutlass::reference::host::TensorFillRandomUniform(tensor_b.host_view(), 1,
                                                    ElementInputB(4),
                                                    ElementInputB(-4), 0);
  cutlass::reference::host::TensorFill(tensor_c.host_view());
  cutlass::reference::host::TensorFill(tensor_d_moonpoly.host_view());
  cutlass::reference::host::TensorFill(tensor_d_cutlass.host_view());

  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();
  tensor_d_moonpoly.sync_device();
  tensor_d_cutlass.sync_device();

  float alpha = 1.0f;
  float beta = 0.0f;

  DefaultCutlassGemm cutlass_gemm;
  typename DefaultCutlassGemm::Arguments cutlass_args{
      problem_size,
      {tensor_a.device_data(), k},
      {tensor_b.device_data(), k},
      {tensor_c.device_data(), m},
      {tensor_d_cutlass.device_data(), m},
      {alpha, beta},
      1};

  CUTLASS_CHECK(cutlass_gemm.can_implement(cutlass_args));
  size_t workspace_size = DefaultCutlassGemm::get_workspace_size(cutlass_args);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  CUTLASS_CHECK(cutlass_gemm.initialize(cutlass_args, workspace.get()));

  moonpoly::run_fp16_gemm(m, n, k, tensor_a.device_data(), false,
                          tensor_b.device_data(), true,
                          tensor_d_moonpoly.device_data(), true, alpha, beta);
  CUTLASS_CHECK(cutlass_gemm());
  CUDA_CHECK(cudaDeviceSynchronize());

  tensor_d_moonpoly.sync_host();
  tensor_d_cutlass.sync_host();

  double cosine_diff = calc_cosine_diff(tensor_d_moonpoly, tensor_d_cutlass);
  std::cout << "Cosine similarity difference: " << std::fixed
            << std::setprecision(6) << cosine_diff << std::endl;
  if (cosine_diff > 1e-3f || std::isnan(cosine_diff)) {
    std::cout << "Result mismatch between moonpoly and default CUTLASS." << std::endl;
    return -1;
  }

  for (int i = 0; i < warmup; ++i) {
    moonpoly::run_fp16_gemm(m, n, k, tensor_a.device_data(), false,
                            tensor_b.device_data(), true,
                            tensor_d_moonpoly.device_data(), true, alpha, beta);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  for (int i = 0; i < warmup; ++i) {
    CUTLASS_CHECK(cutlass_gemm());
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start = nullptr, stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    moonpoly::run_fp16_gemm(m, n, k, tensor_a.device_data(), false,
                            tensor_b.device_data(), true,
                            tensor_d_moonpoly.device_data(), true, alpha, beta);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float moonpoly_total = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&moonpoly_total, start, stop));
  moonpoly_ms = static_cast<double>(moonpoly_total) / static_cast<double>(iters);

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    CUTLASS_CHECK(cutlass_gemm());
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float cutlass_total = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&cutlass_total, start, stop));
  cutlass_ms = static_cast<double>(cutlass_total) / static_cast<double>(iters);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return 0;
}

int main(int argc, char **argv) {
  int m = 4096;
  int n = 4096;
  int k = 4096;
  int warmup = 5;
  int iters = 20;

  if (argc > 1) {
    m = std::stoi(argv[1]);
    n = std::stoi(argv[2]);
    k = std::stoi(argv[3]);
  }
  if (argc > 4) {
    warmup = std::stoi(argv[4]);
  }
  if (argc > 5) {
    iters = std::stoi(argv[5]);
  }

  double cutlass_ms = 0.0;
  double moonpoly_ms = 0.0;

  int ok = run(m, n, k, warmup, iters, cutlass_ms, moonpoly_ms);
  if (ok != 0) {
    std::cout << "Failed" << std::endl;
    return 1;
  }

  std::cout << "(m, n, k) = (" << m << ", " << n << ", " << k << ")" << std::endl;
  std::cout << "warmup: " << warmup << ", iters: " << iters << std::endl;
  std::cout << "cutlass time: " << std::fixed << std::setprecision(6) << cutlass_ms
            << " ms, moonpoly time: " << moonpoly_ms << " ms, speed_up: "
            << (cutlass_ms / moonpoly_ms) << "x" << std::endl;

  return 0;
}
