
#include <iostream>
#include <sstream>
#include <cmath>
#include <cstdlib>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "cuda_runtime.h"

#define CUTLASS_CHECK(status)                                                      \
  {                                                                                \
    cutlass::Status error = status;                                                \
    if (error != cutlass::Status::kSuccess) {                                      \
      std::cerr << "Got bad cutlass status, at line: " << __LINE__ << std::endl;   \
      exit(EXIT_FAILURE);                                                          \
    }                                                                              \
  }

#define CUDA_CHECK(status)                                                         \
  {                                                                                \
    cudaError_t error = status;                                                     \
    if (error != cudaSuccess) {                                                     \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error)             \
                << " at line: " << __LINE__ << std::endl;                          \
      exit(EXIT_FAILURE);                                                           \
    }                                                                               \
  }

using ElementAccumulator = float;
using ElementComputeEpilogue = ElementAccumulator;
using ElementInputA = cutlass::half_t;
using ElementInputB = cutlass::half_t;
using ElementOutput = cutlass::half_t;

using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::ColumnMajor;

using CUTLASS_GEMM_OP0 = cutlass::gemm::device::Gemm<
        cutlass::half_t, cutlass::layout::RowMajor,
        cutlass::half_t, cutlass::layout::ColumnMajor,
        cutlass::half_t, cutlass::layout::ColumnMajor,
        float, cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<256, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
        3, 8, 8, true
>;

using CUTLASS_GEMM_OP1 = cutlass::gemm::device::Gemm<
        cutlass::half_t, cutlass::layout::RowMajor,
        cutlass::half_t, cutlass::layout::ColumnMajor,
        cutlass::half_t, cutlass::layout::ColumnMajor,
        float, cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<64, 64, 64>,
        cutlass::gemm::GemmShape<32, 32, 64>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
        6, 8, 8, false
>;

bool split_n_dimension(int length_n, int& op0_n, int& op1_n, int loop) {
  int kernel0_tile_n = CUTLASS_GEMM_OP0::ThreadblockShape::kN;
  op0_n = loop * kernel0_tile_n;
  op1_n = length_n-op0_n;

  return op0_n > 0 && op1_n > 0;
}

int run(int length_m, int length_n, int length_k,
            cutlass::half_t* A, cutlass::half_t* B,
            double& ct_time, int loop) {

  int area0_n = 0, area1_n = 0;
  if (!split_n_dimension(length_n, area0_n, area1_n, loop)) {
    ct_time = 999999;
    return -1;
  }

  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(0);
  int splitk_val = 1;

  cutlass::gemm::device::TwinGemm<CUTLASS_GEMM_OP0, CUTLASS_GEMM_OP1> twin_gemm;

  // ======================= Primary 0 =================================
  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);

  // Initialize tensors using CUTLASS helper functions
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(problem_size.mn());

  cutlass::half_t* C = tensor_c.device_data();

  cutlass::Status status;
  status = twin_gemm.initialize_op0(length_m, area0_n, length_k,
                                        A, B, C, splitk_val, alpha, beta);
  CUTLASS_CHECK(status);

  status = twin_gemm.initialize_op1(length_m, area1_n, length_k,
                                        A, B+(area0_n*length_k), C+(area0_n*length_m),
                                        splitk_val, alpha, beta);
  CUTLASS_CHECK(status);

  status = twin_gemm.run();
  CUTLASS_CHECK(status);

  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  double total_time = 0.0;
  float milliseconds = 0.0;
  int num_iter = 20;
  for (int i = 0; i < num_iter; i ++) {
    cudaEventRecord(start);
    status = twin_gemm.run();
    CUTLASS_CHECK(status);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    total_time += milliseconds;
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  ct_time = total_time/(num_iter*1.0);

  return 0;
}

int main(int argc, char* argv[]) {
  int m = 1024;
  int n = 4096;
  int k = 4096;
  int loop = 24;

  if (argc == 5) {
    m = std::atoi(argv[1]);
    n = std::atoi(argv[2]);
    k = std::atoi(argv[3]);
    loop = std::atoi(argv[4]);
  } else if (argc != 1) {
    std::cerr << "usage: " << argv[0] << " [m n k split_loop]\n";
    return 1;
  }

  cutlass::gemm::GemmCoord problem_size(m, n, k);
  // Initialize tensors using CUTLASS helper functions
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
      problem_size.mk());  // <- Create matrix A with dimensions M x K
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
      problem_size.kn());  // <- Create matrix B with dimensions K x N
  tensor_a.sync_device();
  tensor_b.sync_device();
  CUDA_CHECK(cudaDeviceSynchronize());

  cutlass::half_t* a_ptr = const_cast<cutlass::half_t*>(tensor_a.device_data());
  cutlass::half_t* b_ptr = const_cast<cutlass::half_t*>(tensor_b.device_data());

  double ct_time = 999999;
  int passed = run(m, n, k, a_ptr, b_ptr, ct_time, loop);
  if (passed != 0) {
    std::cerr << "invalid Pattern2 split: n=" << n << ", loop=" << loop << std::endl;
    return 1;
  }
  std::cout << ct_time << "," << std::flush;
  std::cout << std::endl;

  return 0;
}

    
