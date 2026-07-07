
#include <iostream>
#include <sstream>
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

#define CUTLASS_CHECK(status)                                           \
  {                                                                     \
    cutlass::Status error = status;                                     \
    if (error != cutlass::Status::kSuccess) {                           \
      return -2;                                                        \
    }                                                                   \
  }

#define CUDA_CHECK(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

using ElementAccumulator = float;
using ElementComputeEpilogue = ElementAccumulator;
using ElementInputA = cutlass::half_t;
using ElementInputB = cutlass::half_t;
using ElementOutput = cutlass::half_t;

using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::ColumnMajor;

using MMAOp = cutlass::arch::OpClassTensorOp;

using SmArch = cutlass::arch::Sm80;

using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 64, 64>;
using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 32, 64>;

using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;

using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    1,
    ElementAccumulator,
    ElementComputeEpilogue>;

constexpr int NumStages = 6;

using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutput,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp,
                                         EpilogueOp,
                                         SwizzleThreadBlock,
                                         NumStages,
                                         8, 8,
                                         false>;


int run(int length_m, int length_n, int length_k,
            cutlass::half_t* a_ptr, cutlass::half_t* b_ptr, double& ct_time) {
  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);

  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(
      problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                           // CUTLASS kernel

  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(0);

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                     {a_ptr, length_k}, {b_ptr, length_k},
                                     tensor_d.device_ref(),  // <- reference to matrix C on device
                                     tensor_d.device_ref(),  // <- reference to matrix D on device
                                     {alpha, beta},          // <- tuple of alpha and beta
                                     split_k_slices};        // <- k-dimension split factor

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  // Launch initialized CUTLASS kernel
  status = gemm_op();
  CUTLASS_CHECK(status);

  // Wait for kernels to finish
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  double total_time = 0.0;
  float milliseconds = 0.0;
  int num_iter = 20;
  for (int i = 0; i < num_iter; ++i) {
    cudaEventRecord(start);
    status = gemm_op();
    CUTLASS_CHECK(status);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    total_time += milliseconds;
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  ct_time = total_time/(num_iter*1.0);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 0;
}

int main(int argc, char **argv) {
  int m = 1024;
  int n_begin = 256;
  int n_end = 4096;
  int n_step = 256;
  int k = 4096;

  if (argc == 4) {
    m = std::atoi(argv[1]);
    n_begin = std::atoi(argv[2]);
    n_end = n_begin;
    k = std::atoi(argv[3]);
  } else if (argc == 6) {
    m = std::atoi(argv[1]);
    n_begin = std::atoi(argv[2]);
    n_end = std::atoi(argv[3]);
    n_step = std::atoi(argv[4]);
    k = std::atoi(argv[5]);
  } else if (argc != 1) {
    std::cerr << "usage: " << argv[0] << " [m n k] or [m n_begin n_end n_step k]\n";
    return 1;
  }

  cutlass::gemm::GemmCoord problem_size(m, n_end, k);
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

  // n = 3618
  for (int n = n_begin; n <= n_end; n += n_step) {
    double ct_time = 999999;
    int passed = run(m, n, k, a_ptr, b_ptr, ct_time);
    std::cout << ct_time << "," << std::flush;
  }
  std::cout << std::endl;


  return 0;
}



    
