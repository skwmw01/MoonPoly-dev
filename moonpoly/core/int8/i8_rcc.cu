#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_splitk_parallel.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "moonpoly.cuh"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
/*
============================ CUTLASS LP GEMM PRIMARY ===========================
*/
namespace moonpoly {
namespace int8{
namespace row_col_col{

extern SharedMem int8_gemm_shared_mem[];
extern int int8_gemm_alignment[];

extern float gemm_cost_model(int ims, int ins, int iks, int pid);
extern float gemm_splitk_cost_model(int ims, int ins, int iks,
                                    int split_k_slices);
extern int cutlass_gemm_predict(int m, int n, int k);

struct Int8RccPlan {
  bool use_splitk;
  int pid;
  int split_k;
  float estimated_ms;
};

bool env_flag(const char *name, bool default_value) {
  const char *raw = std::getenv(name);
  if (raw == nullptr) {
    return default_value;
  }
  return !(std::string(raw) == "0" || std::string(raw) == "false" ||
           std::string(raw) == "FALSE" || std::string(raw) == "off" ||
           std::string(raw) == "OFF");
}

float env_float(const char *name, float default_value) {
  const char *raw = std::getenv(name);
  if (raw == nullptr) {
    return default_value;
  }
  char *end = nullptr;
  float value = std::strtof(raw, &end);
  if (end == raw || !std::isfinite(value)) {
    return default_value;
  }
  return value;
}

int env_int(const char *name, int default_value) {
  const char *raw = std::getenv(name);
  if (raw == nullptr) {
    return default_value;
  }
  char *end = nullptr;
  long value = std::strtol(raw, &end, 10);
  if (end == raw) {
    return default_value;
  }
  return static_cast<int>(value);
}

bool int8_pattern3_k_gate(int k, int split_k) {
  const int min_k = env_int("MOONPOLY_INT8_PATTERN3_MIN_K", 2048);
  const int min_part_k = env_int("MOONPOLY_INT8_PATTERN3_MIN_PART_K", 512);
  if (split_k <= 1 || k < min_k || k < 64 * split_k) {
    return false;
  }
  const int part_k = (k + split_k - 1) / split_k;
  return part_k >= min_part_k;
}

struct CUTLASS_INT8_PATTERN3_SPLITK {
  using ElementA = int8_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = int8_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int32_t;
  using LayoutC = cutlass::layout::ColumnMajor;
  using ElementAccumulator = int32_t;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
  using EpilogueOp =
      cutlass::epilogue::thread::LinearCombination<ElementC, 4,
                                                   ElementAccumulator,
                                                   ElementAccumulator>;
  using ConvertOp =
      cutlass::epilogue::thread::Convert<ElementAccumulator, EpilogueOp::kCount,
                                         ElementAccumulator>;
  using ReductionOp = cutlass::reduction::thread::ReduceAdd<
      ElementAccumulator, EpilogueOp::ElementAccumulator, EpilogueOp::kCount>;
  using Gemm = cutlass::gemm::device::GemmSplitKParallel<
      ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
      ElementAccumulator, OperatorClass, ArchTag, ThreadblockShape, WarpShape,
      InstructionShape, EpilogueOp, ConvertOp, ReductionOp,
      cutlass::gemm::threadblock::GemmSplitKHorizontalThreadblockSwizzle, 3, 16,
      16>;

  static void run(int m, int n, int k, ElementA *A, ElementB *B, ElementC *C,
                  int alpha, int beta, int split_k_slices,
                  cudaStream_t stream) {
    if (split_k_slices <= 1) {
      split_k_slices = 2;
    }
    cutlass::gemm::GemmCoord problem_size(m, n, k);
    typename Gemm::Arguments arguments{problem_size,
                                       {A, k},
                                       {B, k},
                                       {C, m},
                                       {C, m},
                                       {alpha, beta},
                                       split_k_slices};
    Gemm gemm_op;
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);
    status = gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};


struct CUTLASS_int8_gemm_P0 {
private:
  CUTLASS_int8_gemm_P0() { }
  ~CUTLASS_int8_gemm_P0() { }

public: 
  // Gemm operator cutlass_tensorop_i8816gemm_s8_64x128_64x2_tn_align4
  using Operation_cutlass_tensorop_i8816gemm_s8_64x128_64x2_tn_align4 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 128, 64>,
    cutlass::gemm::GemmShape<32, 64, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      4,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    2,
    4,
    4
  >;
using int8_gemm = Operation_cutlass_tensorop_i8816gemm_s8_64x128_64x2_tn_align4;
  int8_gemm int8_gemm_op;
  CUTLASS_int8_gemm_P0(const CUTLASS_int8_gemm_P0 &) = delete;
  CUTLASS_int8_gemm_P0 &operator=(const CUTLASS_int8_gemm_P0 &) = delete;
  static CUTLASS_int8_gemm_P0 &get_instance() {
    static CUTLASS_int8_gemm_P0 instance;
    return instance;
  }

  void run_int8_gemm(int length_m, int length_n, int length_k, int8_t *A,
                   int8_t *B, int32_t *C, int alpha, int beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename int8_gemm::Arguments arguments{problem_size,
                                         {A, length_k},
                                         {B, length_k},
                                         {C, length_m},
                                         {C, length_m},
                                         {alpha, beta},
                                         1};
    size_t workspace_size = int8_gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = int8_gemm_op.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = int8_gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_int8_gemm_P0(int length_m, int length_n, int length_k,
                        int8_t *A, int8_t *B, int32_t *C,
                        int alpha, int beta, cudaStream_t stream) {
  CUTLASS_int8_gemm_P0 &instance = CUTLASS_int8_gemm_P0::get_instance();
  instance.run_int8_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_int8_gemm_P1 {
private:
  CUTLASS_int8_gemm_P1() { }
  ~CUTLASS_int8_gemm_P1() { }

public: 
  // Gemm operator cutlass_tensorop_i16832gemm_s8_256x64_64x3_tn_align8
  using Operation_cutlass_tensorop_i16832gemm_s8_256x64_64x3_tn_align8 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 64, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      4,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    8,
    8
  >;
using int8_gemm = Operation_cutlass_tensorop_i16832gemm_s8_256x64_64x3_tn_align8;
  int8_gemm int8_gemm_op;
  CUTLASS_int8_gemm_P1(const CUTLASS_int8_gemm_P1 &) = delete;
  CUTLASS_int8_gemm_P1 &operator=(const CUTLASS_int8_gemm_P1 &) = delete;
  static CUTLASS_int8_gemm_P1 &get_instance() {
    static CUTLASS_int8_gemm_P1 instance;
    return instance;
  }

  void run_int8_gemm(int length_m, int length_n, int length_k, int8_t *A,
                   int8_t *B, int32_t *C, int alpha, int beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename int8_gemm::Arguments arguments{problem_size,
                                         {A, length_k},
                                         {B, length_k},
                                         {C, length_m},
                                         {C, length_m},
                                         {alpha, beta},
                                         1};
    size_t workspace_size = int8_gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = int8_gemm_op.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = int8_gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_int8_gemm_P1(int length_m, int length_n, int length_k,
                        int8_t *A, int8_t *B, int32_t *C,
                        int alpha, int beta, cudaStream_t stream) {
  CUTLASS_int8_gemm_P1 &instance = CUTLASS_int8_gemm_P1::get_instance();
  instance.run_int8_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_int8_gemm_P2 {
private:
  CUTLASS_int8_gemm_P2() { }
  ~CUTLASS_int8_gemm_P2() { }

public: 
  // Gemm operator cutlass_tensorop_i8816gemm_s8_64x64_64x2_tn_align4
  using Operation_cutlass_tensorop_i8816gemm_s8_64x64_64x2_tn_align4 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      4,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    2,
    4,
    4
  >;
using int8_gemm = Operation_cutlass_tensorop_i8816gemm_s8_64x64_64x2_tn_align4;
  int8_gemm int8_gemm_op;
  CUTLASS_int8_gemm_P2(const CUTLASS_int8_gemm_P2 &) = delete;
  CUTLASS_int8_gemm_P2 &operator=(const CUTLASS_int8_gemm_P2 &) = delete;
  static CUTLASS_int8_gemm_P2 &get_instance() {
    static CUTLASS_int8_gemm_P2 instance;
    return instance;
  }

  void run_int8_gemm(int length_m, int length_n, int length_k, int8_t *A,
                   int8_t *B, int32_t *C, int alpha, int beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename int8_gemm::Arguments arguments{problem_size,
                                         {A, length_k},
                                         {B, length_k},
                                         {C, length_m},
                                         {C, length_m},
                                         {alpha, beta},
                                         1};
    size_t workspace_size = int8_gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = int8_gemm_op.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = int8_gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_int8_gemm_P2(int length_m, int length_n, int length_k,
                        int8_t *A, int8_t *B, int32_t *C,
                        int alpha, int beta, cudaStream_t stream) {
  CUTLASS_int8_gemm_P2 &instance = CUTLASS_int8_gemm_P2::get_instance();
  instance.run_int8_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_int8_gemm_P3 {
private:
  CUTLASS_int8_gemm_P3() { }
  ~CUTLASS_int8_gemm_P3() { }

public: 
  // Gemm operator cutlass_tensorop_i8816gemm_s8_64x64_64x2_tn_align1
  using Operation_cutlass_tensorop_i8816gemm_s8_64x64_64x2_tn_align1 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      1,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    2,
    1,
    1
  >;
using int8_gemm = Operation_cutlass_tensorop_i8816gemm_s8_64x64_64x2_tn_align1;
  int8_gemm int8_gemm_op;
  CUTLASS_int8_gemm_P3(const CUTLASS_int8_gemm_P3 &) = delete;
  CUTLASS_int8_gemm_P3 &operator=(const CUTLASS_int8_gemm_P3 &) = delete;
  static CUTLASS_int8_gemm_P3 &get_instance() {
    static CUTLASS_int8_gemm_P3 instance;
    return instance;
  }

  void run_int8_gemm(int length_m, int length_n, int length_k, int8_t *A,
                   int8_t *B, int32_t *C, int alpha, int beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename int8_gemm::Arguments arguments{problem_size,
                                         {A, length_k},
                                         {B, length_k},
                                         {C, length_m},
                                         {C, length_m},
                                         {alpha, beta},
                                         1};
    size_t workspace_size = int8_gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = int8_gemm_op.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = int8_gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_int8_gemm_P3(int length_m, int length_n, int length_k,
                        int8_t *A, int8_t *B, int32_t *C,
                        int alpha, int beta, cudaStream_t stream) {
  CUTLASS_int8_gemm_P3 &instance = CUTLASS_int8_gemm_P3::get_instance();
  instance.run_int8_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_int8_gemm_P4 {
private:
  CUTLASS_int8_gemm_P4() { }
  ~CUTLASS_int8_gemm_P4() { }

public: 
  // Gemm operator cutlass_tensorop_i8816gemm_s8_64x128_64x2_tn_align1
  using Operation_cutlass_tensorop_i8816gemm_s8_64x128_64x2_tn_align1 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 128, 64>,
    cutlass::gemm::GemmShape<32, 64, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      1,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    2,
    1,
    1
  >;
using int8_gemm = Operation_cutlass_tensorop_i8816gemm_s8_64x128_64x2_tn_align1;
  int8_gemm int8_gemm_op;
  CUTLASS_int8_gemm_P4(const CUTLASS_int8_gemm_P4 &) = delete;
  CUTLASS_int8_gemm_P4 &operator=(const CUTLASS_int8_gemm_P4 &) = delete;
  static CUTLASS_int8_gemm_P4 &get_instance() {
    static CUTLASS_int8_gemm_P4 instance;
    return instance;
  }

  void run_int8_gemm(int length_m, int length_n, int length_k, int8_t *A,
                   int8_t *B, int32_t *C, int alpha, int beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename int8_gemm::Arguments arguments{problem_size,
                                         {A, length_k},
                                         {B, length_k},
                                         {C, length_m},
                                         {C, length_m},
                                         {alpha, beta},
                                         1};
    size_t workspace_size = int8_gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = int8_gemm_op.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = int8_gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_int8_gemm_P4(int length_m, int length_n, int length_k,
                        int8_t *A, int8_t *B, int32_t *C,
                        int alpha, int beta, cudaStream_t stream) {
  CUTLASS_int8_gemm_P4 &instance = CUTLASS_int8_gemm_P4::get_instance();
  instance.run_int8_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_int8_gemm_P5 {
private:
  CUTLASS_int8_gemm_P5() { }
  ~CUTLASS_int8_gemm_P5() { }

public: 
  // Gemm operator cutlass_tensorop_i16832gemm_s8_256x128_128x3_tn_align4
  using Operation_cutlass_tensorop_i16832gemm_s8_256x128_128x3_tn_align4 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 128>,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      4,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    4,
    4
  >;
using int8_gemm = Operation_cutlass_tensorop_i16832gemm_s8_256x128_128x3_tn_align4;
  int8_gemm int8_gemm_op;
  CUTLASS_int8_gemm_P5(const CUTLASS_int8_gemm_P5 &) = delete;
  CUTLASS_int8_gemm_P5 &operator=(const CUTLASS_int8_gemm_P5 &) = delete;
  static CUTLASS_int8_gemm_P5 &get_instance() {
    static CUTLASS_int8_gemm_P5 instance;
    return instance;
  }

  void run_int8_gemm(int length_m, int length_n, int length_k, int8_t *A,
                   int8_t *B, int32_t *C, int alpha, int beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename int8_gemm::Arguments arguments{problem_size,
                                         {A, length_k},
                                         {B, length_k},
                                         {C, length_m},
                                         {C, length_m},
                                         {alpha, beta},
                                         1};
    size_t workspace_size = int8_gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = int8_gemm_op.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = int8_gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_int8_gemm_P5(int length_m, int length_n, int length_k,
                        int8_t *A, int8_t *B, int32_t *C,
                        int alpha, int beta, cudaStream_t stream) {
  CUTLASS_int8_gemm_P5 &instance = CUTLASS_int8_gemm_P5::get_instance();
  instance.run_int8_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_int8_gemm_P6 {
private:
  CUTLASS_int8_gemm_P6() { }
  ~CUTLASS_int8_gemm_P6() { }

public: 
  // Gemm operator cutlass_tensorop_i16832gemm_s8_128x128_128x3_tn_align8
  using Operation_cutlass_tensorop_i16832gemm_s8_128x128_128x3_tn_align8 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 128>,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      4,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    8,
    8
  >;
using int8_gemm = Operation_cutlass_tensorop_i16832gemm_s8_128x128_128x3_tn_align8;
  int8_gemm int8_gemm_op;
  CUTLASS_int8_gemm_P6(const CUTLASS_int8_gemm_P6 &) = delete;
  CUTLASS_int8_gemm_P6 &operator=(const CUTLASS_int8_gemm_P6 &) = delete;
  static CUTLASS_int8_gemm_P6 &get_instance() {
    static CUTLASS_int8_gemm_P6 instance;
    return instance;
  }

  void run_int8_gemm(int length_m, int length_n, int length_k, int8_t *A,
                   int8_t *B, int32_t *C, int alpha, int beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename int8_gemm::Arguments arguments{problem_size,
                                         {A, length_k},
                                         {B, length_k},
                                         {C, length_m},
                                         {C, length_m},
                                         {alpha, beta},
                                         1};
    size_t workspace_size = int8_gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = int8_gemm_op.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = int8_gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_int8_gemm_P6(int length_m, int length_n, int length_k,
                        int8_t *A, int8_t *B, int32_t *C,
                        int alpha, int beta, cudaStream_t stream) {
  CUTLASS_int8_gemm_P6 &instance = CUTLASS_int8_gemm_P6::get_instance();
  instance.run_int8_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_int8_gemm_P7 {
private:
  CUTLASS_int8_gemm_P7() { }
  ~CUTLASS_int8_gemm_P7() { }

public: 
  // Gemm operator cutlass_tensorop_i16832gemm_s8_128x256_128x3_tn_align4
  using Operation_cutlass_tensorop_i16832gemm_s8_128x256_128x3_tn_align4 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 128>,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      4,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    4,
    4
  >;
using int8_gemm = Operation_cutlass_tensorop_i16832gemm_s8_128x256_128x3_tn_align4;
  int8_gemm int8_gemm_op;
  CUTLASS_int8_gemm_P7(const CUTLASS_int8_gemm_P7 &) = delete;
  CUTLASS_int8_gemm_P7 &operator=(const CUTLASS_int8_gemm_P7 &) = delete;
  static CUTLASS_int8_gemm_P7 &get_instance() {
    static CUTLASS_int8_gemm_P7 instance;
    return instance;
  }

  void run_int8_gemm(int length_m, int length_n, int length_k, int8_t *A,
                   int8_t *B, int32_t *C, int alpha, int beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename int8_gemm::Arguments arguments{problem_size,
                                         {A, length_k},
                                         {B, length_k},
                                         {C, length_m},
                                         {C, length_m},
                                         {alpha, beta},
                                         1};
    size_t workspace_size = int8_gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = int8_gemm_op.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = int8_gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_int8_gemm_P7(int length_m, int length_n, int length_k,
                        int8_t *A, int8_t *B, int32_t *C,
                        int alpha, int beta, cudaStream_t stream) {
  CUTLASS_int8_gemm_P7 &instance = CUTLASS_int8_gemm_P7::get_instance();
  instance.run_int8_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_int8_gemm_P8 {
private:
  CUTLASS_int8_gemm_P8() { }
  ~CUTLASS_int8_gemm_P8() { }

public: 
  // Gemm operator cutlass_tensorop_i16832gemm_s8_64x128_128x3_tn_align4
  using Operation_cutlass_tensorop_i16832gemm_s8_64x128_128x3_tn_align4 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 128>,
    cutlass::gemm::GemmShape<32, 64, 128>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      4,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    4,
    4
  >;
using int8_gemm = Operation_cutlass_tensorop_i16832gemm_s8_64x128_128x3_tn_align4;
  int8_gemm int8_gemm_op;
  CUTLASS_int8_gemm_P8(const CUTLASS_int8_gemm_P8 &) = delete;
  CUTLASS_int8_gemm_P8 &operator=(const CUTLASS_int8_gemm_P8 &) = delete;
  static CUTLASS_int8_gemm_P8 &get_instance() {
    static CUTLASS_int8_gemm_P8 instance;
    return instance;
  }

  void run_int8_gemm(int length_m, int length_n, int length_k, int8_t *A,
                   int8_t *B, int32_t *C, int alpha, int beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename int8_gemm::Arguments arguments{problem_size,
                                         {A, length_k},
                                         {B, length_k},
                                         {C, length_m},
                                         {C, length_m},
                                         {alpha, beta},
                                         1};
    size_t workspace_size = int8_gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = int8_gemm_op.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = int8_gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_int8_gemm_P8(int length_m, int length_n, int length_k,
                        int8_t *A, int8_t *B, int32_t *C,
                        int alpha, int beta, cudaStream_t stream) {
  CUTLASS_int8_gemm_P8 &instance = CUTLASS_int8_gemm_P8::get_instance();
  instance.run_int8_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_int8_gemm_P9 {
private:
  CUTLASS_int8_gemm_P9() { }
  ~CUTLASS_int8_gemm_P9() { }

public: 
  // Gemm operator cutlass_tensorop_i16832gemm_s8_64x128_64x6_tn_align16
  using Operation_cutlass_tensorop_i16832gemm_s8_64x128_64x6_tn_align16 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 64>,
    cutlass::gemm::GemmShape<32, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      4,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    6,
    16,
    16
  >;
using int8_gemm = Operation_cutlass_tensorop_i16832gemm_s8_64x128_64x6_tn_align16;
  int8_gemm int8_gemm_op;
  CUTLASS_int8_gemm_P9(const CUTLASS_int8_gemm_P9 &) = delete;
  CUTLASS_int8_gemm_P9 &operator=(const CUTLASS_int8_gemm_P9 &) = delete;
  static CUTLASS_int8_gemm_P9 &get_instance() {
    static CUTLASS_int8_gemm_P9 instance;
    return instance;
  }

  void run_int8_gemm(int length_m, int length_n, int length_k, int8_t *A,
                   int8_t *B, int32_t *C, int alpha, int beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename int8_gemm::Arguments arguments{problem_size,
                                         {A, length_k},
                                         {B, length_k},
                                         {C, length_m},
                                         {C, length_m},
                                         {alpha, beta},
                                         1};
    size_t workspace_size = int8_gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = int8_gemm_op.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = int8_gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_int8_gemm_P9(int length_m, int length_n, int length_k,
                        int8_t *A, int8_t *B, int32_t *C,
                        int alpha, int beta, cudaStream_t stream) {
  CUTLASS_int8_gemm_P9 &instance = CUTLASS_int8_gemm_P9::get_instance();
  instance.run_int8_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_int8_gemm_P10 {
private:
  CUTLASS_int8_gemm_P10() { }
  ~CUTLASS_int8_gemm_P10() { }

public: 
  // Gemm operator cutlass_tensorop_i8816gemm_s8_64x128_64x2_tn_align16
  using Operation_cutlass_tensorop_i8816gemm_s8_64x128_64x2_tn_align16 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 128, 64>,
    cutlass::gemm::GemmShape<32, 64, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      4,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    2,
    16,
    16
  >;
using int8_gemm = Operation_cutlass_tensorop_i8816gemm_s8_64x128_64x2_tn_align16;
  int8_gemm int8_gemm_op;
  CUTLASS_int8_gemm_P10(const CUTLASS_int8_gemm_P10 &) = delete;
  CUTLASS_int8_gemm_P10 &operator=(const CUTLASS_int8_gemm_P10 &) = delete;
  static CUTLASS_int8_gemm_P10 &get_instance() {
    static CUTLASS_int8_gemm_P10 instance;
    return instance;
  }

  void run_int8_gemm(int length_m, int length_n, int length_k, int8_t *A,
                   int8_t *B, int32_t *C, int alpha, int beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename int8_gemm::Arguments arguments{problem_size,
                                         {A, length_k},
                                         {B, length_k},
                                         {C, length_m},
                                         {C, length_m},
                                         {alpha, beta},
                                         1};
    size_t workspace_size = int8_gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = int8_gemm_op.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = int8_gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_int8_gemm_P10(int length_m, int length_n, int length_k,
                        int8_t *A, int8_t *B, int32_t *C,
                        int alpha, int beta, cudaStream_t stream) {
  CUTLASS_int8_gemm_P10 &instance = CUTLASS_int8_gemm_P10::get_instance();
  instance.run_int8_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_int8_gemm_P11 {
private:
  CUTLASS_int8_gemm_P11() { }
  ~CUTLASS_int8_gemm_P11() { }

public: 
  // Gemm operator cutlass_tensorop_i8816gemm_s8_128x64_64x2_tn_align8
  using Operation_cutlass_tensorop_i8816gemm_s8_128x64_64x2_tn_align8 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 64, 64>,
    cutlass::gemm::GemmShape<64, 32, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      4,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    2,
    8,
    8
  >;
using int8_gemm = Operation_cutlass_tensorop_i8816gemm_s8_128x64_64x2_tn_align8;
  int8_gemm int8_gemm_op;
  CUTLASS_int8_gemm_P11(const CUTLASS_int8_gemm_P11 &) = delete;
  CUTLASS_int8_gemm_P11 &operator=(const CUTLASS_int8_gemm_P11 &) = delete;
  static CUTLASS_int8_gemm_P11 &get_instance() {
    static CUTLASS_int8_gemm_P11 instance;
    return instance;
  }

  void run_int8_gemm(int length_m, int length_n, int length_k, int8_t *A,
                   int8_t *B, int32_t *C, int alpha, int beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename int8_gemm::Arguments arguments{problem_size,
                                         {A, length_k},
                                         {B, length_k},
                                         {C, length_m},
                                         {C, length_m},
                                         {alpha, beta},
                                         1};
    size_t workspace_size = int8_gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = int8_gemm_op.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = int8_gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_int8_gemm_P11(int length_m, int length_n, int length_k,
                        int8_t *A, int8_t *B, int32_t *C,
                        int alpha, int beta, cudaStream_t stream) {
  CUTLASS_int8_gemm_P11 &instance = CUTLASS_int8_gemm_P11::get_instance();
  instance.run_int8_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_int8_gemm_P12 {
private:
  CUTLASS_int8_gemm_P12() { }
  ~CUTLASS_int8_gemm_P12() { }

public: 
  // Gemm operator cutlass_tensorop_i16832gemm_s8_64x256_64x4_tn_align16
  using Operation_cutlass_tensorop_i16832gemm_s8_64x256_64x4_tn_align16 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      4,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    4,
    16,
    16
  >;
using int8_gemm = Operation_cutlass_tensorop_i16832gemm_s8_64x256_64x4_tn_align16;
  int8_gemm int8_gemm_op;
  CUTLASS_int8_gemm_P12(const CUTLASS_int8_gemm_P12 &) = delete;
  CUTLASS_int8_gemm_P12 &operator=(const CUTLASS_int8_gemm_P12 &) = delete;
  static CUTLASS_int8_gemm_P12 &get_instance() {
    static CUTLASS_int8_gemm_P12 instance;
    return instance;
  }

  void run_int8_gemm(int length_m, int length_n, int length_k, int8_t *A,
                   int8_t *B, int32_t *C, int alpha, int beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename int8_gemm::Arguments arguments{problem_size,
                                         {A, length_k},
                                         {B, length_k},
                                         {C, length_m},
                                         {C, length_m},
                                         {alpha, beta},
                                         1};
    size_t workspace_size = int8_gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = int8_gemm_op.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = int8_gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_int8_gemm_P12(int length_m, int length_n, int length_k,
                        int8_t *A, int8_t *B, int32_t *C,
                        int alpha, int beta, cudaStream_t stream) {
  CUTLASS_int8_gemm_P12 &instance = CUTLASS_int8_gemm_P12::get_instance();
  instance.run_int8_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_int8_gemm_P13 {
private:
  CUTLASS_int8_gemm_P13() { }
  ~CUTLASS_int8_gemm_P13() { }

public: 
  // Gemm operator cutlass_tensorop_i16832gemm_s8_128x256_128x3_tn_align8
  using Operation_cutlass_tensorop_i16832gemm_s8_128x256_128x3_tn_align8 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 128>,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      4,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    8,
    8
  >;
using int8_gemm = Operation_cutlass_tensorop_i16832gemm_s8_128x256_128x3_tn_align8;
  int8_gemm int8_gemm_op;
  CUTLASS_int8_gemm_P13(const CUTLASS_int8_gemm_P13 &) = delete;
  CUTLASS_int8_gemm_P13 &operator=(const CUTLASS_int8_gemm_P13 &) = delete;
  static CUTLASS_int8_gemm_P13 &get_instance() {
    static CUTLASS_int8_gemm_P13 instance;
    return instance;
  }

  void run_int8_gemm(int length_m, int length_n, int length_k, int8_t *A,
                   int8_t *B, int32_t *C, int alpha, int beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename int8_gemm::Arguments arguments{problem_size,
                                         {A, length_k},
                                         {B, length_k},
                                         {C, length_m},
                                         {C, length_m},
                                         {alpha, beta},
                                         1};
    size_t workspace_size = int8_gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = int8_gemm_op.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = int8_gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_int8_gemm_P13(int length_m, int length_n, int length_k,
                        int8_t *A, int8_t *B, int32_t *C,
                        int alpha, int beta, cudaStream_t stream) {
  CUTLASS_int8_gemm_P13 &instance = CUTLASS_int8_gemm_P13::get_instance();
  instance.run_int8_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_int8_gemm_P14 {
private:
  CUTLASS_int8_gemm_P14() { }
  ~CUTLASS_int8_gemm_P14() { }

public: 
  // Gemm operator cutlass_tensorop_i16832gemm_s8_256x128_128x3_tn_align16
  using Operation_cutlass_tensorop_i16832gemm_s8_256x128_128x3_tn_align16 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 128>,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      4,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    16,
    16
  >;
using int8_gemm = Operation_cutlass_tensorop_i16832gemm_s8_256x128_128x3_tn_align16;
  int8_gemm int8_gemm_op;
  CUTLASS_int8_gemm_P14(const CUTLASS_int8_gemm_P14 &) = delete;
  CUTLASS_int8_gemm_P14 &operator=(const CUTLASS_int8_gemm_P14 &) = delete;
  static CUTLASS_int8_gemm_P14 &get_instance() {
    static CUTLASS_int8_gemm_P14 instance;
    return instance;
  }

  void run_int8_gemm(int length_m, int length_n, int length_k, int8_t *A,
                   int8_t *B, int32_t *C, int alpha, int beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename int8_gemm::Arguments arguments{problem_size,
                                         {A, length_k},
                                         {B, length_k},
                                         {C, length_m},
                                         {C, length_m},
                                         {alpha, beta},
                                         1};
    size_t workspace_size = int8_gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = int8_gemm_op.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = int8_gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_int8_gemm_P14(int length_m, int length_n, int length_k,
                        int8_t *A, int8_t *B, int32_t *C,
                        int alpha, int beta, cudaStream_t stream) {
  CUTLASS_int8_gemm_P14 &instance = CUTLASS_int8_gemm_P14::get_instance();
  instance.run_int8_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_int8_gemm_P15 {
private:
  CUTLASS_int8_gemm_P15() { }
  ~CUTLASS_int8_gemm_P15() { }

public: 
  // Gemm operator cutlass_tensorop_i16832gemm_s8_64x256_128x3_tn_align16
  using Operation_cutlass_tensorop_i16832gemm_s8_64x256_128x3_tn_align16 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 256, 128>,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      4,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    16,
    16
  >;
using int8_gemm = Operation_cutlass_tensorop_i16832gemm_s8_64x256_128x3_tn_align16;
  int8_gemm int8_gemm_op;
  CUTLASS_int8_gemm_P15(const CUTLASS_int8_gemm_P15 &) = delete;
  CUTLASS_int8_gemm_P15 &operator=(const CUTLASS_int8_gemm_P15 &) = delete;
  static CUTLASS_int8_gemm_P15 &get_instance() {
    static CUTLASS_int8_gemm_P15 instance;
    return instance;
  }

  void run_int8_gemm(int length_m, int length_n, int length_k, int8_t *A,
                   int8_t *B, int32_t *C, int alpha, int beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename int8_gemm::Arguments arguments{problem_size,
                                         {A, length_k},
                                         {B, length_k},
                                         {C, length_m},
                                         {C, length_m},
                                         {alpha, beta},
                                         1};
    size_t workspace_size = int8_gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = int8_gemm_op.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = int8_gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_int8_gemm_P15(int length_m, int length_n, int length_k,
                        int8_t *A, int8_t *B, int32_t *C,
                        int alpha, int beta, cudaStream_t stream) {
  CUTLASS_int8_gemm_P15 &instance = CUTLASS_int8_gemm_P15::get_instance();
  instance.run_int8_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_int8_gemm_P16 {
private:
  CUTLASS_int8_gemm_P16() { }
  ~CUTLASS_int8_gemm_P16() { }

public: 
  // Gemm operator cutlass_tensorop_i16832gemm_s8_128x64_128x3_tn_align16
  using Operation_cutlass_tensorop_i16832gemm_s8_128x64_128x3_tn_align16 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 128>,
    cutlass::gemm::GemmShape<64, 32, 128>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      4,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    16,
    16
  >;
using int8_gemm = Operation_cutlass_tensorop_i16832gemm_s8_128x64_128x3_tn_align16;
  int8_gemm int8_gemm_op;
  CUTLASS_int8_gemm_P16(const CUTLASS_int8_gemm_P16 &) = delete;
  CUTLASS_int8_gemm_P16 &operator=(const CUTLASS_int8_gemm_P16 &) = delete;
  static CUTLASS_int8_gemm_P16 &get_instance() {
    static CUTLASS_int8_gemm_P16 instance;
    return instance;
  }

  void run_int8_gemm(int length_m, int length_n, int length_k, int8_t *A,
                   int8_t *B, int32_t *C, int alpha, int beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename int8_gemm::Arguments arguments{problem_size,
                                         {A, length_k},
                                         {B, length_k},
                                         {C, length_m},
                                         {C, length_m},
                                         {alpha, beta},
                                         1};
    size_t workspace_size = int8_gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = int8_gemm_op.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = int8_gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_int8_gemm_P16(int length_m, int length_n, int length_k,
                        int8_t *A, int8_t *B, int32_t *C,
                        int alpha, int beta, cudaStream_t stream) {
  CUTLASS_int8_gemm_P16 &instance = CUTLASS_int8_gemm_P16::get_instance();
  instance.run_int8_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_int8_gemm_P17 {
private:
  CUTLASS_int8_gemm_P17() { }
  ~CUTLASS_int8_gemm_P17() { }

public: 
  // Gemm operator cutlass_tensorop_i16832gemm_s8_128x128_64x3_tn_align4
  using Operation_cutlass_tensorop_i16832gemm_s8_128x128_64x3_tn_align4 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      4,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    4,
    4
  >;
using int8_gemm = Operation_cutlass_tensorop_i16832gemm_s8_128x128_64x3_tn_align4;
  int8_gemm int8_gemm_op;
  CUTLASS_int8_gemm_P17(const CUTLASS_int8_gemm_P17 &) = delete;
  CUTLASS_int8_gemm_P17 &operator=(const CUTLASS_int8_gemm_P17 &) = delete;
  static CUTLASS_int8_gemm_P17 &get_instance() {
    static CUTLASS_int8_gemm_P17 instance;
    return instance;
  }

  void run_int8_gemm(int length_m, int length_n, int length_k, int8_t *A,
                   int8_t *B, int32_t *C, int alpha, int beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename int8_gemm::Arguments arguments{problem_size,
                                         {A, length_k},
                                         {B, length_k},
                                         {C, length_m},
                                         {C, length_m},
                                         {alpha, beta},
                                         1};
    size_t workspace_size = int8_gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = int8_gemm_op.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = int8_gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_int8_gemm_P17(int length_m, int length_n, int length_k,
                        int8_t *A, int8_t *B, int32_t *C,
                        int alpha, int beta, cudaStream_t stream) {
  CUTLASS_int8_gemm_P17 &instance = CUTLASS_int8_gemm_P17::get_instance();
  instance.run_int8_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_int8_gemm_P18 {
private:
  CUTLASS_int8_gemm_P18() { }
  ~CUTLASS_int8_gemm_P18() { }

public: 
  // Gemm operator cutlass_tensorop_i8816gemm_s8_64x64_64x2_tn_align16
  using Operation_cutlass_tensorop_i8816gemm_s8_64x64_64x2_tn_align16 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      4,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    2,
    16,
    16
  >;
using int8_gemm = Operation_cutlass_tensorop_i8816gemm_s8_64x64_64x2_tn_align16;
  int8_gemm int8_gemm_op;
  CUTLASS_int8_gemm_P18(const CUTLASS_int8_gemm_P18 &) = delete;
  CUTLASS_int8_gemm_P18 &operator=(const CUTLASS_int8_gemm_P18 &) = delete;
  static CUTLASS_int8_gemm_P18 &get_instance() {
    static CUTLASS_int8_gemm_P18 instance;
    return instance;
  }

  void run_int8_gemm(int length_m, int length_n, int length_k, int8_t *A,
                   int8_t *B, int32_t *C, int alpha, int beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename int8_gemm::Arguments arguments{problem_size,
                                         {A, length_k},
                                         {B, length_k},
                                         {C, length_m},
                                         {C, length_m},
                                         {alpha, beta},
                                         1};
    size_t workspace_size = int8_gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = int8_gemm_op.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = int8_gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_int8_gemm_P18(int length_m, int length_n, int length_k,
                        int8_t *A, int8_t *B, int32_t *C,
                        int alpha, int beta, cudaStream_t stream) {
  CUTLASS_int8_gemm_P18 &instance = CUTLASS_int8_gemm_P18::get_instance();
  instance.run_int8_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_int8_gemm_P19 {
private:
  CUTLASS_int8_gemm_P19() { }
  ~CUTLASS_int8_gemm_P19() { }

public: 
  // Gemm operator cutlass_tensorop_i16832gemm_s8_128x256_64x3_tn_align16
  using Operation_cutlass_tensorop_i16832gemm_s8_128x256_64x3_tn_align16 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      4,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    16,
    16
  >;
using int8_gemm = Operation_cutlass_tensorop_i16832gemm_s8_128x256_64x3_tn_align16;
  int8_gemm int8_gemm_op;
  CUTLASS_int8_gemm_P19(const CUTLASS_int8_gemm_P19 &) = delete;
  CUTLASS_int8_gemm_P19 &operator=(const CUTLASS_int8_gemm_P19 &) = delete;
  static CUTLASS_int8_gemm_P19 &get_instance() {
    static CUTLASS_int8_gemm_P19 instance;
    return instance;
  }

  void run_int8_gemm(int length_m, int length_n, int length_k, int8_t *A,
                   int8_t *B, int32_t *C, int alpha, int beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename int8_gemm::Arguments arguments{problem_size,
                                         {A, length_k},
                                         {B, length_k},
                                         {C, length_m},
                                         {C, length_m},
                                         {alpha, beta},
                                         1};
    size_t workspace_size = int8_gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = int8_gemm_op.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = int8_gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_int8_gemm_P19(int length_m, int length_n, int length_k,
                        int8_t *A, int8_t *B, int32_t *C,
                        int alpha, int beta, cudaStream_t stream) {
  CUTLASS_int8_gemm_P19 &instance = CUTLASS_int8_gemm_P19::get_instance();
  instance.run_int8_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_int8_gemm_P20 {
private:
  CUTLASS_int8_gemm_P20() { }
  ~CUTLASS_int8_gemm_P20() { }

public: 
  // Gemm operator cutlass_tensorop_i16832gemm_s8_64x64_128x5_tn_align16
  using Operation_cutlass_tensorop_i16832gemm_s8_64x64_128x5_tn_align16 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<32, 32, 128>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      4,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    5,
    16,
    16
  >;
using int8_gemm = Operation_cutlass_tensorop_i16832gemm_s8_64x64_128x5_tn_align16;
  int8_gemm int8_gemm_op;
  CUTLASS_int8_gemm_P20(const CUTLASS_int8_gemm_P20 &) = delete;
  CUTLASS_int8_gemm_P20 &operator=(const CUTLASS_int8_gemm_P20 &) = delete;
  static CUTLASS_int8_gemm_P20 &get_instance() {
    static CUTLASS_int8_gemm_P20 instance;
    return instance;
  }

  void run_int8_gemm(int length_m, int length_n, int length_k, int8_t *A,
                   int8_t *B, int32_t *C, int alpha, int beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename int8_gemm::Arguments arguments{problem_size,
                                         {A, length_k},
                                         {B, length_k},
                                         {C, length_m},
                                         {C, length_m},
                                         {alpha, beta},
                                         1};
    size_t workspace_size = int8_gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = int8_gemm_op.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = int8_gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_int8_gemm_P20(int length_m, int length_n, int length_k,
                        int8_t *A, int8_t *B, int32_t *C,
                        int alpha, int beta, cudaStream_t stream) {
  CUTLASS_int8_gemm_P20 &instance = CUTLASS_int8_gemm_P20::get_instance();
  instance.run_int8_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_int8_gemm_P21 {
private:
  CUTLASS_int8_gemm_P21() { }
  ~CUTLASS_int8_gemm_P21() { }

public: 
  // Gemm operator cutlass_tensorop_i16832gemm_s8_256x128_128x3_tn_align8
  using Operation_cutlass_tensorop_i16832gemm_s8_256x128_128x3_tn_align8 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 128>,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      4,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    8,
    8
  >;
using int8_gemm = Operation_cutlass_tensorop_i16832gemm_s8_256x128_128x3_tn_align8;
  int8_gemm int8_gemm_op;
  CUTLASS_int8_gemm_P21(const CUTLASS_int8_gemm_P21 &) = delete;
  CUTLASS_int8_gemm_P21 &operator=(const CUTLASS_int8_gemm_P21 &) = delete;
  static CUTLASS_int8_gemm_P21 &get_instance() {
    static CUTLASS_int8_gemm_P21 instance;
    return instance;
  }

  void run_int8_gemm(int length_m, int length_n, int length_k, int8_t *A,
                   int8_t *B, int32_t *C, int alpha, int beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename int8_gemm::Arguments arguments{problem_size,
                                         {A, length_k},
                                         {B, length_k},
                                         {C, length_m},
                                         {C, length_m},
                                         {alpha, beta},
                                         1};
    size_t workspace_size = int8_gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = int8_gemm_op.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = int8_gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_int8_gemm_P21(int length_m, int length_n, int length_k,
                        int8_t *A, int8_t *B, int32_t *C,
                        int alpha, int beta, cudaStream_t stream) {
  CUTLASS_int8_gemm_P21 &instance = CUTLASS_int8_gemm_P21::get_instance();
  instance.run_int8_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_int8_gemm_P22 {
private:
  CUTLASS_int8_gemm_P22() { }
  ~CUTLASS_int8_gemm_P22() { }

public: 
  // Gemm operator cutlass_tensorop_i16832gemm_s8_128x128_64x3_tn_align8
  using Operation_cutlass_tensorop_i16832gemm_s8_128x128_64x3_tn_align8 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      4,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    8,
    8
  >;
using int8_gemm = Operation_cutlass_tensorop_i16832gemm_s8_128x128_64x3_tn_align8;
  int8_gemm int8_gemm_op;
  CUTLASS_int8_gemm_P22(const CUTLASS_int8_gemm_P22 &) = delete;
  CUTLASS_int8_gemm_P22 &operator=(const CUTLASS_int8_gemm_P22 &) = delete;
  static CUTLASS_int8_gemm_P22 &get_instance() {
    static CUTLASS_int8_gemm_P22 instance;
    return instance;
  }

  void run_int8_gemm(int length_m, int length_n, int length_k, int8_t *A,
                   int8_t *B, int32_t *C, int alpha, int beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename int8_gemm::Arguments arguments{problem_size,
                                         {A, length_k},
                                         {B, length_k},
                                         {C, length_m},
                                         {C, length_m},
                                         {alpha, beta},
                                         1};
    size_t workspace_size = int8_gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = int8_gemm_op.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = int8_gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_int8_gemm_P22(int length_m, int length_n, int length_k,
                        int8_t *A, int8_t *B, int32_t *C,
                        int alpha, int beta, cudaStream_t stream) {
  CUTLASS_int8_gemm_P22 &instance = CUTLASS_int8_gemm_P22::get_instance();
  instance.run_int8_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_int8_gemm_P23 {
private:
  CUTLASS_int8_gemm_P23() { }
  ~CUTLASS_int8_gemm_P23() { }

public: 
  // Gemm operator cutlass_tensorop_i16832gemm_s8_128x128_128x3_tn_align4
  using Operation_cutlass_tensorop_i16832gemm_s8_128x128_128x3_tn_align4 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 128>,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      4,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    4,
    4
  >;
using int8_gemm = Operation_cutlass_tensorop_i16832gemm_s8_128x128_128x3_tn_align4;
  int8_gemm int8_gemm_op;
  CUTLASS_int8_gemm_P23(const CUTLASS_int8_gemm_P23 &) = delete;
  CUTLASS_int8_gemm_P23 &operator=(const CUTLASS_int8_gemm_P23 &) = delete;
  static CUTLASS_int8_gemm_P23 &get_instance() {
    static CUTLASS_int8_gemm_P23 instance;
    return instance;
  }

  void run_int8_gemm(int length_m, int length_n, int length_k, int8_t *A,
                   int8_t *B, int32_t *C, int alpha, int beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename int8_gemm::Arguments arguments{problem_size,
                                         {A, length_k},
                                         {B, length_k},
                                         {C, length_m},
                                         {C, length_m},
                                         {alpha, beta},
                                         1};
    size_t workspace_size = int8_gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = int8_gemm_op.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = int8_gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_int8_gemm_P23(int length_m, int length_n, int length_k,
                        int8_t *A, int8_t *B, int32_t *C,
                        int alpha, int beta, cudaStream_t stream) {
  CUTLASS_int8_gemm_P23 &instance = CUTLASS_int8_gemm_P23::get_instance();
  instance.run_int8_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_int8_gemm_P24 {
private:
  CUTLASS_int8_gemm_P24() { }
  ~CUTLASS_int8_gemm_P24() { }

public: 
  // Gemm operator cutlass_tensorop_i16832gemm_s8_128x64_128x3_tn_align4
  using Operation_cutlass_tensorop_i16832gemm_s8_128x64_128x3_tn_align4 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 128>,
    cutlass::gemm::GemmShape<64, 32, 128>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      4,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    4,
    4
  >;
using int8_gemm = Operation_cutlass_tensorop_i16832gemm_s8_128x64_128x3_tn_align4;
  int8_gemm int8_gemm_op;
  CUTLASS_int8_gemm_P24(const CUTLASS_int8_gemm_P24 &) = delete;
  CUTLASS_int8_gemm_P24 &operator=(const CUTLASS_int8_gemm_P24 &) = delete;
  static CUTLASS_int8_gemm_P24 &get_instance() {
    static CUTLASS_int8_gemm_P24 instance;
    return instance;
  }

  void run_int8_gemm(int length_m, int length_n, int length_k, int8_t *A,
                   int8_t *B, int32_t *C, int alpha, int beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename int8_gemm::Arguments arguments{problem_size,
                                         {A, length_k},
                                         {B, length_k},
                                         {C, length_m},
                                         {C, length_m},
                                         {alpha, beta},
                                         1};
    size_t workspace_size = int8_gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = int8_gemm_op.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = int8_gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_int8_gemm_P24(int length_m, int length_n, int length_k,
                        int8_t *A, int8_t *B, int32_t *C,
                        int alpha, int beta, cudaStream_t stream) {
  CUTLASS_int8_gemm_P24 &instance = CUTLASS_int8_gemm_P24::get_instance();
  instance.run_int8_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_int8_gemm_P25 {
private:
  CUTLASS_int8_gemm_P25() { }
  ~CUTLASS_int8_gemm_P25() { }

public: 
  // Gemm operator cutlass_tensorop_i16832gemm_s8_128x128_64x4_tn_align8
  using Operation_cutlass_tensorop_i16832gemm_s8_128x128_64x4_tn_align8 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      4,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    4,
    8,
    8
  >;
using int8_gemm = Operation_cutlass_tensorop_i16832gemm_s8_128x128_64x4_tn_align8;
  int8_gemm int8_gemm_op;
  CUTLASS_int8_gemm_P25(const CUTLASS_int8_gemm_P25 &) = delete;
  CUTLASS_int8_gemm_P25 &operator=(const CUTLASS_int8_gemm_P25 &) = delete;
  static CUTLASS_int8_gemm_P25 &get_instance() {
    static CUTLASS_int8_gemm_P25 instance;
    return instance;
  }

  void run_int8_gemm(int length_m, int length_n, int length_k, int8_t *A,
                   int8_t *B, int32_t *C, int alpha, int beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename int8_gemm::Arguments arguments{problem_size,
                                         {A, length_k},
                                         {B, length_k},
                                         {C, length_m},
                                         {C, length_m},
                                         {alpha, beta},
                                         1};
    size_t workspace_size = int8_gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = int8_gemm_op.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = int8_gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_int8_gemm_P25(int length_m, int length_n, int length_k,
                        int8_t *A, int8_t *B, int32_t *C,
                        int alpha, int beta, cudaStream_t stream) {
  CUTLASS_int8_gemm_P25 &instance = CUTLASS_int8_gemm_P25::get_instance();
  instance.run_int8_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_int8_gemm_P26 {
private:
  CUTLASS_int8_gemm_P26() { }
  ~CUTLASS_int8_gemm_P26() { }

public: 
  // Gemm operator cutlass_tensorop_i16832gemm_s8_256x128_64x3_tn_align16
  using Operation_cutlass_tensorop_i16832gemm_s8_256x128_64x3_tn_align16 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      4,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    16,
    16
  >;
using int8_gemm = Operation_cutlass_tensorop_i16832gemm_s8_256x128_64x3_tn_align16;
  int8_gemm int8_gemm_op;
  CUTLASS_int8_gemm_P26(const CUTLASS_int8_gemm_P26 &) = delete;
  CUTLASS_int8_gemm_P26 &operator=(const CUTLASS_int8_gemm_P26 &) = delete;
  static CUTLASS_int8_gemm_P26 &get_instance() {
    static CUTLASS_int8_gemm_P26 instance;
    return instance;
  }

  void run_int8_gemm(int length_m, int length_n, int length_k, int8_t *A,
                   int8_t *B, int32_t *C, int alpha, int beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename int8_gemm::Arguments arguments{problem_size,
                                         {A, length_k},
                                         {B, length_k},
                                         {C, length_m},
                                         {C, length_m},
                                         {alpha, beta},
                                         1};
    size_t workspace_size = int8_gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = int8_gemm_op.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = int8_gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_int8_gemm_P26(int length_m, int length_n, int length_k,
                        int8_t *A, int8_t *B, int32_t *C,
                        int alpha, int beta, cudaStream_t stream) {
  CUTLASS_int8_gemm_P26 &instance = CUTLASS_int8_gemm_P26::get_instance();
  instance.run_int8_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_int8_gemm_P27 {
private:
  CUTLASS_int8_gemm_P27() { }
  ~CUTLASS_int8_gemm_P27() { }

public: 
  // Gemm operator cutlass_tensorop_i16832gemm_s8_128x128_64x5_tn_align16
  using Operation_cutlass_tensorop_i16832gemm_s8_128x128_64x5_tn_align16 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      4,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    5,
    16,
    16
  >;
using int8_gemm = Operation_cutlass_tensorop_i16832gemm_s8_128x128_64x5_tn_align16;
  int8_gemm int8_gemm_op;
  CUTLASS_int8_gemm_P27(const CUTLASS_int8_gemm_P27 &) = delete;
  CUTLASS_int8_gemm_P27 &operator=(const CUTLASS_int8_gemm_P27 &) = delete;
  static CUTLASS_int8_gemm_P27 &get_instance() {
    static CUTLASS_int8_gemm_P27 instance;
    return instance;
  }

  void run_int8_gemm(int length_m, int length_n, int length_k, int8_t *A,
                   int8_t *B, int32_t *C, int alpha, int beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename int8_gemm::Arguments arguments{problem_size,
                                         {A, length_k},
                                         {B, length_k},
                                         {C, length_m},
                                         {C, length_m},
                                         {alpha, beta},
                                         1};
    size_t workspace_size = int8_gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = int8_gemm_op.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = int8_gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_int8_gemm_P27(int length_m, int length_n, int length_k,
                        int8_t *A, int8_t *B, int32_t *C,
                        int alpha, int beta, cudaStream_t stream) {
  CUTLASS_int8_gemm_P27 &instance = CUTLASS_int8_gemm_P27::get_instance();
  instance.run_int8_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_int8_gemm_P28 {
private:
  CUTLASS_int8_gemm_P28() { }
  ~CUTLASS_int8_gemm_P28() { }

public: 
  // Gemm operator cutlass_tensorop_i8816gemm_s8_64x64_64x2_tn_align8
  using Operation_cutlass_tensorop_i8816gemm_s8_64x64_64x2_tn_align8 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      4,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    2,
    8,
    8
  >;
using int8_gemm = Operation_cutlass_tensorop_i8816gemm_s8_64x64_64x2_tn_align8;
  int8_gemm int8_gemm_op;
  CUTLASS_int8_gemm_P28(const CUTLASS_int8_gemm_P28 &) = delete;
  CUTLASS_int8_gemm_P28 &operator=(const CUTLASS_int8_gemm_P28 &) = delete;
  static CUTLASS_int8_gemm_P28 &get_instance() {
    static CUTLASS_int8_gemm_P28 instance;
    return instance;
  }

  void run_int8_gemm(int length_m, int length_n, int length_k, int8_t *A,
                   int8_t *B, int32_t *C, int alpha, int beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename int8_gemm::Arguments arguments{problem_size,
                                         {A, length_k},
                                         {B, length_k},
                                         {C, length_m},
                                         {C, length_m},
                                         {alpha, beta},
                                         1};
    size_t workspace_size = int8_gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = int8_gemm_op.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = int8_gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_int8_gemm_P28(int length_m, int length_n, int length_k,
                        int8_t *A, int8_t *B, int32_t *C,
                        int alpha, int beta, cudaStream_t stream) {
  CUTLASS_int8_gemm_P28 &instance = CUTLASS_int8_gemm_P28::get_instance();
  instance.run_int8_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_int8_gemm_P29 {
private:
  CUTLASS_int8_gemm_P29() { }
  ~CUTLASS_int8_gemm_P29() { }

public: 
  // Gemm operator cutlass_tensorop_i16832gemm_s8_128x128_64x3_tn_align16
  using Operation_cutlass_tensorop_i16832gemm_s8_128x128_64x3_tn_align16 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      4,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    16,
    16
  >;
using int8_gemm = Operation_cutlass_tensorop_i16832gemm_s8_128x128_64x3_tn_align16;
  int8_gemm int8_gemm_op;
  CUTLASS_int8_gemm_P29(const CUTLASS_int8_gemm_P29 &) = delete;
  CUTLASS_int8_gemm_P29 &operator=(const CUTLASS_int8_gemm_P29 &) = delete;
  static CUTLASS_int8_gemm_P29 &get_instance() {
    static CUTLASS_int8_gemm_P29 instance;
    return instance;
  }

  void run_int8_gemm(int length_m, int length_n, int length_k, int8_t *A,
                   int8_t *B, int32_t *C, int alpha, int beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename int8_gemm::Arguments arguments{problem_size,
                                         {A, length_k},
                                         {B, length_k},
                                         {C, length_m},
                                         {C, length_m},
                                         {alpha, beta},
                                         1};
    size_t workspace_size = int8_gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = int8_gemm_op.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = int8_gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_int8_gemm_P29(int length_m, int length_n, int length_k,
                        int8_t *A, int8_t *B, int32_t *C,
                        int alpha, int beta, cudaStream_t stream) {
  CUTLASS_int8_gemm_P29 &instance = CUTLASS_int8_gemm_P29::get_instance();
  instance.run_int8_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_int8_gemm_P30 {
private:
  CUTLASS_int8_gemm_P30() { }
  ~CUTLASS_int8_gemm_P30() { }

public: 
  // Gemm operator cutlass_tensorop_i8816gemm_s8_128x128_64x2_tn_align1
  using Operation_cutlass_tensorop_i8816gemm_s8_128x128_64x2_tn_align1 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 32, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      1,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    2,
    1,
    1
  >;
using int8_gemm = Operation_cutlass_tensorop_i8816gemm_s8_128x128_64x2_tn_align1;
  int8_gemm int8_gemm_op;
  CUTLASS_int8_gemm_P30(const CUTLASS_int8_gemm_P30 &) = delete;
  CUTLASS_int8_gemm_P30 &operator=(const CUTLASS_int8_gemm_P30 &) = delete;
  static CUTLASS_int8_gemm_P30 &get_instance() {
    static CUTLASS_int8_gemm_P30 instance;
    return instance;
  }

  void run_int8_gemm(int length_m, int length_n, int length_k, int8_t *A,
                   int8_t *B, int32_t *C, int alpha, int beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename int8_gemm::Arguments arguments{problem_size,
                                         {A, length_k},
                                         {B, length_k},
                                         {C, length_m},
                                         {C, length_m},
                                         {alpha, beta},
                                         1};
    size_t workspace_size = int8_gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = int8_gemm_op.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = int8_gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_int8_gemm_P30(int length_m, int length_n, int length_k,
                        int8_t *A, int8_t *B, int32_t *C,
                        int alpha, int beta, cudaStream_t stream) {
  CUTLASS_int8_gemm_P30 &instance = CUTLASS_int8_gemm_P30::get_instance();
  instance.run_int8_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_int8_gemm_P31 {
private:
  CUTLASS_int8_gemm_P31() { }
  ~CUTLASS_int8_gemm_P31() { }

public: 
  // Gemm operator cutlass_tensorop_i8816gemm_s8_64x128_64x2_tn_align2
  using Operation_cutlass_tensorop_i8816gemm_s8_64x128_64x2_tn_align2 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 128, 64>,
    cutlass::gemm::GemmShape<32, 64, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      2,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    2,
    2,
    2
  >;
using int8_gemm = Operation_cutlass_tensorop_i8816gemm_s8_64x128_64x2_tn_align2;
  int8_gemm int8_gemm_op;
  CUTLASS_int8_gemm_P31(const CUTLASS_int8_gemm_P31 &) = delete;
  CUTLASS_int8_gemm_P31 &operator=(const CUTLASS_int8_gemm_P31 &) = delete;
  static CUTLASS_int8_gemm_P31 &get_instance() {
    static CUTLASS_int8_gemm_P31 instance;
    return instance;
  }

  void run_int8_gemm(int length_m, int length_n, int length_k, int8_t *A,
                   int8_t *B, int32_t *C, int alpha, int beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename int8_gemm::Arguments arguments{problem_size,
                                         {A, length_k},
                                         {B, length_k},
                                         {C, length_m},
                                         {C, length_m},
                                         {alpha, beta},
                                         1};
    size_t workspace_size = int8_gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = int8_gemm_op.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = int8_gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_int8_gemm_P31(int length_m, int length_n, int length_k,
                        int8_t *A, int8_t *B, int32_t *C,
                        int alpha, int beta, cudaStream_t stream) {
  CUTLASS_int8_gemm_P31 &instance = CUTLASS_int8_gemm_P31::get_instance();
  instance.run_int8_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_int8_gemm_P32 {
private:
  CUTLASS_int8_gemm_P32() { }
  ~CUTLASS_int8_gemm_P32() { }

public: 
  // Gemm operator cutlass_tensorop_i8816gemm_s8_64x128_64x2_tn_align8
  using Operation_cutlass_tensorop_i8816gemm_s8_64x128_64x2_tn_align8 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 128, 64>,
    cutlass::gemm::GemmShape<32, 64, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      4,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    2,
    8,
    8
  >;
using int8_gemm = Operation_cutlass_tensorop_i8816gemm_s8_64x128_64x2_tn_align8;
  int8_gemm int8_gemm_op;
  CUTLASS_int8_gemm_P32(const CUTLASS_int8_gemm_P32 &) = delete;
  CUTLASS_int8_gemm_P32 &operator=(const CUTLASS_int8_gemm_P32 &) = delete;
  static CUTLASS_int8_gemm_P32 &get_instance() {
    static CUTLASS_int8_gemm_P32 instance;
    return instance;
  }

  void run_int8_gemm(int length_m, int length_n, int length_k, int8_t *A,
                   int8_t *B, int32_t *C, int alpha, int beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename int8_gemm::Arguments arguments{problem_size,
                                         {A, length_k},
                                         {B, length_k},
                                         {C, length_m},
                                         {C, length_m},
                                         {alpha, beta},
                                         1};
    size_t workspace_size = int8_gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = int8_gemm_op.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = int8_gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_int8_gemm_P32(int length_m, int length_n, int length_k,
                        int8_t *A, int8_t *B, int32_t *C,
                        int alpha, int beta, cudaStream_t stream) {
  CUTLASS_int8_gemm_P32 &instance = CUTLASS_int8_gemm_P32::get_instance();
  instance.run_int8_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_int8_gemm_P33 {
private:
  CUTLASS_int8_gemm_P33() { }
  ~CUTLASS_int8_gemm_P33() { }

public: 
  // Gemm operator cutlass_tensorop_i16832gemm_s8_64x128_128x3_tn_align16
  using Operation_cutlass_tensorop_i16832gemm_s8_64x128_128x3_tn_align16 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 128>,
    cutlass::gemm::GemmShape<32, 64, 128>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      4,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    16,
    16
  >;
using int8_gemm = Operation_cutlass_tensorop_i16832gemm_s8_64x128_128x3_tn_align16;
  int8_gemm int8_gemm_op;
  CUTLASS_int8_gemm_P33(const CUTLASS_int8_gemm_P33 &) = delete;
  CUTLASS_int8_gemm_P33 &operator=(const CUTLASS_int8_gemm_P33 &) = delete;
  static CUTLASS_int8_gemm_P33 &get_instance() {
    static CUTLASS_int8_gemm_P33 instance;
    return instance;
  }

  void run_int8_gemm(int length_m, int length_n, int length_k, int8_t *A,
                   int8_t *B, int32_t *C, int alpha, int beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename int8_gemm::Arguments arguments{problem_size,
                                         {A, length_k},
                                         {B, length_k},
                                         {C, length_m},
                                         {C, length_m},
                                         {alpha, beta},
                                         1};
    size_t workspace_size = int8_gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = int8_gemm_op.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = int8_gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_int8_gemm_P33(int length_m, int length_n, int length_k,
                        int8_t *A, int8_t *B, int32_t *C,
                        int alpha, int beta, cudaStream_t stream) {
  CUTLASS_int8_gemm_P33 &instance = CUTLASS_int8_gemm_P33::get_instance();
  instance.run_int8_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_int8_gemm_P34 {
private:
  CUTLASS_int8_gemm_P34() { }
  ~CUTLASS_int8_gemm_P34() { }

public: 
  // Gemm operator cutlass_tensorop_i16832gemm_s8_64x64_128x5_tn_align4
  using Operation_cutlass_tensorop_i16832gemm_s8_64x64_128x5_tn_align4 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<32, 32, 128>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      4,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    5,
    4,
    4
  >;
using int8_gemm = Operation_cutlass_tensorop_i16832gemm_s8_64x64_128x5_tn_align4;
  int8_gemm int8_gemm_op;
  CUTLASS_int8_gemm_P34(const CUTLASS_int8_gemm_P34 &) = delete;
  CUTLASS_int8_gemm_P34 &operator=(const CUTLASS_int8_gemm_P34 &) = delete;
  static CUTLASS_int8_gemm_P34 &get_instance() {
    static CUTLASS_int8_gemm_P34 instance;
    return instance;
  }

  void run_int8_gemm(int length_m, int length_n, int length_k, int8_t *A,
                   int8_t *B, int32_t *C, int alpha, int beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename int8_gemm::Arguments arguments{problem_size,
                                         {A, length_k},
                                         {B, length_k},
                                         {C, length_m},
                                         {C, length_m},
                                         {alpha, beta},
                                         1};
    size_t workspace_size = int8_gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = int8_gemm_op.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = int8_gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_int8_gemm_P34(int length_m, int length_n, int length_k,
                        int8_t *A, int8_t *B, int32_t *C,
                        int alpha, int beta, cudaStream_t stream) {
  CUTLASS_int8_gemm_P34 &instance = CUTLASS_int8_gemm_P34::get_instance();
  instance.run_int8_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_int8_gemm_P35 {
private:
  CUTLASS_int8_gemm_P35() { }
  ~CUTLASS_int8_gemm_P35() { }

public: 
  // Gemm operator cutlass_tensorop_i16832gemm_s8_128x64_64x6_tn_align16
  using Operation_cutlass_tensorop_i16832gemm_s8_128x64_64x6_tn_align16 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 64>,
    cutlass::gemm::GemmShape<64, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      4,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    6,
    16,
    16
  >;
using int8_gemm = Operation_cutlass_tensorop_i16832gemm_s8_128x64_64x6_tn_align16;
  int8_gemm int8_gemm_op;
  CUTLASS_int8_gemm_P35(const CUTLASS_int8_gemm_P35 &) = delete;
  CUTLASS_int8_gemm_P35 &operator=(const CUTLASS_int8_gemm_P35 &) = delete;
  static CUTLASS_int8_gemm_P35 &get_instance() {
    static CUTLASS_int8_gemm_P35 instance;
    return instance;
  }

  void run_int8_gemm(int length_m, int length_n, int length_k, int8_t *A,
                   int8_t *B, int32_t *C, int alpha, int beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename int8_gemm::Arguments arguments{problem_size,
                                         {A, length_k},
                                         {B, length_k},
                                         {C, length_m},
                                         {C, length_m},
                                         {alpha, beta},
                                         1};
    size_t workspace_size = int8_gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = int8_gemm_op.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = int8_gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_int8_gemm_P35(int length_m, int length_n, int length_k,
                        int8_t *A, int8_t *B, int32_t *C,
                        int alpha, int beta, cudaStream_t stream) {
  CUTLASS_int8_gemm_P35 &instance = CUTLASS_int8_gemm_P35::get_instance();
  instance.run_int8_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_int8_gemm_P36 {
private:
  CUTLASS_int8_gemm_P36() { }
  ~CUTLASS_int8_gemm_P36() { }

public: 
  // Gemm operator cutlass_tensorop_i16832gemm_s8_64x64_128x5_tn_align8
  using Operation_cutlass_tensorop_i16832gemm_s8_64x64_128x5_tn_align8 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<32, 32, 128>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      4,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    5,
    8,
    8
  >;
using int8_gemm = Operation_cutlass_tensorop_i16832gemm_s8_64x64_128x5_tn_align8;
  int8_gemm int8_gemm_op;
  CUTLASS_int8_gemm_P36(const CUTLASS_int8_gemm_P36 &) = delete;
  CUTLASS_int8_gemm_P36 &operator=(const CUTLASS_int8_gemm_P36 &) = delete;
  static CUTLASS_int8_gemm_P36 &get_instance() {
    static CUTLASS_int8_gemm_P36 instance;
    return instance;
  }

  void run_int8_gemm(int length_m, int length_n, int length_k, int8_t *A,
                   int8_t *B, int32_t *C, int alpha, int beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename int8_gemm::Arguments arguments{problem_size,
                                         {A, length_k},
                                         {B, length_k},
                                         {C, length_m},
                                         {C, length_m},
                                         {alpha, beta},
                                         1};
    size_t workspace_size = int8_gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = int8_gemm_op.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = int8_gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_int8_gemm_P36(int length_m, int length_n, int length_k,
                        int8_t *A, int8_t *B, int32_t *C,
                        int alpha, int beta, cudaStream_t stream) {
  CUTLASS_int8_gemm_P36 &instance = CUTLASS_int8_gemm_P36::get_instance();
  instance.run_int8_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_int8_gemm_P37 {
private:
  CUTLASS_int8_gemm_P37() { }
  ~CUTLASS_int8_gemm_P37() { }

public: 
  // Gemm operator cutlass_tensorop_i16832gemm_s8_128x64_128x3_tn_align8
  using Operation_cutlass_tensorop_i16832gemm_s8_128x64_128x3_tn_align8 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 128>,
    cutlass::gemm::GemmShape<64, 32, 128>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      4,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    8,
    8
  >;
using int8_gemm = Operation_cutlass_tensorop_i16832gemm_s8_128x64_128x3_tn_align8;
  int8_gemm int8_gemm_op;
  CUTLASS_int8_gemm_P37(const CUTLASS_int8_gemm_P37 &) = delete;
  CUTLASS_int8_gemm_P37 &operator=(const CUTLASS_int8_gemm_P37 &) = delete;
  static CUTLASS_int8_gemm_P37 &get_instance() {
    static CUTLASS_int8_gemm_P37 instance;
    return instance;
  }

  void run_int8_gemm(int length_m, int length_n, int length_k, int8_t *A,
                   int8_t *B, int32_t *C, int alpha, int beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename int8_gemm::Arguments arguments{problem_size,
                                         {A, length_k},
                                         {B, length_k},
                                         {C, length_m},
                                         {C, length_m},
                                         {alpha, beta},
                                         1};
    size_t workspace_size = int8_gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = int8_gemm_op.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = int8_gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_int8_gemm_P37(int length_m, int length_n, int length_k,
                        int8_t *A, int8_t *B, int32_t *C,
                        int alpha, int beta, cudaStream_t stream) {
  CUTLASS_int8_gemm_P37 &instance = CUTLASS_int8_gemm_P37::get_instance();
  instance.run_int8_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_int8_gemm_P38 {
private:
  CUTLASS_int8_gemm_P38() { }
  ~CUTLASS_int8_gemm_P38() { }

public: 
  // Gemm operator cutlass_tensorop_i8816gemm_s8_128x128_64x2_tn_align4
  using Operation_cutlass_tensorop_i8816gemm_s8_128x128_64x2_tn_align4 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      4,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    2,
    4,
    4
  >;
using int8_gemm = Operation_cutlass_tensorop_i8816gemm_s8_128x128_64x2_tn_align4;
  int8_gemm int8_gemm_op;
  CUTLASS_int8_gemm_P38(const CUTLASS_int8_gemm_P38 &) = delete;
  CUTLASS_int8_gemm_P38 &operator=(const CUTLASS_int8_gemm_P38 &) = delete;
  static CUTLASS_int8_gemm_P38 &get_instance() {
    static CUTLASS_int8_gemm_P38 instance;
    return instance;
  }

  void run_int8_gemm(int length_m, int length_n, int length_k, int8_t *A,
                   int8_t *B, int32_t *C, int alpha, int beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename int8_gemm::Arguments arguments{problem_size,
                                         {A, length_k},
                                         {B, length_k},
                                         {C, length_m},
                                         {C, length_m},
                                         {alpha, beta},
                                         1};
    size_t workspace_size = int8_gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = int8_gemm_op.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = int8_gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_int8_gemm_P38(int length_m, int length_n, int length_k,
                        int8_t *A, int8_t *B, int32_t *C,
                        int alpha, int beta, cudaStream_t stream) {
  CUTLASS_int8_gemm_P38 &instance = CUTLASS_int8_gemm_P38::get_instance();
  instance.run_int8_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_int8_gemm_P39 {
private:
  CUTLASS_int8_gemm_P39() { }
  ~CUTLASS_int8_gemm_P39() { }

public: 
  // Gemm operator cutlass_tensorop_i8816gemm_s8_128x128_64x2_tn_align2
  using Operation_cutlass_tensorop_i8816gemm_s8_128x128_64x2_tn_align2 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      2,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    2,
    2,
    2
  >;
using int8_gemm = Operation_cutlass_tensorop_i8816gemm_s8_128x128_64x2_tn_align2;
  int8_gemm int8_gemm_op;
  CUTLASS_int8_gemm_P39(const CUTLASS_int8_gemm_P39 &) = delete;
  CUTLASS_int8_gemm_P39 &operator=(const CUTLASS_int8_gemm_P39 &) = delete;
  static CUTLASS_int8_gemm_P39 &get_instance() {
    static CUTLASS_int8_gemm_P39 instance;
    return instance;
  }

  void run_int8_gemm(int length_m, int length_n, int length_k, int8_t *A,
                   int8_t *B, int32_t *C, int alpha, int beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename int8_gemm::Arguments arguments{problem_size,
                                         {A, length_k},
                                         {B, length_k},
                                         {C, length_m},
                                         {C, length_m},
                                         {alpha, beta},
                                         1};
    size_t workspace_size = int8_gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = int8_gemm_op.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = int8_gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_int8_gemm_P39(int length_m, int length_n, int length_k,
                        int8_t *A, int8_t *B, int32_t *C,
                        int alpha, int beta, cudaStream_t stream) {
  CUTLASS_int8_gemm_P39 &instance = CUTLASS_int8_gemm_P39::get_instance();
  instance.run_int8_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_int8_gemm_P40 {
private:
  CUTLASS_int8_gemm_P40() { }
  ~CUTLASS_int8_gemm_P40() { }

public: 
  // Gemm operator cutlass_tensorop_i16832gemm_s8_64x128_128x3_tn_align8
  using Operation_cutlass_tensorop_i16832gemm_s8_64x128_128x3_tn_align8 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 128>,
    cutlass::gemm::GemmShape<32, 64, 128>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      4,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    8,
    8
  >;
using int8_gemm = Operation_cutlass_tensorop_i16832gemm_s8_64x128_128x3_tn_align8;
  int8_gemm int8_gemm_op;
  CUTLASS_int8_gemm_P40(const CUTLASS_int8_gemm_P40 &) = delete;
  CUTLASS_int8_gemm_P40 &operator=(const CUTLASS_int8_gemm_P40 &) = delete;
  static CUTLASS_int8_gemm_P40 &get_instance() {
    static CUTLASS_int8_gemm_P40 instance;
    return instance;
  }

  void run_int8_gemm(int length_m, int length_n, int length_k, int8_t *A,
                   int8_t *B, int32_t *C, int alpha, int beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename int8_gemm::Arguments arguments{problem_size,
                                         {A, length_k},
                                         {B, length_k},
                                         {C, length_m},
                                         {C, length_m},
                                         {alpha, beta},
                                         1};
    size_t workspace_size = int8_gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = int8_gemm_op.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = int8_gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_int8_gemm_P40(int length_m, int length_n, int length_k,
                        int8_t *A, int8_t *B, int32_t *C,
                        int alpha, int beta, cudaStream_t stream) {
  CUTLASS_int8_gemm_P40 &instance = CUTLASS_int8_gemm_P40::get_instance();
  instance.run_int8_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_int8_gemm_P41 {
private:
  CUTLASS_int8_gemm_P41() { }
  ~CUTLASS_int8_gemm_P41() { }

public: 
  // Gemm operator cutlass_tensorop_i16832gemm_s8_64x64_64x10_tn_align16
  using Operation_cutlass_tensorop_i16832gemm_s8_64x64_64x10_tn_align16 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      4,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    10,
    16,
    16
  >;
using int8_gemm = Operation_cutlass_tensorop_i16832gemm_s8_64x64_64x10_tn_align16;
  int8_gemm int8_gemm_op;
  CUTLASS_int8_gemm_P41(const CUTLASS_int8_gemm_P41 &) = delete;
  CUTLASS_int8_gemm_P41 &operator=(const CUTLASS_int8_gemm_P41 &) = delete;
  static CUTLASS_int8_gemm_P41 &get_instance() {
    static CUTLASS_int8_gemm_P41 instance;
    return instance;
  }

  void run_int8_gemm(int length_m, int length_n, int length_k, int8_t *A,
                   int8_t *B, int32_t *C, int alpha, int beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename int8_gemm::Arguments arguments{problem_size,
                                         {A, length_k},
                                         {B, length_k},
                                         {C, length_m},
                                         {C, length_m},
                                         {alpha, beta},
                                         1};
    size_t workspace_size = int8_gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = int8_gemm_op.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = int8_gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_int8_gemm_P41(int length_m, int length_n, int length_k,
                        int8_t *A, int8_t *B, int32_t *C,
                        int alpha, int beta, cudaStream_t stream) {
  CUTLASS_int8_gemm_P41 &instance = CUTLASS_int8_gemm_P41::get_instance();
  instance.run_int8_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_int8_gemm_P42 {
private:
  CUTLASS_int8_gemm_P42() { }
  ~CUTLASS_int8_gemm_P42() { }

public: 
  // Gemm operator cutlass_tensorop_i16832gemm_s8_128x128_128x3_tn_align16
  using Operation_cutlass_tensorop_i16832gemm_s8_128x128_128x3_tn_align16 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 128>,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      4,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    16,
    16
  >;
using int8_gemm = Operation_cutlass_tensorop_i16832gemm_s8_128x128_128x3_tn_align16;
  int8_gemm int8_gemm_op;
  CUTLASS_int8_gemm_P42(const CUTLASS_int8_gemm_P42 &) = delete;
  CUTLASS_int8_gemm_P42 &operator=(const CUTLASS_int8_gemm_P42 &) = delete;
  static CUTLASS_int8_gemm_P42 &get_instance() {
    static CUTLASS_int8_gemm_P42 instance;
    return instance;
  }

  void run_int8_gemm(int length_m, int length_n, int length_k, int8_t *A,
                   int8_t *B, int32_t *C, int alpha, int beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename int8_gemm::Arguments arguments{problem_size,
                                         {A, length_k},
                                         {B, length_k},
                                         {C, length_m},
                                         {C, length_m},
                                         {alpha, beta},
                                         1};
    size_t workspace_size = int8_gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = int8_gemm_op.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = int8_gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_int8_gemm_P42(int length_m, int length_n, int length_k,
                        int8_t *A, int8_t *B, int32_t *C,
                        int alpha, int beta, cudaStream_t stream) {
  CUTLASS_int8_gemm_P42 &instance = CUTLASS_int8_gemm_P42::get_instance();
  instance.run_int8_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_int8_gemm_P43 {
private:
  CUTLASS_int8_gemm_P43() { }
  ~CUTLASS_int8_gemm_P43() { }

public: 
  // Gemm operator cutlass_tensorop_i8816gemm_s8_128x64_64x2_tn_align16
  using Operation_cutlass_tensorop_i8816gemm_s8_128x64_64x2_tn_align16 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 64, 64>,
    cutlass::gemm::GemmShape<64, 32, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      4,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    2,
    16,
    16
  >;
using int8_gemm = Operation_cutlass_tensorop_i8816gemm_s8_128x64_64x2_tn_align16;
  int8_gemm int8_gemm_op;
  CUTLASS_int8_gemm_P43(const CUTLASS_int8_gemm_P43 &) = delete;
  CUTLASS_int8_gemm_P43 &operator=(const CUTLASS_int8_gemm_P43 &) = delete;
  static CUTLASS_int8_gemm_P43 &get_instance() {
    static CUTLASS_int8_gemm_P43 instance;
    return instance;
  }

  void run_int8_gemm(int length_m, int length_n, int length_k, int8_t *A,
                   int8_t *B, int32_t *C, int alpha, int beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename int8_gemm::Arguments arguments{problem_size,
                                         {A, length_k},
                                         {B, length_k},
                                         {C, length_m},
                                         {C, length_m},
                                         {alpha, beta},
                                         1};
    size_t workspace_size = int8_gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = int8_gemm_op.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = int8_gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_int8_gemm_P43(int length_m, int length_n, int length_k,
                        int8_t *A, int8_t *B, int32_t *C,
                        int alpha, int beta, cudaStream_t stream) {
  CUTLASS_int8_gemm_P43 &instance = CUTLASS_int8_gemm_P43::get_instance();
  instance.run_int8_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_int8_gemm_P44 {
private:
  CUTLASS_int8_gemm_P44() { }
  ~CUTLASS_int8_gemm_P44() { }

public: 
  // Gemm operator cutlass_tensorop_i8816gemm_s8_64x64_64x2_tn_align2
  using Operation_cutlass_tensorop_i8816gemm_s8_64x64_64x2_tn_align2 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      2,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    2,
    2,
    2
  >;
using int8_gemm = Operation_cutlass_tensorop_i8816gemm_s8_64x64_64x2_tn_align2;
  int8_gemm int8_gemm_op;
  CUTLASS_int8_gemm_P44(const CUTLASS_int8_gemm_P44 &) = delete;
  CUTLASS_int8_gemm_P44 &operator=(const CUTLASS_int8_gemm_P44 &) = delete;
  static CUTLASS_int8_gemm_P44 &get_instance() {
    static CUTLASS_int8_gemm_P44 instance;
    return instance;
  }

  void run_int8_gemm(int length_m, int length_n, int length_k, int8_t *A,
                   int8_t *B, int32_t *C, int alpha, int beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename int8_gemm::Arguments arguments{problem_size,
                                         {A, length_k},
                                         {B, length_k},
                                         {C, length_m},
                                         {C, length_m},
                                         {alpha, beta},
                                         1};
    size_t workspace_size = int8_gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = int8_gemm_op.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = int8_gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_int8_gemm_P44(int length_m, int length_n, int length_k,
                        int8_t *A, int8_t *B, int32_t *C,
                        int alpha, int beta, cudaStream_t stream) {
  CUTLASS_int8_gemm_P44 &instance = CUTLASS_int8_gemm_P44::get_instance();
  instance.run_int8_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_int8_gemm_P45 {
private:
  CUTLASS_int8_gemm_P45() { }
  ~CUTLASS_int8_gemm_P45() { }

public: 
  // Gemm operator cutlass_tensorop_i16832gemm_s8_256x64_64x3_tn_align16
  using Operation_cutlass_tensorop_i16832gemm_s8_256x64_64x3_tn_align16 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 64, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    
    cutlass::epilogue::thread::LinearCombination<
      int32_t,
      4,
      int32_t,
      int32_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    16,
    16
  >;
using int8_gemm = Operation_cutlass_tensorop_i16832gemm_s8_256x64_64x3_tn_align16;
  int8_gemm int8_gemm_op;
  CUTLASS_int8_gemm_P45(const CUTLASS_int8_gemm_P45 &) = delete;
  CUTLASS_int8_gemm_P45 &operator=(const CUTLASS_int8_gemm_P45 &) = delete;
  static CUTLASS_int8_gemm_P45 &get_instance() {
    static CUTLASS_int8_gemm_P45 instance;
    return instance;
  }

  void run_int8_gemm(int length_m, int length_n, int length_k, int8_t *A,
                   int8_t *B, int32_t *C, int alpha, int beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename int8_gemm::Arguments arguments{problem_size,
                                         {A, length_k},
                                         {B, length_k},
                                         {C, length_m},
                                         {C, length_m},
                                         {alpha, beta},
                                         1};
    size_t workspace_size = int8_gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = int8_gemm_op.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = int8_gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_int8_gemm_P45(int length_m, int length_n, int length_k,
                        int8_t *A, int8_t *B, int32_t *C,
                        int alpha, int beta, cudaStream_t stream) {
  CUTLASS_int8_gemm_P45 &instance = CUTLASS_int8_gemm_P45::get_instance();
  instance.run_int8_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

void (*CT_int8_gemm[46])(int, int, int,
                         int8_t *, int8_t *, int32_t *, 
                         int, int, cudaStream_t) = {
    cutlass_int8_gemm_P0, cutlass_int8_gemm_P1, cutlass_int8_gemm_P2, cutlass_int8_gemm_P3,
    cutlass_int8_gemm_P4, cutlass_int8_gemm_P5, cutlass_int8_gemm_P6, cutlass_int8_gemm_P7,
    cutlass_int8_gemm_P8, cutlass_int8_gemm_P9, cutlass_int8_gemm_P10, cutlass_int8_gemm_P11,
    cutlass_int8_gemm_P12, cutlass_int8_gemm_P13, cutlass_int8_gemm_P14, cutlass_int8_gemm_P15,
    cutlass_int8_gemm_P16, cutlass_int8_gemm_P17, cutlass_int8_gemm_P18, cutlass_int8_gemm_P19,
    cutlass_int8_gemm_P20, cutlass_int8_gemm_P21, cutlass_int8_gemm_P22, cutlass_int8_gemm_P23,
    cutlass_int8_gemm_P24, cutlass_int8_gemm_P25, cutlass_int8_gemm_P26, cutlass_int8_gemm_P27,
    cutlass_int8_gemm_P28, cutlass_int8_gemm_P29, cutlass_int8_gemm_P30, cutlass_int8_gemm_P31,
    cutlass_int8_gemm_P32, cutlass_int8_gemm_P33, cutlass_int8_gemm_P34, cutlass_int8_gemm_P35,
    cutlass_int8_gemm_P36, cutlass_int8_gemm_P37, cutlass_int8_gemm_P38, cutlass_int8_gemm_P39,
    cutlass_int8_gemm_P40, cutlass_int8_gemm_P41, cutlass_int8_gemm_P42, cutlass_int8_gemm_P43,
    cutlass_int8_gemm_P44, cutlass_int8_gemm_P45};

Int8RccPlan select_int8_online_plan(int m, int n, int k) {
  const int feature_m = n;
  const int feature_n = m;
  const int pid = cutlass_gemm_predict(feature_m, feature_n, k);
  Int8RccPlan best{false, pid, 1,
                   gemm_cost_model(feature_m, feature_n, k, pid)};

  if (env_flag("MOONPOLY_ENABLE_INT8_PATTERN3", true)) {
    static constexpr int kSplitFactors[] = {2, 4, 8};
    const int forced_split_k =
        env_int("MOONPOLY_INT8_PATTERN3_FORCE_SPLITK", 0);
    if (forced_split_k > 1) {
      best.use_splitk = true;
      best.pid = -1;
      best.split_k = forced_split_k;
      best.estimated_ms =
          gemm_splitk_cost_model(feature_m, feature_n, k, forced_split_k);
      return best;
    }
    const float margin = env_float("MOONPOLY_INT8_PATTERN3_MARGIN", 0.97f);
    for (int split_k : kSplitFactors) {
      if (k % 16 != 0 || !int8_pattern3_k_gate(k, split_k)) {
        continue;
      }
      const float cost =
          gemm_splitk_cost_model(feature_m, feature_n, k, split_k);
      if (std::isfinite(cost) && cost > 0.0f && cost < best.estimated_ms * margin) {
        best.use_splitk = true;
        best.pid = -1;
        best.split_k = split_k;
        best.estimated_ms = cost;
      }
    }
  }

  if (std::getenv("MOONPOLY_DEBUG_ONLINE_SELECTOR")) {
    std::cout << "[moonpoly][int8] online m=" << m << " n=" << n
              << " k=" << k << " -> "
              << (best.use_splitk ? "Pattern3" : "Pattern1")
              << " pid=" << best.pid << " split_k=" << best.split_k
              << " est_ms=" << best.estimated_ms << std::endl;
  }
  return best;
}

} // namespace row_col_col
} // namespace int8

void run_moonpoly_gemm_int8_row_col_col(int m, int n, int k, const int8_t *A,
                                        const int8_t *B, int32_t *C,
                                        int alpha, int beta, cudaStream_t stream) 
{
  auto plan = int8::row_col_col::select_int8_online_plan(m, n, k);
  if (plan.use_splitk) {
    int8::row_col_col::CUTLASS_INT8_PATTERN3_SPLITK::run(
        m, n, k, const_cast<int8_t *>(A), const_cast<int8_t *>(B), C, alpha,
        beta, plan.split_k, stream);
    return;
  }
  int8::row_col_col::CT_int8_gemm[plan.pid](
      m, n, k, const_cast<int8_t *>(A), const_cast<int8_t *>(B), C, alpha,
      beta, stream);
}

// Used for profiling
void run_moonpoly_selected_gemm_int8_row_col_col(int pid, int m, int n, int k, const int8_t *A,
                                                 const int8_t *B, int32_t *C,
                                                 int alpha, int beta, cudaStream_t stream) 
{ 
  if (k % int8::row_col_col::int8_gemm_alignment[pid] != 0) {
    return;
  }
  int8::row_col_col::CT_int8_gemm[pid](m, n, k, const_cast<int8_t *>(A),
                                     const_cast<int8_t *>(B), C, alpha, beta, stream);
}

void get_gemm_shared_mem_int8_row_col_col(int pid, int *shared_mem) 
{
  shared_mem[0] = int8::row_col_col::int8_gemm_shared_mem[pid].pm;
  shared_mem[1] = int8::row_col_col::int8_gemm_shared_mem[pid].pn;
  shared_mem[2] = int8::row_col_col::int8_gemm_shared_mem[pid].pk;
}

void get_gemm_alignment_int8_row_col_col(int pid, int *alignment) 
{
  *alignment = int8::row_col_col::int8_gemm_alignment[pid];
}

} // namespace moonpoly
