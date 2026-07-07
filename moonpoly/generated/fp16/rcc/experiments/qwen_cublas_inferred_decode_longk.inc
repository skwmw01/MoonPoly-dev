// Auto-generated Qwen long-K GemmUniversal candidates.

// pid=278 tag=cb_longk_128x64_s5_sk
// family=gemm_like_longk split_k_serial=True
// notes=Long-K splitKreduce-inspired 128x64, stages=5.
struct CUTLASS_GEMM_P278 {
public:
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = cutlass::half_t;
  using LayoutC = cutlass::layout::ColumnMajor;
  using ElementAccumulator = float;

  using ArchTag = cutlass::arch::Sm80;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ThreadblockSwizzle =
      cutlass::gemm::threadblock::ThreadblockSwizzleStreamK;
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  static constexpr int NumStages = 5;
  static constexpr int AlignmentA = 8;
  static constexpr int AlignmentB = 8;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementC, 1, ElementAccumulator, ElementAccumulator>;
  using Gemm = cutlass::gemm::device::GemmUniversal<
      ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
      ElementAccumulator, OperatorClass, ArchTag, ThreadblockShape, WarpShape,
      InstructionShape, EpilogueOp, ThreadblockSwizzle, NumStages, AlignmentA,
      AlignmentB>;

  static typename Gemm::Arguments make_arguments(
      int m, int n, int k, int split_k_slices, ElementA *ptr_A, ElementB *ptr_B,
      ElementC *ptr_C, float alpha, float beta) {
    cutlass::gemm::GemmCoord problem_size(m, n, k);
    auto gemm_mode = cutlass::gemm::GemmUniversalMode::kGemm;
    return typename Gemm::Arguments(
        gemm_mode, problem_size, split_k_slices,
        {ElementAccumulator(alpha), ElementAccumulator(beta)}, ptr_A, ptr_B,
        ptr_C, ptr_C, 0, 0, 0, 0, LayoutA::packed({m, k}).stride(0),
        LayoutB::packed({k, n}).stride(0), LayoutC::packed({m, n}).stride(0),
        LayoutC::packed({m, n}).stride(0));
  }

  static size_t query_workspace_size(int m, int n, int k, int split_k_slices) {
    auto arguments = make_arguments(
        m, n, k, split_k_slices, nullptr, nullptr, nullptr, 1.0f, 0.0f);
    return Gemm::get_workspace_size(arguments);
  }

  void run_gemm(int m, int n, int k, int split_k_slices, ElementA *ptr_A,
                ElementB *ptr_B, ElementC *ptr_C, float alpha, float beta,
                cudaStream_t stream) {
    Gemm gemm_op_local;
    auto arguments = make_arguments(
        m, n, k, split_k_slices, ptr_A, ptr_B, ptr_C, alpha, beta);
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status =
        gemm_op_local.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = gemm_op_local.run(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_gemm_P278(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P278 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}


// pid=279 tag=cb_longk_128x64_s6_sk
// family=gemm_like_longk split_k_serial=True
// notes=Long-K splitKreduce-inspired 128x64, stages=6.
struct CUTLASS_GEMM_P279 {
public:
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = cutlass::half_t;
  using LayoutC = cutlass::layout::ColumnMajor;
  using ElementAccumulator = float;

  using ArchTag = cutlass::arch::Sm80;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ThreadblockSwizzle =
      cutlass::gemm::threadblock::ThreadblockSwizzleStreamK;
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  static constexpr int NumStages = 6;
  static constexpr int AlignmentA = 8;
  static constexpr int AlignmentB = 8;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementC, 1, ElementAccumulator, ElementAccumulator>;
  using Gemm = cutlass::gemm::device::GemmUniversal<
      ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
      ElementAccumulator, OperatorClass, ArchTag, ThreadblockShape, WarpShape,
      InstructionShape, EpilogueOp, ThreadblockSwizzle, NumStages, AlignmentA,
      AlignmentB>;

  static typename Gemm::Arguments make_arguments(
      int m, int n, int k, int split_k_slices, ElementA *ptr_A, ElementB *ptr_B,
      ElementC *ptr_C, float alpha, float beta) {
    cutlass::gemm::GemmCoord problem_size(m, n, k);
    auto gemm_mode = cutlass::gemm::GemmUniversalMode::kGemm;
    return typename Gemm::Arguments(
        gemm_mode, problem_size, split_k_slices,
        {ElementAccumulator(alpha), ElementAccumulator(beta)}, ptr_A, ptr_B,
        ptr_C, ptr_C, 0, 0, 0, 0, LayoutA::packed({m, k}).stride(0),
        LayoutB::packed({k, n}).stride(0), LayoutC::packed({m, n}).stride(0),
        LayoutC::packed({m, n}).stride(0));
  }

  static size_t query_workspace_size(int m, int n, int k, int split_k_slices) {
    auto arguments = make_arguments(
        m, n, k, split_k_slices, nullptr, nullptr, nullptr, 1.0f, 0.0f);
    return Gemm::get_workspace_size(arguments);
  }

  void run_gemm(int m, int n, int k, int split_k_slices, ElementA *ptr_A,
                ElementB *ptr_B, ElementC *ptr_C, float alpha, float beta,
                cudaStream_t stream) {
    Gemm gemm_op_local;
    auto arguments = make_arguments(
        m, n, k, split_k_slices, ptr_A, ptr_B, ptr_C, alpha, beta);
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status =
        gemm_op_local.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = gemm_op_local.run(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_gemm_P279(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P279 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}


// pid=280 tag=cb_longk_64x64_s5_sk
// family=gemm_like_longk split_k_serial=True
// notes=Long-K 64x64 fallback.
struct CUTLASS_GEMM_P280 {
public:
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = cutlass::half_t;
  using LayoutC = cutlass::layout::ColumnMajor;
  using ElementAccumulator = float;

  using ArchTag = cutlass::arch::Sm80;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ThreadblockSwizzle =
      cutlass::gemm::threadblock::ThreadblockSwizzleStreamK;
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  static constexpr int NumStages = 5;
  static constexpr int AlignmentA = 8;
  static constexpr int AlignmentB = 8;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementC, 1, ElementAccumulator, ElementAccumulator>;
  using Gemm = cutlass::gemm::device::GemmUniversal<
      ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
      ElementAccumulator, OperatorClass, ArchTag, ThreadblockShape, WarpShape,
      InstructionShape, EpilogueOp, ThreadblockSwizzle, NumStages, AlignmentA,
      AlignmentB>;

  static typename Gemm::Arguments make_arguments(
      int m, int n, int k, int split_k_slices, ElementA *ptr_A, ElementB *ptr_B,
      ElementC *ptr_C, float alpha, float beta) {
    cutlass::gemm::GemmCoord problem_size(m, n, k);
    auto gemm_mode = cutlass::gemm::GemmUniversalMode::kGemm;
    return typename Gemm::Arguments(
        gemm_mode, problem_size, split_k_slices,
        {ElementAccumulator(alpha), ElementAccumulator(beta)}, ptr_A, ptr_B,
        ptr_C, ptr_C, 0, 0, 0, 0, LayoutA::packed({m, k}).stride(0),
        LayoutB::packed({k, n}).stride(0), LayoutC::packed({m, n}).stride(0),
        LayoutC::packed({m, n}).stride(0));
  }

  static size_t query_workspace_size(int m, int n, int k, int split_k_slices) {
    auto arguments = make_arguments(
        m, n, k, split_k_slices, nullptr, nullptr, nullptr, 1.0f, 0.0f);
    return Gemm::get_workspace_size(arguments);
  }

  void run_gemm(int m, int n, int k, int split_k_slices, ElementA *ptr_A,
                ElementB *ptr_B, ElementC *ptr_C, float alpha, float beta,
                cudaStream_t stream) {
    Gemm gemm_op_local;
    auto arguments = make_arguments(
        m, n, k, split_k_slices, ptr_A, ptr_B, ptr_C, alpha, beta);
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status =
        gemm_op_local.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = gemm_op_local.run(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_gemm_P280(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P280 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}


// pid=281 tag=cb_longk_64x64_s6_sk
// family=gemm_like_longk split_k_serial=True
// notes=Long-K 64x64, stages=6.
struct CUTLASS_GEMM_P281 {
public:
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = cutlass::half_t;
  using LayoutC = cutlass::layout::ColumnMajor;
  using ElementAccumulator = float;

  using ArchTag = cutlass::arch::Sm80;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ThreadblockSwizzle =
      cutlass::gemm::threadblock::ThreadblockSwizzleStreamK;
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  static constexpr int NumStages = 6;
  static constexpr int AlignmentA = 8;
  static constexpr int AlignmentB = 8;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementC, 1, ElementAccumulator, ElementAccumulator>;
  using Gemm = cutlass::gemm::device::GemmUniversal<
      ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
      ElementAccumulator, OperatorClass, ArchTag, ThreadblockShape, WarpShape,
      InstructionShape, EpilogueOp, ThreadblockSwizzle, NumStages, AlignmentA,
      AlignmentB>;

  static typename Gemm::Arguments make_arguments(
      int m, int n, int k, int split_k_slices, ElementA *ptr_A, ElementB *ptr_B,
      ElementC *ptr_C, float alpha, float beta) {
    cutlass::gemm::GemmCoord problem_size(m, n, k);
    auto gemm_mode = cutlass::gemm::GemmUniversalMode::kGemm;
    return typename Gemm::Arguments(
        gemm_mode, problem_size, split_k_slices,
        {ElementAccumulator(alpha), ElementAccumulator(beta)}, ptr_A, ptr_B,
        ptr_C, ptr_C, 0, 0, 0, 0, LayoutA::packed({m, k}).stride(0),
        LayoutB::packed({k, n}).stride(0), LayoutC::packed({m, n}).stride(0),
        LayoutC::packed({m, n}).stride(0));
  }

  static size_t query_workspace_size(int m, int n, int k, int split_k_slices) {
    auto arguments = make_arguments(
        m, n, k, split_k_slices, nullptr, nullptr, nullptr, 1.0f, 0.0f);
    return Gemm::get_workspace_size(arguments);
  }

  void run_gemm(int m, int n, int k, int split_k_slices, ElementA *ptr_A,
                ElementB *ptr_B, ElementC *ptr_C, float alpha, float beta,
                cudaStream_t stream) {
    Gemm gemm_op_local;
    auto arguments = make_arguments(
        m, n, k, split_k_slices, ptr_A, ptr_B, ptr_C, alpha, beta);
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status =
        gemm_op_local.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = gemm_op_local.run(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_gemm_P281(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P281 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}


// pid=282 tag=cb_longk_128x64_s6_sw4_sk
// family=gemm_like_longk split_k_serial=True
// notes=Long-K with stronger swizzle.
struct CUTLASS_GEMM_P282 {
public:
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = cutlass::half_t;
  using LayoutC = cutlass::layout::ColumnMajor;
  using ElementAccumulator = float;

  using ArchTag = cutlass::arch::Sm80;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ThreadblockSwizzle =
      cutlass::gemm::threadblock::ThreadblockSwizzleStreamK;
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  static constexpr int NumStages = 6;
  static constexpr int AlignmentA = 8;
  static constexpr int AlignmentB = 8;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementC, 1, ElementAccumulator, ElementAccumulator>;
  using Gemm = cutlass::gemm::device::GemmUniversal<
      ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
      ElementAccumulator, OperatorClass, ArchTag, ThreadblockShape, WarpShape,
      InstructionShape, EpilogueOp, ThreadblockSwizzle, NumStages, AlignmentA,
      AlignmentB>;

  static typename Gemm::Arguments make_arguments(
      int m, int n, int k, int split_k_slices, ElementA *ptr_A, ElementB *ptr_B,
      ElementC *ptr_C, float alpha, float beta) {
    cutlass::gemm::GemmCoord problem_size(m, n, k);
    auto gemm_mode = cutlass::gemm::GemmUniversalMode::kGemm;
    return typename Gemm::Arguments(
        gemm_mode, problem_size, split_k_slices,
        {ElementAccumulator(alpha), ElementAccumulator(beta)}, ptr_A, ptr_B,
        ptr_C, ptr_C, 0, 0, 0, 0, LayoutA::packed({m, k}).stride(0),
        LayoutB::packed({k, n}).stride(0), LayoutC::packed({m, n}).stride(0),
        LayoutC::packed({m, n}).stride(0));
  }

  static size_t query_workspace_size(int m, int n, int k, int split_k_slices) {
    auto arguments = make_arguments(
        m, n, k, split_k_slices, nullptr, nullptr, nullptr, 1.0f, 0.0f);
    return Gemm::get_workspace_size(arguments);
  }

  void run_gemm(int m, int n, int k, int split_k_slices, ElementA *ptr_A,
                ElementB *ptr_B, ElementC *ptr_C, float alpha, float beta,
                cudaStream_t stream) {
    Gemm gemm_op_local;
    auto arguments = make_arguments(
        m, n, k, split_k_slices, ptr_A, ptr_B, ptr_C, alpha, beta);
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status =
        gemm_op_local.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = gemm_op_local.run(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_gemm_P282(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P282 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}


// pid=283 tag=cb_longk_128x64_s6_a4_sk
// family=gemm_like_longk split_k_serial=True
// notes=Long-K alignment-4 fallback.
struct CUTLASS_GEMM_P283 {
public:
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = cutlass::half_t;
  using LayoutC = cutlass::layout::ColumnMajor;
  using ElementAccumulator = float;

  using ArchTag = cutlass::arch::Sm80;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ThreadblockSwizzle =
      cutlass::gemm::threadblock::ThreadblockSwizzleStreamK;
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  static constexpr int NumStages = 6;
  static constexpr int AlignmentA = 4;
  static constexpr int AlignmentB = 4;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementC, 1, ElementAccumulator, ElementAccumulator>;
  using Gemm = cutlass::gemm::device::GemmUniversal<
      ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
      ElementAccumulator, OperatorClass, ArchTag, ThreadblockShape, WarpShape,
      InstructionShape, EpilogueOp, ThreadblockSwizzle, NumStages, AlignmentA,
      AlignmentB>;

  static typename Gemm::Arguments make_arguments(
      int m, int n, int k, int split_k_slices, ElementA *ptr_A, ElementB *ptr_B,
      ElementC *ptr_C, float alpha, float beta) {
    cutlass::gemm::GemmCoord problem_size(m, n, k);
    auto gemm_mode = cutlass::gemm::GemmUniversalMode::kGemm;
    return typename Gemm::Arguments(
        gemm_mode, problem_size, split_k_slices,
        {ElementAccumulator(alpha), ElementAccumulator(beta)}, ptr_A, ptr_B,
        ptr_C, ptr_C, 0, 0, 0, 0, LayoutA::packed({m, k}).stride(0),
        LayoutB::packed({k, n}).stride(0), LayoutC::packed({m, n}).stride(0),
        LayoutC::packed({m, n}).stride(0));
  }

  static size_t query_workspace_size(int m, int n, int k, int split_k_slices) {
    auto arguments = make_arguments(
        m, n, k, split_k_slices, nullptr, nullptr, nullptr, 1.0f, 0.0f);
    return Gemm::get_workspace_size(arguments);
  }

  void run_gemm(int m, int n, int k, int split_k_slices, ElementA *ptr_A,
                ElementB *ptr_B, ElementC *ptr_C, float alpha, float beta,
                cudaStream_t stream) {
    Gemm gemm_op_local;
    auto arguments = make_arguments(
        m, n, k, split_k_slices, ptr_A, ptr_B, ptr_C, alpha, beta);
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status =
        gemm_op_local.initialize(arguments, workspace.get(), stream);
    CUTLASS_CHECK(status);
    status = gemm_op_local.run(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_gemm_P283(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P283 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}

