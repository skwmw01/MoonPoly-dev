// Auto-generated legacy FP16 split-K GemmUniversal candidates.
// Cleaned for runtime split_k and stream dispatch; kept as the stable generated source.

// pid=400 tag=legacy_splitk_p0
// family=fp16_legacy_splitk_universal runtime_split_k=True
struct CUTLASS_GEMM_P400 {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<16, 96, 64>;
  using WarpShape = cutlass::gemm::GemmShape<16, 48, 64>;
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
    return typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm, problem_size, split_k_slices,
        {ElementAccumulator(alpha), ElementAccumulator(beta)}, ptr_A, ptr_B,
        ptr_C, ptr_C, 0, 0, 0, 0, LayoutA::packed({m, k}).stride(0),
        LayoutB::packed({k, n}).stride(0), LayoutC::packed({m, n}).stride(0),
        LayoutC::packed({m, n}).stride(0));
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

void cutlass_gemm_P400(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P400 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}

// pid=401 tag=legacy_splitk_p1
// family=fp16_legacy_splitk_universal runtime_split_k=True
struct CUTLASS_GEMM_P401 {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
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
    return typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm, problem_size, split_k_slices,
        {ElementAccumulator(alpha), ElementAccumulator(beta)}, ptr_A, ptr_B,
        ptr_C, ptr_C, 0, 0, 0, 0, LayoutA::packed({m, k}).stride(0),
        LayoutB::packed({k, n}).stride(0), LayoutC::packed({m, n}).stride(0),
        LayoutC::packed({m, n}).stride(0));
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

void cutlass_gemm_P401(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P401 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}

// pid=402 tag=legacy_splitk_p2
// family=fp16_legacy_splitk_universal runtime_split_k=True
struct CUTLASS_GEMM_P402 {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  static constexpr int NumStages = 4;
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
    return typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm, problem_size, split_k_slices,
        {ElementAccumulator(alpha), ElementAccumulator(beta)}, ptr_A, ptr_B,
        ptr_C, ptr_C, 0, 0, 0, 0, LayoutA::packed({m, k}).stride(0),
        LayoutB::packed({k, n}).stride(0), LayoutC::packed({m, n}).stride(0),
        LayoutC::packed({m, n}).stride(0));
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

void cutlass_gemm_P402(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P402 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}

// pid=403 tag=legacy_splitk_p3
// family=fp16_legacy_splitk_universal runtime_split_k=True
struct CUTLASS_GEMM_P403 {
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
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 64>;
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
    return typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm, problem_size, split_k_slices,
        {ElementAccumulator(alpha), ElementAccumulator(beta)}, ptr_A, ptr_B,
        ptr_C, ptr_C, 0, 0, 0, 0, LayoutA::packed({m, k}).stride(0),
        LayoutB::packed({k, n}).stride(0), LayoutC::packed({m, n}).stride(0),
        LayoutC::packed({m, n}).stride(0));
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

void cutlass_gemm_P403(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P403 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}

// pid=404 tag=legacy_splitk_p4
// family=fp16_legacy_splitk_universal runtime_split_k=True
struct CUTLASS_GEMM_P404 {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<16, 32, 64>;
  using WarpShape = cutlass::gemm::GemmShape<16, 32, 32>;
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
    return typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm, problem_size, split_k_slices,
        {ElementAccumulator(alpha), ElementAccumulator(beta)}, ptr_A, ptr_B,
        ptr_C, ptr_C, 0, 0, 0, 0, LayoutA::packed({m, k}).stride(0),
        LayoutB::packed({k, n}).stride(0), LayoutC::packed({m, n}).stride(0),
        LayoutC::packed({m, n}).stride(0));
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

void cutlass_gemm_P404(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P404 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}

// pid=405 tag=legacy_splitk_p5
// family=fp16_legacy_splitk_universal runtime_split_k=True
struct CUTLASS_GEMM_P405 {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 16, 32>;
  using WarpShape = cutlass::gemm::GemmShape<16, 16, 32>;
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
    return typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm, problem_size, split_k_slices,
        {ElementAccumulator(alpha), ElementAccumulator(beta)}, ptr_A, ptr_B,
        ptr_C, ptr_C, 0, 0, 0, 0, LayoutA::packed({m, k}).stride(0),
        LayoutB::packed({k, n}).stride(0), LayoutC::packed({m, n}).stride(0),
        LayoutC::packed({m, n}).stride(0));
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

void cutlass_gemm_P405(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P405 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}

// pid=406 tag=legacy_splitk_p6
// family=fp16_legacy_splitk_universal runtime_split_k=True
struct CUTLASS_GEMM_P406 {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<16, 16, 64>;
  using WarpShape = cutlass::gemm::GemmShape<16, 16, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  static constexpr int NumStages = 10;
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
    return typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm, problem_size, split_k_slices,
        {ElementAccumulator(alpha), ElementAccumulator(beta)}, ptr_A, ptr_B,
        ptr_C, ptr_C, 0, 0, 0, 0, LayoutA::packed({m, k}).stride(0),
        LayoutB::packed({k, n}).stride(0), LayoutC::packed({m, n}).stride(0),
        LayoutC::packed({m, n}).stride(0));
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

void cutlass_gemm_P406(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P406 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}

// pid=407 tag=legacy_splitk_p7
// family=fp16_legacy_splitk_universal runtime_split_k=True
struct CUTLASS_GEMM_P407 {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 8, 64>;
  using WarpShape = cutlass::gemm::GemmShape<32, 8, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  static constexpr int NumStages = 4;
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
    return typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm, problem_size, split_k_slices,
        {ElementAccumulator(alpha), ElementAccumulator(beta)}, ptr_A, ptr_B,
        ptr_C, ptr_C, 0, 0, 0, 0, LayoutA::packed({m, k}).stride(0),
        LayoutB::packed({k, n}).stride(0), LayoutC::packed({m, n}).stride(0),
        LayoutC::packed({m, n}).stride(0));
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

void cutlass_gemm_P407(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P407 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}

// pid=408 tag=legacy_splitk_p8
// family=fp16_legacy_splitk_universal runtime_split_k=True
struct CUTLASS_GEMM_P408 {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 8, 64>;
  using WarpShape = cutlass::gemm::GemmShape<32, 8, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  static constexpr int NumStages = 4;
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
    return typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm, problem_size, split_k_slices,
        {ElementAccumulator(alpha), ElementAccumulator(beta)}, ptr_A, ptr_B,
        ptr_C, ptr_C, 0, 0, 0, 0, LayoutA::packed({m, k}).stride(0),
        LayoutB::packed({k, n}).stride(0), LayoutC::packed({m, n}).stride(0),
        LayoutC::packed({m, n}).stride(0));
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

void cutlass_gemm_P408(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P408 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}

// pid=409 tag=legacy_splitk_p9
// family=fp16_legacy_splitk_universal runtime_split_k=True
struct CUTLASS_GEMM_P409 {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 16, 64>;
  using WarpShape = cutlass::gemm::GemmShape<16, 16, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  static constexpr int NumStages = 2;
  static constexpr int AlignmentA = 2;
  static constexpr int AlignmentB = 2;
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
    return typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm, problem_size, split_k_slices,
        {ElementAccumulator(alpha), ElementAccumulator(beta)}, ptr_A, ptr_B,
        ptr_C, ptr_C, 0, 0, 0, 0, LayoutA::packed({m, k}).stride(0),
        LayoutB::packed({k, n}).stride(0), LayoutC::packed({m, n}).stride(0),
        LayoutC::packed({m, n}).stride(0));
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

void cutlass_gemm_P409(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P409 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}

// pid=410 tag=legacy_splitk_p10
// family=fp16_legacy_splitk_universal runtime_split_k=True
struct CUTLASS_GEMM_P410 {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 64, 64>;
  using WarpShape = cutlass::gemm::GemmShape<16, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  static constexpr int NumStages = 2;
  static constexpr int AlignmentA = 1;
  static constexpr int AlignmentB = 1;
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
    return typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm, problem_size, split_k_slices,
        {ElementAccumulator(alpha), ElementAccumulator(beta)}, ptr_A, ptr_B,
        ptr_C, ptr_C, 0, 0, 0, 0, LayoutA::packed({m, k}).stride(0),
        LayoutB::packed({k, n}).stride(0), LayoutC::packed({m, n}).stride(0),
        LayoutC::packed({m, n}).stride(0));
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

void cutlass_gemm_P410(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P410 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}

// pid=411 tag=legacy_splitk_p11
// family=fp16_legacy_splitk_universal runtime_split_k=True
struct CUTLASS_GEMM_P411 {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 96, 64>;
  using WarpShape = cutlass::gemm::GemmShape<32, 96, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  static constexpr int NumStages = 2;
  static constexpr int AlignmentA = 2;
  static constexpr int AlignmentB = 2;
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
    return typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm, problem_size, split_k_slices,
        {ElementAccumulator(alpha), ElementAccumulator(beta)}, ptr_A, ptr_B,
        ptr_C, ptr_C, 0, 0, 0, 0, LayoutA::packed({m, k}).stride(0),
        LayoutB::packed({k, n}).stride(0), LayoutC::packed({m, n}).stride(0),
        LayoutC::packed({m, n}).stride(0));
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

void cutlass_gemm_P411(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P411 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}

// pid=412 tag=legacy_splitk_p12
// family=fp16_legacy_splitk_universal runtime_split_k=True
struct CUTLASS_GEMM_P412 {
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
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  static constexpr int NumStages = 2;
  static constexpr int AlignmentA = 2;
  static constexpr int AlignmentB = 2;
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
    return typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm, problem_size, split_k_slices,
        {ElementAccumulator(alpha), ElementAccumulator(beta)}, ptr_A, ptr_B,
        ptr_C, ptr_C, 0, 0, 0, 0, LayoutA::packed({m, k}).stride(0),
        LayoutB::packed({k, n}).stride(0), LayoutC::packed({m, n}).stride(0),
        LayoutC::packed({m, n}).stride(0));
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

void cutlass_gemm_P412(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P412 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}

// pid=413 tag=legacy_splitk_p13
// family=fp16_legacy_splitk_universal runtime_split_k=True
struct CUTLASS_GEMM_P413 {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<16, 8, 32>;
  using WarpShape = cutlass::gemm::GemmShape<16, 8, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  static constexpr int NumStages = 5;
  static constexpr int AlignmentA = 2;
  static constexpr int AlignmentB = 2;
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
    return typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm, problem_size, split_k_slices,
        {ElementAccumulator(alpha), ElementAccumulator(beta)}, ptr_A, ptr_B,
        ptr_C, ptr_C, 0, 0, 0, 0, LayoutA::packed({m, k}).stride(0),
        LayoutB::packed({k, n}).stride(0), LayoutC::packed({m, n}).stride(0),
        LayoutC::packed({m, n}).stride(0));
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

void cutlass_gemm_P413(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P413 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}

// pid=414 tag=legacy_splitk_p14
// family=fp16_legacy_splitk_universal runtime_split_k=True
struct CUTLASS_GEMM_P414 {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<96, 160, 32>;
  using WarpShape = cutlass::gemm::GemmShape<48, 80, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  static constexpr int NumStages = 3;
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
    return typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm, problem_size, split_k_slices,
        {ElementAccumulator(alpha), ElementAccumulator(beta)}, ptr_A, ptr_B,
        ptr_C, ptr_C, 0, 0, 0, 0, LayoutA::packed({m, k}).stride(0),
        LayoutB::packed({k, n}).stride(0), LayoutC::packed({m, n}).stride(0),
        LayoutC::packed({m, n}).stride(0));
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

void cutlass_gemm_P414(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P414 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}

// pid=415 tag=legacy_splitk_p15
// family=fp16_legacy_splitk_universal runtime_split_k=True
struct CUTLASS_GEMM_P415 {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<80, 128, 64>;
  using WarpShape = cutlass::gemm::GemmShape<80, 32, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  static constexpr int NumStages = 3;
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
    return typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm, problem_size, split_k_slices,
        {ElementAccumulator(alpha), ElementAccumulator(beta)}, ptr_A, ptr_B,
        ptr_C, ptr_C, 0, 0, 0, 0, LayoutA::packed({m, k}).stride(0),
        LayoutB::packed({k, n}).stride(0), LayoutC::packed({m, n}).stride(0),
        LayoutC::packed({m, n}).stride(0));
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

void cutlass_gemm_P415(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P415 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}

// pid=416 tag=legacy_splitk_p16
// family=fp16_legacy_splitk_universal runtime_split_k=True
struct CUTLASS_GEMM_P416 {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<48, 32, 64>;
  using WarpShape = cutlass::gemm::GemmShape<48, 16, 64>;
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
    return typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm, problem_size, split_k_slices,
        {ElementAccumulator(alpha), ElementAccumulator(beta)}, ptr_A, ptr_B,
        ptr_C, ptr_C, 0, 0, 0, 0, LayoutA::packed({m, k}).stride(0),
        LayoutB::packed({k, n}).stride(0), LayoutC::packed({m, n}).stride(0),
        LayoutC::packed({m, n}).stride(0));
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

void cutlass_gemm_P416(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P416 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}

// pid=417 tag=legacy_splitk_p17
// family=fp16_legacy_splitk_universal runtime_split_k=True
struct CUTLASS_GEMM_P417 {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<256, 32, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  static constexpr int NumStages = 3;
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
    return typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm, problem_size, split_k_slices,
        {ElementAccumulator(alpha), ElementAccumulator(beta)}, ptr_A, ptr_B,
        ptr_C, ptr_C, 0, 0, 0, 0, LayoutA::packed({m, k}).stride(0),
        LayoutB::packed({k, n}).stride(0), LayoutC::packed({m, n}).stride(0),
        LayoutC::packed({m, n}).stride(0));
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

void cutlass_gemm_P417(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P417 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}

// pid=418 tag=legacy_splitk_p18
// family=fp16_legacy_splitk_universal runtime_split_k=True
struct CUTLASS_GEMM_P418 {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<16, 64, 32>;
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
    return typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm, problem_size, split_k_slices,
        {ElementAccumulator(alpha), ElementAccumulator(beta)}, ptr_A, ptr_B,
        ptr_C, ptr_C, 0, 0, 0, 0, LayoutA::packed({m, k}).stride(0),
        LayoutB::packed({k, n}).stride(0), LayoutC::packed({m, n}).stride(0),
        LayoutC::packed({m, n}).stride(0));
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

void cutlass_gemm_P418(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P418 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}

// pid=419 tag=legacy_splitk_p19
// family=fp16_legacy_splitk_universal runtime_split_k=True
struct CUTLASS_GEMM_P419 {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<16, 128, 64>;
  using WarpShape = cutlass::gemm::GemmShape<16, 32, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  static constexpr int NumStages = 4;
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
    return typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm, problem_size, split_k_slices,
        {ElementAccumulator(alpha), ElementAccumulator(beta)}, ptr_A, ptr_B,
        ptr_C, ptr_C, 0, 0, 0, 0, LayoutA::packed({m, k}).stride(0),
        LayoutB::packed({k, n}).stride(0), LayoutC::packed({m, n}).stride(0),
        LayoutC::packed({m, n}).stride(0));
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

void cutlass_gemm_P419(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P419 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}

// pid=420 tag=legacy_splitk_p20
// family=fp16_legacy_splitk_universal runtime_split_k=True
struct CUTLASS_GEMM_P420 {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<80, 96, 64>;
  using WarpShape = cutlass::gemm::GemmShape<80, 48, 32>;
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
    return typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm, problem_size, split_k_slices,
        {ElementAccumulator(alpha), ElementAccumulator(beta)}, ptr_A, ptr_B,
        ptr_C, ptr_C, 0, 0, 0, 0, LayoutA::packed({m, k}).stride(0),
        LayoutB::packed({k, n}).stride(0), LayoutC::packed({m, n}).stride(0),
        LayoutC::packed({m, n}).stride(0));
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

void cutlass_gemm_P420(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P420 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}

// pid=421 tag=legacy_splitk_p21
// family=fp16_legacy_splitk_universal runtime_split_k=True
struct CUTLASS_GEMM_P421 {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<16, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<16, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  static constexpr int NumStages = 2;
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
    return typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm, problem_size, split_k_slices,
        {ElementAccumulator(alpha), ElementAccumulator(beta)}, ptr_A, ptr_B,
        ptr_C, ptr_C, 0, 0, 0, 0, LayoutA::packed({m, k}).stride(0),
        LayoutB::packed({k, n}).stride(0), LayoutC::packed({m, n}).stride(0),
        LayoutC::packed({m, n}).stride(0));
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

void cutlass_gemm_P421(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P421 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}

// pid=422 tag=legacy_splitk_p22
// family=fp16_legacy_splitk_universal runtime_split_k=True
struct CUTLASS_GEMM_P422 {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 96, 32>;
  using WarpShape = cutlass::gemm::GemmShape<32, 96, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  static constexpr int NumStages = 3;
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
    return typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm, problem_size, split_k_slices,
        {ElementAccumulator(alpha), ElementAccumulator(beta)}, ptr_A, ptr_B,
        ptr_C, ptr_C, 0, 0, 0, 0, LayoutA::packed({m, k}).stride(0),
        LayoutB::packed({k, n}).stride(0), LayoutC::packed({m, n}).stride(0),
        LayoutC::packed({m, n}).stride(0));
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

void cutlass_gemm_P422(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P422 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}

// pid=423 tag=legacy_splitk_p23
// family=fp16_legacy_splitk_universal runtime_split_k=True
struct CUTLASS_GEMM_P423 {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  static constexpr int NumStages = 3;
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
    return typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm, problem_size, split_k_slices,
        {ElementAccumulator(alpha), ElementAccumulator(beta)}, ptr_A, ptr_B,
        ptr_C, ptr_C, 0, 0, 0, 0, LayoutA::packed({m, k}).stride(0),
        LayoutB::packed({k, n}).stride(0), LayoutC::packed({m, n}).stride(0),
        LayoutC::packed({m, n}).stride(0));
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

void cutlass_gemm_P423(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P423 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}

// pid=424 tag=legacy_splitk_p24
// family=fp16_legacy_splitk_universal runtime_split_k=True
struct CUTLASS_GEMM_P424 {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 16, 64>;
  using WarpShape = cutlass::gemm::GemmShape<8, 16, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  static constexpr int NumStages = 2;
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
    return typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm, problem_size, split_k_slices,
        {ElementAccumulator(alpha), ElementAccumulator(beta)}, ptr_A, ptr_B,
        ptr_C, ptr_C, 0, 0, 0, 0, LayoutA::packed({m, k}).stride(0),
        LayoutB::packed({k, n}).stride(0), LayoutC::packed({m, n}).stride(0),
        LayoutC::packed({m, n}).stride(0));
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

void cutlass_gemm_P424(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P424 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}

// pid=425 tag=legacy_splitk_p25
// family=fp16_legacy_splitk_universal runtime_split_k=True
struct CUTLASS_GEMM_P425 {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<16, 8, 64>;
  using WarpShape = cutlass::gemm::GemmShape<16, 8, 64>;
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
    return typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm, problem_size, split_k_slices,
        {ElementAccumulator(alpha), ElementAccumulator(beta)}, ptr_A, ptr_B,
        ptr_C, ptr_C, 0, 0, 0, 0, LayoutA::packed({m, k}).stride(0),
        LayoutB::packed({k, n}).stride(0), LayoutC::packed({m, n}).stride(0),
        LayoutC::packed({m, n}).stride(0));
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

void cutlass_gemm_P425(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P425 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}

// pid=426 tag=legacy_splitk_p26
// family=fp16_legacy_splitk_universal runtime_split_k=True
struct CUTLASS_GEMM_P426 {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<16, 64, 64>;
  using WarpShape = cutlass::gemm::GemmShape<16, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  static constexpr int NumStages = 3;
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
    return typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm, problem_size, split_k_slices,
        {ElementAccumulator(alpha), ElementAccumulator(beta)}, ptr_A, ptr_B,
        ptr_C, ptr_C, 0, 0, 0, 0, LayoutA::packed({m, k}).stride(0),
        LayoutB::packed({k, n}).stride(0), LayoutC::packed({m, n}).stride(0),
        LayoutC::packed({m, n}).stride(0));
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

void cutlass_gemm_P426(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P426 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}

// pid=427 tag=legacy_splitk_p27
// family=fp16_legacy_splitk_universal runtime_split_k=True
struct CUTLASS_GEMM_P427 {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  static constexpr int NumStages = 2;
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
    return typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm, problem_size, split_k_slices,
        {ElementAccumulator(alpha), ElementAccumulator(beta)}, ptr_A, ptr_B,
        ptr_C, ptr_C, 0, 0, 0, 0, LayoutA::packed({m, k}).stride(0),
        LayoutB::packed({k, n}).stride(0), LayoutC::packed({m, n}).stride(0),
        LayoutC::packed({m, n}).stride(0));
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

void cutlass_gemm_P427(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P427 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}

// pid=428 tag=legacy_splitk_p28
// family=fp16_legacy_splitk_universal runtime_split_k=True
struct CUTLASS_GEMM_P428 {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<16, 32, 32>;
  using WarpShape = cutlass::gemm::GemmShape<16, 16, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  static constexpr int NumStages = 4;
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
    return typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm, problem_size, split_k_slices,
        {ElementAccumulator(alpha), ElementAccumulator(beta)}, ptr_A, ptr_B,
        ptr_C, ptr_C, 0, 0, 0, 0, LayoutA::packed({m, k}).stride(0),
        LayoutB::packed({k, n}).stride(0), LayoutC::packed({m, n}).stride(0),
        LayoutC::packed({m, n}).stride(0));
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

void cutlass_gemm_P428(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P428 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}

// pid=429 tag=legacy_splitk_p29
// family=fp16_legacy_splitk_universal runtime_split_k=True
struct CUTLASS_GEMM_P429 {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<8, 16, 64>;
  using WarpShape = cutlass::gemm::GemmShape<8, 16, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  static constexpr int NumStages = 2;
  static constexpr int AlignmentA = 1;
  static constexpr int AlignmentB = 1;
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
    return typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm, problem_size, split_k_slices,
        {ElementAccumulator(alpha), ElementAccumulator(beta)}, ptr_A, ptr_B,
        ptr_C, ptr_C, 0, 0, 0, 0, LayoutA::packed({m, k}).stride(0),
        LayoutB::packed({k, n}).stride(0), LayoutC::packed({m, n}).stride(0),
        LayoutC::packed({m, n}).stride(0));
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

void cutlass_gemm_P429(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P429 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}

// pid=430 tag=legacy_splitk_p30
// family=fp16_legacy_splitk_universal runtime_split_k=True
struct CUTLASS_GEMM_P430 {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<192, 192, 32>;
  using WarpShape = cutlass::gemm::GemmShape<96, 48, 32>;
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
    return typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm, problem_size, split_k_slices,
        {ElementAccumulator(alpha), ElementAccumulator(beta)}, ptr_A, ptr_B,
        ptr_C, ptr_C, 0, 0, 0, 0, LayoutA::packed({m, k}).stride(0),
        LayoutB::packed({k, n}).stride(0), LayoutC::packed({m, n}).stride(0),
        LayoutC::packed({m, n}).stride(0));
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

void cutlass_gemm_P430(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P430 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}

// pid=431 tag=legacy_splitk_p31
// family=fp16_legacy_splitk_universal runtime_split_k=True
struct CUTLASS_GEMM_P431 {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 192, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 48, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  static constexpr int NumStages = 3;
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
    return typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm, problem_size, split_k_slices,
        {ElementAccumulator(alpha), ElementAccumulator(beta)}, ptr_A, ptr_B,
        ptr_C, ptr_C, 0, 0, 0, 0, LayoutA::packed({m, k}).stride(0),
        LayoutB::packed({k, n}).stride(0), LayoutC::packed({m, n}).stride(0),
        LayoutC::packed({m, n}).stride(0));
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

void cutlass_gemm_P431(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P431 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}

// pid=432 tag=legacy_splitk_p32
// family=fp16_legacy_splitk_universal runtime_split_k=True
struct CUTLASS_GEMM_P432 {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  static constexpr int NumStages = 2;
  static constexpr int AlignmentA = 2;
  static constexpr int AlignmentB = 2;
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
    return typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm, problem_size, split_k_slices,
        {ElementAccumulator(alpha), ElementAccumulator(beta)}, ptr_A, ptr_B,
        ptr_C, ptr_C, 0, 0, 0, 0, LayoutA::packed({m, k}).stride(0),
        LayoutB::packed({k, n}).stride(0), LayoutC::packed({m, n}).stride(0),
        LayoutC::packed({m, n}).stride(0));
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

void cutlass_gemm_P432(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P432 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}

// pid=433 tag=legacy_splitk_p33
// family=fp16_legacy_splitk_universal runtime_split_k=True
struct CUTLASS_GEMM_P433 {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<96, 192, 64>;
  using WarpShape = cutlass::gemm::GemmShape<48, 48, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  static constexpr int NumStages = 2;
  static constexpr int AlignmentA = 1;
  static constexpr int AlignmentB = 1;
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
    return typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm, problem_size, split_k_slices,
        {ElementAccumulator(alpha), ElementAccumulator(beta)}, ptr_A, ptr_B,
        ptr_C, ptr_C, 0, 0, 0, 0, LayoutA::packed({m, k}).stride(0),
        LayoutB::packed({k, n}).stride(0), LayoutC::packed({m, n}).stride(0),
        LayoutC::packed({m, n}).stride(0));
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

void cutlass_gemm_P433(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P433 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}

// pid=434 tag=legacy_splitk_p34
// family=fp16_legacy_splitk_universal runtime_split_k=True
struct CUTLASS_GEMM_P434 {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 96, 32>;
  using WarpShape = cutlass::gemm::GemmShape<32, 48, 32>;
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
    return typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm, problem_size, split_k_slices,
        {ElementAccumulator(alpha), ElementAccumulator(beta)}, ptr_A, ptr_B,
        ptr_C, ptr_C, 0, 0, 0, 0, LayoutA::packed({m, k}).stride(0),
        LayoutB::packed({k, n}).stride(0), LayoutC::packed({m, n}).stride(0),
        LayoutC::packed({m, n}).stride(0));
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

void cutlass_gemm_P434(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P434 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}

// pid=435 tag=legacy_splitk_p35
// family=fp16_legacy_splitk_universal runtime_split_k=True
struct CUTLASS_GEMM_P435 {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<8, 8, 64>;
  using WarpShape = cutlass::gemm::GemmShape<8, 8, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  static constexpr int NumStages = 2;
  static constexpr int AlignmentA = 2;
  static constexpr int AlignmentB = 2;
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
    return typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm, problem_size, split_k_slices,
        {ElementAccumulator(alpha), ElementAccumulator(beta)}, ptr_A, ptr_B,
        ptr_C, ptr_C, 0, 0, 0, 0, LayoutA::packed({m, k}).stride(0),
        LayoutB::packed({k, n}).stride(0), LayoutC::packed({m, n}).stride(0),
        LayoutC::packed({m, n}).stride(0));
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

void cutlass_gemm_P435(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P435 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}

// pid=436 tag=legacy_splitk_p36
// family=fp16_legacy_splitk_universal runtime_split_k=True
struct CUTLASS_GEMM_P436 {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 32, 32>;
  using WarpShape = cutlass::gemm::GemmShape<8, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  static constexpr int NumStages = 2;
  static constexpr int AlignmentA = 2;
  static constexpr int AlignmentB = 2;
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
    return typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm, problem_size, split_k_slices,
        {ElementAccumulator(alpha), ElementAccumulator(beta)}, ptr_A, ptr_B,
        ptr_C, ptr_C, 0, 0, 0, 0, LayoutA::packed({m, k}).stride(0),
        LayoutB::packed({k, n}).stride(0), LayoutC::packed({m, n}).stride(0),
        LayoutC::packed({m, n}).stride(0));
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

void cutlass_gemm_P436(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P436 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}

// pid=437 tag=legacy_splitk_p37
// family=fp16_legacy_splitk_universal runtime_split_k=True
struct CUTLASS_GEMM_P437 {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 32, 64>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  static constexpr int NumStages = 4;
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
    return typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm, problem_size, split_k_slices,
        {ElementAccumulator(alpha), ElementAccumulator(beta)}, ptr_A, ptr_B,
        ptr_C, ptr_C, 0, 0, 0, 0, LayoutA::packed({m, k}).stride(0),
        LayoutB::packed({k, n}).stride(0), LayoutC::packed({m, n}).stride(0),
        LayoutC::packed({m, n}).stride(0));
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

void cutlass_gemm_P437(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P437 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}

// pid=438 tag=legacy_splitk_p38
// family=fp16_legacy_splitk_universal runtime_split_k=True
struct CUTLASS_GEMM_P438 {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<16, 32, 16>;
  using WarpShape = cutlass::gemm::GemmShape<16, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  static constexpr int NumStages = 2;
  static constexpr int AlignmentA = 1;
  static constexpr int AlignmentB = 1;
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
    return typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm, problem_size, split_k_slices,
        {ElementAccumulator(alpha), ElementAccumulator(beta)}, ptr_A, ptr_B,
        ptr_C, ptr_C, 0, 0, 0, 0, LayoutA::packed({m, k}).stride(0),
        LayoutB::packed({k, n}).stride(0), LayoutC::packed({m, n}).stride(0),
        LayoutC::packed({m, n}).stride(0));
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

void cutlass_gemm_P438(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P438 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}

// pid=439 tag=legacy_splitk_p39
// family=fp16_legacy_splitk_universal runtime_split_k=True
struct CUTLASS_GEMM_P439 {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 256, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  static constexpr int NumStages = 2;
  static constexpr int AlignmentA = 2;
  static constexpr int AlignmentB = 2;
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
    return typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm, problem_size, split_k_slices,
        {ElementAccumulator(alpha), ElementAccumulator(beta)}, ptr_A, ptr_B,
        ptr_C, ptr_C, 0, 0, 0, 0, LayoutA::packed({m, k}).stride(0),
        LayoutB::packed({k, n}).stride(0), LayoutC::packed({m, n}).stride(0),
        LayoutC::packed({m, n}).stride(0));
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

void cutlass_gemm_P439(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P439 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}

bool is_legacy_splitk_pid(int pid) {
  return pid >= 400 && pid <= 439;
}

int legacy_splitk_alignment(int pid) {
  switch (pid) {
    case 400: return 8;
    case 401: return 8;
    case 402: return 8;
    case 403: return 8;
    case 404: return 8;
    case 405: return 8;
    case 406: return 8;
    case 407: return 8;
    case 408: return 8;
    case 409: return 2;
    case 410: return 1;
    case 411: return 2;
    case 412: return 2;
    case 413: return 2;
    case 414: return 8;
    case 415: return 8;
    case 416: return 8;
    case 417: return 8;
    case 418: return 8;
    case 419: return 8;
    case 420: return 8;
    case 421: return 4;
    case 422: return 8;
    case 423: return 8;
    case 424: return 4;
    case 425: return 8;
    case 426: return 4;
    case 427: return 4;
    case 428: return 8;
    case 429: return 1;
    case 430: return 8;
    case 431: return 8;
    case 432: return 2;
    case 433: return 1;
    case 434: return 8;
    case 435: return 2;
    case 436: return 2;
    case 437: return 8;
    case 438: return 1;
    case 439: return 2;
    default: return 1;
  }
}

SharedMem legacy_splitk_shared_mem(int pid) {
  switch (pid) {
    case 400: return SharedMem(16, 96, 64);
    case 401: return SharedMem(128, 128, 32);
    case 402: return SharedMem(64, 128, 64);
    case 403: return SharedMem(64, 64, 64);
    case 404: return SharedMem(16, 32, 64);
    case 405: return SharedMem(32, 16, 32);
    case 406: return SharedMem(16, 16, 64);
    case 407: return SharedMem(64, 8, 64);
    case 408: return SharedMem(64, 8, 64);
    case 409: return SharedMem(64, 16, 64);
    case 410: return SharedMem(32, 64, 64);
    case 411: return SharedMem(64, 96, 64);
    case 412: return SharedMem(128, 64, 64);
    case 413: return SharedMem(16, 8, 32);
    case 414: return SharedMem(96, 160, 32);
    case 415: return SharedMem(80, 128, 64);
    case 416: return SharedMem(48, 32, 64);
    case 417: return SharedMem(256, 32, 64);
    case 418: return SharedMem(32, 128, 32);
    case 419: return SharedMem(16, 128, 64);
    case 420: return SharedMem(80, 96, 64);
    case 421: return SharedMem(16, 64, 16);
    case 422: return SharedMem(128, 96, 32);
    case 423: return SharedMem(256, 128, 32);
    case 424: return SharedMem(32, 16, 64);
    case 425: return SharedMem(16, 8, 64);
    case 426: return SharedMem(16, 64, 64);
    case 427: return SharedMem(64, 64, 16);
    case 428: return SharedMem(16, 32, 32);
    case 429: return SharedMem(8, 16, 64);
    case 430: return SharedMem(192, 192, 32);
    case 431: return SharedMem(128, 192, 64);
    case 432: return SharedMem(32, 64, 16);
    case 433: return SharedMem(96, 192, 64);
    case 434: return SharedMem(64, 96, 32);
    case 435: return SharedMem(8, 8, 64);
    case 436: return SharedMem(32, 32, 32);
    case 437: return SharedMem(64, 32, 64);
    case 438: return SharedMem(16, 32, 16);
    case 439: return SharedMem(128, 256, 64);
    default: return SharedMem();
  }
}

void run_legacy_splitk_gemm(
    int pid, int m, int n, int k, cutlass::half_t *A, cutlass::half_t *B,
    cutlass::half_t *C, float alpha, float beta, int split_k_slices,
    cudaStream_t stream) {
  switch (pid) {
    case 400: cutlass_gemm_P400(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 401: cutlass_gemm_P401(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 402: cutlass_gemm_P402(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 403: cutlass_gemm_P403(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 404: cutlass_gemm_P404(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 405: cutlass_gemm_P405(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 406: cutlass_gemm_P406(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 407: cutlass_gemm_P407(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 408: cutlass_gemm_P408(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 409: cutlass_gemm_P409(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 410: cutlass_gemm_P410(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 411: cutlass_gemm_P411(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 412: cutlass_gemm_P412(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 413: cutlass_gemm_P413(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 414: cutlass_gemm_P414(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 415: cutlass_gemm_P415(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 416: cutlass_gemm_P416(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 417: cutlass_gemm_P417(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 418: cutlass_gemm_P418(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 419: cutlass_gemm_P419(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 420: cutlass_gemm_P420(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 421: cutlass_gemm_P421(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 422: cutlass_gemm_P422(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 423: cutlass_gemm_P423(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 424: cutlass_gemm_P424(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 425: cutlass_gemm_P425(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 426: cutlass_gemm_P426(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 427: cutlass_gemm_P427(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 428: cutlass_gemm_P428(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 429: cutlass_gemm_P429(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 430: cutlass_gemm_P430(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 431: cutlass_gemm_P431(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 432: cutlass_gemm_P432(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 433: cutlass_gemm_P433(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 434: cutlass_gemm_P434(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 435: cutlass_gemm_P435(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 436: cutlass_gemm_P436(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 437: cutlass_gemm_P437(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 438: cutlass_gemm_P438(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 439: cutlass_gemm_P439(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    default: return;
  }
}
