// Auto-generated Qwen long-K GemmUniversal candidates.

// pid=216 tag=qku_tb64_n8_k64_s6_sw4
// family=gemm_universal_longk split_k_serial=True
// notes=Long-K family with tiny N tile and taller M tile.
struct CUTLASS_GEMM_P216 {
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

void cutlass_gemm_P216(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P216 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}


// pid=217 tag=qku_tb64_n8_k64_s8_sw4
// family=gemm_universal_longk split_k_serial=True
// notes=Deeper pipeline variant.
struct CUTLASS_GEMM_P217 {
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
  static constexpr int NumStages = 8;
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

void cutlass_gemm_P217(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P217 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}


// pid=218 tag=qku_tb64_n16_k64_s6_sw4
// family=gemm_universal_longk split_k_serial=True
// notes=N=16 universal long-K candidate.
struct CUTLASS_GEMM_P218 {
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
  using WarpShape = cutlass::gemm::GemmShape<32, 16, 64>;
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

void cutlass_gemm_P218(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P218 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}


// pid=219 tag=qku_tb64_n16_k64_s8_sw4
// family=gemm_universal_longk split_k_serial=True
// notes=N=16, deeper pipeline.
struct CUTLASS_GEMM_P219 {
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
  using WarpShape = cutlass::gemm::GemmShape<32, 16, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  static constexpr int NumStages = 8;
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

void cutlass_gemm_P219(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P219 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}


// pid=220 tag=qku_tb128_n8_k64_s6_sw4
// family=gemm_universal_longk split_k_serial=True
// notes=Very tall M tile for output reuse.
struct CUTLASS_GEMM_P220 {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 8, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 8, 64>;
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

void cutlass_gemm_P220(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P220 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}


// pid=221 tag=qku_tb128_n8_k64_s8_sw4
// family=gemm_universal_longk split_k_serial=True
// notes=Very tall M tile with deeper pipeline.
struct CUTLASS_GEMM_P221 {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 8, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 8, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  static constexpr int NumStages = 8;
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

void cutlass_gemm_P221(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P221 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}


// pid=222 tag=qku_tb64_n8_k64_s6_sw2
// family=gemm_universal_longk split_k_serial=True
// notes=Lower swizzle variant.
struct CUTLASS_GEMM_P222 {
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

void cutlass_gemm_P222(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P222 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}


// pid=223 tag=qku_tb64_n8_k64_s6_sw1
// family=gemm_universal_longk split_k_serial=True
// notes=Swizzle-1 for tiny N.
struct CUTLASS_GEMM_P223 {
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

void cutlass_gemm_P223(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P223 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}


// pid=224 tag=qku_tb64_n8_k64_s6_a4
// family=gemm_universal_longk split_k_serial=True
// notes=Alignment-4 fallback.
struct CUTLASS_GEMM_P224 {
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

void cutlass_gemm_P224(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P224 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}


// pid=225 tag=qku_tb64_n8_k64_parallel
// family=gemm_universal_longk split_k_serial=False
// notes=Non-serial runtime split-k variant.
struct CUTLASS_GEMM_P225 {
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

void cutlass_gemm_P225(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P225 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}

