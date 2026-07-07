// Auto-generated Qwen long-K GemmUniversal candidates.

// pid=226 tag=qku3_tb64_n8_k64_s10_sw4
// family=gemm_universal_longk split_k_serial=True
// notes=Tiny-N=8 baseline with deeper pipeline than batch2.
struct CUTLASS_GEMM_P226 {
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

void cutlass_gemm_P226(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P226 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}


// pid=227 tag=qku3_tb64_n8_k64_s12_sw4
// family=gemm_universal_longk split_k_serial=True
// notes=Tiny-N=8 with even deeper pipeline.
struct CUTLASS_GEMM_P227 {
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
  static constexpr int NumStages = 12;
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

void cutlass_gemm_P227(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P227 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}


// pid=228 tag=qku3_tb64_n8_k64_s10_sw2
// family=gemm_universal_longk split_k_serial=True
// notes=Tiny-N=8 with reduced swizzle.
struct CUTLASS_GEMM_P228 {
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

void cutlass_gemm_P228(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P228 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}


// pid=229 tag=qku3_tb64_n8_k64_s10_sw8
// family=gemm_universal_longk split_k_serial=True
// notes=Tiny-N=8 with higher swizzle.
struct CUTLASS_GEMM_P229 {
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

void cutlass_gemm_P229(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P229 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}


// pid=230 tag=qku3_tb128_n8_k64_s10_sw4
// family=gemm_universal_longk split_k_serial=True
// notes=Tall-M tiny-N=8 with deeper pipeline.
struct CUTLASS_GEMM_P230 {
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

void cutlass_gemm_P230(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P230 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}


// pid=231 tag=qku3_tb32_n8_k64_s10_sw4
// family=gemm_universal_longk split_k_serial=True
// notes=Shorter M-tile for tiny batch cases.
struct CUTLASS_GEMM_P231 {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 8, 64>;
  using WarpShape = cutlass::gemm::GemmShape<32, 8, 64>;
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

void cutlass_gemm_P231(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P231 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}


// pid=232 tag=qku3_tb64_n8_k64_s10_a4
// family=gemm_universal_longk split_k_serial=True
// notes=Tiny-N=8 alignment-4 fallback.
struct CUTLASS_GEMM_P232 {
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
  static constexpr int NumStages = 10;
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

void cutlass_gemm_P232(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P232 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}


// pid=233 tag=qku3_tb64_n8_k64_parallel
// family=gemm_universal_longk split_k_serial=False
// notes=Tiny-N=8 runtime split-K variant.
struct CUTLASS_GEMM_P233 {
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

void cutlass_gemm_P233(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B,
                         cutlass::half_t *C, float alpha, float beta,
                         int split_k_slices, cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  CUTLASS_GEMM_P233 instance;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices, A, B, C,
                    alpha, beta, stream);
}

