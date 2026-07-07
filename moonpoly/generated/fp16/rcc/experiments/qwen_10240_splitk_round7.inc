// Auto-generated focused split-k candidates for Qwen shape (8,10240,5120).

// pid=284 tag=q10240_pid96_sw4
// notes=pid96 with swizzle-4.
struct CUTLASS_GEMM_P284 {
private:
  CUTLASS_GEMM_P284() {}
  ~CUTLASS_GEMM_P284() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 48, 64>,
      cutlass::gemm::GemmShape<16, 48, 32>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
      6, 8, 8, true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P284(const CUTLASS_GEMM_P284 &) = delete;
  CUTLASS_GEMM_P284 &operator=(const CUTLASS_GEMM_P284 &) = delete;
  static CUTLASS_GEMM_P284 &get_instance() {
    static CUTLASS_GEMM_P284 instance;
    return instance;
  }
  void run_gemm(int length_m, int length_n, int length_k, cutlass::half_t *A,
                cutlass::half_t *B, cutlass::half_t *C, float alpha,
                float beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename Gemm::Arguments arguments{problem_size, {A, length_k}, {B, length_k},
                                       {C, length_m}, {C, length_m}, {alpha, beta}, 1};
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);
    status = gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};
void cutlass_gemm_P284(int length_m, int length_n, int length_k,
                       cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                       float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P284 &instance = CUTLASS_GEMM_P284::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

// pid=285 tag=q10240_pid96_stage5
// notes=pid96 with stage-5.
struct CUTLASS_GEMM_P285 {
private:
  CUTLASS_GEMM_P285() {}
  ~CUTLASS_GEMM_P285() {}
public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 48, 64>,
      cutlass::gemm::GemmShape<16, 48, 32>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>,
      5, 8, 8, true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P285(const CUTLASS_GEMM_P285 &) = delete;
  CUTLASS_GEMM_P285 &operator=(const CUTLASS_GEMM_P285 &) = delete;
  static CUTLASS_GEMM_P285 &get_instance() { static CUTLASS_GEMM_P285 instance; return instance; }
  void run_gemm(int length_m, int length_n, int length_k, cutlass::half_t *A,
                cutlass::half_t *B, cutlass::half_t *C, float alpha,
                float beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename Gemm::Arguments arguments{problem_size, {A, length_k}, {B, length_k},
                                       {C, length_m}, {C, length_m}, {alpha, beta}, 1};
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);
    status = gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};
void cutlass_gemm_P285(int length_m, int length_n, int length_k,
                       cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                       float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P285 &instance = CUTLASS_GEMM_P285::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

// pid=286 tag=q10240_pid96_stage7
// notes=pid96 with stage-7.
struct CUTLASS_GEMM_P286 {
private:
  CUTLASS_GEMM_P286() {}
  ~CUTLASS_GEMM_P286() {}
public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 48, 64>,
      cutlass::gemm::GemmShape<16, 48, 32>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>,
      7, 8, 8, true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P286(const CUTLASS_GEMM_P286 &) = delete;
  CUTLASS_GEMM_P286 &operator=(const CUTLASS_GEMM_P286 &) = delete;
  static CUTLASS_GEMM_P286 &get_instance() { static CUTLASS_GEMM_P286 instance; return instance; }
  void run_gemm(int length_m, int length_n, int length_k, cutlass::half_t *A,
                cutlass::half_t *B, cutlass::half_t *C, float alpha,
                float beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename Gemm::Arguments arguments{problem_size, {A, length_k}, {B, length_k},
                                       {C, length_m}, {C, length_m}, {alpha, beta}, 1};
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);
    status = gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};
void cutlass_gemm_P286(int length_m, int length_n, int length_k,
                       cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                       float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P286 &instance = CUTLASS_GEMM_P286::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

// pid=287 tag=q10240_pid166_stage5
// notes=pid166 with stage-5.
struct CUTLASS_GEMM_P287 {
private:
  CUTLASS_GEMM_P287() {}
  ~CUTLASS_GEMM_P287() {}
public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 32, 64>,
      cutlass::gemm::GemmShape<16, 32, 64>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
      5, 8, 8, true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P287(const CUTLASS_GEMM_P287 &) = delete;
  CUTLASS_GEMM_P287 &operator=(const CUTLASS_GEMM_P287 &) = delete;
  static CUTLASS_GEMM_P287 &get_instance() { static CUTLASS_GEMM_P287 instance; return instance; }
  void run_gemm(int length_m, int length_n, int length_k, cutlass::half_t *A,
                cutlass::half_t *B, cutlass::half_t *C, float alpha,
                float beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename Gemm::Arguments arguments{problem_size, {A, length_k}, {B, length_k},
                                       {C, length_m}, {C, length_m}, {alpha, beta}, 1};
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);
    status = gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};
void cutlass_gemm_P287(int length_m, int length_n, int length_k,
                       cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                       float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P287 &instance = CUTLASS_GEMM_P287::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

// pid=288 tag=q10240_pid166_stage7
// notes=pid166 with stage-7.
struct CUTLASS_GEMM_P288 {
private:
  CUTLASS_GEMM_P288() {}
  ~CUTLASS_GEMM_P288() {}
public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 32, 64>,
      cutlass::gemm::GemmShape<16, 32, 64>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
      7, 8, 8, true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P288(const CUTLASS_GEMM_P288 &) = delete;
  CUTLASS_GEMM_P288 &operator=(const CUTLASS_GEMM_P288 &) = delete;
  static CUTLASS_GEMM_P288 &get_instance() { static CUTLASS_GEMM_P288 instance; return instance; }
  void run_gemm(int length_m, int length_n, int length_k, cutlass::half_t *A,
                cutlass::half_t *B, cutlass::half_t *C, float alpha,
                float beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename Gemm::Arguments arguments{problem_size, {A, length_k}, {B, length_k},
                                       {C, length_m}, {C, length_m}, {alpha, beta}, 1};
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);
    status = gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};
void cutlass_gemm_P288(int length_m, int length_n, int length_k,
                       cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                       float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P288 &instance = CUTLASS_GEMM_P288::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

// pid=289 tag=q10240_pid166_sw8
// notes=pid166 with swizzle-8.
struct CUTLASS_GEMM_P289 {
private:
  CUTLASS_GEMM_P289() {}
  ~CUTLASS_GEMM_P289() {}
public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 32, 64>,
      cutlass::gemm::GemmShape<16, 32, 64>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      6, 8, 8, true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P289(const CUTLASS_GEMM_P289 &) = delete;
  CUTLASS_GEMM_P289 &operator=(const CUTLASS_GEMM_P289 &) = delete;
  static CUTLASS_GEMM_P289 &get_instance() { static CUTLASS_GEMM_P289 instance; return instance; }
  void run_gemm(int length_m, int length_n, int length_k, cutlass::half_t *A,
                cutlass::half_t *B, cutlass::half_t *C, float alpha,
                float beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename Gemm::Arguments arguments{problem_size, {A, length_k}, {B, length_k},
                                       {C, length_m}, {C, length_m}, {alpha, beta}, 1};
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);
    status = gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};
void cutlass_gemm_P289(int length_m, int length_n, int length_k,
                       cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                       float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P289 &instance = CUTLASS_GEMM_P289::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

// pid=290 tag=q10240_pid166_a4
// notes=pid166 with alignment-4.
struct CUTLASS_GEMM_P290 {
private:
  CUTLASS_GEMM_P290() {}
  ~CUTLASS_GEMM_P290() {}
public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 32, 64>,
      cutlass::gemm::GemmShape<16, 32, 64>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
      6, 4, 4, true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P290(const CUTLASS_GEMM_P290 &) = delete;
  CUTLASS_GEMM_P290 &operator=(const CUTLASS_GEMM_P290 &) = delete;
  static CUTLASS_GEMM_P290 &get_instance() { static CUTLASS_GEMM_P290 instance; return instance; }
  void run_gemm(int length_m, int length_n, int length_k, cutlass::half_t *A,
                cutlass::half_t *B, cutlass::half_t *C, float alpha,
                float beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename Gemm::Arguments arguments{problem_size, {A, length_k}, {B, length_k},
                                       {C, length_m}, {C, length_m}, {alpha, beta}, 1};
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);
    status = gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};
void cutlass_gemm_P290(int length_m, int length_n, int length_k,
                       cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                       float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P290 &instance = CUTLASS_GEMM_P290::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

// pid=291 tag=q10240_pid96_stage8
// notes=pid96 with stage-8.
struct CUTLASS_GEMM_P291 {
private:
  CUTLASS_GEMM_P291() {}
  ~CUTLASS_GEMM_P291() {}
public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 48, 64>,
      cutlass::gemm::GemmShape<16, 48, 32>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>,
      8, 8, 8, true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P291(const CUTLASS_GEMM_P291 &) = delete;
  CUTLASS_GEMM_P291 &operator=(const CUTLASS_GEMM_P291 &) = delete;
  static CUTLASS_GEMM_P291 &get_instance() { static CUTLASS_GEMM_P291 instance; return instance; }
  void run_gemm(int length_m, int length_n, int length_k, cutlass::half_t *A,
                cutlass::half_t *B, cutlass::half_t *C, float alpha,
                float beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename Gemm::Arguments arguments{problem_size, {A, length_k}, {B, length_k},
                                       {C, length_m}, {C, length_m}, {alpha, beta}, 1};
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);
    status = gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};
void cutlass_gemm_P291(int length_m, int length_n, int length_k,
                       cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                       float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P291 &instance = CUTLASS_GEMM_P291::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

// pid=292 tag=q10240_pid96_stage7_sw4
// notes=pid96 with stage-7 and swizzle-4.
struct CUTLASS_GEMM_P292 {
private:
  CUTLASS_GEMM_P292() {}
  ~CUTLASS_GEMM_P292() {}
public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 48, 64>,
      cutlass::gemm::GemmShape<16, 48, 32>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
      7, 8, 8, true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P292(const CUTLASS_GEMM_P292 &) = delete;
  CUTLASS_GEMM_P292 &operator=(const CUTLASS_GEMM_P292 &) = delete;
  static CUTLASS_GEMM_P292 &get_instance() { static CUTLASS_GEMM_P292 instance; return instance; }
  void run_gemm(int length_m, int length_n, int length_k, cutlass::half_t *A,
                cutlass::half_t *B, cutlass::half_t *C, float alpha,
                float beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename Gemm::Arguments arguments{problem_size, {A, length_k}, {B, length_k},
                                       {C, length_m}, {C, length_m}, {alpha, beta}, 1};
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);
    status = gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};
void cutlass_gemm_P292(int length_m, int length_n, int length_k,
                       cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                       float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P292 &instance = CUTLASS_GEMM_P292::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

// pid=293 tag=q10240_pid96_stage8_sw4
// notes=pid96 with stage-8 and swizzle-4.
struct CUTLASS_GEMM_P293 {
private:
  CUTLASS_GEMM_P293() {}
  ~CUTLASS_GEMM_P293() {}
public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 48, 64>,
      cutlass::gemm::GemmShape<16, 48, 32>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
      8, 8, 8, true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P293(const CUTLASS_GEMM_P293 &) = delete;
  CUTLASS_GEMM_P293 &operator=(const CUTLASS_GEMM_P293 &) = delete;
  static CUTLASS_GEMM_P293 &get_instance() { static CUTLASS_GEMM_P293 instance; return instance; }
  void run_gemm(int length_m, int length_n, int length_k, cutlass::half_t *A,
                cutlass::half_t *B, cutlass::half_t *C, float alpha,
                float beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename Gemm::Arguments arguments{problem_size, {A, length_k}, {B, length_k},
                                       {C, length_m}, {C, length_m}, {alpha, beta}, 1};
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);
    status = gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};
void cutlass_gemm_P293(int length_m, int length_n, int length_k,
                       cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                       float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P293 &instance = CUTLASS_GEMM_P293::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}
