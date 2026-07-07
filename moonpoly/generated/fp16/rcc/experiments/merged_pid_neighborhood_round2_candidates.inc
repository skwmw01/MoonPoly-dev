// Auto-generated round-2 merged pid-neighborhood candidates.

// pid=101 tag=r2_pid94_n40 base_pid=94
// notes=Tighter N around pid94 for Llama gate_up and Qwen batch-1 family.
struct CUTLASS_GEMM_P101 {
private:
  CUTLASS_GEMM_P101() {}
  ~CUTLASS_GEMM_P101() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 40, 64>,
      cutlass::gemm::GemmShape<16, 40, 32>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>,
      6, 8, 8, true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P101(const CUTLASS_GEMM_P101 &) = delete;
  CUTLASS_GEMM_P101 &operator=(const CUTLASS_GEMM_P101 &) = delete;
  static CUTLASS_GEMM_P101 &get_instance() {
    static CUTLASS_GEMM_P101 instance;
    return instance;
  }

  void run_gemm(int length_m, int length_n, int length_k, cutlass::half_t *A,
                cutlass::half_t *B, cutlass::half_t *C, float alpha,
                float beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename Gemm::Arguments arguments{problem_size,
                                       {A, length_k},
                                       {B, length_k},
                                       {C, length_m},
                                       {C, length_m},
                                       {alpha, beta},
                                       1};
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);
    status = gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_gemm_P101(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P101 &instance = CUTLASS_GEMM_P101::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=102 tag=r2_pid94_stage8 base_pid=94
// notes=Deeper pipeline around pid94.
struct CUTLASS_GEMM_P102 {
private:
  CUTLASS_GEMM_P102() {}
  ~CUTLASS_GEMM_P102() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 32, 64>,
      cutlass::gemm::GemmShape<16, 32, 32>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>,
      8, 8, 8, true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P102(const CUTLASS_GEMM_P102 &) = delete;
  CUTLASS_GEMM_P102 &operator=(const CUTLASS_GEMM_P102 &) = delete;
  static CUTLASS_GEMM_P102 &get_instance() {
    static CUTLASS_GEMM_P102 instance;
    return instance;
  }

  void run_gemm(int length_m, int length_n, int length_k, cutlass::half_t *A,
                cutlass::half_t *B, cutlass::half_t *C, float alpha,
                float beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename Gemm::Arguments arguments{problem_size,
                                       {A, length_k},
                                       {B, length_k},
                                       {C, length_m},
                                       {C, length_m},
                                       {alpha, beta},
                                       1};
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);
    status = gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_gemm_P102(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P102 &instance = CUTLASS_GEMM_P102::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=103 tag=r2_pid95_tb48 base_pid=95
// notes=More N coverage while preserving full-K warp reuse.
struct CUTLASS_GEMM_P103 {
private:
  CUTLASS_GEMM_P103() {}
  ~CUTLASS_GEMM_P103() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 48, 64>,
      cutlass::gemm::GemmShape<16, 48, 64>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>,
      6, 8, 8, true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P103(const CUTLASS_GEMM_P103 &) = delete;
  CUTLASS_GEMM_P103 &operator=(const CUTLASS_GEMM_P103 &) = delete;
  static CUTLASS_GEMM_P103 &get_instance() {
    static CUTLASS_GEMM_P103 instance;
    return instance;
  }

  void run_gemm(int length_m, int length_n, int length_k, cutlass::half_t *A,
                cutlass::half_t *B, cutlass::half_t *C, float alpha,
                float beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename Gemm::Arguments arguments{problem_size,
                                       {A, length_k},
                                       {B, length_k},
                                       {C, length_m},
                                       {C, length_m},
                                       {alpha, beta},
                                       1};
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);
    status = gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_gemm_P103(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P103 &instance = CUTLASS_GEMM_P103::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=104 tag=r2_pid96_tb64 base_pid=96
// notes=Expand pid96 for Qwen 51200x5120 family.
struct CUTLASS_GEMM_P104 {
private:
  CUTLASS_GEMM_P104() {}
  ~CUTLASS_GEMM_P104() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 64, 64>,
      cutlass::gemm::GemmShape<16, 32, 64>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>,
      6, 8, 8, true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P104(const CUTLASS_GEMM_P104 &) = delete;
  CUTLASS_GEMM_P104 &operator=(const CUTLASS_GEMM_P104 &) = delete;
  static CUTLASS_GEMM_P104 &get_instance() {
    static CUTLASS_GEMM_P104 instance;
    return instance;
  }

  void run_gemm(int length_m, int length_n, int length_k, cutlass::half_t *A,
                cutlass::half_t *B, cutlass::half_t *C, float alpha,
                float beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename Gemm::Arguments arguments{problem_size,
                                       {A, length_k},
                                       {B, length_k},
                                       {C, length_m},
                                       {C, length_m},
                                       {alpha, beta},
                                       1};
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);
    status = gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_gemm_P104(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P104 &instance = CUTLASS_GEMM_P104::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=105 tag=r2_pid96_tb80 base_pid=96
// notes=Probe wider N tile for large-output hot families.
struct CUTLASS_GEMM_P105 {
private:
  CUTLASS_GEMM_P105() {}
  ~CUTLASS_GEMM_P105() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 80, 64>,
      cutlass::gemm::GemmShape<16, 40, 32>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>,
      6, 8, 8, true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P105(const CUTLASS_GEMM_P105 &) = delete;
  CUTLASS_GEMM_P105 &operator=(const CUTLASS_GEMM_P105 &) = delete;
  static CUTLASS_GEMM_P105 &get_instance() {
    static CUTLASS_GEMM_P105 instance;
    return instance;
  }

  void run_gemm(int length_m, int length_n, int length_k, cutlass::half_t *A,
                cutlass::half_t *B, cutlass::half_t *C, float alpha,
                float beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename Gemm::Arguments arguments{problem_size,
                                       {A, length_k},
                                       {B, length_k},
                                       {C, length_m},
                                       {C, length_m},
                                       {alpha, beta},
                                       1};
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);
    status = gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_gemm_P105(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P105 &instance = CUTLASS_GEMM_P105::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=106 tag=r2_pid96_k32 base_pid=96
// notes=Reduce K tile latency for 5120-reduction families.
struct CUTLASS_GEMM_P106 {
private:
  CUTLASS_GEMM_P106() {}
  ~CUTLASS_GEMM_P106() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 48, 32>,
      cutlass::gemm::GemmShape<16, 48, 32>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>,
      8, 8, 8, true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P106(const CUTLASS_GEMM_P106 &) = delete;
  CUTLASS_GEMM_P106 &operator=(const CUTLASS_GEMM_P106 &) = delete;
  static CUTLASS_GEMM_P106 &get_instance() {
    static CUTLASS_GEMM_P106 instance;
    return instance;
  }

  void run_gemm(int length_m, int length_n, int length_k, cutlass::half_t *A,
                cutlass::half_t *B, cutlass::half_t *C, float alpha,
                float beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename Gemm::Arguments arguments{problem_size,
                                       {A, length_k},
                                       {B, length_k},
                                       {C, length_m},
                                       {C, length_m},
                                       {alpha, beta},
                                       1};
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);
    status = gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_gemm_P106(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P106 &instance = CUTLASS_GEMM_P106::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=107 tag=r2_pid91_split base_pid=91
// notes=Turn pid91 into split-K candidate for Qwen medium-batch family.
struct CUTLASS_GEMM_P107 {
private:
  CUTLASS_GEMM_P107() {}
  ~CUTLASS_GEMM_P107() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 64, 64>,
      cutlass::gemm::GemmShape<16, 32, 64>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>,
      5, 8, 8, true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P107(const CUTLASS_GEMM_P107 &) = delete;
  CUTLASS_GEMM_P107 &operator=(const CUTLASS_GEMM_P107 &) = delete;
  static CUTLASS_GEMM_P107 &get_instance() {
    static CUTLASS_GEMM_P107 instance;
    return instance;
  }

  void run_gemm(int length_m, int length_n, int length_k, cutlass::half_t *A,
                cutlass::half_t *B, cutlass::half_t *C, float alpha,
                float beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename Gemm::Arguments arguments{problem_size,
                                       {A, length_k},
                                       {B, length_k},
                                       {C, length_m},
                                       {C, length_m},
                                       {alpha, beta},
                                       1};
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);
    status = gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_gemm_P107(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P107 &instance = CUTLASS_GEMM_P107::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=108 tag=r2_pid92_split base_pid=92
// notes=Split-K version of current Qwen large-batch winner.
struct CUTLASS_GEMM_P108 {
private:
  CUTLASS_GEMM_P108() {}
  ~CUTLASS_GEMM_P108() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 128, 64>,
      cutlass::gemm::GemmShape<16, 64, 64>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>,
      5, 8, 8, true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P108(const CUTLASS_GEMM_P108 &) = delete;
  CUTLASS_GEMM_P108 &operator=(const CUTLASS_GEMM_P108 &) = delete;
  static CUTLASS_GEMM_P108 &get_instance() {
    static CUTLASS_GEMM_P108 instance;
    return instance;
  }

  void run_gemm(int length_m, int length_n, int length_k, cutlass::half_t *A,
                cutlass::half_t *B, cutlass::half_t *C, float alpha,
                float beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename Gemm::Arguments arguments{problem_size,
                                       {A, length_k},
                                       {B, length_k},
                                       {C, length_m},
                                       {C, length_m},
                                       {alpha, beta},
                                       1};
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);
    status = gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_gemm_P108(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P108 &instance = CUTLASS_GEMM_P108::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=109 tag=r2_pid92_tb96 base_pid=92
// notes=Intermediate N between pid91 and pid92 for Qwen batch sweep.
struct CUTLASS_GEMM_P109 {
private:
  CUTLASS_GEMM_P109() {}
  ~CUTLASS_GEMM_P109() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 96, 64>,
      cutlass::gemm::GemmShape<16, 48, 64>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>,
      5, 8, 8, false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P109(const CUTLASS_GEMM_P109 &) = delete;
  CUTLASS_GEMM_P109 &operator=(const CUTLASS_GEMM_P109 &) = delete;
  static CUTLASS_GEMM_P109 &get_instance() {
    static CUTLASS_GEMM_P109 instance;
    return instance;
  }

  void run_gemm(int length_m, int length_n, int length_k, cutlass::half_t *A,
                cutlass::half_t *B, cutlass::half_t *C, float alpha,
                float beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename Gemm::Arguments arguments{problem_size,
                                       {A, length_k},
                                       {B, length_k},
                                       {C, length_m},
                                       {C, length_m},
                                       {alpha, beta},
                                       1};
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);
    status = gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_gemm_P109(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P109 &instance = CUTLASS_GEMM_P109::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=110 tag=r2_pid25_stage6 base_pid=25
// notes=Small-N family refinement for 5120x5120.
struct CUTLASS_GEMM_P110 {
private:
  CUTLASS_GEMM_P110() {}
  ~CUTLASS_GEMM_P110() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 8, 64>,
      cutlass::gemm::GemmShape<16, 8, 64>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      6, 8, 8, true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P110(const CUTLASS_GEMM_P110 &) = delete;
  CUTLASS_GEMM_P110 &operator=(const CUTLASS_GEMM_P110 &) = delete;
  static CUTLASS_GEMM_P110 &get_instance() {
    static CUTLASS_GEMM_P110 instance;
    return instance;
  }

  void run_gemm(int length_m, int length_n, int length_k, cutlass::half_t *A,
                cutlass::half_t *B, cutlass::half_t *C, float alpha,
                float beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename Gemm::Arguments arguments{problem_size,
                                       {A, length_k},
                                       {B, length_k},
                                       {C, length_m},
                                       {C, length_m},
                                       {alpha, beta},
                                       1};
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);
    status = gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_gemm_P110(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P110 &instance = CUTLASS_GEMM_P110::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=111 tag=r2_pid25_sw4 base_pid=25
// notes=Less aggressive swizzle for 5120x5120/13824 tails.
struct CUTLASS_GEMM_P111 {
private:
  CUTLASS_GEMM_P111() {}
  ~CUTLASS_GEMM_P111() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 8, 64>,
      cutlass::gemm::GemmShape<16, 8, 64>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
      5, 8, 8, true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P111(const CUTLASS_GEMM_P111 &) = delete;
  CUTLASS_GEMM_P111 &operator=(const CUTLASS_GEMM_P111 &) = delete;
  static CUTLASS_GEMM_P111 &get_instance() {
    static CUTLASS_GEMM_P111 instance;
    return instance;
  }

  void run_gemm(int length_m, int length_n, int length_k, cutlass::half_t *A,
                cutlass::half_t *B, cutlass::half_t *C, float alpha,
                float beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename Gemm::Arguments arguments{problem_size,
                                       {A, length_k},
                                       {B, length_k},
                                       {C, length_m},
                                       {C, length_m},
                                       {alpha, beta},
                                       1};
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);
    status = gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_gemm_P111(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P111 &instance = CUTLASS_GEMM_P111::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=112 tag=r2_pid6_sw8 base_pid=6
// notes=More aggressive swizzle for 5120x13824.
struct CUTLASS_GEMM_P112 {
private:
  CUTLASS_GEMM_P112() {}
  ~CUTLASS_GEMM_P112() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 16, 64>,
      cutlass::gemm::GemmShape<16, 16, 64>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      10, 8, 8, true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P112(const CUTLASS_GEMM_P112 &) = delete;
  CUTLASS_GEMM_P112 &operator=(const CUTLASS_GEMM_P112 &) = delete;
  static CUTLASS_GEMM_P112 &get_instance() {
    static CUTLASS_GEMM_P112 instance;
    return instance;
  }

  void run_gemm(int length_m, int length_n, int length_k, cutlass::half_t *A,
                cutlass::half_t *B, cutlass::half_t *C, float alpha,
                float beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename Gemm::Arguments arguments{problem_size,
                                       {A, length_k},
                                       {B, length_k},
                                       {C, length_m},
                                       {C, length_m},
                                       {alpha, beta},
                                       1};
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);
    status = gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_gemm_P112(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P112 &instance = CUTLASS_GEMM_P112::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=113 tag=r2_pid6_stage8_sw2 base_pid=6
// notes=Combine lower stage depth and narrower swizzle for medium-K family.
struct CUTLASS_GEMM_P113 {
private:
  CUTLASS_GEMM_P113() {}
  ~CUTLASS_GEMM_P113() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 16, 64>,
      cutlass::gemm::GemmShape<16, 16, 64>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>,
      8, 8, 8, true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P113(const CUTLASS_GEMM_P113 &) = delete;
  CUTLASS_GEMM_P113 &operator=(const CUTLASS_GEMM_P113 &) = delete;
  static CUTLASS_GEMM_P113 &get_instance() {
    static CUTLASS_GEMM_P113 instance;
    return instance;
  }

  void run_gemm(int length_m, int length_n, int length_k, cutlass::half_t *A,
                cutlass::half_t *B, cutlass::half_t *C, float alpha,
                float beta, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    typename Gemm::Arguments arguments{problem_size,
                                       {A, length_k},
                                       {B, length_k},
                                       {C, length_m},
                                       {C, length_m},
                                       {alpha, beta},
                                       1};
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);
    status = gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_gemm_P113(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P113 &instance = CUTLASS_GEMM_P113::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

