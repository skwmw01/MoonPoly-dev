// Auto-generated round-2 merged pid-neighborhood candidates.

// pid=314 tag=r9_pid195_stage3 base_pid=195
// notes=More aggressive shallow pipeline.
struct CUTLASS_GEMM_P314 {
private:
  CUTLASS_GEMM_P314() {}
  ~CUTLASS_GEMM_P314() {}

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
      3, 8, 8, true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P314(const CUTLASS_GEMM_P314 &) = delete;
  CUTLASS_GEMM_P314 &operator=(const CUTLASS_GEMM_P314 &) = delete;
  static CUTLASS_GEMM_P314 &get_instance() {
    static CUTLASS_GEMM_P314 instance;
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

void cutlass_gemm_P314(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P314 &instance = CUTLASS_GEMM_P314::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=315 tag=r9_pid195_stage7 base_pid=195
// notes=Deeper pipeline than pid195.
struct CUTLASS_GEMM_P315 {
private:
  CUTLASS_GEMM_P315() {}
  ~CUTLASS_GEMM_P315() {}

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
      7, 8, 8, true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P315(const CUTLASS_GEMM_P315 &) = delete;
  CUTLASS_GEMM_P315 &operator=(const CUTLASS_GEMM_P315 &) = delete;
  static CUTLASS_GEMM_P315 &get_instance() {
    static CUTLASS_GEMM_P315 instance;
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

void cutlass_gemm_P315(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P315 &instance = CUTLASS_GEMM_P315::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=316 tag=r9_pid195_stage8 base_pid=195
// notes=Very deep pipeline around pid195.
struct CUTLASS_GEMM_P316 {
private:
  CUTLASS_GEMM_P316() {}
  ~CUTLASS_GEMM_P316() {}

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
      8, 8, 8, true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P316(const CUTLASS_GEMM_P316 &) = delete;
  CUTLASS_GEMM_P316 &operator=(const CUTLASS_GEMM_P316 &) = delete;
  static CUTLASS_GEMM_P316 &get_instance() {
    static CUTLASS_GEMM_P316 instance;
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

void cutlass_gemm_P316(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P316 &instance = CUTLASS_GEMM_P316::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=317 tag=r9_pid195_sw1 base_pid=195
// notes=Minimal swizzle around pid195.
struct CUTLASS_GEMM_P317 {
private:
  CUTLASS_GEMM_P317() {}
  ~CUTLASS_GEMM_P317() {}

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
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
      5, 8, 8, true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P317(const CUTLASS_GEMM_P317 &) = delete;
  CUTLASS_GEMM_P317 &operator=(const CUTLASS_GEMM_P317 &) = delete;
  static CUTLASS_GEMM_P317 &get_instance() {
    static CUTLASS_GEMM_P317 instance;
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

void cutlass_gemm_P317(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P317 &instance = CUTLASS_GEMM_P317::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=318 tag=r9_pid195_sw8 base_pid=195
// notes=Maximum swizzle around pid195.
struct CUTLASS_GEMM_P318 {
private:
  CUTLASS_GEMM_P318() {}
  ~CUTLASS_GEMM_P318() {}

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
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      5, 8, 8, true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P318(const CUTLASS_GEMM_P318 &) = delete;
  CUTLASS_GEMM_P318 &operator=(const CUTLASS_GEMM_P318 &) = delete;
  static CUTLASS_GEMM_P318 &get_instance() {
    static CUTLASS_GEMM_P318 instance;
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

void cutlass_gemm_P318(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P318 &instance = CUTLASS_GEMM_P318::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=319 tag=r9_pid195_a4 base_pid=195
// notes=Alignment-4 around pid195.
struct CUTLASS_GEMM_P319 {
private:
  CUTLASS_GEMM_P319() {}
  ~CUTLASS_GEMM_P319() {}

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
      5, 4, 4, true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P319(const CUTLASS_GEMM_P319 &) = delete;
  CUTLASS_GEMM_P319 &operator=(const CUTLASS_GEMM_P319 &) = delete;
  static CUTLASS_GEMM_P319 &get_instance() {
    static CUTLASS_GEMM_P319 instance;
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

void cutlass_gemm_P319(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P319 &instance = CUTLASS_GEMM_P319::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=320 tag=r9_pid195_nosks base_pid=195
// notes=Disable split_k_serial while keeping runtime split_k.
struct CUTLASS_GEMM_P320 {
private:
  CUTLASS_GEMM_P320() {}
  ~CUTLASS_GEMM_P320() {}

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
      5, 8, 8, false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P320(const CUTLASS_GEMM_P320 &) = delete;
  CUTLASS_GEMM_P320 &operator=(const CUTLASS_GEMM_P320 &) = delete;
  static CUTLASS_GEMM_P320 &get_instance() {
    static CUTLASS_GEMM_P320 instance;
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

void cutlass_gemm_P320(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P320 &instance = CUTLASS_GEMM_P320::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=321 tag=r9_pid195_nosks_sw4 base_pid=195
// notes=Disable split_k_serial and raise swizzle.
struct CUTLASS_GEMM_P321 {
private:
  CUTLASS_GEMM_P321() {}
  ~CUTLASS_GEMM_P321() {}

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
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
      5, 8, 8, false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P321(const CUTLASS_GEMM_P321 &) = delete;
  CUTLASS_GEMM_P321 &operator=(const CUTLASS_GEMM_P321 &) = delete;
  static CUTLASS_GEMM_P321 &get_instance() {
    static CUTLASS_GEMM_P321 instance;
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

void cutlass_gemm_P321(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P321 &instance = CUTLASS_GEMM_P321::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

