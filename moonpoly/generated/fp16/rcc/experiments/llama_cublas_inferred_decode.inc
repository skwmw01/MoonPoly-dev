// Auto-generated cuBLAS-inferred decode candidates.

// pid=262 tag=cb128x64_s5_sw2
// family=gemm_like_decode
// notes=cuBLAS decode-inspired 128x64 family, stages=5, sliced/split-friendly.
struct CUTLASS_GEMM_P262 {
private:
  CUTLASS_GEMM_P262() {}
  ~CUTLASS_GEMM_P262() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 64, 64>,
      cutlass::gemm::GemmShape<64, 64, 64>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>,
      5, 8, 8, false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P262(const CUTLASS_GEMM_P262 &) = delete;
  CUTLASS_GEMM_P262 &operator=(const CUTLASS_GEMM_P262 &) = delete;
  static CUTLASS_GEMM_P262 &get_instance() {
    static CUTLASS_GEMM_P262 instance;
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

void cutlass_gemm_P262(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P262 &instance = CUTLASS_GEMM_P262::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=263 tag=cb128x64_s6_sw2
// family=gemm_like_decode
// notes=cuBLAS decode-inspired 128x64 family, stages=6.
struct CUTLASS_GEMM_P263 {
private:
  CUTLASS_GEMM_P263() {}
  ~CUTLASS_GEMM_P263() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 64, 64>,
      cutlass::gemm::GemmShape<64, 64, 64>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>,
      6, 8, 8, false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P263(const CUTLASS_GEMM_P263 &) = delete;
  CUTLASS_GEMM_P263 &operator=(const CUTLASS_GEMM_P263 &) = delete;
  static CUTLASS_GEMM_P263 &get_instance() {
    static CUTLASS_GEMM_P263 instance;
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

void cutlass_gemm_P263(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P263 &instance = CUTLASS_GEMM_P263::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=264 tag=cb64x64_s5_sw2
// family=gemm_like_decode
// notes=cuBLAS decode-inspired 64x64 family, sliced1x2-like baseline.
struct CUTLASS_GEMM_P264 {
private:
  CUTLASS_GEMM_P264() {}
  ~CUTLASS_GEMM_P264() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<64, 64, 64>,
      cutlass::gemm::GemmShape<64, 64, 64>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>,
      5, 8, 8, false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P264(const CUTLASS_GEMM_P264 &) = delete;
  CUTLASS_GEMM_P264 &operator=(const CUTLASS_GEMM_P264 &) = delete;
  static CUTLASS_GEMM_P264 &get_instance() {
    static CUTLASS_GEMM_P264 instance;
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

void cutlass_gemm_P264(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P264 &instance = CUTLASS_GEMM_P264::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=265 tag=cb64x64_s6_sw2
// family=gemm_like_decode
// notes=cuBLAS decode-inspired 64x64 family, stages=6.
struct CUTLASS_GEMM_P265 {
private:
  CUTLASS_GEMM_P265() {}
  ~CUTLASS_GEMM_P265() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<64, 64, 64>,
      cutlass::gemm::GemmShape<64, 64, 64>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>,
      6, 8, 8, false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P265(const CUTLASS_GEMM_P265 &) = delete;
  CUTLASS_GEMM_P265 &operator=(const CUTLASS_GEMM_P265 &) = delete;
  static CUTLASS_GEMM_P265 &get_instance() {
    static CUTLASS_GEMM_P265 instance;
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

void cutlass_gemm_P265(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P265 &instance = CUTLASS_GEMM_P265::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=266 tag=cb128x64_s6_sw4
// family=gemm_like_decode
// notes=Same tile, higher swizzle for launch distribution.
struct CUTLASS_GEMM_P266 {
private:
  CUTLASS_GEMM_P266() {}
  ~CUTLASS_GEMM_P266() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 64, 64>,
      cutlass::gemm::GemmShape<64, 64, 64>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
      6, 8, 8, false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P266(const CUTLASS_GEMM_P266 &) = delete;
  CUTLASS_GEMM_P266 &operator=(const CUTLASS_GEMM_P266 &) = delete;
  static CUTLASS_GEMM_P266 &get_instance() {
    static CUTLASS_GEMM_P266 instance;
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

void cutlass_gemm_P266(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P266 &instance = CUTLASS_GEMM_P266::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=267 tag=cb64x64_s6_sw4
// family=gemm_like_decode
// notes=64x64 with stronger swizzle.
struct CUTLASS_GEMM_P267 {
private:
  CUTLASS_GEMM_P267() {}
  ~CUTLASS_GEMM_P267() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<64, 64, 64>,
      cutlass::gemm::GemmShape<64, 64, 64>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
      6, 8, 8, false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P267(const CUTLASS_GEMM_P267 &) = delete;
  CUTLASS_GEMM_P267 &operator=(const CUTLASS_GEMM_P267 &) = delete;
  static CUTLASS_GEMM_P267 &get_instance() {
    static CUTLASS_GEMM_P267 instance;
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

void cutlass_gemm_P267(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P267 &instance = CUTLASS_GEMM_P267::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=268 tag=cb128x64_s6_a4
// family=gemm_like_decode
// notes=Alignment-4 fallback variant.
struct CUTLASS_GEMM_P268 {
private:
  CUTLASS_GEMM_P268() {}
  ~CUTLASS_GEMM_P268() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 64, 64>,
      cutlass::gemm::GemmShape<64, 64, 64>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>,
      6, 4, 4, false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P268(const CUTLASS_GEMM_P268 &) = delete;
  CUTLASS_GEMM_P268 &operator=(const CUTLASS_GEMM_P268 &) = delete;
  static CUTLASS_GEMM_P268 &get_instance() {
    static CUTLASS_GEMM_P268 instance;
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

void cutlass_gemm_P268(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P268 &instance = CUTLASS_GEMM_P268::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=269 tag=cb64x64_s6_a4
// family=gemm_like_decode
// notes=64x64 alignment-4 fallback.
struct CUTLASS_GEMM_P269 {
private:
  CUTLASS_GEMM_P269() {}
  ~CUTLASS_GEMM_P269() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<64, 64, 64>,
      cutlass::gemm::GemmShape<64, 64, 64>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>,
      6, 4, 4, false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P269(const CUTLASS_GEMM_P269 &) = delete;
  CUTLASS_GEMM_P269 &operator=(const CUTLASS_GEMM_P269 &) = delete;
  static CUTLASS_GEMM_P269 &get_instance() {
    static CUTLASS_GEMM_P269 instance;
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

void cutlass_gemm_P269(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P269 &instance = CUTLASS_GEMM_P269::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

