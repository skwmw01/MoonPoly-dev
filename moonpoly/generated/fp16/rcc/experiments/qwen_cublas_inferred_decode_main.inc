// Auto-generated cuBLAS-inferred decode candidates.

// pid=270 tag=cb128x64_s5_sw2
// family=gemm_like_decode
// notes=cuBLAS decode-inspired 128x64 family, stages=5, sliced/split-friendly.
struct CUTLASS_GEMM_P270 {
private:
  CUTLASS_GEMM_P270() {}
  ~CUTLASS_GEMM_P270() {}

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
  CUTLASS_GEMM_P270(const CUTLASS_GEMM_P270 &) = delete;
  CUTLASS_GEMM_P270 &operator=(const CUTLASS_GEMM_P270 &) = delete;
  static CUTLASS_GEMM_P270 &get_instance() {
    static CUTLASS_GEMM_P270 instance;
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

void cutlass_gemm_P270(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P270 &instance = CUTLASS_GEMM_P270::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=271 tag=cb128x64_s6_sw2
// family=gemm_like_decode
// notes=cuBLAS decode-inspired 128x64 family, stages=6.
struct CUTLASS_GEMM_P271 {
private:
  CUTLASS_GEMM_P271() {}
  ~CUTLASS_GEMM_P271() {}

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
  CUTLASS_GEMM_P271(const CUTLASS_GEMM_P271 &) = delete;
  CUTLASS_GEMM_P271 &operator=(const CUTLASS_GEMM_P271 &) = delete;
  static CUTLASS_GEMM_P271 &get_instance() {
    static CUTLASS_GEMM_P271 instance;
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

void cutlass_gemm_P271(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P271 &instance = CUTLASS_GEMM_P271::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=272 tag=cb64x64_s5_sw2
// family=gemm_like_decode
// notes=cuBLAS decode-inspired 64x64 family, sliced1x2-like baseline.
struct CUTLASS_GEMM_P272 {
private:
  CUTLASS_GEMM_P272() {}
  ~CUTLASS_GEMM_P272() {}

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
  CUTLASS_GEMM_P272(const CUTLASS_GEMM_P272 &) = delete;
  CUTLASS_GEMM_P272 &operator=(const CUTLASS_GEMM_P272 &) = delete;
  static CUTLASS_GEMM_P272 &get_instance() {
    static CUTLASS_GEMM_P272 instance;
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

void cutlass_gemm_P272(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P272 &instance = CUTLASS_GEMM_P272::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=273 tag=cb64x64_s6_sw2
// family=gemm_like_decode
// notes=cuBLAS decode-inspired 64x64 family, stages=6.
struct CUTLASS_GEMM_P273 {
private:
  CUTLASS_GEMM_P273() {}
  ~CUTLASS_GEMM_P273() {}

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
  CUTLASS_GEMM_P273(const CUTLASS_GEMM_P273 &) = delete;
  CUTLASS_GEMM_P273 &operator=(const CUTLASS_GEMM_P273 &) = delete;
  static CUTLASS_GEMM_P273 &get_instance() {
    static CUTLASS_GEMM_P273 instance;
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

void cutlass_gemm_P273(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P273 &instance = CUTLASS_GEMM_P273::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=274 tag=cb128x64_s6_sw4
// family=gemm_like_decode
// notes=Same tile, higher swizzle for launch distribution.
struct CUTLASS_GEMM_P274 {
private:
  CUTLASS_GEMM_P274() {}
  ~CUTLASS_GEMM_P274() {}

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
  CUTLASS_GEMM_P274(const CUTLASS_GEMM_P274 &) = delete;
  CUTLASS_GEMM_P274 &operator=(const CUTLASS_GEMM_P274 &) = delete;
  static CUTLASS_GEMM_P274 &get_instance() {
    static CUTLASS_GEMM_P274 instance;
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

void cutlass_gemm_P274(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P274 &instance = CUTLASS_GEMM_P274::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=275 tag=cb64x64_s6_sw4
// family=gemm_like_decode
// notes=64x64 with stronger swizzle.
struct CUTLASS_GEMM_P275 {
private:
  CUTLASS_GEMM_P275() {}
  ~CUTLASS_GEMM_P275() {}

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
  CUTLASS_GEMM_P275(const CUTLASS_GEMM_P275 &) = delete;
  CUTLASS_GEMM_P275 &operator=(const CUTLASS_GEMM_P275 &) = delete;
  static CUTLASS_GEMM_P275 &get_instance() {
    static CUTLASS_GEMM_P275 instance;
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

void cutlass_gemm_P275(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P275 &instance = CUTLASS_GEMM_P275::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=276 tag=cb128x64_s6_a4
// family=gemm_like_decode
// notes=Alignment-4 fallback variant.
struct CUTLASS_GEMM_P276 {
private:
  CUTLASS_GEMM_P276() {}
  ~CUTLASS_GEMM_P276() {}

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
  CUTLASS_GEMM_P276(const CUTLASS_GEMM_P276 &) = delete;
  CUTLASS_GEMM_P276 &operator=(const CUTLASS_GEMM_P276 &) = delete;
  static CUTLASS_GEMM_P276 &get_instance() {
    static CUTLASS_GEMM_P276 instance;
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

void cutlass_gemm_P276(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P276 &instance = CUTLASS_GEMM_P276::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=277 tag=cb64x64_s6_a4
// family=gemm_like_decode
// notes=64x64 alignment-4 fallback.
struct CUTLASS_GEMM_P277 {
private:
  CUTLASS_GEMM_P277() {}
  ~CUTLASS_GEMM_P277() {}

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
  CUTLASS_GEMM_P277(const CUTLASS_GEMM_P277 &) = delete;
  CUTLASS_GEMM_P277 &operator=(const CUTLASS_GEMM_P277 &) = delete;
  static CUTLASS_GEMM_P277 &get_instance() {
    static CUTLASS_GEMM_P277 instance;
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

void cutlass_gemm_P277(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P277 &instance = CUTLASS_GEMM_P277::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

