// Auto-generated round-2 merged pid-neighborhood candidates.

// pid=184 tag=f5_pid175_stage5 base_pid=175
// notes=Reduce stages from pid175.
struct CUTLASS_GEMM_P184 {
private:
  CUTLASS_GEMM_P184() {}
  ~CUTLASS_GEMM_P184() {}

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
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
      5, 8, 8, false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P184(const CUTLASS_GEMM_P184 &) = delete;
  CUTLASS_GEMM_P184 &operator=(const CUTLASS_GEMM_P184 &) = delete;
  static CUTLASS_GEMM_P184 &get_instance() {
    static CUTLASS_GEMM_P184 instance;
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

void cutlass_gemm_P184(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P184 &instance = CUTLASS_GEMM_P184::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=185 tag=f5_pid175_stage7 base_pid=175
// notes=Increase stages from pid175.
struct CUTLASS_GEMM_P185 {
private:
  CUTLASS_GEMM_P185() {}
  ~CUTLASS_GEMM_P185() {}

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
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
      7, 8, 8, false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P185(const CUTLASS_GEMM_P185 &) = delete;
  CUTLASS_GEMM_P185 &operator=(const CUTLASS_GEMM_P185 &) = delete;
  static CUTLASS_GEMM_P185 &get_instance() {
    static CUTLASS_GEMM_P185 instance;
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

void cutlass_gemm_P185(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P185 &instance = CUTLASS_GEMM_P185::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=186 tag=f5_pid175_stage8 base_pid=175
// notes=Deep pipeline around pid175.
struct CUTLASS_GEMM_P186 {
private:
  CUTLASS_GEMM_P186() {}
  ~CUTLASS_GEMM_P186() {}

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
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
      8, 8, 8, false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P186(const CUTLASS_GEMM_P186 &) = delete;
  CUTLASS_GEMM_P186 &operator=(const CUTLASS_GEMM_P186 &) = delete;
  static CUTLASS_GEMM_P186 &get_instance() {
    static CUTLASS_GEMM_P186 instance;
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

void cutlass_gemm_P186(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P186 &instance = CUTLASS_GEMM_P186::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=187 tag=f5_pid175_sw2 base_pid=175
// notes=Swizzle-2 around pid175.
struct CUTLASS_GEMM_P187 {
private:
  CUTLASS_GEMM_P187() {}
  ~CUTLASS_GEMM_P187() {}

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
      6, 8, 8, false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P187(const CUTLASS_GEMM_P187 &) = delete;
  CUTLASS_GEMM_P187 &operator=(const CUTLASS_GEMM_P187 &) = delete;
  static CUTLASS_GEMM_P187 &get_instance() {
    static CUTLASS_GEMM_P187 instance;
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

void cutlass_gemm_P187(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P187 &instance = CUTLASS_GEMM_P187::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=188 tag=f5_pid175_sw8 base_pid=175
// notes=Swizzle-8 around pid175.
struct CUTLASS_GEMM_P188 {
private:
  CUTLASS_GEMM_P188() {}
  ~CUTLASS_GEMM_P188() {}

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
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      6, 8, 8, false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P188(const CUTLASS_GEMM_P188 &) = delete;
  CUTLASS_GEMM_P188 &operator=(const CUTLASS_GEMM_P188 &) = delete;
  static CUTLASS_GEMM_P188 &get_instance() {
    static CUTLASS_GEMM_P188 instance;
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

void cutlass_gemm_P188(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P188 &instance = CUTLASS_GEMM_P188::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=189 tag=f5_pid175_a4 base_pid=175
// notes=Alignment-4 around pid175.
struct CUTLASS_GEMM_P189 {
private:
  CUTLASS_GEMM_P189() {}
  ~CUTLASS_GEMM_P189() {}

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
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
      6, 4, 4, false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P189(const CUTLASS_GEMM_P189 &) = delete;
  CUTLASS_GEMM_P189 &operator=(const CUTLASS_GEMM_P189 &) = delete;
  static CUTLASS_GEMM_P189 &get_instance() {
    static CUTLASS_GEMM_P189 instance;
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

void cutlass_gemm_P189(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P189 &instance = CUTLASS_GEMM_P189::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=190 tag=f5_pid177_stage4 base_pid=177
// notes=Reduce stages from pid177.
struct CUTLASS_GEMM_P190 {
private:
  CUTLASS_GEMM_P190() {}
  ~CUTLASS_GEMM_P190() {}

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
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
      4, 8, 8, false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P190(const CUTLASS_GEMM_P190 &) = delete;
  CUTLASS_GEMM_P190 &operator=(const CUTLASS_GEMM_P190 &) = delete;
  static CUTLASS_GEMM_P190 &get_instance() {
    static CUTLASS_GEMM_P190 instance;
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

void cutlass_gemm_P190(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P190 &instance = CUTLASS_GEMM_P190::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=191 tag=f5_pid177_stage6 base_pid=177
// notes=Increase stages from pid177.
struct CUTLASS_GEMM_P191 {
private:
  CUTLASS_GEMM_P191() {}
  ~CUTLASS_GEMM_P191() {}

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
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
      6, 8, 8, false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P191(const CUTLASS_GEMM_P191 &) = delete;
  CUTLASS_GEMM_P191 &operator=(const CUTLASS_GEMM_P191 &) = delete;
  static CUTLASS_GEMM_P191 &get_instance() {
    static CUTLASS_GEMM_P191 instance;
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

void cutlass_gemm_P191(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P191 &instance = CUTLASS_GEMM_P191::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=192 tag=f5_pid177_sw2 base_pid=177
// notes=Swizzle-2 around pid177.
struct CUTLASS_GEMM_P192 {
private:
  CUTLASS_GEMM_P192() {}
  ~CUTLASS_GEMM_P192() {}

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
      5, 8, 8, false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P192(const CUTLASS_GEMM_P192 &) = delete;
  CUTLASS_GEMM_P192 &operator=(const CUTLASS_GEMM_P192 &) = delete;
  static CUTLASS_GEMM_P192 &get_instance() {
    static CUTLASS_GEMM_P192 instance;
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

void cutlass_gemm_P192(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P192 &instance = CUTLASS_GEMM_P192::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=193 tag=f5_pid177_sw8 base_pid=177
// notes=Swizzle-8 around pid177.
struct CUTLASS_GEMM_P193 {
private:
  CUTLASS_GEMM_P193() {}
  ~CUTLASS_GEMM_P193() {}

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
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      5, 8, 8, false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P193(const CUTLASS_GEMM_P193 &) = delete;
  CUTLASS_GEMM_P193 &operator=(const CUTLASS_GEMM_P193 &) = delete;
  static CUTLASS_GEMM_P193 &get_instance() {
    static CUTLASS_GEMM_P193 instance;
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

void cutlass_gemm_P193(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P193 &instance = CUTLASS_GEMM_P193::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=194 tag=f5_pid177_a4 base_pid=177
// notes=Alignment-4 around pid177.
struct CUTLASS_GEMM_P194 {
private:
  CUTLASS_GEMM_P194() {}
  ~CUTLASS_GEMM_P194() {}

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
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
      5, 4, 4, false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P194(const CUTLASS_GEMM_P194 &) = delete;
  CUTLASS_GEMM_P194 &operator=(const CUTLASS_GEMM_P194 &) = delete;
  static CUTLASS_GEMM_P194 &get_instance() {
    static CUTLASS_GEMM_P194 instance;
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

void cutlass_gemm_P194(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P194 &instance = CUTLASS_GEMM_P194::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=195 tag=f5_pid103_stage5 base_pid=103
// notes=Mid stage around pid103.
struct CUTLASS_GEMM_P195 {
private:
  CUTLASS_GEMM_P195() {}
  ~CUTLASS_GEMM_P195() {}

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
      5, 8, 8, true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P195(const CUTLASS_GEMM_P195 &) = delete;
  CUTLASS_GEMM_P195 &operator=(const CUTLASS_GEMM_P195 &) = delete;
  static CUTLASS_GEMM_P195 &get_instance() {
    static CUTLASS_GEMM_P195 instance;
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

void cutlass_gemm_P195(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P195 &instance = CUTLASS_GEMM_P195::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=196 tag=f5_pid103_stage7 base_pid=103
// notes=Slightly deeper pid103.
struct CUTLASS_GEMM_P196 {
private:
  CUTLASS_GEMM_P196() {}
  ~CUTLASS_GEMM_P196() {}

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
  CUTLASS_GEMM_P196(const CUTLASS_GEMM_P196 &) = delete;
  CUTLASS_GEMM_P196 &operator=(const CUTLASS_GEMM_P196 &) = delete;
  static CUTLASS_GEMM_P196 &get_instance() {
    static CUTLASS_GEMM_P196 instance;
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

void cutlass_gemm_P196(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P196 &instance = CUTLASS_GEMM_P196::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=197 tag=f5_pid103_sw8 base_pid=103
// notes=Swizzle-8 around pid103.
struct CUTLASS_GEMM_P197 {
private:
  CUTLASS_GEMM_P197() {}
  ~CUTLASS_GEMM_P197() {}

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
      6, 8, 8, true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P197(const CUTLASS_GEMM_P197 &) = delete;
  CUTLASS_GEMM_P197 &operator=(const CUTLASS_GEMM_P197 &) = delete;
  static CUTLASS_GEMM_P197 &get_instance() {
    static CUTLASS_GEMM_P197 instance;
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

void cutlass_gemm_P197(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P197 &instance = CUTLASS_GEMM_P197::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=198 tag=f5_pid103_a4 base_pid=103
// notes=Alignment-4 around pid103.
struct CUTLASS_GEMM_P198 {
private:
  CUTLASS_GEMM_P198() {}
  ~CUTLASS_GEMM_P198() {}

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
      6, 4, 4, true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P198(const CUTLASS_GEMM_P198 &) = delete;
  CUTLASS_GEMM_P198 &operator=(const CUTLASS_GEMM_P198 &) = delete;
  static CUTLASS_GEMM_P198 &get_instance() {
    static CUTLASS_GEMM_P198 instance;
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

void cutlass_gemm_P198(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P198 &instance = CUTLASS_GEMM_P198::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

