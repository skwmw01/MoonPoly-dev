// Auto-generated round-2 merged pid-neighborhood candidates.

// pid=114 tag=seed_pid95_s4_sw1_a4_sk0 base_pid=95
// notes=Full CUTLASS param sweep around seed_pid95: stages=4, swizzle=1, align=4, split_k_serial=False
struct CUTLASS_GEMM_P114 {
private:
  CUTLASS_GEMM_P114() {}
  ~CUTLASS_GEMM_P114() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 32, 64>,
      cutlass::gemm::GemmShape<16, 32, 64>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
      4, 4, 4, false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P114(const CUTLASS_GEMM_P114 &) = delete;
  CUTLASS_GEMM_P114 &operator=(const CUTLASS_GEMM_P114 &) = delete;
  static CUTLASS_GEMM_P114 &get_instance() {
    static CUTLASS_GEMM_P114 instance;
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

void cutlass_gemm_P114(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P114 &instance = CUTLASS_GEMM_P114::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=115 tag=seed_pid103_s4_sw1_a4_sk0 base_pid=103
// notes=Full CUTLASS param sweep around seed_pid103: stages=4, swizzle=1, align=4, split_k_serial=False
struct CUTLASS_GEMM_P115 {
private:
  CUTLASS_GEMM_P115() {}
  ~CUTLASS_GEMM_P115() {}

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
      4, 4, 4, false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P115(const CUTLASS_GEMM_P115 &) = delete;
  CUTLASS_GEMM_P115 &operator=(const CUTLASS_GEMM_P115 &) = delete;
  static CUTLASS_GEMM_P115 &get_instance() {
    static CUTLASS_GEMM_P115 instance;
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

void cutlass_gemm_P115(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P115 &instance = CUTLASS_GEMM_P115::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=116 tag=seed_pid92_s4_sw1_a4_sk0 base_pid=92
// notes=Full CUTLASS param sweep around seed_pid92: stages=4, swizzle=1, align=4, split_k_serial=False
struct CUTLASS_GEMM_P116 {
private:
  CUTLASS_GEMM_P116() {}
  ~CUTLASS_GEMM_P116() {}

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
      4, 4, 4, false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P116(const CUTLASS_GEMM_P116 &) = delete;
  CUTLASS_GEMM_P116 &operator=(const CUTLASS_GEMM_P116 &) = delete;
  static CUTLASS_GEMM_P116 &get_instance() {
    static CUTLASS_GEMM_P116 instance;
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

void cutlass_gemm_P116(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P116 &instance = CUTLASS_GEMM_P116::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=117 tag=seed_pid101_s4_sw1_a4_sk0 base_pid=101
// notes=Full CUTLASS param sweep around seed_pid101: stages=4, swizzle=1, align=4, split_k_serial=False
struct CUTLASS_GEMM_P117 {
private:
  CUTLASS_GEMM_P117() {}
  ~CUTLASS_GEMM_P117() {}

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
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
      4, 4, 4, false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P117(const CUTLASS_GEMM_P117 &) = delete;
  CUTLASS_GEMM_P117 &operator=(const CUTLASS_GEMM_P117 &) = delete;
  static CUTLASS_GEMM_P117 &get_instance() {
    static CUTLASS_GEMM_P117 instance;
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

void cutlass_gemm_P117(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P117 &instance = CUTLASS_GEMM_P117::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=118 tag=seed_pid6_s8_sw1_a4_sk0 base_pid=6
// notes=Full CUTLASS param sweep around seed_pid6: stages=8, swizzle=1, align=4, split_k_serial=False
struct CUTLASS_GEMM_P118 {
private:
  CUTLASS_GEMM_P118() {}
  ~CUTLASS_GEMM_P118() {}

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
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
      8, 4, 4, false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P118(const CUTLASS_GEMM_P118 &) = delete;
  CUTLASS_GEMM_P118 &operator=(const CUTLASS_GEMM_P118 &) = delete;
  static CUTLASS_GEMM_P118 &get_instance() {
    static CUTLASS_GEMM_P118 instance;
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

void cutlass_gemm_P118(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P118 &instance = CUTLASS_GEMM_P118::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=119 tag=seed_pid4_s4_sw1_a4_sk0 base_pid=4
// notes=Full CUTLASS param sweep around seed_pid4: stages=4, swizzle=1, align=4, split_k_serial=False
struct CUTLASS_GEMM_P119 {
private:
  CUTLASS_GEMM_P119() {}
  ~CUTLASS_GEMM_P119() {}

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
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
      4, 4, 4, false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P119(const CUTLASS_GEMM_P119 &) = delete;
  CUTLASS_GEMM_P119 &operator=(const CUTLASS_GEMM_P119 &) = delete;
  static CUTLASS_GEMM_P119 &get_instance() {
    static CUTLASS_GEMM_P119 instance;
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

void cutlass_gemm_P119(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P119 &instance = CUTLASS_GEMM_P119::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=120 tag=seed_pid95_s4_sw1_a8_sk0 base_pid=95
// notes=Full CUTLASS param sweep around seed_pid95: stages=4, swizzle=1, align=8, split_k_serial=False
struct CUTLASS_GEMM_P120 {
private:
  CUTLASS_GEMM_P120() {}
  ~CUTLASS_GEMM_P120() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 32, 64>,
      cutlass::gemm::GemmShape<16, 32, 64>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
      4, 8, 8, false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P120(const CUTLASS_GEMM_P120 &) = delete;
  CUTLASS_GEMM_P120 &operator=(const CUTLASS_GEMM_P120 &) = delete;
  static CUTLASS_GEMM_P120 &get_instance() {
    static CUTLASS_GEMM_P120 instance;
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

void cutlass_gemm_P120(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P120 &instance = CUTLASS_GEMM_P120::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=121 tag=seed_pid103_s4_sw1_a8_sk0 base_pid=103
// notes=Full CUTLASS param sweep around seed_pid103: stages=4, swizzle=1, align=8, split_k_serial=False
struct CUTLASS_GEMM_P121 {
private:
  CUTLASS_GEMM_P121() {}
  ~CUTLASS_GEMM_P121() {}

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
      4, 8, 8, false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P121(const CUTLASS_GEMM_P121 &) = delete;
  CUTLASS_GEMM_P121 &operator=(const CUTLASS_GEMM_P121 &) = delete;
  static CUTLASS_GEMM_P121 &get_instance() {
    static CUTLASS_GEMM_P121 instance;
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

void cutlass_gemm_P121(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P121 &instance = CUTLASS_GEMM_P121::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=122 tag=seed_pid92_s4_sw1_a8_sk0 base_pid=92
// notes=Full CUTLASS param sweep around seed_pid92: stages=4, swizzle=1, align=8, split_k_serial=False
struct CUTLASS_GEMM_P122 {
private:
  CUTLASS_GEMM_P122() {}
  ~CUTLASS_GEMM_P122() {}

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
  CUTLASS_GEMM_P122(const CUTLASS_GEMM_P122 &) = delete;
  CUTLASS_GEMM_P122 &operator=(const CUTLASS_GEMM_P122 &) = delete;
  static CUTLASS_GEMM_P122 &get_instance() {
    static CUTLASS_GEMM_P122 instance;
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

void cutlass_gemm_P122(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P122 &instance = CUTLASS_GEMM_P122::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=123 tag=seed_pid101_s4_sw1_a8_sk0 base_pid=101
// notes=Full CUTLASS param sweep around seed_pid101: stages=4, swizzle=1, align=8, split_k_serial=False
struct CUTLASS_GEMM_P123 {
private:
  CUTLASS_GEMM_P123() {}
  ~CUTLASS_GEMM_P123() {}

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
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
      4, 8, 8, false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P123(const CUTLASS_GEMM_P123 &) = delete;
  CUTLASS_GEMM_P123 &operator=(const CUTLASS_GEMM_P123 &) = delete;
  static CUTLASS_GEMM_P123 &get_instance() {
    static CUTLASS_GEMM_P123 instance;
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

void cutlass_gemm_P123(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P123 &instance = CUTLASS_GEMM_P123::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=124 tag=seed_pid6_s8_sw1_a8_sk0 base_pid=6
// notes=Full CUTLASS param sweep around seed_pid6: stages=8, swizzle=1, align=8, split_k_serial=False
struct CUTLASS_GEMM_P124 {
private:
  CUTLASS_GEMM_P124() {}
  ~CUTLASS_GEMM_P124() {}

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
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
      8, 8, 8, false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P124(const CUTLASS_GEMM_P124 &) = delete;
  CUTLASS_GEMM_P124 &operator=(const CUTLASS_GEMM_P124 &) = delete;
  static CUTLASS_GEMM_P124 &get_instance() {
    static CUTLASS_GEMM_P124 instance;
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

void cutlass_gemm_P124(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P124 &instance = CUTLASS_GEMM_P124::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=125 tag=seed_pid4_s4_sw1_a8_sk0 base_pid=4
// notes=Full CUTLASS param sweep around seed_pid4: stages=4, swizzle=1, align=8, split_k_serial=False
struct CUTLASS_GEMM_P125 {
private:
  CUTLASS_GEMM_P125() {}
  ~CUTLASS_GEMM_P125() {}

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
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
      4, 8, 8, false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P125(const CUTLASS_GEMM_P125 &) = delete;
  CUTLASS_GEMM_P125 &operator=(const CUTLASS_GEMM_P125 &) = delete;
  static CUTLASS_GEMM_P125 &get_instance() {
    static CUTLASS_GEMM_P125 instance;
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

void cutlass_gemm_P125(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P125 &instance = CUTLASS_GEMM_P125::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=126 tag=seed_pid95_s4_sw1_a4_sk1 base_pid=95
// notes=Full CUTLASS param sweep around seed_pid95: stages=4, swizzle=1, align=4, split_k_serial=True
struct CUTLASS_GEMM_P126 {
private:
  CUTLASS_GEMM_P126() {}
  ~CUTLASS_GEMM_P126() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 32, 64>,
      cutlass::gemm::GemmShape<16, 32, 64>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
      4, 4, 4, true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P126(const CUTLASS_GEMM_P126 &) = delete;
  CUTLASS_GEMM_P126 &operator=(const CUTLASS_GEMM_P126 &) = delete;
  static CUTLASS_GEMM_P126 &get_instance() {
    static CUTLASS_GEMM_P126 instance;
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

void cutlass_gemm_P126(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P126 &instance = CUTLASS_GEMM_P126::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=127 tag=seed_pid103_s4_sw1_a4_sk1 base_pid=103
// notes=Full CUTLASS param sweep around seed_pid103: stages=4, swizzle=1, align=4, split_k_serial=True
struct CUTLASS_GEMM_P127 {
private:
  CUTLASS_GEMM_P127() {}
  ~CUTLASS_GEMM_P127() {}

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
      4, 4, 4, true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P127(const CUTLASS_GEMM_P127 &) = delete;
  CUTLASS_GEMM_P127 &operator=(const CUTLASS_GEMM_P127 &) = delete;
  static CUTLASS_GEMM_P127 &get_instance() {
    static CUTLASS_GEMM_P127 instance;
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

void cutlass_gemm_P127(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P127 &instance = CUTLASS_GEMM_P127::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=128 tag=seed_pid92_s4_sw1_a4_sk1 base_pid=92
// notes=Full CUTLASS param sweep around seed_pid92: stages=4, swizzle=1, align=4, split_k_serial=True
struct CUTLASS_GEMM_P128 {
private:
  CUTLASS_GEMM_P128() {}
  ~CUTLASS_GEMM_P128() {}

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
      4, 4, 4, true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P128(const CUTLASS_GEMM_P128 &) = delete;
  CUTLASS_GEMM_P128 &operator=(const CUTLASS_GEMM_P128 &) = delete;
  static CUTLASS_GEMM_P128 &get_instance() {
    static CUTLASS_GEMM_P128 instance;
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

void cutlass_gemm_P128(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P128 &instance = CUTLASS_GEMM_P128::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}


// pid=129 tag=seed_pid101_s4_sw1_a4_sk1 base_pid=101
// notes=Full CUTLASS param sweep around seed_pid101: stages=4, swizzle=1, align=4, split_k_serial=True
struct CUTLASS_GEMM_P129 {
private:
  CUTLASS_GEMM_P129() {}
  ~CUTLASS_GEMM_P129() {}

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
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
      4, 4, 4, true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P129(const CUTLASS_GEMM_P129 &) = delete;
  CUTLASS_GEMM_P129 &operator=(const CUTLASS_GEMM_P129 &) = delete;
  static CUTLASS_GEMM_P129 &get_instance() {
    static CUTLASS_GEMM_P129 instance;
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

void cutlass_gemm_P129(int length_m, int length_n, int length_k,
                         cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                         float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P129 &instance = CUTLASS_GEMM_P129::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

