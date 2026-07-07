#include "cutlass/cutlass.h"
#include "cutlass/half.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_splitk_parallel.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/device/gemm.h"
#include "moonpoly.cuh"
#include "../../generated/fp16/rcc/candidates.cuh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <iostream>
#include <chrono>

namespace moonpoly {
namespace fp16 {
namespace row_col_col {
/*
   ============================ CUTLASS GEMM UTILITY ===========================
*/
// Cost model components are defined in fp16_rcc_cost.cu
// Include declarations for external linkage
extern const int num_gemm_primary;
extern SharedMem gemm_shared_mem[];
extern int gemm_alignment[];
extern float gemm_sensitive[];
extern float gemm_cost_model(int ims, int ins, int iks, int pid);
extern float gemm_splitk_cost_model(int ims, int ins, int iks, int pid,
                                    int split_k_slices);
extern int cutlass_gemm_predict(int m, int n, int k);
extern void dump_cost_model(int ims, int ins, int iks, int pid);

using CUTLASS_PATTERN2_GEMM_OP0 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
    cutlass::layout::ColumnMajor, cutlass::half_t,
    cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80, cutlass::gemm::GemmShape<256, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                 float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 3, 8, 8,
    true>;

using CUTLASS_PATTERN2_GEMM_OP1 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
    cutlass::layout::ColumnMajor, cutlass::half_t,
    cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80, cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                 float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 6, 8, 8,
    false>;

int choose_pattern2_split_n(int n) {
  constexpr int tile_n = CUTLASS_PATTERN2_GEMM_OP0::ThreadblockShape::kN;
  int split_n = (n * 3 / 4 / tile_n) * tile_n;
  if (split_n <= 0) {
    split_n = tile_n;
  }
  if (split_n >= n) {
    split_n = ((n / tile_n) - 1) * tile_n;
  }
  return split_n;
}

void run_pattern2_twin_gemm(int m, int n, int k, int split_n,
                            cutlass::half_t *A, cutlass::half_t *B,
                            cutlass::half_t *C, float alpha, float beta,
                            cudaStream_t stream) {
  constexpr int tile_n = CUTLASS_PATTERN2_GEMM_OP0::ThreadblockShape::kN;
  if (split_n <= 0) {
    split_n = choose_pattern2_split_n(n);
  }
  if (split_n <= 0 || split_n >= n || split_n % tile_n != 0) {
    throw std::invalid_argument(
        "Pattern2 FP16 RCC requires 0 < split_n < n and split_n aligned to "
        "the OP0 N tile");
  }
  if (k % 8 != 0) {
    throw std::invalid_argument("Pattern2 FP16 RCC requires K aligned to 8");
  }

  cutlass::gemm::device::TwinGemm<CUTLASS_PATTERN2_GEMM_OP0,
                                  CUTLASS_PATTERN2_GEMM_OP1>
      twin_gemm;
  constexpr int split_k_slices = 1;

  cutlass::Status status = twin_gemm.initialize_op0(
      m, split_n, k, A, B, C, split_k_slices, alpha, beta, stream);
  CUTLASS_CHECK(status);

  status = twin_gemm.initialize_op1(m, n - split_n, k, A, B + split_n * k,
                                    C + split_n * m, split_k_slices, alpha,
                                    beta, stream);
  CUTLASS_CHECK(status);

  status = twin_gemm.run(stream);
  CUTLASS_CHECK(status);
}

bool env_flag(const char *name, bool default_value) {
  const char *raw = std::getenv(name);
  if (raw == nullptr) {
    return default_value;
  }
  return !(std::string(raw) == "0" || std::string(raw) == "false" ||
           std::string(raw) == "FALSE" || std::string(raw) == "off" ||
           std::string(raw) == "OFF");
}

float env_float(const char *name, float default_value) {
  const char *raw = std::getenv(name);
  if (raw == nullptr) {
    return default_value;
  }
  char *end = nullptr;
  float value = std::strtof(raw, &end);
  if (end == raw || !std::isfinite(value)) {
    return default_value;
  }
  return value;
}

int env_int(const char *name, int default_value) {
  const char *raw = std::getenv(name);
  if (raw == nullptr) {
    return default_value;
  }
  char *end = nullptr;
  long value = std::strtol(raw, &end, 10);
  if (end == raw) {
    return default_value;
  }
  return static_cast<int>(value);
}

bool pattern3_k_gate(int k, int split_k, int min_tile_k) {
  const int min_k = env_int("MOONPOLY_PATTERN3_MIN_K", 2048);
  const int min_part_k = env_int("MOONPOLY_PATTERN3_MIN_PART_K", 512);
  if (split_k <= 1 || k < min_k || k < min_tile_k * split_k) {
    return false;
  }
  const int part_k = (k + split_k - 1) / split_k;
  return part_k >= min_part_k;
}

int pattern3_min_tile_k_for_pid(int pid) {
  if (pid >= 0 && pid < 40) {
    return gemm_shared_mem[pid].pk;
  }
  if (is_explicit_neighborhood_pid(pid)) {
    SharedMem meta = explicit_neighborhood_shared_mem(pid);
    if (meta.pk > 0) {
      return meta.pk;
    }
  }
  return 1;
}

const char *pattern_name(Fp16PolyPattern pattern) {
  switch (pattern) {
  case Fp16PolyPattern::Single:
    return "Pattern1";
  case Fp16PolyPattern::Combine:
    return "Pattern2";
  case Fp16PolyPattern::SplitK:
    return "Pattern3";
  default:
    return "Unknown";
  }
}

struct DecodeSelectorEntry {
  int m;
  int n;
  int k;
  int pid;
  int split_k;
};

#include "../../generated/fp16/rcc/llm_decode_selector.inc"

struct Fp16RccGeneratedPlanEntry {
  int m;
  int n;
  int k;
  int pid;
  int split_k;
  bool requires_ablation_env;
  float estimated_ms;
};

#include "../../generated/fp16/rcc/rcc_generated_plans.inc"
#include "../../generated/fp16/rcc/fp16_rcc_pattern2_cost.inc"

bool is_primary_runtime_splitk_pid(int pid);

Fp16PolyPattern pattern_for_pid_split(int pid, int split_k) {
  const bool split_capable_pid =
      is_primary_runtime_splitk_pid(pid) || is_explicit_neighborhood_pid(pid);
  if (split_k > 1 && split_capable_pid) {
    return Fp16PolyPattern::SplitK;
  }
  if (is_primary_runtime_splitk_pid(pid)) {
    return Fp16PolyPattern::SplitK;
  }
  return Fp16PolyPattern::Single;
}

float estimate_explicit_pid_cost(int feature_m, int feature_n, int k, int pid,
                                 int split_k) {
  if (!is_explicit_neighborhood_pid(pid)) {
    return std::numeric_limits<float>::infinity();
  }
  SharedMem meta = explicit_neighborhood_shared_mem(pid);
  if (meta.pm <= 0 || meta.pn <= 0 || meta.pk <= 0) {
    return std::numeric_limits<float>::infinity();
  }

  const int split = std::max(1, split_k);
  const int lm =
      std::max(1, static_cast<int>(std::ceil(1.0 * feature_m / meta.pm)));
  const int ln =
      std::max(1, static_cast<int>(std::ceil(1.0 * feature_n / meta.pn)));
  const int part_k =
      std::max(1, static_cast<int>(std::ceil(1.0 * k / split)));
  const int lk =
      std::max(1, static_cast<int>(std::ceil(1.0 * part_k / meta.pk)));
  const int waves = std::max(
      1, static_cast<int>(std::ceil(1.0 * lm * ln * split / num_SM)));

  const double padded_flops =
      2.0 * lm * meta.pm * ln * meta.pn * lk * meta.pk * split;
  const double actual_flops = 2.0 * feature_m * feature_n * k;
  const float tail_eff = static_cast<float>(std::max(
      0.10, std::min(1.0, actual_flops / std::max(1.0, padded_flops))));
  const float peak_tflops =
      env_float("MOONPOLY_EXPLICIT_SELECTOR_PEAK_TFLOPS", 120.0f);
  const float wave_overhead_ms =
      env_float("MOONPOLY_EXPLICIT_SELECTOR_WAVE_OVERHEAD_MS", 0.0015f);
  const float reduction_ms =
      split > 1 ? env_float("MOONPOLY_EXPLICIT_SELECTOR_REDUCTION_MS", 0.0030f) *
                      (split - 1)
                : 0.0f;
  const double compute_ms =
      padded_flops / (static_cast<double>(peak_tflops) * tail_eff * 1.0e12) *
      1.0e3;
  return static_cast<float>(compute_ms + wave_overhead_ms * waves +
                            reduction_ms);
}

float estimate_pattern2_cost(int m, int n, int k, int split_n) {
  if (split_n <= 0 || split_n >= n) {
    return std::numeric_limits<float>::infinity();
  }
  int op0_pid = cutlass_gemm_predict(split_n, m, k);
  int op1_pid = cutlass_gemm_predict(n - split_n, m, k);
  float op0_ms = gemm_cost_model(split_n, m, k, op0_pid);
  float op1_ms = gemm_cost_model(n - split_n, m, k, op1_pid);
  // TwinGemm runs both regions in one launch; model the critical path plus
  // the fitted launch/imbalance overhead for the irregular tail region.
  const int tail_n = n - split_n;
  const int tail_tiles = std::max(
      1, static_cast<int>(
             std::ceil(1.0 * m / pattern2_fitted_cost::kTailTileM)) *
             static_cast<int>(
                 std::ceil(1.0 * tail_n / pattern2_fitted_cost::kTailTileN)));
  const float default_overhead =
      env_float("MOONPOLY_PATTERN2_DEFAULT_OVERHEAD_MS", 0.006f);
  const float overhead =
      pattern2_fitted_cost::kHasMeasurements
          ? static_cast<float>(pattern2_fitted_cost::kFixedOverheadMs +
                               pattern2_fitted_cost::kTailTileOverheadMs *
                                   tail_tiles)
          : default_overhead;
  return std::max(op0_ms, op1_ms) + std::max(0.0f, overhead);
}

bool pattern2_tail_gate(int m, int n, int k, int *split_n) {
  if (!env_flag("MOONPOLY_ENABLE_PATTERN2", true)) {
    return false;
  }
  if (k % 8 != 0) {
    return false;
  }
  const int min_n = env_int("MOONPOLY_PATTERN2_MIN_N", 512);
  if (n < min_n) {
    return false;
  }

  constexpr int tile_n = CUTLASS_PATTERN2_GEMM_OP0::ThreadblockShape::kN;
  const int tail_n = n % tile_n;
  if (tail_n == 0) {
    if (!env_flag("MOONPOLY_PATTERN2_ALLOW_REGULAR", false)) {
      return false;
    }
    *split_n = choose_pattern2_split_n(n);
    return *split_n > 0 && *split_n < n && *split_n % tile_n == 0;
  }

  const float max_tail_ratio =
      env_float("MOONPOLY_PATTERN2_MAX_TAIL_RATIO", 0.25f);
  const float tail_ratio = static_cast<float>(tail_n) / static_cast<float>(tile_n);
  if (tail_ratio > max_tail_ratio) {
    return false;
  }

  const int main_n = (n / tile_n) * tile_n;
  if (main_n <= 0 || main_n >= n) {
    return false;
  }
  *split_n = main_n;
  (void)m;
  return true;
}

void consider_plan(Fp16OnlinePlan &best, Fp16PolyPattern pattern, int pid,
                   int split_k, int split_n, float estimated_ms,
                   float margin) {
  if (!std::isfinite(estimated_ms) || estimated_ms <= 0.0f) {
    return;
  }
  if (estimated_ms < best.estimated_ms * margin) {
    best.pattern = pattern;
    best.pid = pid;
    best.split_k = split_k;
    best.split_n = split_n;
    best.estimated_ms = estimated_ms;
  }
}

float estimate_pid_split_cost(int feature_m, int feature_n, int k, int pid,
                              int split_k) {
  if (pid < 0) {
    return std::numeric_limits<float>::infinity();
  }
  if (split_k > 1 && pattern_for_pid_split(pid, split_k) ==
                         Fp16PolyPattern::SplitK) {
    if (pid >= 0 && pid < 40) {
      return gemm_splitk_cost_model(feature_m, feature_n, k, pid, split_k);
    }
    return estimate_explicit_pid_cost(feature_m, feature_n, k, pid, split_k);
  }
  if (pid >= 0 && pid < 40) {
    return gemm_cost_model(feature_m, feature_n, k, pid);
  }
  return estimate_explicit_pid_cost(feature_m, feature_n, k, pid,
                                    std::max(1, split_k));
}

bool consider_generated_selector_plans(int m, int n, int k,
                                       Fp16OnlinePlan &best,
                                       bool *matched_decode_selector) {
  if (!env_flag("MOONPOLY_ENABLE_GENERATED_SELECTOR", true)) {
    return false;
  }

  bool considered = false;
  if (matched_decode_selector != nullptr) {
    *matched_decode_selector = false;
  }
  const int feature_m = n;
  const int feature_n = m;
  const float margin = env_float("MOONPOLY_GENERATED_SELECTOR_MARGIN", 1.0f);

  // kMergedDecodeSelectorEntries is generated for Linear(A[M,K], W[N,K]).
  // The core RCC API receives the mapped GEMM as (N, M, K), so the lookup
  // intentionally compares selector.M with runtime n and selector.N with
  // runtime m. These entries are measured, profile-backed decode plans; by
  // default exact matches short-circuit the generic cost model in
  // select_online_plan().
  for (const auto &entry : kMergedDecodeSelectorEntries) {
    if (entry.m == n && entry.n == m && entry.k == k) {
      int split_k = std::max(1, entry.split_k);
      if (split_k > 1 &&
          !pattern3_k_gate(k, split_k, pattern3_min_tile_k_for_pid(entry.pid))) {
        continue;
      }
      float cost = estimate_pid_split_cost(feature_m, feature_n, k, entry.pid,
                                           split_k);
      consider_plan(best, pattern_for_pid_split(entry.pid, split_k), entry.pid,
                    split_k, 0, cost, margin);
      considered = true;
      if (matched_decode_selector != nullptr) {
        *matched_decode_selector = true;
      }
    }
  }

  for (const auto &entry : kFp16RccGeneratedPlanEntries) {
    if (entry.m == m && entry.n == n && entry.k == k) {
      if (entry.requires_ablation_env &&
          !env_flag("MOONPOLY_ENABLE_QWEN_P286_RCC_ABLATION", false)) {
        continue;
      }
      int split_k = std::max(1, entry.split_k);
      if (split_k > 1 &&
          !pattern3_k_gate(k, split_k, pattern3_min_tile_k_for_pid(entry.pid))) {
        continue;
      }
      float cost = entry.estimated_ms > 0.0f
                       ? entry.estimated_ms
                       : estimate_pid_split_cost(feature_m, feature_n, k,
                                                 entry.pid, split_k);
      consider_plan(best, pattern_for_pid_split(entry.pid, split_k), entry.pid,
                    split_k, 0, cost, margin);
      considered = true;
    }
  }

  return considered;
}

Fp16OnlinePlan select_online_plan(int m, int n, int k) {
  constexpr int kNumPrimaryPredict = 40;
  Fp16OnlinePlan best{Fp16PolyPattern::Single, 0, 1, 0,
                      std::numeric_limits<float>::infinity()};

  bool matched_decode_selector = false;
  const bool generated_considered =
      consider_generated_selector_plans(m, n, k, best, &matched_decode_selector);
  const bool strict_decode_selector =
      matched_decode_selector &&
      env_flag("MOONPOLY_LLM_DECODE_SELECTOR_STRICT", true);
  const bool strict_generated_selector =
      generated_considered &&
      env_flag("MOONPOLY_GENERATED_SELECTOR_STRICT", false);
  if (strict_decode_selector || strict_generated_selector) {
    if (std::getenv("MOONPOLY_DEBUG_ONLINE_SELECTOR")) {
      std::cout << "[moonpoly][fp16] online m=" << m << " n=" << n
                << " k=" << k << " -> " << pattern_name(best.pattern)
                << (strict_decode_selector ? " strict_decode_selector"
                                           : " strict_generated_selector")
                << " pid=" << best.pid
                << " split_k=" << best.split_k
                << " split_n=" << best.split_n
                << " est_ms=" << best.estimated_ms << std::endl;
    }
    return best;
  }

  // The inherited predictor convention uses (N, M, K) for RCC kernels.
  int feature_m = n;
  int feature_n = m;
  for (int pid = 0; pid < kNumPrimaryPredict; ++pid) {
    if (k % gemm_alignment[pid] != 0) {
      continue;
    }
    float cost = gemm_cost_model(feature_m, feature_n, k, pid);
    consider_plan(best, Fp16PolyPattern::Single, pid, 1, 0, cost, 1.0f);
  }
  if (!std::isfinite(best.estimated_ms)) {
    best.pid = cutlass_gemm_predict(feature_m, feature_n, k);
    best.estimated_ms = gemm_cost_model(feature_m, feature_n, k, best.pid);
  }

  if (env_flag("MOONPOLY_ENABLE_PATTERN3", true)) {
    static constexpr int kSplitPids[] = {
        4,  6,  7,  12, 13, 15, 19, 20, 23,
        24, 25, 29, 31, 32, 33, 37, 39};
    static constexpr int kSplitFactors[] = {2, 4, 8};
    float margin = env_float("MOONPOLY_PATTERN3_MARGIN", 0.97f);
    for (int pid : kSplitPids) {
      if (k % gemm_alignment[pid] != 0) {
        continue;
      }
      for (int split_k : kSplitFactors) {
        if (!pattern3_k_gate(k, split_k, gemm_shared_mem[pid].pk)) {
          continue;
        }
        float cost =
            gemm_splitk_cost_model(feature_m, feature_n, k, pid, split_k);
        consider_plan(best, Fp16PolyPattern::SplitK, pid, split_k, 0, cost,
                      margin);
      }
    }
  }

  int pattern2_split_n = 0;
  if (pattern2_tail_gate(m, n, k, &pattern2_split_n)) {
    float margin = env_float("MOONPOLY_PATTERN2_MARGIN", 0.90f);
    float cost = estimate_pattern2_cost(m, n, k, pattern2_split_n);
    consider_plan(best, Fp16PolyPattern::Combine, -1, 1, pattern2_split_n,
                  cost, margin);
  }

  if (std::getenv("MOONPOLY_DEBUG_ONLINE_SELECTOR")) {
    std::cout << "[moonpoly][fp16] online m=" << m << " n=" << n
              << " k=" << k << " -> " << pattern_name(best.pattern)
              << " pid=" << best.pid << " split_k=" << best.split_k
              << " split_n=" << best.split_n
              << " est_ms=" << best.estimated_ms << std::endl;
  }

  return best;
}
/*
   ============================ CUTLASS GEMM PRIMARY ===========================
*/

struct CUTLASS_GEMM_P0 {
private:
  CUTLASS_GEMM_P0() {}
  ~CUTLASS_GEMM_P0() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 96, 64>,
      cutlass::gemm::GemmShape<16, 48, 64>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>, 5, 8, 8,
      false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P0(const CUTLASS_GEMM_P0 &) = delete;
  CUTLASS_GEMM_P0 &operator=(const CUTLASS_GEMM_P0 &) = delete;
  static CUTLASS_GEMM_P0 &get_instance() {
    static CUTLASS_GEMM_P0 instance;
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

void cutlass_gemm_P0(int length_m, int length_n, int length_k,
                     cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                     float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P0 &instance = CUTLASS_GEMM_P0::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_GEMM_P1 {
private:
  CUTLASS_GEMM_P1() {}
  ~CUTLASS_GEMM_P1() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 128, 32>,
      cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>, 5, 8, 8,
      false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P1(const CUTLASS_GEMM_P1 &) = delete;
  CUTLASS_GEMM_P1 &operator=(const CUTLASS_GEMM_P1 &) = delete;
  static CUTLASS_GEMM_P1 &get_instance() {
    static CUTLASS_GEMM_P1 instance;
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

void cutlass_gemm_P1(int length_m, int length_n, int length_k,
                     cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                     float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P1 &instance = CUTLASS_GEMM_P1::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_GEMM_P2 {
private:
  CUTLASS_GEMM_P2() {}
  ~CUTLASS_GEMM_P2() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<64, 128, 64>,
      cutlass::gemm::GemmShape<64, 32, 64>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 4, 8, 8,
      false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P2(const CUTLASS_GEMM_P2 &) = delete;
  CUTLASS_GEMM_P2 &operator=(const CUTLASS_GEMM_P2 &) = delete;
  static CUTLASS_GEMM_P2 &get_instance() {
    static CUTLASS_GEMM_P2 instance;
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

void cutlass_gemm_P2(int length_m, int length_n, int length_k,
                     cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                     float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P2 &instance = CUTLASS_GEMM_P2::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_GEMM_P3 {
private:
  CUTLASS_GEMM_P3() {}
  ~CUTLASS_GEMM_P3() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<64, 64, 64>,
      cutlass::gemm::GemmShape<32, 32, 64>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 6, 8, 8,
      false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P3(const CUTLASS_GEMM_P3 &) = delete;
  CUTLASS_GEMM_P3 &operator=(const CUTLASS_GEMM_P3 &) = delete;
  static CUTLASS_GEMM_P3 &get_instance() {
    static CUTLASS_GEMM_P3 instance;
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

void cutlass_gemm_P3(int length_m, int length_n, int length_k,
                     cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                     float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P3 &instance = CUTLASS_GEMM_P3::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_GEMM_P4 {
private:
  CUTLASS_GEMM_P4() {}
  ~CUTLASS_GEMM_P4() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 32, 64>,
      cutlass::gemm::GemmShape<16, 32, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>, 6, 8, 8,
      true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P4(const CUTLASS_GEMM_P4 &) = delete;
  CUTLASS_GEMM_P4 &operator=(const CUTLASS_GEMM_P4 &) = delete;
  static CUTLASS_GEMM_P4 &get_instance() {
    static CUTLASS_GEMM_P4 instance;
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

void cutlass_gemm_P4(int length_m, int length_n, int length_k,
                     cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                     float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P4 &instance = CUTLASS_GEMM_P4::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_GEMM_P5 {
private:
  CUTLASS_GEMM_P5() {}
  ~CUTLASS_GEMM_P5() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<32, 16, 32>,
      cutlass::gemm::GemmShape<16, 16, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 6, 8, 8,
      false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P5(const CUTLASS_GEMM_P5 &) = delete;
  CUTLASS_GEMM_P5 &operator=(const CUTLASS_GEMM_P5 &) = delete;
  static CUTLASS_GEMM_P5 &get_instance() {
    static CUTLASS_GEMM_P5 instance;
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

void cutlass_gemm_P5(int length_m, int length_n, int length_k,
                     cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                     float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P5 &instance = CUTLASS_GEMM_P5::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_GEMM_P6 {
private:
  CUTLASS_GEMM_P6() {}
  ~CUTLASS_GEMM_P6() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 16, 64>,
      cutlass::gemm::GemmShape<16, 16, 64>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, 10, 8, 8,
      true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P6(const CUTLASS_GEMM_P6 &) = delete;
  CUTLASS_GEMM_P6 &operator=(const CUTLASS_GEMM_P6 &) = delete;
  static CUTLASS_GEMM_P6 &get_instance() {
    static CUTLASS_GEMM_P6 instance;
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

void cutlass_gemm_P6(int length_m, int length_n, int length_k,
                     cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                     float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P6 &instance = CUTLASS_GEMM_P6::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_GEMM_P7 {
private:
  CUTLASS_GEMM_P7() {}
  ~CUTLASS_GEMM_P7() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm75, cutlass::gemm::GemmShape<128, 256, 64>,
      cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 8>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, 2, 2, 2,
      true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P7(const CUTLASS_GEMM_P7 &) = delete;
  CUTLASS_GEMM_P7 &operator=(const CUTLASS_GEMM_P7 &) = delete;
  static CUTLASS_GEMM_P7 &get_instance() {
    static CUTLASS_GEMM_P7 instance;
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

void cutlass_gemm_P7(int length_m, int length_n, int length_k,
                     cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                     float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P7 &instance = CUTLASS_GEMM_P7::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_GEMM_P8 {
private:
  CUTLASS_GEMM_P8() {}
  ~CUTLASS_GEMM_P8() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<64, 8, 64>,
      cutlass::gemm::GemmShape<32, 8, 64>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, 4, 8, 8,
      false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P8(const CUTLASS_GEMM_P8 &) = delete;
  CUTLASS_GEMM_P8 &operator=(const CUTLASS_GEMM_P8 &) = delete;
  static CUTLASS_GEMM_P8 &get_instance() {
    static CUTLASS_GEMM_P8 instance;
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

void cutlass_gemm_P8(int length_m, int length_n, int length_k,
                     cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                     float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P8 &instance = CUTLASS_GEMM_P8::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_GEMM_P9 {
private:
  CUTLASS_GEMM_P9() {}
  ~CUTLASS_GEMM_P9() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm75, cutlass::gemm::GemmShape<64, 16, 64>,
      cutlass::gemm::GemmShape<16, 16, 64>, cutlass::gemm::GemmShape<16, 8, 8>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, 2, 2, 2,
      false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P9(const CUTLASS_GEMM_P9 &) = delete;
  CUTLASS_GEMM_P9 &operator=(const CUTLASS_GEMM_P9 &) = delete;
  static CUTLASS_GEMM_P9 &get_instance() {
    static CUTLASS_GEMM_P9 instance;
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

void cutlass_gemm_P9(int length_m, int length_n, int length_k,
                     cutlass::half_t *A, cutlass::half_t *B, cutlass::half_t *C,
                     float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P9 &instance = CUTLASS_GEMM_P9::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_GEMM_P10 {
private:
  CUTLASS_GEMM_P10() {}
  ~CUTLASS_GEMM_P10() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<32, 64, 64>,
      cutlass::gemm::GemmShape<16, 32, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, 2, 1, 1,
      false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P10(const CUTLASS_GEMM_P10 &) = delete;
  CUTLASS_GEMM_P10 &operator=(const CUTLASS_GEMM_P10 &) = delete;
  static CUTLASS_GEMM_P10 &get_instance() {
    static CUTLASS_GEMM_P10 instance;
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

void cutlass_gemm_P10(int length_m, int length_n, int length_k,
                      cutlass::half_t *A, cutlass::half_t *B,
                      cutlass::half_t *C, float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P10 &instance = CUTLASS_GEMM_P10::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_GEMM_P11 {
private:
  CUTLASS_GEMM_P11() {}
  ~CUTLASS_GEMM_P11() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm75, cutlass::gemm::GemmShape<64, 96, 64>,
      cutlass::gemm::GemmShape<32, 96, 32>, cutlass::gemm::GemmShape<16, 8, 8>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>, 2, 2, 2,
      false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P11(const CUTLASS_GEMM_P11 &) = delete;
  CUTLASS_GEMM_P11 &operator=(const CUTLASS_GEMM_P11 &) = delete;
  static CUTLASS_GEMM_P11 &get_instance() {
    static CUTLASS_GEMM_P11 instance;
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

void cutlass_gemm_P11(int length_m, int length_n, int length_k,
                      cutlass::half_t *A, cutlass::half_t *B,
                      cutlass::half_t *C, float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P11 &instance = CUTLASS_GEMM_P11::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_GEMM_P12 {
private:
  CUTLASS_GEMM_P12() {}
  ~CUTLASS_GEMM_P12() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm75, cutlass::gemm::GemmShape<128, 64, 64>,
      cutlass::gemm::GemmShape<32, 32, 64>, cutlass::gemm::GemmShape<16, 8, 8>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>, 2, 2, 2,
      true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P12(const CUTLASS_GEMM_P12 &) = delete;
  CUTLASS_GEMM_P12 &operator=(const CUTLASS_GEMM_P12 &) = delete;
  static CUTLASS_GEMM_P12 &get_instance() {
    static CUTLASS_GEMM_P12 instance;
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

void cutlass_gemm_P12(int length_m, int length_n, int length_k,
                      cutlass::half_t *A, cutlass::half_t *B,
                      cutlass::half_t *C, float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P12 &instance = CUTLASS_GEMM_P12::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_GEMM_P13 {
private:
  CUTLASS_GEMM_P13() {}
  ~CUTLASS_GEMM_P13() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 8, 32>,
      cutlass::gemm::GemmShape<16, 8, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 5, 2, 2,
      true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P13(const CUTLASS_GEMM_P13 &) = delete;
  CUTLASS_GEMM_P13 &operator=(const CUTLASS_GEMM_P13 &) = delete;
  static CUTLASS_GEMM_P13 &get_instance() {
    static CUTLASS_GEMM_P13 instance;
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

void cutlass_gemm_P13(int length_m, int length_n, int length_k,
                      cutlass::half_t *A, cutlass::half_t *B,
                      cutlass::half_t *C, float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P13 &instance = CUTLASS_GEMM_P13::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_GEMM_P14 {
private:
  CUTLASS_GEMM_P14() {}
  ~CUTLASS_GEMM_P14() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<96, 160, 32>,
      cutlass::gemm::GemmShape<48, 80, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>, 3, 8, 8,
      false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P14(const CUTLASS_GEMM_P14 &) = delete;
  CUTLASS_GEMM_P14 &operator=(const CUTLASS_GEMM_P14 &) = delete;
  static CUTLASS_GEMM_P14 &get_instance() {
    static CUTLASS_GEMM_P14 instance;
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

void cutlass_gemm_P14(int length_m, int length_n, int length_k,
                      cutlass::half_t *A, cutlass::half_t *B,
                      cutlass::half_t *C, float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P14 &instance = CUTLASS_GEMM_P14::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_GEMM_P15 {
private:
  CUTLASS_GEMM_P15() {}
  ~CUTLASS_GEMM_P15() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<80, 128, 64>,
      cutlass::gemm::GemmShape<80, 32, 64>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>, 3, 8, 8,
      true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P15(const CUTLASS_GEMM_P15 &) = delete;
  CUTLASS_GEMM_P15 &operator=(const CUTLASS_GEMM_P15 &) = delete;
  static CUTLASS_GEMM_P15 &get_instance() {
    static CUTLASS_GEMM_P15 instance;
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

void cutlass_gemm_P15(int length_m, int length_n, int length_k,
                      cutlass::half_t *A, cutlass::half_t *B,
                      cutlass::half_t *C, float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P15 &instance = CUTLASS_GEMM_P15::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_GEMM_P16 {
private:
  CUTLASS_GEMM_P16() {}
  ~CUTLASS_GEMM_P16() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<48, 32, 64>,
      cutlass::gemm::GemmShape<48, 16, 64>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>, 6, 8, 8,
      false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P16(const CUTLASS_GEMM_P16 &) = delete;
  CUTLASS_GEMM_P16 &operator=(const CUTLASS_GEMM_P16 &) = delete;
  static CUTLASS_GEMM_P16 &get_instance() {
    static CUTLASS_GEMM_P16 instance;
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

void cutlass_gemm_P16(int length_m, int length_n, int length_k,
                      cutlass::half_t *A, cutlass::half_t *B,
                      cutlass::half_t *C, float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P16 &instance = CUTLASS_GEMM_P16::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_GEMM_P17 {
private:
  CUTLASS_GEMM_P17() {}
  ~CUTLASS_GEMM_P17() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<256, 32, 64>,
      cutlass::gemm::GemmShape<64, 32, 64>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>, 3, 8, 8,
      false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P17(const CUTLASS_GEMM_P17 &) = delete;
  CUTLASS_GEMM_P17 &operator=(const CUTLASS_GEMM_P17 &) = delete;
  static CUTLASS_GEMM_P17 &get_instance() {
    static CUTLASS_GEMM_P17 instance;
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

void cutlass_gemm_P17(int length_m, int length_n, int length_k,
                      cutlass::half_t *A, cutlass::half_t *B,
                      cutlass::half_t *C, float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P17 &instance = CUTLASS_GEMM_P17::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_GEMM_P18 {
private:
  CUTLASS_GEMM_P18() {}
  ~CUTLASS_GEMM_P18() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<32, 128, 32>,
      cutlass::gemm::GemmShape<16, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, 5, 8, 8,
      false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P18(const CUTLASS_GEMM_P18 &) = delete;
  CUTLASS_GEMM_P18 &operator=(const CUTLASS_GEMM_P18 &) = delete;
  static CUTLASS_GEMM_P18 &get_instance() {
    static CUTLASS_GEMM_P18 instance;
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

void cutlass_gemm_P18(int length_m, int length_n, int length_k,
                      cutlass::half_t *A, cutlass::half_t *B,
                      cutlass::half_t *C, float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P18 &instance = CUTLASS_GEMM_P18::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_GEMM_P19 {
private:
  CUTLASS_GEMM_P19() {}
  ~CUTLASS_GEMM_P19() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 128, 64>,
      cutlass::gemm::GemmShape<16, 32, 64>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 4, 8, 8,
      true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P19(const CUTLASS_GEMM_P19 &) = delete;
  CUTLASS_GEMM_P19 &operator=(const CUTLASS_GEMM_P19 &) = delete;
  static CUTLASS_GEMM_P19 &get_instance() {
    static CUTLASS_GEMM_P19 instance;
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

void cutlass_gemm_P19(int length_m, int length_n, int length_k,
                      cutlass::half_t *A, cutlass::half_t *B,
                      cutlass::half_t *C, float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P19 &instance = CUTLASS_GEMM_P19::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_GEMM_P20 {
private:
  CUTLASS_GEMM_P20() {}
  ~CUTLASS_GEMM_P20() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<80, 96, 64>,
      cutlass::gemm::GemmShape<80, 48, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>, 6, 8, 8,
      true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P20(const CUTLASS_GEMM_P20 &) = delete;
  CUTLASS_GEMM_P20 &operator=(const CUTLASS_GEMM_P20 &) = delete;
  static CUTLASS_GEMM_P20 &get_instance() {
    static CUTLASS_GEMM_P20 instance;
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

void cutlass_gemm_P20(int length_m, int length_n, int length_k,
                      cutlass::half_t *A, cutlass::half_t *B,
                      cutlass::half_t *C, float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P20 &instance = CUTLASS_GEMM_P20::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_GEMM_P21 {
private:
  CUTLASS_GEMM_P21() {}
  ~CUTLASS_GEMM_P21() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm75, cutlass::gemm::GemmShape<16, 64, 16>,
      cutlass::gemm::GemmShape<16, 64, 16>, cutlass::gemm::GemmShape<16, 8, 8>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>, 2, 4, 4,
      false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P21(const CUTLASS_GEMM_P21 &) = delete;
  CUTLASS_GEMM_P21 &operator=(const CUTLASS_GEMM_P21 &) = delete;
  static CUTLASS_GEMM_P21 &get_instance() {
    static CUTLASS_GEMM_P21 instance;
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

void cutlass_gemm_P21(int length_m, int length_n, int length_k,
                      cutlass::half_t *A, cutlass::half_t *B,
                      cutlass::half_t *C, float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P21 &instance = CUTLASS_GEMM_P21::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_GEMM_P22 {
private:
  CUTLASS_GEMM_P22() {}
  ~CUTLASS_GEMM_P22() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 96, 32>,
      cutlass::gemm::GemmShape<32, 96, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>, 3, 8, 8,
      false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P22(const CUTLASS_GEMM_P22 &) = delete;
  CUTLASS_GEMM_P22 &operator=(const CUTLASS_GEMM_P22 &) = delete;
  static CUTLASS_GEMM_P22 &get_instance() {
    static CUTLASS_GEMM_P22 instance;
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

void cutlass_gemm_P22(int length_m, int length_n, int length_k,
                      cutlass::half_t *A, cutlass::half_t *B,
                      cutlass::half_t *C, float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P22 &instance = CUTLASS_GEMM_P22::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_GEMM_P23 {
private:
  CUTLASS_GEMM_P23() {}
  ~CUTLASS_GEMM_P23() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<256, 128, 32>,
      cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 3, 8, 8,
      true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P23(const CUTLASS_GEMM_P23 &) = delete;
  CUTLASS_GEMM_P23 &operator=(const CUTLASS_GEMM_P23 &) = delete;
  static CUTLASS_GEMM_P23 &get_instance() {
    static CUTLASS_GEMM_P23 instance;
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

void cutlass_gemm_P23(int length_m, int length_n, int length_k,
                      cutlass::half_t *A, cutlass::half_t *B,
                      cutlass::half_t *C, float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P23 &instance = CUTLASS_GEMM_P23::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_GEMM_P24 {
private:
  CUTLASS_GEMM_P24() {}
  ~CUTLASS_GEMM_P24() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm75, cutlass::gemm::GemmShape<32, 16, 64>,
      cutlass::gemm::GemmShape<8, 16, 64>, cutlass::gemm::GemmShape<16, 8, 8>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, 2, 4, 4,
      true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P24(const CUTLASS_GEMM_P24 &) = delete;
  CUTLASS_GEMM_P24 &operator=(const CUTLASS_GEMM_P24 &) = delete;
  static CUTLASS_GEMM_P24 &get_instance() {
    static CUTLASS_GEMM_P24 instance;
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

void cutlass_gemm_P24(int length_m, int length_n, int length_k,
                      cutlass::half_t *A, cutlass::half_t *B,
                      cutlass::half_t *C, float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P24 &instance = CUTLASS_GEMM_P24::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_GEMM_P25 {
private:
  CUTLASS_GEMM_P25() {}
  ~CUTLASS_GEMM_P25() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 8, 64>,
      cutlass::gemm::GemmShape<16, 8, 64>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 5, 8, 8,
      true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P25(const CUTLASS_GEMM_P25 &) = delete;
  CUTLASS_GEMM_P25 &operator=(const CUTLASS_GEMM_P25 &) = delete;
  static CUTLASS_GEMM_P25 &get_instance() {
    static CUTLASS_GEMM_P25 instance;
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

void cutlass_gemm_P25(int length_m, int length_n, int length_k,
                      cutlass::half_t *A, cutlass::half_t *B,
                      cutlass::half_t *C, float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P25 &instance = CUTLASS_GEMM_P25::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_GEMM_P26 {
private:
  CUTLASS_GEMM_P26() {}
  ~CUTLASS_GEMM_P26() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 64, 64>,
      cutlass::gemm::GemmShape<16, 32, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, 3, 4, 4,
      false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P26(const CUTLASS_GEMM_P26 &) = delete;
  CUTLASS_GEMM_P26 &operator=(const CUTLASS_GEMM_P26 &) = delete;
  static CUTLASS_GEMM_P26 &get_instance() {
    static CUTLASS_GEMM_P26 instance;
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

void cutlass_gemm_P26(int length_m, int length_n, int length_k,
                      cutlass::half_t *A, cutlass::half_t *B,
                      cutlass::half_t *C, float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P26 &instance = CUTLASS_GEMM_P26::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_GEMM_P27 {
private:
  CUTLASS_GEMM_P27() {}
  ~CUTLASS_GEMM_P27() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm75, cutlass::gemm::GemmShape<64, 64, 16>,
      cutlass::gemm::GemmShape<32, 32, 16>, cutlass::gemm::GemmShape<16, 8, 8>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>, 2, 4, 4,
      false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P27(const CUTLASS_GEMM_P27 &) = delete;
  CUTLASS_GEMM_P27 &operator=(const CUTLASS_GEMM_P27 &) = delete;
  static CUTLASS_GEMM_P27 &get_instance() {
    static CUTLASS_GEMM_P27 instance;
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

void cutlass_gemm_P27(int length_m, int length_n, int length_k,
                      cutlass::half_t *A, cutlass::half_t *B,
                      cutlass::half_t *C, float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P27 &instance = CUTLASS_GEMM_P27::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_GEMM_P28 {
private:
  CUTLASS_GEMM_P28() {}
  ~CUTLASS_GEMM_P28() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 32, 32>,
      cutlass::gemm::GemmShape<16, 16, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, 4, 8, 8,
      false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P28(const CUTLASS_GEMM_P28 &) = delete;
  CUTLASS_GEMM_P28 &operator=(const CUTLASS_GEMM_P28 &) = delete;
  static CUTLASS_GEMM_P28 &get_instance() {
    static CUTLASS_GEMM_P28 instance;
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

void cutlass_gemm_P28(int length_m, int length_n, int length_k,
                      cutlass::half_t *A, cutlass::half_t *B,
                      cutlass::half_t *C, float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P28 &instance = CUTLASS_GEMM_P28::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_GEMM_P29 {
private:
  CUTLASS_GEMM_P29() {}
  ~CUTLASS_GEMM_P29() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm75, cutlass::gemm::GemmShape<8, 16, 64>,
      cutlass::gemm::GemmShape<8, 16, 32>, cutlass::gemm::GemmShape<16, 8, 8>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>, 2, 1, 1,
      true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P29(const CUTLASS_GEMM_P29 &) = delete;
  CUTLASS_GEMM_P29 &operator=(const CUTLASS_GEMM_P29 &) = delete;
  static CUTLASS_GEMM_P29 &get_instance() {
    static CUTLASS_GEMM_P29 instance;
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

void cutlass_gemm_P29(int length_m, int length_n, int length_k,
                      cutlass::half_t *A, cutlass::half_t *B,
                      cutlass::half_t *C, float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P29 &instance = CUTLASS_GEMM_P29::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_GEMM_P30 {
private:
  CUTLASS_GEMM_P30() {}
  ~CUTLASS_GEMM_P30() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<192, 192, 32>,
      cutlass::gemm::GemmShape<96, 48, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>, 5, 8, 8,
      false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P30(const CUTLASS_GEMM_P30 &) = delete;
  CUTLASS_GEMM_P30 &operator=(const CUTLASS_GEMM_P30 &) = delete;
  static CUTLASS_GEMM_P30 &get_instance() {
    static CUTLASS_GEMM_P30 instance;
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

void cutlass_gemm_P30(int length_m, int length_n, int length_k,
                      cutlass::half_t *A, cutlass::half_t *B,
                      cutlass::half_t *C, float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P30 &instance = CUTLASS_GEMM_P30::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_GEMM_P31 {
private:
  CUTLASS_GEMM_P31() {}
  ~CUTLASS_GEMM_P31() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 192, 64>,
      cutlass::gemm::GemmShape<64, 48, 64>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>, 3, 8, 8,
      true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P31(const CUTLASS_GEMM_P31 &) = delete;
  CUTLASS_GEMM_P31 &operator=(const CUTLASS_GEMM_P31 &) = delete;
  static CUTLASS_GEMM_P31 &get_instance() {
    static CUTLASS_GEMM_P31 instance;
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

void cutlass_gemm_P31(int length_m, int length_n, int length_k,
                      cutlass::half_t *A, cutlass::half_t *B,
                      cutlass::half_t *C, float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P31 &instance = CUTLASS_GEMM_P31::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_GEMM_P32 {
private:
  CUTLASS_GEMM_P32() {}
  ~CUTLASS_GEMM_P32() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm75, cutlass::gemm::GemmShape<32, 64, 16>,
      cutlass::gemm::GemmShape<32, 32, 16>, cutlass::gemm::GemmShape<16, 8, 8>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>, 2, 2, 2,
      true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P32(const CUTLASS_GEMM_P32 &) = delete;
  CUTLASS_GEMM_P32 &operator=(const CUTLASS_GEMM_P32 &) = delete;
  static CUTLASS_GEMM_P32 &get_instance() {
    static CUTLASS_GEMM_P32 instance;
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

void cutlass_gemm_P32(int length_m, int length_n, int length_k,
                      cutlass::half_t *A, cutlass::half_t *B,
                      cutlass::half_t *C, float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P32 &instance = CUTLASS_GEMM_P32::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_GEMM_P33 {
private:
  CUTLASS_GEMM_P33() {}
  ~CUTLASS_GEMM_P33() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<96, 192, 64>,
      cutlass::gemm::GemmShape<48, 48, 64>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>, 2, 1, 1,
      true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P33(const CUTLASS_GEMM_P33 &) = delete;
  CUTLASS_GEMM_P33 &operator=(const CUTLASS_GEMM_P33 &) = delete;
  static CUTLASS_GEMM_P33 &get_instance() {
    static CUTLASS_GEMM_P33 instance;
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

void cutlass_gemm_P33(int length_m, int length_n, int length_k,
                      cutlass::half_t *A, cutlass::half_t *B,
                      cutlass::half_t *C, float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P33 &instance = CUTLASS_GEMM_P33::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_GEMM_P34 {
private:
  CUTLASS_GEMM_P34() {}
  ~CUTLASS_GEMM_P34() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<64, 96, 32>,
      cutlass::gemm::GemmShape<32, 48, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>, 5, 8, 8,
      false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P34(const CUTLASS_GEMM_P34 &) = delete;
  CUTLASS_GEMM_P34 &operator=(const CUTLASS_GEMM_P34 &) = delete;
  static CUTLASS_GEMM_P34 &get_instance() {
    static CUTLASS_GEMM_P34 instance;
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

void cutlass_gemm_P34(int length_m, int length_n, int length_k,
                      cutlass::half_t *A, cutlass::half_t *B,
                      cutlass::half_t *C, float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P34 &instance = CUTLASS_GEMM_P34::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_GEMM_P35 {
private:
  CUTLASS_GEMM_P35() {}
  ~CUTLASS_GEMM_P35() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm75, cutlass::gemm::GemmShape<8, 8, 64>,
      cutlass::gemm::GemmShape<8, 8, 32>, cutlass::gemm::GemmShape<16, 8, 8>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, 2, 2, 2,
      false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P35(const CUTLASS_GEMM_P35 &) = delete;
  CUTLASS_GEMM_P35 &operator=(const CUTLASS_GEMM_P35 &) = delete;
  static CUTLASS_GEMM_P35 &get_instance() {
    static CUTLASS_GEMM_P35 instance;
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

void cutlass_gemm_P35(int length_m, int length_n, int length_k,
                      cutlass::half_t *A, cutlass::half_t *B,
                      cutlass::half_t *C, float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P35 &instance = CUTLASS_GEMM_P35::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_GEMM_P36 {
private:
  CUTLASS_GEMM_P36() {}
  ~CUTLASS_GEMM_P36() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm75, cutlass::gemm::GemmShape<32, 32, 32>,
      cutlass::gemm::GemmShape<8, 32, 32>, cutlass::gemm::GemmShape<16, 8, 8>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 2, 2, 2,
      false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P36(const CUTLASS_GEMM_P36 &) = delete;
  CUTLASS_GEMM_P36 &operator=(const CUTLASS_GEMM_P36 &) = delete;
  static CUTLASS_GEMM_P36 &get_instance() {
    static CUTLASS_GEMM_P36 instance;
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

void cutlass_gemm_P36(int length_m, int length_n, int length_k,
                      cutlass::half_t *A, cutlass::half_t *B,
                      cutlass::half_t *C, float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P36 &instance = CUTLASS_GEMM_P36::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_GEMM_P37 {
private:
  CUTLASS_GEMM_P37() {}
  ~CUTLASS_GEMM_P37() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<64, 32, 64>,
      cutlass::gemm::GemmShape<32, 32, 64>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>, 4, 8, 8,
      true>;
  Gemm gemm_op;
  CUTLASS_GEMM_P37(const CUTLASS_GEMM_P37 &) = delete;
  CUTLASS_GEMM_P37 &operator=(const CUTLASS_GEMM_P37 &) = delete;
  static CUTLASS_GEMM_P37 &get_instance() {
    static CUTLASS_GEMM_P37 instance;
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

void cutlass_gemm_P37(int length_m, int length_n, int length_k,
                      cutlass::half_t *A, cutlass::half_t *B,
                      cutlass::half_t *C, float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P37 &instance = CUTLASS_GEMM_P37::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_GEMM_P38 {
private:
  CUTLASS_GEMM_P38() {}
  ~CUTLASS_GEMM_P38() {}

public:
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm75, cutlass::gemm::GemmShape<16, 32, 16>,
      cutlass::gemm::GemmShape<16, 32, 16>, cutlass::gemm::GemmShape<16, 8, 8>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 2, 1, 1,
      false>;
  Gemm gemm_op;
  CUTLASS_GEMM_P38(const CUTLASS_GEMM_P38 &) = delete;
  CUTLASS_GEMM_P38 &operator=(const CUTLASS_GEMM_P38 &) = delete;
  static CUTLASS_GEMM_P38 &get_instance() {
    static CUTLASS_GEMM_P38 instance;
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

void cutlass_gemm_P38(int length_m, int length_n, int length_k,
                      cutlass::half_t *A, cutlass::half_t *B,
                      cutlass::half_t *C, float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P38 &instance = CUTLASS_GEMM_P38::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_GEMM_P39 {
private:
  CUTLASS_GEMM_P39() {}
  ~CUTLASS_GEMM_P39() {}

public:
  using ElementAccumulator = float;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = cutlass::half_t;
  using ElementInputB = cutlass::half_t;
  using ElementOutput = cutlass::half_t;

  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;

  using MMAOp = cutlass::arch::OpClassTensorOp;

  using SmArch = cutlass::arch::Sm80;

  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32, 32, 64>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<16, 32, 32>;

  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput, 1, ElementAccumulator, ElementComputeEpilogue>;

  using ConvertOp =
      cutlass::epilogue::thread::Convert<ElementAccumulator, EpilogueOp::kCount,
                                         ElementAccumulator>;

  using ReductionOp = cutlass::reduction::thread::ReduceAdd<
      ElementAccumulator, EpilogueOp::ElementAccumulator, EpilogueOp::kCount>;

  using Gemm = cutlass::gemm::device::GemmSplitKParallel<
      ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput,
      LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock,
      ShapeMMAWarp, ShapeMMAOp, EpilogueOp, ConvertOp, ReductionOp,
      cutlass::gemm::threadblock::GemmSplitKHorizontalThreadblockSwizzle, 2, 8,
      8>;

  Gemm gemm_op;
  CUTLASS_GEMM_P39(const CUTLASS_GEMM_P39 &) = delete;
  CUTLASS_GEMM_P39 &operator=(const CUTLASS_GEMM_P39 &) = delete;
  static CUTLASS_GEMM_P39 &get_instance() {
    static CUTLASS_GEMM_P39 instance;
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
                                       16};
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);
    status = gemm_op(stream);
    CUTLASS_CHECK(status);
  }
};

void cutlass_gemm_P39(int length_m, int length_n, int length_k,
                      cutlass::half_t *A, cutlass::half_t *B,
                      cutlass::half_t *C, float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P39 &instance = CUTLASS_GEMM_P39::get_instance();
  instance.run_gemm(length_m, length_n, length_k, A, B, C, alpha, beta, stream);
}

struct CUTLASS_GEMM_P40 {
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

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 24, 32>;
  using WarpShape = cutlass::gemm::GemmShape<32, 24, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  static constexpr int NumStages = 10;
  static constexpr int AlignmentA = 8;
  static constexpr int AlignmentB = 8;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementC, 8, ElementAccumulator, ElementAccumulator>;

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

  static void run_gemm_with_external_workspace(
      int m, int n, int k, int split_k_slices, ElementA *ptr_A, ElementB *ptr_B,
      ElementC *ptr_C, float alpha, float beta, void *workspace_ptr,
      size_t workspace_size_bytes, cudaStream_t stream) {
    Gemm gemm_op_local;
    auto arguments = make_arguments(
        m, n, k, split_k_slices, ptr_A, ptr_B, ptr_C, alpha, beta);

    size_t required_workspace = Gemm::get_workspace_size(arguments);
    if (required_workspace > 0) {
      if (workspace_ptr == nullptr) {
        throw std::runtime_error(
            "P40 streamK requires non-null workspace pointer");
      }
      if (workspace_size_bytes < required_workspace) {
        throw std::runtime_error(
            "Provided workspace is too small for P40 streamK");
      }
    }

    cutlass::Status status =
        gemm_op_local.initialize(arguments, workspace_ptr, stream);
    CUTLASS_CHECK(status);
    status = gemm_op_local.run(stream);
    CUTLASS_CHECK(status);
  }

private:
  Gemm gemm_op;
  void *workspace_ptr;
  // cutlass::device_memory::allocation<uint8_t> workspace;
  bool gemm_initialized;
  static std::unordered_set<const void*> weight_cache;
  size_t max_workspace_size;

  int cached_m, cached_n, cached_k, cached_split_k_slices;
  ElementA *cached_ptr_A;
  ElementB *cached_ptr_B;
  ElementC *cached_ptr_C;
  float cached_alpha, cached_beta;

  CUTLASS_GEMM_P40()
      : gemm_initialized(false), workspace_ptr(nullptr), cached_m(0),
        cached_n(0), cached_k(0), cached_split_k_slices(0),
        cached_ptr_A(nullptr), cached_ptr_B(nullptr), cached_ptr_C(nullptr),
        cached_alpha(0.0f), cached_beta(0.0f) {
    preallocate_workspace();
  }
  ~CUTLASS_GEMM_P40() {
    // 不在析构函数中释放CUDA内存，避免程序退出时的问题
    // 对于单例模式，内存会在程序结束时自动释放
    // if (workspace_ptr) {
    //   cudaFree(workspace_ptr);
    //   workspace_ptr = nullptr;
    // }
  }

  void preallocate_workspace() {
    // Best-effort internal workspace for legacy API path.
    static constexpr int MAX_M = 5120;
    static constexpr int MAX_N = 1024;
    static constexpr int MAX_K = 25600;
    static constexpr int MAX_SPLIT_K_SLICES = 2;
    // 创建临时的虚拟设备指针
    void* dummy_ptr_A = nullptr;
    void* dummy_ptr_B = nullptr;
    void* dummy_ptr_C = nullptr;
    cudaMalloc(&dummy_ptr_A, sizeof(ElementA));
    cudaMalloc(&dummy_ptr_B, sizeof(ElementB));
    cudaMalloc(&dummy_ptr_C, sizeof(ElementC));

    cutlass::gemm::GemmCoord max_problem_size(MAX_M, MAX_N, MAX_K);
    typename Gemm::Arguments max_arguments(
        cutlass::gemm::GemmUniversalMode::kGemm, max_problem_size,
        MAX_SPLIT_K_SLICES,
        {ElementAccumulator(1.0f), ElementAccumulator(0.0f)},
        reinterpret_cast<ElementA*>(dummy_ptr_A),
        reinterpret_cast<ElementB*>(dummy_ptr_B),
        reinterpret_cast<ElementC*>(dummy_ptr_C),
        reinterpret_cast<ElementC*>(dummy_ptr_C),
        0, 0, 0, 0,
        LayoutA::packed({MAX_M, MAX_K}).stride(0),
        LayoutB::packed({MAX_K, MAX_N}).stride(0),
        LayoutC::packed({MAX_M, MAX_N}).stride(0),
        LayoutC::packed({MAX_M, MAX_N}).stride(0));

    max_workspace_size = Gemm::get_workspace_size(max_arguments);
    if (max_workspace_size > 0) {
      cudaError_t error = cudaMalloc(reinterpret_cast<void**>(&workspace_ptr), max_workspace_size);
      cudaMemset(workspace_ptr, 0, max_workspace_size);
      if (error != cudaSuccess) {
        std::cerr << "[ERROR] cudaMalloc failed: " << cudaGetErrorString(error) << std::endl;
        workspace_ptr = nullptr;
      }
    }

    // 清理临时指针
    cudaFree(dummy_ptr_A);
    cudaFree(dummy_ptr_B);
    cudaFree(dummy_ptr_C);
  }

public:
  CUTLASS_GEMM_P40(const CUTLASS_GEMM_P40 &) = delete;
  CUTLASS_GEMM_P40 &operator=(const CUTLASS_GEMM_P40 &) = delete;

  static CUTLASS_GEMM_P40 &get_instance() {
    static CUTLASS_GEMM_P40 instance;
    return instance;
  }

  void cleanup() {
    if (workspace_ptr) {
      cudaError_t error = cudaFree(workspace_ptr);
      if (error != cudaSuccess) {
        std::cerr << "[WARNING] cudaFree failed in cleanup: " << cudaGetErrorString(error) << std::endl;
      }
      workspace_ptr = nullptr;
    }
  }

  void run_gemm(int m, int n, int k, int split_k_slices,
                ElementA *ptr_A, ElementB *ptr_B, ElementC *ptr_C,
                float alpha, float beta, cudaStream_t stream) {
    cutlass::Status status;
    cutlass::gemm::GemmCoord problem_size(m, n, k);
    auto gemm_mode = cutlass::gemm::GemmUniversalMode::kGemm;
    typename Gemm::Arguments arguments(
      gemm_mode, problem_size, split_k_slices,
      {ElementAccumulator(alpha), ElementAccumulator(beta)},
      ptr_A, ptr_B, ptr_C, ptr_C,
      0, 0, 0, 0,
      LayoutA::packed({m, k}).stride(0),
      LayoutB::packed({k, n}).stride(0),
      LayoutC::packed({m, n}).stride(0),
      LayoutC::packed({m, n}).stride(0)
    );
    size_t workspace_size = gemm_op.get_workspace_size(arguments);
    status = gemm_op.initialize(arguments, workspace_ptr, stream);
    CUTLASS_CHECK(status);

    // std::cout << "[DEBUG] Entering run_gemm() with m=" << m << ", n=" << n
    //           << ", k=" << k << ", split_k_slices=" << split_k_slices
    //           << ", workspace_size=" << workspace_size << " bytes" << std::endl;
    // 检查 workspace 是否大于已经分配的
    if (workspace_size > max_workspace_size) throw std::runtime_error("Problem size exceeds pre-allocated workspace limits.");

    if (workspace_ptr == nullptr) {
      //   std::cout << "[WARNING] workspace_ptr is null, calculating required size..." << std::endl;

      //   // 为当前实际问题大小计算 workspace
      //   size_t required_workspace_size = Gemm::get_workspace_size(arguments);
      //   std::cout << "[DEBUG] Required workspace size: " << required_workspace_size << " bytes" << std::endl;
      //   if (required_workspace_size > 0) {
      //     cudaError_t error = cudaMalloc(&workspace_ptr, required_workspace_size);
      //     if (error != cudaSuccess) {
      //       std::cerr << "[ERROR] cudaMalloc failed: " << cudaGetErrorString(error) << std::endl;
      //       throw std::runtime_error("Failed to allocate workspace");
      //     }
      //     std::cout << "[DEBUG] Allocated workspace at: " << workspace_ptr << std::endl;
      //     cudaMemset(workspace_ptr, 0, required_workspace_size);
      //   }
      // }
          //   cached_m = m;
    //   cached_n = n;
    //   cached_k = k;
    //   cached_split_k_slices = split_k_slices;
    //   cached_ptr_A = ptr_A;
    //   cached_ptr_B = ptr_B;
    //   cached_ptr_C = ptr_C;
    //   cached_alpha = alpha;
    //   cached_beta = beta;
    //   gemm_initialized = true;
    }

    status = gemm_op.run(stream);
    CUTLASS_CHECK(status);
  }
};

struct CUTLASS_GEMM_P66 {
public:
  using ElementA = CUTLASS_GEMM_P40::ElementA;
  using LayoutA = CUTLASS_GEMM_P40::LayoutA;
  using ElementB = CUTLASS_GEMM_P40::ElementB;
  using LayoutB = CUTLASS_GEMM_P40::LayoutB;
  using ElementC = CUTLASS_GEMM_P40::ElementC;
  using LayoutC = CUTLASS_GEMM_P40::LayoutC;
  using ElementAccumulator = CUTLASS_GEMM_P40::ElementAccumulator;
  using ArchTag = CUTLASS_GEMM_P40::ArchTag;
  using OperatorClass = CUTLASS_GEMM_P40::OperatorClass;
  using ThreadblockSwizzle = CUTLASS_GEMM_P40::ThreadblockSwizzle;
  using ThreadblockShape = CUTLASS_GEMM_P40::ThreadblockShape;
  using WarpShape = CUTLASS_GEMM_P40::WarpShape;
  using InstructionShape = CUTLASS_GEMM_P40::InstructionShape;
  static constexpr int NumStages = CUTLASS_GEMM_P40::NumStages;
  static constexpr int AlignmentA = CUTLASS_GEMM_P40::AlignmentA;
  static constexpr int AlignmentB = CUTLASS_GEMM_P40::AlignmentB;
  using EpilogueOp = CUTLASS_GEMM_P40::EpilogueOp;
  using Gemm = CUTLASS_GEMM_P40::Gemm;

  static typename Gemm::Arguments make_arguments(
      int m, int n, int k, int split_k_slices, ElementA *ptr_A, ElementB *ptr_B,
      ElementC *ptr_C, float alpha, float beta) {
    return CUTLASS_GEMM_P40::make_arguments(
        m, n, k, split_k_slices, ptr_A, ptr_B, ptr_C, alpha, beta);
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

struct CUTLASS_GEMM_P42 {
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

struct CUTLASS_GEMM_P43 {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 16, 32>;
  using WarpShape = cutlass::gemm::GemmShape<32, 16, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  static constexpr int NumStages = 10;
  static constexpr int AlignmentA = 8;
  static constexpr int AlignmentB = 8;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementC, 8, ElementAccumulator, ElementAccumulator>;
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

struct CUTLASS_GEMM_P44 {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 32, 32>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  static constexpr int NumStages = 10;
  static constexpr int AlignmentA = 8;
  static constexpr int AlignmentB = 8;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementC, 8, ElementAccumulator, ElementAccumulator>;
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

void cutlass_gemm_P40(int length_m, int length_n, int length_k,
                      cutlass::half_t *A, cutlass::half_t *B,
                      cutlass::half_t *C, float alpha, float beta, cudaStream_t stream) {
  CUTLASS_GEMM_P40 &instance = CUTLASS_GEMM_P40::get_instance();
  const int split_k_slices = 2;
  instance.run_gemm(length_m, length_n, length_k, split_k_slices,
                    A, B, C, alpha, beta, stream);
}

void (*CT_GEMM[41])(int, int, int, cutlass::half_t *, cutlass::half_t *,
                    cutlass::half_t *, float, float, cudaStream_t) = {
    cutlass_gemm_P0,  cutlass_gemm_P1,  cutlass_gemm_P2,  cutlass_gemm_P3,
    cutlass_gemm_P4,  cutlass_gemm_P5,  cutlass_gemm_P6,  cutlass_gemm_P7,
    cutlass_gemm_P8,  cutlass_gemm_P9,  cutlass_gemm_P10, cutlass_gemm_P11,
    cutlass_gemm_P12, cutlass_gemm_P13, cutlass_gemm_P14, cutlass_gemm_P15,
    cutlass_gemm_P16, cutlass_gemm_P17, cutlass_gemm_P18, cutlass_gemm_P19,
    cutlass_gemm_P20, cutlass_gemm_P21, cutlass_gemm_P22, cutlass_gemm_P23,
    cutlass_gemm_P24, cutlass_gemm_P25, cutlass_gemm_P26, cutlass_gemm_P27,
    cutlass_gemm_P28, cutlass_gemm_P29, cutlass_gemm_P30, cutlass_gemm_P31,
    cutlass_gemm_P32, cutlass_gemm_P33, cutlass_gemm_P34, cutlass_gemm_P35,
    cutlass_gemm_P36, cutlass_gemm_P37, cutlass_gemm_P38, cutlass_gemm_P39,
    cutlass_gemm_P40};

bool is_primary_runtime_splitk_pid(int pid) {
  switch (pid) {
  case 4:
  case 6:
  case 7:
  case 12:
  case 13:
  case 15:
  case 19:
  case 20:
  case 23:
  case 24:
  case 25:
  case 29:
  case 31:
  case 32:
  case 33:
  case 37:
  case 39:
    return true;
  default:
    return false;
  }
}

template <typename Kernel>
void run_primary_runtime_splitk_kernel(int length_m, int length_n,
                                       int length_k, cutlass::half_t *A,
                                       cutlass::half_t *B, cutlass::half_t *C,
                                       float alpha, float beta,
                                       int split_k_slices,
                                       cudaStream_t stream) {
  if (split_k_slices <= 0) {
    split_k_slices = 1;
  }
  cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
  typename Kernel::Gemm::Arguments arguments{problem_size,
                                             {A, length_k},
                                             {B, length_k},
                                             {C, length_m},
                                             {C, length_m},
                                             {alpha, beta},
                                             split_k_slices};
  typename Kernel::Gemm gemm_op;
  size_t workspace_size = Kernel::Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = gemm_op(stream);
  CUTLASS_CHECK(status);
}

void run_primary_runtime_splitk_gemm(int pid, int length_m, int length_n,
                                     int length_k, cutlass::half_t *A,
                                     cutlass::half_t *B, cutlass::half_t *C,
                                     float alpha, float beta,
                                     int split_k_slices,
                                     cudaStream_t stream) {
  switch (pid) {
  case 4:
    run_primary_runtime_splitk_kernel<CUTLASS_GEMM_P4>(
        length_m, length_n, length_k, A, B, C, alpha, beta, split_k_slices,
        stream);
    return;
  case 6:
    run_primary_runtime_splitk_kernel<CUTLASS_GEMM_P6>(
        length_m, length_n, length_k, A, B, C, alpha, beta, split_k_slices,
        stream);
    return;
  case 7:
    run_primary_runtime_splitk_kernel<CUTLASS_GEMM_P7>(
        length_m, length_n, length_k, A, B, C, alpha, beta, split_k_slices,
        stream);
    return;
  case 12:
    run_primary_runtime_splitk_kernel<CUTLASS_GEMM_P12>(
        length_m, length_n, length_k, A, B, C, alpha, beta, split_k_slices,
        stream);
    return;
  case 13:
    run_primary_runtime_splitk_kernel<CUTLASS_GEMM_P13>(
        length_m, length_n, length_k, A, B, C, alpha, beta, split_k_slices,
        stream);
    return;
  case 15:
    run_primary_runtime_splitk_kernel<CUTLASS_GEMM_P15>(
        length_m, length_n, length_k, A, B, C, alpha, beta, split_k_slices,
        stream);
    return;
  case 19:
    run_primary_runtime_splitk_kernel<CUTLASS_GEMM_P19>(
        length_m, length_n, length_k, A, B, C, alpha, beta, split_k_slices,
        stream);
    return;
  case 20:
    run_primary_runtime_splitk_kernel<CUTLASS_GEMM_P20>(
        length_m, length_n, length_k, A, B, C, alpha, beta, split_k_slices,
        stream);
    return;
  case 23:
    run_primary_runtime_splitk_kernel<CUTLASS_GEMM_P23>(
        length_m, length_n, length_k, A, B, C, alpha, beta, split_k_slices,
        stream);
    return;
  case 24:
    run_primary_runtime_splitk_kernel<CUTLASS_GEMM_P24>(
        length_m, length_n, length_k, A, B, C, alpha, beta, split_k_slices,
        stream);
    return;
  case 25:
    run_primary_runtime_splitk_kernel<CUTLASS_GEMM_P25>(
        length_m, length_n, length_k, A, B, C, alpha, beta, split_k_slices,
        stream);
    return;
  case 29:
    run_primary_runtime_splitk_kernel<CUTLASS_GEMM_P29>(
        length_m, length_n, length_k, A, B, C, alpha, beta, split_k_slices,
        stream);
    return;
  case 31:
    run_primary_runtime_splitk_kernel<CUTLASS_GEMM_P31>(
        length_m, length_n, length_k, A, B, C, alpha, beta, split_k_slices,
        stream);
    return;
  case 32:
    run_primary_runtime_splitk_kernel<CUTLASS_GEMM_P32>(
        length_m, length_n, length_k, A, B, C, alpha, beta, split_k_slices,
        stream);
    return;
  case 33:
    run_primary_runtime_splitk_kernel<CUTLASS_GEMM_P33>(
        length_m, length_n, length_k, A, B, C, alpha, beta, split_k_slices,
        stream);
    return;
  case 37:
    run_primary_runtime_splitk_kernel<CUTLASS_GEMM_P37>(
        length_m, length_n, length_k, A, B, C, alpha, beta, split_k_slices,
        stream);
    return;
  case 39:
    run_primary_runtime_splitk_kernel<CUTLASS_GEMM_P39>(
        length_m, length_n, length_k, A, B, C, alpha, beta, split_k_slices,
        stream);
    return;
  default:
    CT_GEMM[pid](length_m, length_n, length_k, A, B, C, alpha, beta, stream);
    return;
  }
}

} // namespace row_col_col
} // namespace fp16
std::unordered_set<const void*> fp16::row_col_col::CUTLASS_GEMM_P40::weight_cache;
Fp16OnlinePlan select_fp16_online_plan_row_col_col(int m, int n, int k) {
  return fp16::row_col_col::select_online_plan(m, n, k);
}

void run_fp16_online_gemm_row_col_col(
    int m, int n, int k, const cutlass::half_t *A, const cutlass::half_t *B,
    cutlass::half_t *C, float falpha, float fbeta, cudaStream_t stream) {
  Fp16OnlinePlan plan = select_fp16_online_plan_row_col_col(m, n, k);
  switch (plan.pattern) {
  case Fp16PolyPattern::Combine:
    fp16::row_col_col::run_pattern2_twin_gemm(
        m, n, k, plan.split_n, const_cast<cutlass::half_t *>(A),
        const_cast<cutlass::half_t *>(B), C, falpha, fbeta, stream);
    return;
  case Fp16PolyPattern::SplitK:
    if (fp16::row_col_col::is_explicit_neighborhood_pid(plan.pid)) {
      fp16::row_col_col::run_explicit_neighborhood_gemm(
          plan.pid, m, n, k, const_cast<cutlass::half_t *>(A),
          const_cast<cutlass::half_t *>(B), C, falpha, fbeta, plan.split_k,
          stream);
      return;
    }
    fp16::row_col_col::run_primary_runtime_splitk_gemm(
        plan.pid, m, n, k, const_cast<cutlass::half_t *>(A),
        const_cast<cutlass::half_t *>(B), C, falpha, fbeta, plan.split_k,
        stream);
    return;
  case Fp16PolyPattern::Single:
  default:
    if (fp16::row_col_col::is_explicit_neighborhood_pid(plan.pid)) {
      fp16::row_col_col::run_explicit_neighborhood_gemm(
          plan.pid, m, n, k, const_cast<cutlass::half_t *>(A),
          const_cast<cutlass::half_t *>(B), C, falpha, fbeta,
          std::max(1, plan.split_k), stream);
      return;
    }
    fp16::row_col_col::CT_GEMM[plan.pid](
        m, n, k, const_cast<cutlass::half_t *>(A),
        const_cast<cutlass::half_t *>(B), C, falpha, fbeta, stream);
    return;
  }
}

void run_moonpoly_gemm_fp16_row_col_col(int m, int n, int k,
                                        const cutlass::half_t *A,
                                        const cutlass::half_t *B,
                                        cutlass::half_t *C, float falpha,
                                        float fbeta, cudaStream_t stream) {
  run_fp16_online_gemm_row_col_col(m, n, k, A, B, C, falpha, fbeta, stream);
}

void run_moonpoly_selected_gemm_fp16_row_col_col(int pid, int m, int n, int k,
                                                 const cutlass::half_t *A,
                                                 const cutlass::half_t *B,
                                                 cutlass::half_t *C,
                                                 float falpha, float fbeta, cudaStream_t stream) {
  // std::cout << "pid: " << pid << " m: " << m << " n: " << n << " k: " << k;
  int alignment = fp16::row_col_col::is_explicit_neighborhood_pid(pid)
                      ? fp16::row_col_col::explicit_neighborhood_alignment(pid)
                      : fp16::row_col_col::gemm_alignment[pid];
  if (k % alignment != 0) {
    std::cout << " k is not aligned, please check!" << std::endl;
  } else {
    if (fp16::row_col_col::is_explicit_neighborhood_pid(pid)) {
      fp16::row_col_col::run_explicit_neighborhood_gemm(
          pid, m, n, k, const_cast<cutlass::half_t *>(A),
          const_cast<cutlass::half_t *>(B), C, falpha, fbeta, 1, stream);
    } else {
      fp16::row_col_col::CT_GEMM[pid](m, n, k, const_cast<cutlass::half_t *>(A),
                                      const_cast<cutlass::half_t *>(B), C, falpha,
                                      fbeta, stream);
    }
    // std::cout << " Est time: "
    //           << fp16::row_col_col::gemm_cost_model(m, n, k, pid) << std::endl;
  }
}

void run_fp16_pattern2_twin_gemm_row_col_col(
    int m, int n, int k, int split_n, const cutlass::half_t *A,
    const cutlass::half_t *B, cutlass::half_t *C, float falpha, float fbeta,
    cudaStream_t stream) {
  fp16::row_col_col::run_pattern2_twin_gemm(
      m, n, k, split_n, const_cast<cutlass::half_t *>(A),
      const_cast<cutlass::half_t *>(B), C, falpha, fbeta, stream);
}

size_t get_gemm_workspace_size_fp16_row_col_col(int pid, int m, int n, int k,
                                                 int split_k_slices) {
  if (pid != 40) {
    return 0;
  }
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  return fp16::row_col_col::CUTLASS_GEMM_P40::query_workspace_size(
      m, n, k, split_k_slices);
}

size_t get_fp16_p40_streamk_workspace_size_row_col_col(int m, int n, int k,
                                                       int split_k_slices) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  return fp16::row_col_col::CUTLASS_GEMM_P40::query_workspace_size(
      m, n, k, split_k_slices);
}

size_t get_fp16_p66_streamk_workspace_size_row_col_col(int m, int n, int k,
                                                       int split_k_slices) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  return fp16::row_col_col::CUTLASS_GEMM_P66::query_workspace_size(
      m, n, k, split_k_slices);
}

size_t get_fp16_p42_streamk_workspace_size_row_col_col(int m, int n, int k,
                                                       int split_k_slices) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  return fp16::row_col_col::CUTLASS_GEMM_P42::query_workspace_size(
      m, n, k, split_k_slices);
}

size_t get_fp16_p43_streamk_workspace_size_row_col_col(int m, int n, int k,
                                                       int split_k_slices) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  return fp16::row_col_col::CUTLASS_GEMM_P43::query_workspace_size(
      m, n, k, split_k_slices);
}

size_t get_fp16_p44_streamk_workspace_size_row_col_col(int m, int n, int k,
                                                       int split_k_slices) {
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  return fp16::row_col_col::CUTLASS_GEMM_P44::query_workspace_size(
      m, n, k, split_k_slices);
}

void run_fp16_p40_streamk_singleton_row_col_col(
    int m, int n, int k, const cutlass::half_t *A, const cutlass::half_t *B,
    cutlass::half_t *C, float falpha, float fbeta, int split_k_slices,
    cudaStream_t stream) {
  constexpr int pid = 40;
  if (k % fp16::row_col_col::gemm_alignment[pid] != 0) {
    std::cout << " k is not aligned, please check!" << std::endl;
    return;
  }
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  fp16::row_col_col::CUTLASS_GEMM_P40 &instance =
      fp16::row_col_col::CUTLASS_GEMM_P40::get_instance();
  instance.run_gemm(m, n, k, split_k_slices, const_cast<cutlass::half_t *>(A),
                    const_cast<cutlass::half_t *>(B), C, falpha, fbeta,
                    stream);
}

void run_fp16_p66_streamk_row_col_col(
    int m, int n, int k, const cutlass::half_t *A, const cutlass::half_t *B,
    cutlass::half_t *C, float falpha, float fbeta, int split_k_slices,
    cudaStream_t stream) {
  constexpr int alignment = fp16::row_col_col::CUTLASS_GEMM_P66::AlignmentA;
  if (k % alignment != 0) {
    std::cout << " k is not aligned, please check!" << std::endl;
    return;
  }
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  fp16::row_col_col::CUTLASS_GEMM_P66 instance;
  instance.run_gemm(m, n, k, split_k_slices, const_cast<cutlass::half_t *>(A),
                    const_cast<cutlass::half_t *>(B), C, falpha, fbeta,
                    stream);
}

void run_fp16_p42_streamk_row_col_col(
    int m, int n, int k, const cutlass::half_t *A, const cutlass::half_t *B,
    cutlass::half_t *C, float falpha, float fbeta, int split_k_slices,
    cudaStream_t stream) {
  constexpr int alignment = fp16::row_col_col::CUTLASS_GEMM_P42::AlignmentA;
  if (k % alignment != 0) {
    std::cout << " k is not aligned, please check!" << std::endl;
    return;
  }
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  fp16::row_col_col::CUTLASS_GEMM_P42 instance;
  instance.run_gemm(m, n, k, split_k_slices, const_cast<cutlass::half_t *>(A),
                    const_cast<cutlass::half_t *>(B), C, falpha, fbeta,
                    stream);
}

void run_fp16_p43_streamk_row_col_col(
    int m, int n, int k, const cutlass::half_t *A, const cutlass::half_t *B,
    cutlass::half_t *C, float falpha, float fbeta, int split_k_slices,
    cudaStream_t stream) {
  constexpr int alignment = fp16::row_col_col::CUTLASS_GEMM_P43::AlignmentA;
  if (k % alignment != 0) {
    std::cout << " k is not aligned, please check!" << std::endl;
    return;
  }
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  fp16::row_col_col::CUTLASS_GEMM_P43 instance;
  instance.run_gemm(m, n, k, split_k_slices, const_cast<cutlass::half_t *>(A),
                    const_cast<cutlass::half_t *>(B), C, falpha, fbeta,
                    stream);
}

void run_fp16_p44_streamk_row_col_col(
    int m, int n, int k, const cutlass::half_t *A, const cutlass::half_t *B,
    cutlass::half_t *C, float falpha, float fbeta, int split_k_slices,
    cudaStream_t stream) {
  constexpr int alignment = fp16::row_col_col::CUTLASS_GEMM_P44::AlignmentA;
  if (k % alignment != 0) {
    std::cout << " k is not aligned, please check!" << std::endl;
    return;
  }
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  fp16::row_col_col::CUTLASS_GEMM_P44 instance;
  instance.run_gemm(m, n, k, split_k_slices, const_cast<cutlass::half_t *>(A),
                    const_cast<cutlass::half_t *>(B), C, falpha, fbeta,
                    stream);
}

void run_moonpoly_selected_gemm_fp16_row_col_col_with_workspace(
    int pid, int m, int n, int k, const cutlass::half_t *A,
    const cutlass::half_t *B, cutlass::half_t *C, float falpha, float fbeta,
    void *workspace_ptr, size_t workspace_size_bytes, int split_k_slices,
    cudaStream_t stream) {
  int alignment = fp16::row_col_col::is_explicit_neighborhood_pid(pid)
                      ? fp16::row_col_col::explicit_neighborhood_alignment(pid)
                      : fp16::row_col_col::gemm_alignment[pid];
  if (k % alignment != 0) {
    std::cout << " k is not aligned, please check!" << std::endl;
    return;
  }

  if (fp16::row_col_col::is_explicit_neighborhood_pid(pid)) {
    (void)workspace_ptr;
    (void)workspace_size_bytes;
    fp16::row_col_col::run_explicit_neighborhood_gemm(
        pid, m, n, k, const_cast<cutlass::half_t *>(A),
        const_cast<cutlass::half_t *>(B), C, falpha, fbeta, split_k_slices,
        stream);
    return;
  }

  if (pid == 40) {
    if (split_k_slices <= 0) {
      split_k_slices = 2;
    }
    // NOTE:
    // The external-workspace StreamK path may be sensitive to runtime allocator
    // behavior in integration scenarios. For deployment stability, route pid=40
    // through the validated singleton path, which manages workspace internally.
    // This keeps StreamK enabled while avoiding non-finite outputs observed in
    // Python/vLLM integration.
    (void)workspace_ptr;
    (void)workspace_size_bytes;
    fp16::row_col_col::CUTLASS_GEMM_P40 &instance =
        fp16::row_col_col::CUTLASS_GEMM_P40::get_instance();
    instance.run_gemm(m, n, k, split_k_slices, const_cast<cutlass::half_t *>(A),
                      const_cast<cutlass::half_t *>(B), C, falpha, fbeta,
                      stream);
    return;
  }

  if (split_k_slices > 1 &&
      fp16::row_col_col::is_primary_runtime_splitk_pid(pid)) {
    // Primary fp16 SplitKSerial/GemmSplitKParallel kernels manage their own
    // workspace here. The external workspace path remains reserved for
    // generated StreamK/explicit kernels whose workspace ABI is stable.
    (void)workspace_ptr;
    (void)workspace_size_bytes;
    fp16::row_col_col::run_primary_runtime_splitk_gemm(
        pid, m, n, k, const_cast<cutlass::half_t *>(A),
        const_cast<cutlass::half_t *>(B), C, falpha, fbeta, split_k_slices,
        stream);
    return;
  }

  fp16::row_col_col::CT_GEMM[pid](m, n, k, const_cast<cutlass::half_t *>(A),
                                  const_cast<cutlass::half_t *>(B), C, falpha,
                                  fbeta, stream);
}

void run_fp16_p40_streamk_row_col_col(
    int m, int n, int k, const cutlass::half_t *A, const cutlass::half_t *B,
    cutlass::half_t *C, float falpha, float fbeta, void *workspace_ptr,
    size_t workspace_size_bytes, int split_k_slices, cudaStream_t stream) {
  constexpr int pid = 40;
  if (k % fp16::row_col_col::gemm_alignment[pid] != 0) {
    std::cout << " k is not aligned, please check!" << std::endl;
    return;
  }
  if (split_k_slices <= 0) {
    split_k_slices = 2;
  }
  fp16::row_col_col::CUTLASS_GEMM_P40::run_gemm_with_external_workspace(
      m, n, k, split_k_slices, const_cast<cutlass::half_t *>(A),
      const_cast<cutlass::half_t *>(B), C, falpha, fbeta, workspace_ptr,
      workspace_size_bytes, stream);
}

void get_gemm_shared_mem_fp16_row_col_col(int pid, int *shared_mem) {
  if (fp16::row_col_col::is_explicit_neighborhood_pid(pid)) {
    SharedMem meta = fp16::row_col_col::explicit_neighborhood_shared_mem(pid);
    shared_mem[0] = meta.pm;
    shared_mem[1] = meta.pn;
    shared_mem[2] = meta.pk;
    return;
  }
  shared_mem[0] = fp16::row_col_col::gemm_shared_mem[pid].pm;
  shared_mem[1] = fp16::row_col_col::gemm_shared_mem[pid].pn;
  shared_mem[2] = fp16::row_col_col::gemm_shared_mem[pid].pk;
}

void get_gemm_alignment_fp16_row_col_col(int pid, int *alignment) {
  if (fp16::row_col_col::is_explicit_neighborhood_pid(pid)) {
    *alignment = fp16::row_col_col::explicit_neighborhood_alignment(pid);
    return;
  }
  *alignment = fp16::row_col_col::gemm_alignment[pid];
}

} // namespace moonpoly
