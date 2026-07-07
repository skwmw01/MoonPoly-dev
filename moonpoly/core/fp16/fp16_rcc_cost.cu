#include "moonpoly.cuh"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

namespace moonpoly {
namespace fp16 {
namespace row_col_col {

/*
   ============================ COST MODEL DATA ===========================
 */
const int num_gemm_primary = 41;
const int num_gemm_active_predict = 40;

SharedMem gemm_shared_mem[] = {
    {16, 64, 96},  {128, 32, 128}, {64, 64, 128},  {64, 64, 64},
    {16, 64, 32},  {32, 32, 16},   {16, 64, 16},   {128, 64, 256},
    {64, 64, 8},   {64, 64, 16},   {32, 64, 64},   {64, 64, 96},
    {128, 64, 64}, {16, 32, 8},    {96, 32, 160},  {80, 64, 128},
    {48, 64, 32},  {256, 64, 32},  {32, 32, 128},  {16, 64, 128},
    {80, 64, 96},  {16, 16, 64},   {128, 32, 96},  {256, 32, 128},
    {32, 64, 16},  {16, 64, 8},    {16, 64, 64},   {64, 16, 64},
    {16, 32, 32},  {8, 64, 16},    {192, 32, 192}, {128, 64, 192},
    {32, 16, 64},  {96, 64, 192},  {64, 32, 96},   {8, 64, 8},
    {32, 32, 32},  {64, 64, 32},   {16, 16, 32},   {32, 64, 32},
    {64, 32, 24},
};

int gemm_alignment[] = {
    8, 8, 8, 8, 8, 8, 8, 2, 8, 2, 1, 2, 2, 2, 8, 8, 8, 8, 8, 8,
    8, 4, 8, 8, 4, 8, 4, 4, 8, 1, 8, 8, 2, 1, 8, 2, 2, 8, 1, 8,
    8,
};

float gemm_sensitive[] = {
    0.11842875652870526,  0.0613320075902433,   0.1055658078834371,
    0.09591179158326782,  0.09070913684741375,  0.0965595088049647,
    0.07693120950333342,  0.2070700309113034,  0.15812832752355302,
    0.05922031114097802,  0.033075562378095266, 0.06558300893960126,
    0.048569173340831245, 0.06520710784650098,  0.07147350565609607,
    0.07787661563109313,  0.08033603378138311,  0.1806139794072268,
    0.08790790593374154,  0.12474726273116464,  0.1501519296838761,
    0.15133489849052625,  0.1520139889904713,   0.09187240701743796,
    0.16466707211441003,  0.15718633306255653,  0.16252272050516134,
    0.18986757808818097,  0.1725991405995812,   0.19533348158538472,
    0.09369063233447512,  0.24459406819231305,  0.18216720416107465,
    0.13410371544285968,  0.19984246590433383,  0.10744442922999696,
    0.16417469663322423,  0.11515271738983768,  0.17577460217204416,
    0.11970271370789869,  0.12};

#include "../../generated/fp16/rcc/fp16_rcc_fitted_cost.inc"
#include "../../generated/fp16/rcc/fp16_rcc_splitk_reduction_cost.inc"

// Legacy handwritten FP16 cost fallback. Keep it isolated from the
// runtime selector so generated fitted models remain the default path.
#include "fp16_rcc_legacy_cost.inc"

/*
   ============================ COST MODEL API ===========================
 */

inline float get_gemm_utime(int k_iter, int pid) {
  float utime = GEMM_K_FUNC[pid](k_iter);
  return utime * (1 + gemm_sensitive[pid]);
}

namespace {

inline bool use_fitted_cost_model() {
  static bool initialized = false;
  static bool enabled = true;
  if (!initialized) {
    const char *disable = std::getenv("MOONPOLY_DISABLE_FITTED_COST");
    const char *legacy = std::getenv("MOONPOLY_USE_FITTED_COST");
    enabled = true;
    if (disable != nullptr && disable[0] != '\0' && disable[0] != '0') {
      enabled = false;
    }
    // Backward-compatible escape hatch for older scripts.
    if (legacy != nullptr && legacy[0] == '0') {
      enabled = false;
    }
    initialized = true;
  }
  return enabled;
}

inline bool fitted_pid_valid(int pid) {
  return pid >= 0 && pid < num_gemm_primary &&
         pid < fitted_cost::kCandidateCount &&
         fitted_cost::kPidValid[pid] != 0 &&
         fitted_cost::kSegmentCount[pid] > 0;
}

inline int fitted_kiter_segment_index(int pid, int k_iter) {
  const int segment_count = fitted_cost::kSegmentCount[pid];
  for (int i = 0; i < segment_count; ++i) {
    if (k_iter <= fitted_cost::kSegmentMaxKiter[pid][i]) {
      return i;
    }
  }
  return segment_count - 1;
}

inline float fitted_unit_ms(int pid, int k_iter) {
  if (!fitted_pid_valid(pid)) {
    return std::numeric_limits<float>::infinity();
  }
  const int segment = fitted_kiter_segment_index(pid, k_iter);
  const double unit_ms =
      fitted_cost::kSegmentSlope[pid][segment] * k_iter +
      fitted_cost::kSegmentIntercept[pid][segment];
  return static_cast<float>(unit_ms > 0.0 ? unit_ms
                                          : std::numeric_limits<float>::infinity());
}

inline bool fitted_reduction_pid_valid(int pid) {
  return pid >= 0 && pid < num_gemm_primary &&
         pid < splitk_reduction_fitted_cost::kCandidateCount &&
         splitk_reduction_fitted_cost::kPidValid[pid] != 0;
}

inline float fitted_reduction_cost_model(int output_tiles, int pid,
                                         int split_k_slices) {
  if (split_k_slices <= 1 || !fitted_reduction_pid_valid(pid)) {
    return 0.0f;
  }
  const int extra = split_k_slices - 1;
  const double fixed_ms =
      splitk_reduction_fitted_cost::kFixedMsPerExtraSlice[pid] * extra;
  const double tile_ms =
      splitk_reduction_fitted_cost::kTileMsPerExtraSlice[pid] *
      static_cast<double>(std::max(1, output_tiles)) * extra;
  const double cost = fixed_ms + tile_ms;
  return static_cast<float>(cost > 0.0 ? cost : 0.0);
}

}  // namespace

float analytic_gemm_cost_model(int ims, int ins, int iks, int pid) {
  int pm = gemm_shared_mem[pid].pm;
  int pn = gemm_shared_mem[pid].pn;
  int pk = gemm_shared_mem[pid].pk;
  int lm = ceil((1.0 * ims) / pm), ln = ceil((1.0 * ins) / pn);
  int k_iter = 0, sm_iteration = 0;
  if (pid == 39) {
    k_iter = ceil((1.0 * iks) / (pk * 16));
    sm_iteration = ceil((lm * ln * 16.0) / num_SM);
  } else {
    k_iter = ceil((1.0 * iks) / pk);
    sm_iteration = ceil((lm * ln * 1.0) / num_SM);
  }

  float utime = get_gemm_utime(k_iter, pid);
  return sm_iteration * utime;
}

float fitted_gemm_cost_model(int ims, int ins, int iks, int pid) {
  if (!fitted_pid_valid(pid)) {
    return std::numeric_limits<float>::infinity();
  }
  const int pm = gemm_shared_mem[pid].pm;
  const int pn = gemm_shared_mem[pid].pn;
  const int pk = gemm_shared_mem[pid].pk;
  const int lm = std::max(1, static_cast<int>(ceil((1.0 * ims) / pm)));
  const int ln = std::max(1, static_cast<int>(ceil((1.0 * ins) / pn)));
  const int k_iter = std::max(1, static_cast<int>(ceil((1.0 * iks) / pk)));
  const int sm_iteration = std::max(
      1, static_cast<int>(ceil((lm * ln * 1.0) / fitted_cost::kNumSms)));
  const float unit_ms = fitted_unit_ms(pid, k_iter);
  if (!std::isfinite(unit_ms)) {
    return std::numeric_limits<float>::infinity();
  }
  return sm_iteration * unit_ms;
}

float gemm_cost_model(int ims, int ins, int iks, int pid) {
  if (use_fitted_cost_model()) {
    const float fitted = fitted_gemm_cost_model(ims, ins, iks, pid);
    if (std::isfinite(fitted) && fitted > 0.0f) {
      return fitted;
    }
  }
  return analytic_gemm_cost_model(ims, ins, iks, pid);
}

float gemm_reduction_cost_model(int ims, int ins, int, int pid,
                                int split_k_slices) {
  if (split_k_slices <= 1) {
    return 0.0f;
  }
  int pm = gemm_shared_mem[pid].pm;
  int pn = gemm_shared_mem[pid].pn;
  int lm = ceil((1.0 * ims) / pm);
  int ln = ceil((1.0 * ins) / pn);
  int output_tiles = std::max(1, lm * ln);

  if (use_fitted_cost_model() && fitted_reduction_pid_valid(pid)) {
    return fitted_reduction_cost_model(output_tiles, pid, split_k_slices);
  }

  // Empirical prior for Pattern-Split&Merge overhead. The fitted r_predict
  // path above uses the same feature form and is enabled by default.
  float fixed_ms = 0.0030f * (split_k_slices - 1);
  float tile_ms = 0.000020f * output_tiles * (split_k_slices - 1);
  return fixed_ms + tile_ms;
}

float gemm_splitk_cost_model(int ims, int ins, int iks, int pid,
                             int split_k_slices) {
  if (split_k_slices <= 1) {
    return gemm_cost_model(ims, ins, iks, pid);
  }
  int pm = gemm_shared_mem[pid].pm;
  int pn = gemm_shared_mem[pid].pn;
  int pk = gemm_shared_mem[pid].pk;
  int lm = ceil((1.0 * ims) / pm);
  int ln = ceil((1.0 * ins) / pn);
  int k_iter = std::max(1, (int)ceil((1.0 * iks) / (pk * split_k_slices)));
  int sm_iteration =
      std::max(1, (int)ceil((lm * ln * 1.0 * split_k_slices) / num_SM));

  float unit_ms = get_gemm_utime(k_iter, pid);
  if (use_fitted_cost_model()) {
    const float fitted_unit = fitted_unit_ms(pid, k_iter);
    if (std::isfinite(fitted_unit) && fitted_unit > 0.0f) {
      unit_ms = fitted_unit;
    }
  }
  float pipe_ms = sm_iteration * unit_ms;
  float reduction_ms =
      gemm_reduction_cost_model(ims, ins, iks, pid, split_k_slices);
  return pipe_ms + reduction_ms;
}

void dump_cost_model(int ims, int ins, int iks, int pid) {
  int pm = gemm_shared_mem[pid].pm;
  int pn = gemm_shared_mem[pid].pn;
  int pk = gemm_shared_mem[pid].pk;
  int lm = ceil((1.0 * ims) / pm), ln = ceil((1.0 * ins) / pn);
  int k_iter = 0, sm_iteration = 0;
  if (pid == 39) {
    k_iter = ceil((1.0 * iks) / (pk * 16));
    sm_iteration = ceil((lm * ln * 16.0) / num_SM);
  } else {
    k_iter = ceil((1.0 * iks) / pk);
    sm_iteration = ceil((lm * ln * 1.0) / num_SM);
  }
  float utime = get_gemm_utime(k_iter, pid);
  std::cout << "f_wave:" << sm_iteration << "; f_pipe:" << utime << std::endl;
}

int cutlass_gemm_predict(int m, int n, int k) {
  int best_id = 0;
  float min_time = 1000.0, time = 0;

  for (int i = 0; i < num_gemm_active_predict; i++) {
    if (k % gemm_alignment[i] != 0)
      continue;
    time = gemm_cost_model(m, n, k, i);
    if (time < min_time) {
      min_time = time;
      best_id = i;
    }
  }
  // Temp cost model
  // if (m < 25 && n > 4000 && n<25601 && k<25601 && k>4000) {
  //   best_id = 40; // P40
  // }
  // dump_cost_model(m, n, k, best_id);
  if (std::getenv("MOONPOLY_DEBUG_PREDICT")) {
    std::cout << "[moonpoly][fp16] predict m=" << m << " n=" << n
              << " k=" << k << " -> pid=" << best_id << std::endl;
  }
  return best_id;
}

} // namespace row_col_col
} // namespace fp16
} // namespace moonpoly
