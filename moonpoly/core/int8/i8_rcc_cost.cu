#include "moonpoly.cuh"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

namespace moonpoly {
namespace int8 {
namespace row_col_col {

/*
   ============================ COST MODEL DATA ===========================
 */
constexpr int num_int8_gemm_primary = 46;
SharedMem int8_gemm_shared_mem[] = {
  {64, 128, 64}, {256, 64, 64}, {64, 64, 64}, {64, 64, 64},
  {64, 128, 64}, {256, 128, 128}, {128, 128, 128}, {128, 256, 128},
  {64, 128, 128}, {64, 128, 64}, {64, 128, 64}, {128, 64, 64},
  {64, 256, 64}, {128, 256, 128}, {256, 128, 128}, {64, 256, 128},
  {128, 64, 128}, {128, 128, 64}, {64, 64, 64}, {128, 256, 64},
  {64, 64, 128}, {256, 128, 128}, {128, 128, 64}, {128, 128, 128},
  {128, 64, 128}, {128, 128, 64}, {256, 128, 64}, {128, 128, 64},
  {64, 64, 64}, {128, 128, 64}, {128, 128, 64}, {64, 128, 64},
  {64, 128, 64}, {64, 128, 128}, {64, 64, 128}, {128, 64, 64},
  {64, 64, 128}, {128, 64, 128}, {128, 128, 64}, {128, 128, 64},
  {64, 128, 128}, {64, 64, 64}, {128, 128, 128}, {128, 64, 64},
  {64, 64, 64}, {256, 64, 64}
};

int int8_gemm_alignment[] = {
  4, 8, 4, 1, 1, 4, 8, 4,
  4, 16, 16, 8, 16, 8, 16, 16,
  16, 4, 16, 16, 16, 8, 8, 4,
  4, 8, 16, 16, 8, 16, 1, 2,
  8, 16, 4, 16, 8, 8, 4, 2,
  8, 16, 16, 16, 2, 16
};

#include "../../generated/int8/rcc/i8_rcc_fitted_cost.inc"
#include "../../generated/int8/rcc/i8_rcc_splitk_cost.inc"

/*
   ============================ COST MODEL API ===========================
 */

namespace {

constexpr int kPattern3TileM = 128;
constexpr int kPattern3TileN = 128;
constexpr int kPattern3TileK = 64;

inline int ceil_div(int x, int y) {
  return (x + y - 1) / y;
}

inline float clampf(float x, float lo, float hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

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

inline int fitted_kiter_segment_index(int pid, int k_iter) {
  const int segment_count = fitted_cost::kSegmentCount[pid];
  for (int i = 0; i < segment_count; ++i) {
    if (k_iter <= fitted_cost::kSegmentMaxKiter[pid][i]) {
      return i;
    }
  }
  return segment_count - 1;
}

inline float int8_tile_efficiency(int pid) {
  const int pm = int8_gemm_shared_mem[pid].pm;
  const int pn = int8_gemm_shared_mem[pid].pn;
  const int pk = int8_gemm_shared_mem[pid].pk;
  const int alignment = int8_gemm_alignment[pid];
  const float area = static_cast<float>(pm * pn);
  const float area_eff = clampf(area / (128.0f * 128.0f), 0.35f, 1.20f);
  const float k_eff = pk >= 128 ? 1.00f : 0.82f;
  const float align_eff = alignment >= 16 ? 1.00f : (alignment >= 8 ? 0.95f : (alignment >= 4 ? 0.88f : 0.74f));
  const float large_tile_penalty = (pm >= 256 || pn >= 256) ? 0.92f : 1.0f;
  return clampf((0.42f + 0.58f * area_eff) * k_eff * align_eff * large_tile_penalty, 0.22f, 1.05f);
}

}  // namespace

float analytic_gemm_cost_model(int ims, int ins, int iks, int pid) {
  const int pm = int8_gemm_shared_mem[pid].pm;
  const int pn = int8_gemm_shared_mem[pid].pn;
  const int pk = int8_gemm_shared_mem[pid].pk;
  const int lm = ceil_div(ims, pm);
  const int ln = ceil_div(ins, pn);
  const int lk = ceil_div(iks, pk);
  const int waves = ceil_div(lm * ln, num_SM);

  const double padded_ops = 2.0 * lm * pm * ln * pn * lk * pk;
  const double actual_ops = 2.0 * ims * ins * iks;
  const float tail_eff = static_cast<float>(actual_ops / padded_ops);
  const float effective_tops = 312.0f * int8_tile_efficiency(pid) * clampf(tail_eff, 0.35f, 1.0f);
  const float compute_ms = static_cast<float>(padded_ops / (static_cast<double>(effective_tops) * 1.0e9));

  const float launch_wave_overhead_ms = 0.0012f * waves;
  const float tile_overhead_ms = 0.0000015f * static_cast<float>(lm * ln * lk);
  return compute_ms + launch_wave_overhead_ms + tile_overhead_ms;
}

float fitted_gemm_cost_model(int ims, int ins, int iks, int pid) {
  const int pm = int8_gemm_shared_mem[pid].pm;
  const int pn = int8_gemm_shared_mem[pid].pn;
  const int pk = int8_gemm_shared_mem[pid].pk;
  const int lm = ceil_div(ims, pm);
  const int ln = ceil_div(ins, pn);
  const int k_iter = ceil_div(iks, pk);
  const int waves = ceil_div(lm * ln, fitted_cost::kNumSms);
  const int segment = fitted_kiter_segment_index(pid, k_iter);
  const double unit_ms =
      fitted_cost::kSegmentSlope[pid][segment] * k_iter +
      fitted_cost::kSegmentIntercept[pid][segment];
  const double cost = static_cast<double>(waves) * unit_ms;
  return static_cast<float>(cost > 0.0 ? cost : 0.0);
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

float gemm_splitk_cost_model(int ims, int ins, int iks, int split_k_slices) {
  if (use_fitted_cost_model() &&
      splitk_gr_fitted_cost::kHasMeasurements != 0) {
    for (int i = 0; i < splitk_gr_fitted_cost::kProfileAnchorSampleCount; ++i) {
      if (splitk_gr_fitted_cost::kProfileAnchorSplitK[i] == split_k_slices &&
          splitk_gr_fitted_cost::kProfileAnchorFeatureM[i] == ims &&
          splitk_gr_fitted_cost::kProfileAnchorFeatureN[i] == ins &&
          splitk_gr_fitted_cost::kProfileAnchorFeatureK[i] == iks) {
        return static_cast<float>(
            splitk_gr_fitted_cost::kProfileAnchorLatencyMs[i]);
      }
    }

    for (int i = 0; i < splitk_gr_fitted_cost::kSplitFactorCount; ++i) {
      if (splitk_gr_fitted_cost::kSplitFactors[i] != split_k_slices ||
          splitk_gr_fitted_cost::kSplitValid[i] == 0) {
        continue;
      }
      const int lm = ceil_div(ims, splitk_gr_fitted_cost::kTileM);
      const int ln = ceil_div(ins, splitk_gr_fitted_cost::kTileN);
      const int output_tiles = std::max(1, lm * ln);
      const int part_k = ceil_div(iks, split_k_slices);
      const int k_iter = ceil_div(part_k, splitk_gr_fitted_cost::kTileK);
      const int waves = ceil_div(
          output_tiles * split_k_slices, splitk_gr_fitted_cost::kNumSms);
      const int extra = split_k_slices - 1;
      int segment = splitk_gr_fitted_cost::kSegmentCount[i] - 1;
      for (int s = 0; s < splitk_gr_fitted_cost::kSegmentCount[i]; ++s) {
        if (k_iter <= splitk_gr_fitted_cost::kSegmentMaxKiter[i][s]) {
          segment = s;
          break;
        }
      }
      if (segment < 0 ||
          splitk_gr_fitted_cost::kSegmentValid[i][segment] == 0) {
        continue;
      }
      const double g_predict =
          static_cast<double>(waves) *
          (splitk_gr_fitted_cost::kGemmSlopeMsPerKiter[i][segment] * k_iter +
           splitk_gr_fitted_cost::kGemmInterceptMsPerWave[i][segment]);
      const double r_predict =
          splitk_gr_fitted_cost::kReductionFixedMsPerExtraSlice[i][segment] *
              extra +
          splitk_gr_fitted_cost::kReductionTileMsPerExtraSlice[i][segment] *
              output_tiles * extra;
      const double cost = g_predict + r_predict;
      if (std::isfinite(cost) && cost > 0.0) {
        return static_cast<float>(cost);
      }
    }
  }

  if (split_k_slices <= 1) {
    return std::numeric_limits<float>::infinity();
  }

  const int lm = ceil_div(ims, kPattern3TileM);
  const int ln = ceil_div(ins, kPattern3TileN);
  const int part_k = ceil_div(iks, split_k_slices);
  const int lk = ceil_div(part_k, kPattern3TileK);
  const int waves = ceil_div(lm * ln * split_k_slices, num_SM);

  const double padded_ops =
      2.0 * lm * kPattern3TileM * ln * kPattern3TileN * lk * kPattern3TileK;
  const double actual_ops =
      2.0 * static_cast<double>(ims) * static_cast<double>(ins) *
      static_cast<double>(part_k);
  const float tail_eff = static_cast<float>(actual_ops / padded_ops);
  const float effective_tops = 312.0f * 0.70f * clampf(tail_eff, 0.35f, 1.0f);
  const float pipe_ms = static_cast<float>(
      (padded_ops * split_k_slices) /
      (static_cast<double>(effective_tops) * 1.0e9));

  const float wave_overhead_ms = 0.0012f * waves;
  const float reduction_ms =
      0.0030f * (split_k_slices - 1) +
      0.0000020f * static_cast<float>(lm * ln) * (split_k_slices - 1);
  return pipe_ms + wave_overhead_ms + reduction_ms;
}

int cutlass_gemm_predict(int m, int n, int k) {
  int best_id = 0;
  float min_time = std::numeric_limits<float>::infinity();
  for (int i = 0; i < num_int8_gemm_primary; i++) {
    if (k % int8_gemm_alignment[i] != 0)
      continue;
    const float time = gemm_cost_model(m, n, k, i);
    if (time < min_time) {
      min_time = time;
      best_id = i;
    }
  }
  return best_id;
}

} // namespace row_col_col
} // namespace int8
} // namespace moonpoly
