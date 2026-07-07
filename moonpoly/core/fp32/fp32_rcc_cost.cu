#include "moonpoly.cuh"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

namespace moonpoly {
namespace fp32 {
namespace simt {
namespace row_col_col {

/*
   ============================ COST MODEL DATA ===========================
 */
const int num_gemm_primary = 40;

extern SharedMem gemm_shared_mem[];


int gemm_alignment[] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
};

#include "../../generated/fp32/rcc/fp32_rcc_fitted_cost.inc"
#include "../../generated/fp32/rcc/fp32_rcc_splitk_cost.inc"


/*
   ============================ COST MODEL API ===========================
 */

namespace {

constexpr int kPattern3TileM = 128;
constexpr int kPattern3TileN = 64;
constexpr int kPattern3TileK = 8;

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

inline float fp32_tile_efficiency(int pid) {
  const int pm = gemm_shared_mem[pid].pm;
  const int pn = gemm_shared_mem[pid].pn;
  const int pk = gemm_shared_mem[pid].pk;
  const float area = static_cast<float>(pm * pn);
  const float area_eff = clampf(area / (128.0f * 128.0f), 0.20f, 1.00f);
  const float k_eff = pk >= 64 ? 0.92f : (pk >= 16 ? 0.78f : 0.64f);
  const float skinny_penalty = (pm < 16 || pn < 32) ? 0.82f : 1.0f;
  return clampf((0.34f + 0.66f * area_eff) * k_eff * skinny_penalty, 0.18f, 0.95f);
}

}  // namespace

float analytic_gemm_cost_model(int ims, int ins, int iks, int pid) {
  const int pm = gemm_shared_mem[pid].pm;
  const int pn = gemm_shared_mem[pid].pn;
  const int pk = gemm_shared_mem[pid].pk;
  const int lm = ceil_div(ims, pm);
  const int ln = ceil_div(ins, pn);
  const int lk = ceil_div(iks, pk);
  const int waves = ceil_div(lm * ln, num_SM);

  const double padded_flops = 2.0 * lm * pm * ln * pn * lk * pk;
  const double actual_flops = 2.0 * ims * ins * iks;
  const float tail_eff = static_cast<float>(actual_flops / padded_flops);
  const float effective_tflops = 19.5f * fp32_tile_efficiency(pid) * clampf(tail_eff, 0.35f, 1.0f);
  const float compute_ms = static_cast<float>(padded_flops / (static_cast<double>(effective_tflops) * 1.0e9));

  const float launch_wave_overhead_ms = 0.0018f * waves;
  const float tile_overhead_ms = 0.0000025f * static_cast<float>(lm * ln * lk);
  return compute_ms + launch_wave_overhead_ms + tile_overhead_ms;
}

float fitted_gemm_cost_model(int ims, int ins, int iks, int pid) {
  const int pm = gemm_shared_mem[pid].pm;
  const int pn = gemm_shared_mem[pid].pn;
  const int pk = gemm_shared_mem[pid].pk;
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

  const double padded_flops =
      2.0 * lm * kPattern3TileM * ln * kPattern3TileN * lk * kPattern3TileK;
  const double actual_flops =
      2.0 * static_cast<double>(ims) * static_cast<double>(ins) *
      static_cast<double>(part_k);
  const float tail_eff = static_cast<float>(actual_flops / padded_flops);
  const float effective_tflops = 19.5f * 0.70f * clampf(tail_eff, 0.35f, 1.0f);
  const float pipe_ms = static_cast<float>(
      (padded_flops * split_k_slices) /
      (static_cast<double>(effective_tflops) * 1.0e9));

  const float wave_overhead_ms = 0.0018f * waves;
  const float reduction_ms =
      0.0040f * (split_k_slices - 1) +
      0.0000030f * static_cast<float>(lm * ln) * (split_k_slices - 1);
  return pipe_ms + wave_overhead_ms + reduction_ms;
}

int cutlass_gemm_predict(int m, int n, int k) {
  int best_id = 0;
  float min_time = std::numeric_limits<float>::infinity();

  for (int i = 0; i < num_gemm_primary; i++) {
    if (k % gemm_alignment[i] != 0)
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
} // namespace simt
} // namespace fp32
} // namespace moonpoly
