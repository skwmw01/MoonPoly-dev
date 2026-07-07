#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <cutlass/cutlass.h>

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"

#include "scaled_mm_epilogues_mikpoly.hpp"

#include <array>
#include <cstdio>
#include <limits>
#include <optional>
#include <tuple>

namespace moonpoly::int8 {

namespace {

#define CUTLASS_CHECK(status)                       \
  {                                                 \
    cutlass::Status error = status;                 \
    TORCH_CHECK(error == cutlass::Status::kSuccess, \
                cutlassGetStatusString(error));     \
  }

template <typename Arch, typename ElementAB, typename ElementD,
          template <typename, typename> typename Epilogue,
          typename TileShape, typename WarpShape, typename InstructionShape,
          int32_t MainLoopStages>
struct cutlass_2x_gemm_evt {
  using ElementDType = ElementD;
  using ElementAcc = int32_t;
  using Operator = cutlass::arch::OpMultiplyAddSaturate;
  static constexpr int AlignmentAB = 128 / cutlass::sizeof_bits<ElementAB>::value;
  static constexpr int AlignmentCD = 4;

  using OutputTileThreadMap =
      cutlass::epilogue::threadblock::OutputTileThreadLayout<
          TileShape, WarpShape, float, 4, 1>;

  using EpilogueImpl = Epilogue<ElementD, OutputTileThreadMap>;
  using EVTCompute = typename EpilogueImpl::EVTCompute;

  using D = cutlass::epilogue::threadblock::VisitorAuxStore<
      OutputTileThreadMap, ElementD, cutlass::FloatRoundStyle::round_to_nearest,
      Stride<int64_t, Int<1>, Int<0>>>;

  using EVTD = cutlass::epilogue::threadblock::Sm80EVT<D, EVTCompute>;

  using Kernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      ElementAB, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone,
      AlignmentAB,
      ElementAB, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone,
      AlignmentAB,
      float, cutlass::layout::RowMajor, AlignmentCD,
      ElementAcc, float, cutlass::arch::OpClassTensorOp, Arch,
      TileShape, WarpShape, InstructionShape, EVTD,
      cutlass::gemm::threadblock::ThreadblockSwizzleStreamK,
      MainLoopStages, Operator, 1>::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

struct Sm80Cfg0 {
  static constexpr int tile_m = 128;
  static constexpr int tile_n = 128;
  static constexpr int tile_k = 64;
  static constexpr int stages = 5;
  template <typename ElementD, template <typename, typename> typename Epilogue>
  using Gemm = cutlass_2x_gemm_evt<cutlass::arch::Sm80, int8_t, ElementD,
                                   Epilogue,
                                   cutlass::gemm::GemmShape<128, 128, 64>,
                                   cutlass::gemm::GemmShape<64, 64, 64>,
                                   cutlass::gemm::GemmShape<16, 8, 32>, 5>;
};

struct Sm80Cfg1 {
  static constexpr int tile_m = 64;
  static constexpr int tile_n = 128;
  static constexpr int tile_k = 128;
  static constexpr int stages = 5;
  template <typename ElementD, template <typename, typename> typename Epilogue>
  using Gemm = cutlass_2x_gemm_evt<cutlass::arch::Sm80, int8_t, ElementD,
                                   Epilogue,
                                   cutlass::gemm::GemmShape<64, 128, 128>,
                                   cutlass::gemm::GemmShape<64, 64, 64>,
                                   cutlass::gemm::GemmShape<16, 8, 32>, 5>;
};

struct Sm80Cfg2 {
  static constexpr int tile_m = 32;
  static constexpr int tile_n = 64;
  static constexpr int tile_k = 128;
  static constexpr int stages = 5;
  template <typename ElementD, template <typename, typename> typename Epilogue>
  using Gemm = cutlass_2x_gemm_evt<cutlass::arch::Sm80, int8_t, ElementD,
                                   Epilogue,
                                   cutlass::gemm::GemmShape<32, 64, 128>,
                                   cutlass::gemm::GemmShape<32, 64, 64>,
                                   cutlass::gemm::GemmShape<16, 8, 32>, 5>;
};

struct Sm80Cfg3 {
  static constexpr int tile_m = 16;
  static constexpr int tile_n = 64;
  static constexpr int tile_k = 128;
  static constexpr int stages = 5;
  template <typename ElementD, template <typename, typename> typename Epilogue>
  using Gemm = cutlass_2x_gemm_evt<cutlass::arch::Sm80, int8_t, ElementD,
                                   Epilogue,
                                   cutlass::gemm::GemmShape<16, 64, 128>,
                                   cutlass::gemm::GemmShape<16, 64, 64>,
                                   cutlass::gemm::GemmShape<16, 8, 32>, 5>;
};

struct Sm80Cfg4 {
  static constexpr int tile_m = 128;
  static constexpr int tile_n = 64;
  static constexpr int tile_k = 64;
  static constexpr int stages = 4;
  template <typename ElementD, template <typename, typename> typename Epilogue>
  using Gemm = cutlass_2x_gemm_evt<cutlass::arch::Sm80, int8_t, ElementD,
                                   Epilogue,
                                   cutlass::gemm::GemmShape<128, 64, 64>,
                                   cutlass::gemm::GemmShape<64, 32, 64>,
                                   cutlass::gemm::GemmShape<16, 8, 32>, 4>;
};

struct Sm80Cfg5 {
  static constexpr int tile_m = 64;
  static constexpr int tile_n = 64;
  static constexpr int tile_k = 128;
  static constexpr int stages = 5;
  template <typename ElementD, template <typename, typename> typename Epilogue>
  using Gemm = cutlass_2x_gemm_evt<cutlass::arch::Sm80, int8_t, ElementD,
                                   Epilogue,
                                   cutlass::gemm::GemmShape<64, 64, 128>,
                                   cutlass::gemm::GemmShape<32, 32, 64>,
                                   cutlass::gemm::GemmShape<16, 8, 32>, 5>;
};

struct Sm80ConfigMeta {
  int tile_m;
  int tile_n;
  int tile_k;
  int stages;
};

constexpr std::array<Sm80ConfigMeta, 6> kSm80Configs = {{
    {Sm80Cfg0::tile_m, Sm80Cfg0::tile_n, Sm80Cfg0::tile_k, Sm80Cfg0::stages},
    {Sm80Cfg1::tile_m, Sm80Cfg1::tile_n, Sm80Cfg1::tile_k, Sm80Cfg1::stages},
    {Sm80Cfg2::tile_m, Sm80Cfg2::tile_n, Sm80Cfg2::tile_k, Sm80Cfg2::stages},
    {Sm80Cfg3::tile_m, Sm80Cfg3::tile_n, Sm80Cfg3::tile_k, Sm80Cfg3::stages},
    {Sm80Cfg4::tile_m, Sm80Cfg4::tile_n, Sm80Cfg4::tile_k, Sm80Cfg4::stages},
    {Sm80Cfg5::tile_m, Sm80Cfg5::tile_n, Sm80Cfg5::tile_k, Sm80Cfg5::stages},
}};

template <typename Gemm, typename... EpilogueArgs>
void run_cutlass_evt_gemm(torch::Tensor& out,
                          torch::Tensor const& a,
                          torch::Tensor const& b,
                          EpilogueArgs&&... epilogue_args) {
  using ElementAB = typename Gemm::Gemm::ElementA;
  using ElementD = typename Gemm::ElementDType;
  using EpilogueType = typename Gemm::EpilogueImpl;

  int32_t m = a.size(0);
  int32_t n = out.size(1);
  int32_t k = a.size(1);
  cutlass::gemm::GemmCoord problem_size{m, n, k};

  int64_t lda = a.stride(0);
  int64_t ldb = b.stride(0);
  int64_t ldc = out.stride(0);

  using StrideC = Stride<int64_t, Int<1>, Int<0>>;
  StrideC c_stride{ldc, Int<1>{}, Int<0>{}};

  auto evt_args =
      EpilogueType::prepare_args(std::forward<EpilogueArgs>(epilogue_args)...);

  typename Gemm::D::Arguments d_args{
      static_cast<ElementD*>(out.data_ptr()), c_stride};

  typename Gemm::EVTD::Arguments evtd_args{evt_args, d_args};

  auto a_ptr = static_cast<ElementAB const*>(a.data_ptr());
  auto b_ptr = static_cast<ElementAB const*>(b.data_ptr());

  typename Gemm::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel,
      problem_size,
      1,
      evtd_args,
      a_ptr,
      b_ptr,
      nullptr,
      nullptr,
      0,
      0,
      0,
      0,
      lda,
      ldb,
      ldc,
      ldc};

  typename Gemm::Gemm gemm_op;
  size_t workspace_size = gemm_op.get_workspace_size(arguments);
  auto workspace_options =
      torch::TensorOptions().dtype(torch::kUInt8).device(a.device());
  auto workspace = torch::empty(workspace_size, workspace_options);

  auto stream = at::cuda::getCurrentCUDAStream(a.get_device());
  CUTLASS_CHECK(gemm_op.can_implement(arguments));
  auto status = gemm_op(arguments, workspace.data_ptr(), stream);
  CUTLASS_CHECK(status);
}

template <typename ElementD>
using RunnerNoBiasFn = void (*)(torch::Tensor&, torch::Tensor const&,
                                torch::Tensor const&, torch::Tensor const&,
                                torch::Tensor const&);

template <typename ElementD>
using RunnerBiasFn = void (*)(torch::Tensor&, torch::Tensor const&,
                              torch::Tensor const&, torch::Tensor const&,
                              torch::Tensor const&, torch::Tensor const&);

template <typename Config, typename ElementD>
void run_cfg_no_bias(torch::Tensor& out,
                     torch::Tensor const& a,
                     torch::Tensor const& b,
                     torch::Tensor const& a_scales,
                     torch::Tensor const& b_scales) {
  run_cutlass_evt_gemm<typename Config::template Gemm<ElementD, ScaledEpilogue>>(
      out, a, b, a_scales, b_scales);
}

template <typename Config, typename ElementD>
void run_cfg_bias(torch::Tensor& out,
                  torch::Tensor const& a,
                  torch::Tensor const& b,
                  torch::Tensor const& a_scales,
                  torch::Tensor const& b_scales,
                  torch::Tensor const& bias) {
  run_cutlass_evt_gemm<
      typename Config::template Gemm<ElementD, ScaledEpilogueBias>>(
      out, a, b, a_scales, b_scales, bias);
}

struct Sm80EvtScheduler {
  static constexpr int kConfigCount = static_cast<int>(kSm80Configs.size());

  static int select_best_config(int m, int n, int k, bool verbose = false) {
    int cfg = select_by_analytic_cost_model(m, n, k);
    if (cfg < 0 || cfg >= kConfigCount) {
      cfg = 0;
    }

    if (verbose) {
      auto meta = kSm80Configs[cfg];
      std::printf("Problem size: (%d, %d, %d)\n", m, n, k);
      std::printf("Selected SM80 EVT config: %d -> TB(%d,%d,%d), stages=%d\n",
                  cfg, meta.tile_m, meta.tile_n, meta.tile_k, meta.stages);
    }

    return cfg;
  }

 private:
  static int select_by_analytic_cost_model(int m, int n, int k) {
    float best_cost = std::numeric_limits<float>::max();
    int best_cfg = 0;

    for (int i = 0; i < kConfigCount; ++i) {
      auto meta = kSm80Configs[i];
      float c = compute_proxy_cost(meta, m, n, k);
      if (c < best_cost) {
        best_cost = c;
        best_cfg = i;
      }
    }

    return best_cfg;
  }

  static float compute_proxy_cost(Sm80ConfigMeta const& cfg, int m, int n,
                                  int k) {
    if (k % 16 != 0) {
      return std::numeric_limits<float>::max() / 2;
    }

    int tiles_m = (m + cfg.tile_m - 1) / cfg.tile_m;
    int tiles_n = (n + cfg.tile_n - 1) / cfg.tile_n;
    int waves = (tiles_m * tiles_n + 108 - 1) / 108;

    float util_m = static_cast<float>(m) / (tiles_m * cfg.tile_m);
    float util_n = static_cast<float>(n) / (tiles_n * cfg.tile_n);
    float util_k = 1.0f -
                   static_cast<float>(k % cfg.tile_k) /
                       static_cast<float>(cfg.tile_k);
    float wave_util = static_cast<float>(tiles_m * tiles_n) / (waves * 108.0f);

    float util_score = 0.45f * util_m + 0.35f * util_n + 0.20f * util_k;
    float stage_bonus = 1.0f + 0.03f * cfg.stages;

    return 1.0f / (1e-4f + util_score * wave_util * stage_bonus);
  }
};

template <typename ElementD>
constexpr std::array<RunnerNoBiasFn<ElementD>,
                     Sm80EvtScheduler::kConfigCount>
    kNoBiasRunners = {
        &run_cfg_no_bias<Sm80Cfg0, ElementD>,
        &run_cfg_no_bias<Sm80Cfg1, ElementD>,
        &run_cfg_no_bias<Sm80Cfg2, ElementD>,
        &run_cfg_no_bias<Sm80Cfg3, ElementD>,
        &run_cfg_no_bias<Sm80Cfg4, ElementD>,
        &run_cfg_no_bias<Sm80Cfg5, ElementD>,
};

template <typename ElementD>
constexpr std::array<RunnerBiasFn<ElementD>,
                     Sm80EvtScheduler::kConfigCount>
    kBiasRunners = {
        &run_cfg_bias<Sm80Cfg0, ElementD>,
        &run_cfg_bias<Sm80Cfg1, ElementD>,
        &run_cfg_bias<Sm80Cfg2, ElementD>,
        &run_cfg_bias<Sm80Cfg3, ElementD>,
        &run_cfg_bias<Sm80Cfg4, ElementD>,
        &run_cfg_bias<Sm80Cfg5, ElementD>,
};

template <typename ElementD>
void cutlass_scaled_mm_mikpoly_dispatch_unified(
    torch::Tensor& out, torch::Tensor const& a, torch::Tensor const& b,
    torch::Tensor const& a_scales, torch::Tensor const& b_scales,
    std::optional<torch::Tensor> const& bias, bool verbose, bool use_bias_epilogue) {
  int m = a.size(0);
  int n = b.size(1);
  int k = a.size(1);

  int cfg = Sm80EvtScheduler::select_best_config(m, n, k, verbose);
  TORCH_CHECK(cfg >= 0 && cfg < Sm80EvtScheduler::kConfigCount,
              "Invalid SM80 EVT config id: ", cfg);

  if (use_bias_epilogue) {
    TORCH_CHECK(bias.has_value(), "Bias epilogue selected but bias is missing");
    kBiasRunners<ElementD>[cfg](out, a, b, a_scales, b_scales, bias.value());
  } else {
    kNoBiasRunners<ElementD>[cfg](out, a, b, a_scales, b_scales);
  }
}

}  // namespace

void cutlass_scaled_mm_mikpoly_unified(
    torch::Tensor& out, torch::Tensor const& a, torch::Tensor const& b,
    torch::Tensor const& a_scales, torch::Tensor const& b_scales,
    std::optional<torch::Tensor> const& bias, bool verbose) {
  TORCH_CHECK(a.dtype() == torch::kInt8, "A must be int8");
  TORCH_CHECK(b.dtype() == torch::kInt8, "B must be int8");
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32,
              "A scales must be float32");
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32,
              "B scales must be float32");

  auto a_contig = a.contiguous();
  auto b_contig = b.contiguous();

  if (out.dtype() == torch::kFloat16) {
    using ElementD = cutlass::half_t;
    if (bias.has_value()) {
      TORCH_CHECK(bias->dtype() == out.dtype(), "Bias dtype mismatch");
      cutlass_scaled_mm_mikpoly_dispatch_unified<ElementD>(
          out, a_contig, b_contig, a_scales, b_scales, bias, verbose, true);
    } else {
      cutlass_scaled_mm_mikpoly_dispatch_unified<ElementD>(
          out, a_contig, b_contig, a_scales, b_scales, std::nullopt, verbose, false);
    }
    return;
  }

  if (out.dtype() == torch::kBFloat16) {
    using ElementD = cutlass::bfloat16_t;
    if (bias.has_value()) {
      TORCH_CHECK(bias->dtype() == out.dtype(), "Bias dtype mismatch");
      cutlass_scaled_mm_mikpoly_dispatch_unified<ElementD>(
          out, a_contig, b_contig, a_scales, b_scales, bias, verbose, true);
    } else {
      cutlass_scaled_mm_mikpoly_dispatch_unified<ElementD>(
          out, a_contig, b_contig, a_scales, b_scales, std::nullopt, verbose, false);
    }
    return;
  }

  TORCH_CHECK(false, "Unsupported output dtype");
}

int predict_best_sm80_evt_config(int m, int n, int k) {
  return Sm80EvtScheduler::select_best_config(m, n, k, false);
}

std::tuple<int, int, int, int> get_sm80_evt_config(int config_id) {
  TORCH_CHECK(config_id >= 0 && config_id < Sm80EvtScheduler::kConfigCount,
              "Invalid config id: ", config_id);
  auto cfg = kSm80Configs[config_id];
  return std::make_tuple(cfg.tile_m, cfg.tile_n, cfg.tile_k, 16);
}

void analyze_kernel_selection(
    std::vector<std::tuple<int, int, int>> problem_sizes) {
  std::printf("\n=== SM80 EVT Config Selection Analysis ===\n");
  std::printf("%-15s %-10s %-20s\n", "Problem Size", "Config", "Tile");

  for (const auto& [m, n, k] : problem_sizes) {
    int cfg = Sm80EvtScheduler::select_best_config(m, n, k, false);
    auto meta = kSm80Configs[cfg];
    std::printf("(%4d,%4d,%4d) %-10d %3dx%3dx%3d\n", m, n, k, cfg,
                meta.tile_m, meta.tile_n, meta.tile_k);
  }
}

}  // namespace moonpoly::int8
