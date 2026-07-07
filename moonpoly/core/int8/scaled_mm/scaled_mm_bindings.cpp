#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <optional>
#include <tuple>

// Forward declarations
namespace moonpoly::int8 {
    void cutlass_scaled_mm_mikpoly_unified(
        torch::Tensor& out,
        torch::Tensor const& a,
        torch::Tensor const& b,
        torch::Tensor const& a_scales,
        torch::Tensor const& b_scales,
        std::optional<torch::Tensor> const& bias,
        bool verbose);

    int predict_best_sm80_evt_config(int m, int n, int k);
    std::tuple<int, int, int, int> get_sm80_evt_config(int config_id);
}

torch::Tensor scaled_mm_mikpoly(
    torch::Tensor& out,
    torch::Tensor const& a,
    torch::Tensor const& b,
    torch::Tensor const& a_scales,
    torch::Tensor const& b_scales,
    std::optional<torch::Tensor> const& bias) {

    // Ensure we're on the correct device
    c10::cuda::CUDAGuard device_guard(a.device());

    // Input validation
    TORCH_CHECK(a.is_cuda(), "Input A must be on CUDA device");
    TORCH_CHECK(b.is_cuda(), "Input B must be on CUDA device");
    TORCH_CHECK(a.dtype() == torch::kInt8, "Input A must be INT8");
    TORCH_CHECK(b.dtype() == torch::kInt8, "Input B must be INT8");
    TORCH_CHECK(a_scales.dtype() == torch::kFloat32, "A scales must be FP32");
    TORCH_CHECK(b_scales.dtype() == torch::kFloat32, "B scales must be FP32");

    // Check dimensions
    TORCH_CHECK(a.dim() == 2, "Input A must be 2D");
    TORCH_CHECK(b.dim() == 2, "Input B must be 2D");
    // B is expected as [N, K] row-major (equivalent to logical [K, N] column-major).
    TORCH_CHECK(a.size(1) == b.size(1), "Dimension mismatch for matrix multiplication");

    // Check output
    TORCH_CHECK(out.is_cuda(), "Output must be on CUDA device");
    TORCH_CHECK(out.dim() == 2, "Output must be 2D");
    TORCH_CHECK(out.size(0) == a.size(0), "Output M dimension mismatch");
    TORCH_CHECK(out.size(1) == b.size(0), "Output N dimension mismatch");
    TORCH_CHECK(out.dtype() == torch::kFloat16 || out.dtype() == torch::kBFloat16,
                "Output must be FP16 or BF16");

    // Check scales dimensions
    TORCH_CHECK(a_scales.numel() == 1 ||
                (a_scales.dim() == 2 && a_scales.size(0) == a.size(0) && a_scales.size(1) == 1),
                "A scales must be scalar or per-row");
    TORCH_CHECK(b_scales.numel() == 1 ||
                (b_scales.dim() == 2 && b_scales.size(0) == 1 && b_scales.size(1) == out.size(1)),
                "B scales must be scalar or per-column");

    // Check bias if provided
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda(), "Bias must be on CUDA device");
        TORCH_CHECK(bias->dtype() == out.dtype(), "Bias dtype must match output dtype");
        TORCH_CHECK(bias->dim() <= 2, "Bias must be 1D or 2D");
        if (bias->dim() == 2) {
            TORCH_CHECK(bias->size(0) == 1, "Bias must be broadcastable");
        }
        TORCH_CHECK(bias->size(-1) == out.size(1), "Bias dimension mismatch");
    }

    // Get CUDA device properties
    cudaDeviceProp props;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&props, device);

    // Check compute capability
    int cc = props.major * 10 + props.minor;
    if (cc >= 80 && cc < 90) {
        // SM80 path (Ampere): unified path with cost-model-first scheduler.
        moonpoly::int8::cutlass_scaled_mm_mikpoly_unified(
            out, a, b, a_scales, b_scales, bias, false);
    } else {
        TORCH_CHECK(false,
            "Unsupported GPU architecture. This implementation requires SM80 (Ampere). "
            "Your GPU has compute capability ", props.major, ".", props.minor);
    }

    return out;
}

// Kernel prediction function
int predict_best_kernel(int m, int n, int k) {
    return moonpoly::int8::predict_best_sm80_evt_config(m, n, k);
}

// Get kernel configuration
std::tuple<int, int, int, int> get_kernel_config(int kernel_id) {
    return moonpoly::int8::get_sm80_evt_config(kernel_id);
}
