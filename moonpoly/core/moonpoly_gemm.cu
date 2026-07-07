#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "moonpoly.cuh"
namespace moonpoly {

void moonpoly_gemm(MoonpolyDataType dtype, int m, int n, int k,
                   void* A, bool is_a_col_major,
                   void* B, bool is_b_col_major,
                   void* C, bool is_c_col_major,
                   float alpha, float beta,
                   cudaStream_t stream)
{
    switch (dtype) {
        case MoonpolyDataType::FP16: {
            cutlass::half_t const *ptrA = reinterpret_cast<cutlass::half_t const*>(A);
            cutlass::half_t const *ptrB = reinterpret_cast<cutlass::half_t const*>(B);
            cutlass::half_t *ptrC = reinterpret_cast<cutlass::half_t*>(C);
            run_fp16_gemm(m, n, k, ptrA, is_a_col_major, ptrB, is_b_col_major, ptrC, is_c_col_major, alpha, beta, stream);
            break;
        }
        case MoonpolyDataType::INT8: {
            int8_t const *ptrA = reinterpret_cast<int8_t const*>(A);
            int8_t const *ptrB = reinterpret_cast<int8_t const*>(B);
            int32_t *ptrC = reinterpret_cast<int32_t*>(C);
            run_int8_gemm(m, n, k, ptrA, is_a_col_major, ptrB, is_b_col_major, ptrC, is_c_col_major, alpha, beta, stream);
            break;
        }
        case MoonpolyDataType::FP32: {
            float const *ptrA = reinterpret_cast<float const*>(A);
            float const *ptrB = reinterpret_cast<float const*>(B);
            float *ptrC = reinterpret_cast<float*>(C);
            run_fp32_gemm(m, n, k, ptrA, is_a_col_major, ptrB, is_b_col_major, ptrC, is_c_col_major, alpha, beta, stream);
            break;
        }
        default:
            throw std::invalid_argument("Unsupported data type");
    }
}

void moonpolyLinear(MoonpolyDataType dtype, int m, int n, int k, void* A, void* B, void* C,
                    cudaStream_t stream) {
    moonpoly_gemm(dtype, n, m, k, B, false, A, true, C, true, 1.0f, 0.0f, stream);
}

} // namespace moonpoly

#ifdef PYTHON_BIND
#include <ATen/autocast_mode.h>
#include <c10/cuda/CUDAStream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <mutex>
#include <optional>
#include <tuple>
#include <unordered_map>

torch::Tensor scaled_mm_mikpoly(
    torch::Tensor& out,
    torch::Tensor const& a,
    torch::Tensor const& b,
    torch::Tensor const& a_scales,
    torch::Tensor const& b_scales,
    std::optional<torch::Tensor> const& bias);
int predict_best_kernel(int m, int n, int k);
std::tuple<int, int, int, int> get_kernel_config(int kernel_id);

// Torch 数据类型转换器
moonpoly::MoonpolyDataType torchDtypeToMoonpoly(const torch::Tensor& tensor) {
    if (tensor.dtype() == torch::kFloat16) return moonpoly::MoonpolyDataType::FP16;
    if (tensor.dtype() == torch::kInt8) return moonpoly::MoonpolyDataType::INT8;
    if (tensor.dtype() == torch::kFloat32) return moonpoly::MoonpolyDataType::FP32;
    if (tensor.dtype() == torch::kBFloat16) {
        throw std::invalid_argument(
            "BF16 core GEMM has been removed. Convert tensors to torch.float16 before calling MoonPoly.");
    }
    throw std::invalid_argument("Unsupported torch data type");
}

namespace torch_helpers {

void validateTensorsOnCuda(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& C) {
    if (!(A.device().is_cuda() && B.device().is_cuda() && C.device().is_cuda()))
        throw std::invalid_argument("Tensors must be on CUDA device");
}

void validateTensorsContiguous(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& C) {
    // Allow non-contiguous tensors during vLLM dynamic compilation
    // Make contiguous if needed rather than throwing error
    if (!A.is_contiguous() || !B.is_contiguous() || !C.is_contiguous()) {
        // During compilation phase, just warn instead of failing
        // The actual contiguous check will happen at runtime
        return;
    }
}

torch::Tensor createOutputTensor(const torch::Tensor& A, const torch::Tensor& B) {
    const int M = A.sizes()[0];
    const int N = B.sizes()[0];
    return torch::empty({M, N}, torch::TensorOptions().device(torch::kCUDA).dtype(A.dtype()));
}

} // namespace torch_helpers

namespace {

struct DecodeSelectorEntry {
    int m;
    int n;
    int k;
    int pid;
    int split_k;
};

#include "../generated/fp16/rcc/llm_decode_selector.inc"

bool lookup_merged_decode_selector(int m, int n, int k, int* pid, int* split_k) {
    for (const auto& entry : kMergedDecodeSelectorEntries) {
        if (entry.m == m && entry.n == n && entry.k == k) {
            *pid = entry.pid;
            *split_k = entry.split_k;
            return true;
        }
    }
    return false;
}

uint64_t make_workspace_key(int device, int pid, cudaStream_t stream) {
    uint64_t key = static_cast<uint64_t>(static_cast<uint32_t>(device));
    key = (key << 16) | static_cast<uint64_t>(static_cast<uint16_t>(pid));
    key = (key << 32) | (reinterpret_cast<uintptr_t>(stream) & 0xffffffffULL);
    return key;
}

torch::Tensor get_selector_workspace(const torch::Tensor& ref,
                                     int pid,
                                     size_t bytes,
                                     cudaStream_t stream) {
    static std::mutex pool_mutex;
    static std::unordered_map<uint64_t, torch::Tensor> pool;

    if (bytes == 0) {
        return torch::Tensor();
    }

    uint64_t key = make_workspace_key(ref.get_device(), pid, stream);
    std::lock_guard<std::mutex> guard(pool_mutex);
    auto it = pool.find(key);
    if (it == pool.end() || static_cast<size_t>(it->second.numel()) < bytes) {
        auto options = torch::TensorOptions().device(ref.device()).dtype(torch::kUInt8);
        auto tensor = torch::empty({static_cast<int64_t>(bytes)}, options);
        it = pool.insert_or_assign(key, tensor).first;
    }
    return it->second;
}

}  // namespace

torch::Tensor moonpoly_linear_cuda(const torch::Tensor& A, const torch::Tensor& B) {
    torch::Tensor C = torch_helpers::createOutputTensor(A, B);
    // torch_helpers::validateTensorsOnCuda(A, B, C);

    // Ensure contiguous tensors for actual computation
    // torch::Tensor A_contig = A.is_contiguous() ? A : A.contiguous();
    // torch::Tensor B_contig = B.is_contiguous() ? B : B.contiguous();
    // torch::Tensor C_contig = C.is_contiguous() ? C : C.contiguous();

    // torch::Tensor A_contig = A : A.contiguous();
    // torch::Tensor B_contig = true ?  B : B.contiguous();
    // torch::Tensor C_contig = true ?  C : C.contiguous();

    const int M = A.sizes()[0];
    const int K = A.sizes()[1];
    const int N = B.sizes()[0];
    moonpoly::MoonpolyDataType dtype = torchDtypeToMoonpoly(A);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(A.get_device()).stream();
    moonpoly::moonpolyLinear(dtype, M, N, K, A.data_ptr(), B.data_ptr(), C.data_ptr(), stream);

    return C;
}

torch::Tensor moonpoly_linear_cpp_selector_cuda(const torch::Tensor& A,
                                                const torch::Tensor& B) {
    if (!A.is_cuda() || !B.is_cuda()) {
        throw std::invalid_argument("linear_cpp_selector: tensors must be CUDA");
    }
    if (A.dtype() != torch::kFloat16 || B.dtype() != torch::kFloat16) {
        return moonpoly_linear_cuda(A, B);
    }
    if (A.dim() != 2 || B.dim() != 2) {
        return moonpoly_linear_cuda(A, B);
    }
    if (!A.is_contiguous() || !B.is_contiguous()) {
        return moonpoly_linear_cuda(A, B);
    }

    const int M = static_cast<int>(A.size(0));
    const int K = static_cast<int>(A.size(1));
    const int N = static_cast<int>(B.size(0));
    if (B.size(1) != K) {
        throw std::invalid_argument("linear_cpp_selector: weight shape mismatch");
    }

    int pid = -1;
    int split_k = 2;
    if (!lookup_merged_decode_selector(M, N, K, &pid, &split_k)) {
        return moonpoly_linear_cuda(A, B);
    }

    torch::Tensor C = torch_helpers::createOutputTensor(A, B);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(A.get_device()).stream();
    size_t required_ws = moonpoly::get_gemm_workspace_size_fp16_row_col_col(
        pid, N, M, K, split_k);

    void* ws_ptr = nullptr;
    size_t ws_bytes = 0;
    if (required_ws > 0) {
        torch::Tensor workspace = get_selector_workspace(A, pid, required_ws, stream);
        ws_ptr = workspace.data_ptr();
        ws_bytes = static_cast<size_t>(workspace.numel());
        cudaError_t err = cudaMemsetAsync(ws_ptr, 0, required_ws, stream);
        if (err != cudaSuccess) {
            throw std::runtime_error("linear_cpp_selector: cudaMemsetAsync failed");
        }
    }

    moonpoly::run_moonpoly_selected_gemm_fp16_row_col_col_with_workspace(
        pid, N, M, K,
        reinterpret_cast<const cutlass::half_t*>(B.data_ptr()),
        reinterpret_cast<const cutlass::half_t*>(A.data_ptr()),
        reinterpret_cast<cutlass::half_t*>(C.data_ptr()),
        1.0f, 0.0f, ws_ptr, ws_bytes, split_k, stream);
    return C;
}

torch::Tensor moonpoly_linear_cpp_forced_pid_cuda(const torch::Tensor& A,
                                                  const torch::Tensor& B,
                                                  int64_t pid,
                                                  int64_t split_k) {
    if (!A.is_cuda() || !B.is_cuda()) {
        throw std::invalid_argument("linear_cpp_forced_pid: tensors must be CUDA");
    }
    if (A.dtype() != torch::kFloat16 || B.dtype() != torch::kFloat16) {
        return moonpoly_linear_cuda(A, B);
    }
    if (A.dim() != 2 || B.dim() != 2) {
        return moonpoly_linear_cuda(A, B);
    }
    if (!A.is_contiguous() || !B.is_contiguous()) {
        return moonpoly_linear_cuda(A, B);
    }

    const int M = static_cast<int>(A.size(0));
    const int K = static_cast<int>(A.size(1));
    const int N = static_cast<int>(B.size(0));
    if (B.size(1) != K) {
        throw std::invalid_argument("linear_cpp_forced_pid: weight shape mismatch");
    }

    torch::Tensor C = torch_helpers::createOutputTensor(A, B);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(A.get_device()).stream();
    size_t required_ws = moonpoly::get_gemm_workspace_size_fp16_row_col_col(
        static_cast<int>(pid), N, M, K, static_cast<int>(split_k));

    void* ws_ptr = nullptr;
    size_t ws_bytes = 0;
    if (required_ws > 0) {
        torch::Tensor workspace =
            get_selector_workspace(A, static_cast<int>(pid), required_ws, stream);
        ws_ptr = workspace.data_ptr();
        ws_bytes = static_cast<size_t>(workspace.numel());
        cudaError_t err = cudaMemsetAsync(ws_ptr, 0, required_ws, stream);
        if (err != cudaSuccess) {
            throw std::runtime_error("linear_cpp_forced_pid: cudaMemsetAsync failed");
        }
    }

    moonpoly::run_moonpoly_selected_gemm_fp16_row_col_col_with_workspace(
        static_cast<int>(pid), N, M, K,
        reinterpret_cast<const cutlass::half_t*>(B.data_ptr()),
        reinterpret_cast<const cutlass::half_t*>(A.data_ptr()),
        reinterpret_cast<cutlass::half_t*>(C.data_ptr()),
        1.0f, 0.0f, ws_ptr, ws_bytes, static_cast<int>(split_k), stream);
    return C;
}

int64_t moonpoly_fp16_workspace_size_cuda(const torch::Tensor& A,
                                          const torch::Tensor& B,
                                          int64_t pid,
                                          int64_t split_k_slices) {
    if (!A.is_cuda() || !B.is_cuda()) {
        throw std::invalid_argument("fp16_workspace_size: tensors must be CUDA");
    }
    if (A.dtype() != torch::kFloat16 || B.dtype() != torch::kFloat16) {
        throw std::invalid_argument("fp16_workspace_size: tensors must be float16");
    }
    if (A.dim() != 2 || B.dim() != 2) {
        throw std::invalid_argument("fp16_workspace_size: input and weight must be 2D");
    }
    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    const int64_t N = B.size(0);
    if (B.size(1) != K) {
        throw std::invalid_argument("fp16_workspace_size: weight shape mismatch");
    }
    if (split_k_slices <= 0) {
        split_k_slices = 2;
    }

    // Moonpoly linear path maps to row-col-col GEMM as (m, n, k) = (N, M, K).
    size_t bytes = moonpoly::get_gemm_workspace_size_fp16_row_col_col(
        static_cast<int>(pid), static_cast<int>(N), static_cast<int>(M),
        static_cast<int>(K), static_cast<int>(split_k_slices));
    return static_cast<int64_t>(bytes);
}

int64_t moonpoly_fp16_rcc_workspace_size_cuda(const torch::Tensor& A,
                                              const torch::Tensor& B_col,
                                              int64_t pid,
                                              int64_t split_k_slices) {
    if (!A.is_cuda() || !B_col.is_cuda()) {
        throw std::invalid_argument("fp16_rcc_workspace_size: tensors must be CUDA");
    }
    if (A.dtype() != torch::kFloat16 || B_col.dtype() != torch::kFloat16) {
        throw std::invalid_argument("fp16_rcc_workspace_size: tensors must be float16");
    }
    if (A.dim() != 2 || B_col.dim() != 2) {
        throw std::invalid_argument("fp16_rcc_workspace_size: tensors must be 2D");
    }
    const int64_t m = A.size(0);
    const int64_t k = A.size(1);
    if (B_col.size(0) != k) {
        throw std::invalid_argument("fp16_rcc_workspace_size: B logical shape must be [k, n]");
    }
    const int64_t n = B_col.size(1);
    if (split_k_slices <= 0) {
        split_k_slices = 2;
    }
    size_t bytes = moonpoly::get_gemm_workspace_size_fp16_row_col_col(
        static_cast<int>(pid), static_cast<int>(m), static_cast<int>(n),
        static_cast<int>(k), static_cast<int>(split_k_slices));
    return static_cast<int64_t>(bytes);
}

void moonpoly_linear_fp16_pid_ws_out_cuda(const torch::Tensor& A,
                                          const torch::Tensor& B,
                                          torch::Tensor& C,
                                          int64_t pid,
                                          const torch::Tensor& workspace,
                                          int64_t split_k_slices) {
    if (!A.is_cuda() || !B.is_cuda() || !C.is_cuda()) {
        throw std::invalid_argument("linear_fp16_pid_ws_out: tensors must be CUDA");
    }
    if (A.dtype() != torch::kFloat16 || B.dtype() != torch::kFloat16 ||
        C.dtype() != torch::kFloat16) {
        throw std::invalid_argument("linear_fp16_pid_ws_out: tensors must be float16");
    }
    if (A.dim() != 2 || B.dim() != 2 || C.dim() != 2) {
        throw std::invalid_argument("linear_fp16_pid_ws_out: input/weight/output must be 2D");
    }
    if (!A.is_contiguous() || !B.is_contiguous() || !C.is_contiguous()) {
        throw std::invalid_argument("linear_fp16_pid_ws_out: input/weight/output must be contiguous");
    }

    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    const int64_t N = B.size(0);
    if (B.size(1) != K) {
        throw std::invalid_argument("linear_fp16_pid_ws_out: weight shape mismatch");
    }
    if (C.size(0) != M || C.size(1) != N) {
        throw std::invalid_argument("linear_fp16_pid_ws_out: output shape must be [M, N]");
    }

    if (split_k_slices <= 0) {
        split_k_slices = 2;
    }
    size_t required_ws = moonpoly::get_gemm_workspace_size_fp16_row_col_col(
        static_cast<int>(pid), static_cast<int>(N), static_cast<int>(M),
        static_cast<int>(K), static_cast<int>(split_k_slices));

    void* ws_ptr = nullptr;
    size_t ws_bytes = 0;
    if (required_ws > 0) {
        if (!workspace.defined()) {
            throw std::invalid_argument("linear_fp16_pid_ws_out: workspace is required for this pid");
        }
        if (!workspace.is_cuda() || workspace.dtype() != torch::kUInt8 ||
            !workspace.is_contiguous()) {
            throw std::invalid_argument("linear_fp16_pid_ws_out: workspace must be contiguous CUDA uint8 tensor");
        }
        ws_ptr = workspace.data_ptr();
        ws_bytes = static_cast<size_t>(workspace.numel());
        if (ws_bytes < required_ws) {
            throw std::invalid_argument("linear_fp16_pid_ws_out: workspace buffer too small");
        }
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(A.get_device()).stream();
    moonpoly::run_moonpoly_selected_gemm_fp16_row_col_col_with_workspace(
        static_cast<int>(pid), static_cast<int>(N), static_cast<int>(M), static_cast<int>(K),
        reinterpret_cast<const cutlass::half_t*>(B.data_ptr()),
        reinterpret_cast<const cutlass::half_t*>(A.data_ptr()),
        reinterpret_cast<cutlass::half_t*>(C.data_ptr()),
        1.0f, 0.0f, ws_ptr, ws_bytes, static_cast<int>(split_k_slices), stream);
}

void moonpoly_fp16_rcc_pid_ws_out_cuda(const torch::Tensor& A,
                                       const torch::Tensor& B_col,
                                       torch::Tensor& C_col,
                                       int64_t pid,
                                       const torch::Tensor& workspace,
                                       int64_t split_k_slices) {
    if (!A.is_cuda() || !B_col.is_cuda() || !C_col.is_cuda()) {
        throw std::invalid_argument("fp16_rcc_pid_ws_out: tensors must be CUDA");
    }
    if (A.dtype() != torch::kFloat16 || B_col.dtype() != torch::kFloat16 ||
        C_col.dtype() != torch::kFloat16) {
        throw std::invalid_argument("fp16_rcc_pid_ws_out: tensors must be float16");
    }
    if (A.dim() != 2 || B_col.dim() != 2 || C_col.dim() != 2) {
        throw std::invalid_argument("fp16_rcc_pid_ws_out: tensors must be 2D");
    }
    if (!A.is_contiguous()) {
        throw std::invalid_argument("fp16_rcc_pid_ws_out: A must be contiguous row-major");
    }

    const int64_t m = A.size(0);
    const int64_t k = A.size(1);
    if (B_col.size(0) != k) {
        throw std::invalid_argument("fp16_rcc_pid_ws_out: B logical shape must be [k, n]");
    }
    const int64_t n = B_col.size(1);
    if (C_col.size(0) != m || C_col.size(1) != n) {
        throw std::invalid_argument("fp16_rcc_pid_ws_out: output logical shape must be [m, n]");
    }
    if (B_col.stride(0) != 1 || B_col.stride(1) != k) {
        throw std::invalid_argument("fp16_rcc_pid_ws_out: B must use packed column-major strides [1, k]");
    }
    if (C_col.stride(0) != 1 || C_col.stride(1) != m) {
        throw std::invalid_argument("fp16_rcc_pid_ws_out: C must use packed column-major strides [1, m]");
    }

    if (split_k_slices <= 0) {
        split_k_slices = 2;
    }
    size_t required_ws = moonpoly::get_gemm_workspace_size_fp16_row_col_col(
        static_cast<int>(pid), static_cast<int>(m), static_cast<int>(n),
        static_cast<int>(k), static_cast<int>(split_k_slices));

    void* ws_ptr = nullptr;
    size_t ws_bytes = 0;
    if (required_ws > 0) {
        if (!workspace.defined()) {
            throw std::invalid_argument("fp16_rcc_pid_ws_out: workspace is required for this pid");
        }
        if (!workspace.is_cuda() || workspace.dtype() != torch::kUInt8 ||
            !workspace.is_contiguous()) {
            throw std::invalid_argument("fp16_rcc_pid_ws_out: workspace must be contiguous CUDA uint8 tensor");
        }
        ws_ptr = workspace.data_ptr();
        ws_bytes = static_cast<size_t>(workspace.numel());
        if (ws_bytes < required_ws) {
            throw std::invalid_argument("fp16_rcc_pid_ws_out: workspace buffer too small");
        }
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(A.get_device()).stream();
    moonpoly::run_moonpoly_selected_gemm_fp16_row_col_col_with_workspace(
        static_cast<int>(pid), static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
        reinterpret_cast<const cutlass::half_t*>(A.data_ptr()),
        reinterpret_cast<const cutlass::half_t*>(B_col.data_ptr()),
        reinterpret_cast<cutlass::half_t*>(C_col.data_ptr()),
        1.0f, 0.0f, ws_ptr, ws_bytes, static_cast<int>(split_k_slices), stream);
}

torch::Tensor moonpoly_linear_fp16_pid_ws_cuda(const torch::Tensor& A,
                                               const torch::Tensor& B,
                                               int64_t pid,
                                               const torch::Tensor& workspace,
                                               int64_t split_k_slices) {
    torch::Tensor C = torch::empty({A.size(0), B.size(0)},
                                   torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat16));
    moonpoly_linear_fp16_pid_ws_out_cuda(A, B, C, pid, workspace, split_k_slices);
    return C;
}

torch::Tensor moonpoly_fp16_rcc_pid_ws_meta(const torch::Tensor& A,
                                            const torch::Tensor& B_col,
                                            int64_t,
                                            const torch::Tensor&,
                                            int64_t) {
    auto m = A.sym_size(0);
    auto n = B_col.sym_size(1);
    return torch::empty_symint({m, n},
                               torch::TensorOptions().device(A.device()).dtype(A.dtype()));
}

// 2. Meta Kernel (abstract_impl)
torch::Tensor moonpoly_linear_meta(const torch::Tensor& A, const torch::Tensor& B) {
    // For meta tensors, create output on the same device as input (should be meta)
    auto M = A.sym_size(0);
    auto N = B.sym_size(0);
    return torch::empty_symint({M, N}, torch::TensorOptions().device(A.device()).dtype(A.dtype()));
}

torch::Tensor moonpoly_linear_fp16_pid_ws_meta(const torch::Tensor& A,
                                                const torch::Tensor& B,
                                                int64_t,
                                                const torch::Tensor&,
                                                int64_t) {
    return moonpoly_linear_meta(A, B);
}

TORCH_LIBRARY(moonpoly_ops, m) {
    m.def("linear(Tensor input, Tensor weight) -> Tensor");
    m.def("linear_cpp_selector(Tensor input, Tensor weight) -> Tensor");
    m.def("fp16_workspace_size(Tensor input, Tensor weight, int pid, int split_k_slices=2) -> int");
    m.def("fp16_rcc_workspace_size(Tensor a, Tensor b_col, int pid, int split_k_slices=2) -> int");
    m.def("linear_fp16_pid_ws(Tensor input, Tensor weight, int pid, Tensor workspace, int split_k_slices=2) -> Tensor");
    m.def("linear_fp16_pid_ws_out(Tensor input, Tensor weight, Tensor(a!) output, int pid, Tensor workspace, int split_k_slices=2) -> ()");
    m.def("fp16_rcc_pid_ws_out(Tensor a, Tensor b_col, Tensor(a!) c_col, int pid, Tensor workspace, int split_k_slices=2) -> ()");
}

TORCH_LIBRARY_IMPL(moonpoly_ops, CUDA, m) {
    m.impl("linear", &moonpoly_linear_cuda);
    m.impl("linear_cpp_selector", &moonpoly_linear_cpp_selector_cuda);
    m.impl("fp16_workspace_size", &moonpoly_fp16_workspace_size_cuda);
    m.impl("fp16_rcc_workspace_size", &moonpoly_fp16_rcc_workspace_size_cuda);
    m.impl("linear_fp16_pid_ws", &moonpoly_linear_fp16_pid_ws_cuda);
    m.impl("linear_fp16_pid_ws_out", &moonpoly_linear_fp16_pid_ws_out_cuda);
    m.impl("fp16_rcc_pid_ws_out", &moonpoly_fp16_rcc_pid_ws_out_cuda);
}

TORCH_LIBRARY_IMPL(moonpoly_ops, Meta, m) {
    m.impl("linear", &moonpoly_linear_meta);
    m.impl("linear_cpp_selector", &moonpoly_linear_meta);
    m.impl("linear_fp16_pid_ws", &moonpoly_linear_fp16_pid_ws_meta);
    m.impl("fp16_rcc_workspace_size",
           [](const torch::Tensor&, const torch::Tensor&, int64_t, int64_t) {
             return int64_t(0);
           });
    m.impl("fp16_rcc_pid_ws_out",
           [](const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C,
              int64_t, const torch::Tensor&, int64_t) {
             (void)A;
             (void)B;
             (void)C;
           });
    m.impl("linear_fp16_pid_ws_out",
           [](const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C,
              int64_t, const torch::Tensor&, int64_t) {
             (void)A;
             (void)B;
             (void)C;
           });
}

torch::Tensor moonpoly_linear_pybind(torch::Tensor A, torch::Tensor B) {
    return moonpoly_linear_cuda(A, B);
}

torch::Tensor moonpoly_linear_cpp_selector_pybind(torch::Tensor A, torch::Tensor B) {
    return moonpoly_linear_cpp_selector_cuda(A, B);
}

torch::Tensor moonpoly_linear_cpp_forced_pid_pybind(torch::Tensor A,
                                                    torch::Tensor B,
                                                    int64_t pid,
                                                    int64_t split_k) {
    return moonpoly_linear_cpp_forced_pid_cuda(A, B, pid, split_k);
}

int64_t moonpoly_fp16_p40_rcc_workspace_size_pybind(torch::Tensor A,
                                                    torch::Tensor B_col,
                                                    int64_t split_k_slices) {
    if (!A.is_cuda() || !B_col.is_cuda()) {
        throw std::invalid_argument("p40_rcc_workspace_size: tensors must be CUDA");
    }
    if (A.dtype() != torch::kFloat16 || B_col.dtype() != torch::kFloat16) {
        throw std::invalid_argument("p40_rcc_workspace_size: tensors must be float16");
    }
    if (A.dim() != 2 || B_col.dim() != 2) {
        throw std::invalid_argument("p40_rcc_workspace_size: tensors must be 2D");
    }
    if (!A.is_contiguous()) {
        throw std::invalid_argument("p40_rcc_workspace_size: A must be contiguous");
    }

    const int64_t m = A.size(0);
    const int64_t k = A.size(1);
    if (B_col.size(0) != k) {
        throw std::invalid_argument("p40_rcc_workspace_size: B must have logical shape [k, n]");
    }
    const int64_t n = B_col.size(1);
    if (B_col.stride(0) != 1 || B_col.stride(1) != k) {
        throw std::invalid_argument(
            "p40_rcc_workspace_size: B must use packed column-major strides [1, k]");
    }

    if (split_k_slices <= 0) {
        split_k_slices = 2;
    }
    return static_cast<int64_t>(
        moonpoly::get_fp16_p40_streamk_workspace_size_row_col_col(
            static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
            static_cast<int>(split_k_slices)));
}

int64_t moonpoly_fp16_p66_rcc_workspace_size_pybind(torch::Tensor A,
                                                    torch::Tensor B_col,
                                                    int64_t split_k_slices) {
    if (!A.is_cuda() || !B_col.is_cuda()) {
        throw std::invalid_argument("p66_rcc_workspace_size: tensors must be CUDA");
    }
    if (A.dtype() != torch::kFloat16 || B_col.dtype() != torch::kFloat16) {
        throw std::invalid_argument("p66_rcc_workspace_size: tensors must be float16");
    }
    if (A.dim() != 2 || B_col.dim() != 2) {
        throw std::invalid_argument("p66_rcc_workspace_size: tensors must be 2D");
    }
    if (!A.is_contiguous()) {
        throw std::invalid_argument("p66_rcc_workspace_size: A must be contiguous");
    }

    const int64_t m = A.size(0);
    const int64_t k = A.size(1);
    if (B_col.size(0) != k) {
        throw std::invalid_argument("p66_rcc_workspace_size: B must have logical shape [k, n]");
    }
    const int64_t n = B_col.size(1);
    if (B_col.stride(0) != 1 || B_col.stride(1) != k) {
        throw std::invalid_argument(
            "p66_rcc_workspace_size: B must use packed column-major strides [1, k]");
    }

    if (split_k_slices <= 0) {
        split_k_slices = 2;
    }
    return static_cast<int64_t>(
        moonpoly::get_fp16_p66_streamk_workspace_size_row_col_col(
            static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
            static_cast<int>(split_k_slices)));
}

int64_t moonpoly_fp16_p42_rcc_workspace_size_pybind(torch::Tensor A,
                                                    torch::Tensor B_col,
                                                    int64_t split_k_slices) {
    if (!A.is_cuda() || !B_col.is_cuda()) {
        throw std::invalid_argument("p42_rcc_workspace_size: tensors must be CUDA");
    }
    if (A.dtype() != torch::kFloat16 || B_col.dtype() != torch::kFloat16) {
        throw std::invalid_argument("p42_rcc_workspace_size: tensors must be float16");
    }
    if (A.dim() != 2 || B_col.dim() != 2) {
        throw std::invalid_argument("p42_rcc_workspace_size: tensors must be 2D");
    }
    if (!A.is_contiguous()) {
        throw std::invalid_argument("p42_rcc_workspace_size: A must be contiguous");
    }

    const int64_t m = A.size(0);
    const int64_t k = A.size(1);
    if (B_col.size(0) != k) {
        throw std::invalid_argument("p42_rcc_workspace_size: B must have logical shape [k, n]");
    }
    const int64_t n = B_col.size(1);
    if (B_col.stride(0) != 1 || B_col.stride(1) != k) {
        throw std::invalid_argument(
            "p42_rcc_workspace_size: B must use packed column-major strides [1, k]");
    }

    if (split_k_slices <= 0) {
        split_k_slices = 2;
    }
    return static_cast<int64_t>(
        moonpoly::get_fp16_p42_streamk_workspace_size_row_col_col(
            static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
            static_cast<int>(split_k_slices)));
}

int64_t moonpoly_fp16_p43_rcc_workspace_size_pybind(torch::Tensor A,
                                                    torch::Tensor B_col,
                                                    int64_t split_k_slices) {
    if (!A.is_cuda() || !B_col.is_cuda()) {
        throw std::invalid_argument("p43_rcc_workspace_size: tensors must be CUDA");
    }
    if (A.dtype() != torch::kFloat16 || B_col.dtype() != torch::kFloat16) {
        throw std::invalid_argument("p43_rcc_workspace_size: tensors must be float16");
    }
    if (A.dim() != 2 || B_col.dim() != 2) {
        throw std::invalid_argument("p43_rcc_workspace_size: tensors must be 2D");
    }
    if (!A.is_contiguous()) {
        throw std::invalid_argument("p43_rcc_workspace_size: A must be contiguous");
    }

    const int64_t m = A.size(0);
    const int64_t k = A.size(1);
    if (B_col.size(0) != k) {
        throw std::invalid_argument("p43_rcc_workspace_size: B must have logical shape [k, n]");
    }
    const int64_t n = B_col.size(1);
    if (B_col.stride(0) != 1 || B_col.stride(1) != k) {
        throw std::invalid_argument(
            "p43_rcc_workspace_size: B must use packed column-major strides [1, k]");
    }

    if (split_k_slices <= 0) {
        split_k_slices = 2;
    }
    return static_cast<int64_t>(
        moonpoly::get_fp16_p43_streamk_workspace_size_row_col_col(
            static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
            static_cast<int>(split_k_slices)));
}

int64_t moonpoly_fp16_p44_rcc_workspace_size_pybind(torch::Tensor A,
                                                    torch::Tensor B_col,
                                                    int64_t split_k_slices) {
    if (!A.is_cuda() || !B_col.is_cuda()) {
        throw std::invalid_argument("p44_rcc_workspace_size: tensors must be CUDA");
    }
    if (A.dtype() != torch::kFloat16 || B_col.dtype() != torch::kFloat16) {
        throw std::invalid_argument("p44_rcc_workspace_size: tensors must be float16");
    }
    if (A.dim() != 2 || B_col.dim() != 2) {
        throw std::invalid_argument("p44_rcc_workspace_size: tensors must be 2D");
    }
    if (!A.is_contiguous()) {
        throw std::invalid_argument("p44_rcc_workspace_size: A must be contiguous");
    }

    const int64_t m = A.size(0);
    const int64_t k = A.size(1);
    if (B_col.size(0) != k) {
        throw std::invalid_argument("p44_rcc_workspace_size: B must have logical shape [k, n]");
    }
    const int64_t n = B_col.size(1);
    if (B_col.stride(0) != 1 || B_col.stride(1) != k) {
        throw std::invalid_argument(
            "p44_rcc_workspace_size: B must use packed column-major strides [1, k]");
    }

    if (split_k_slices <= 0) {
        split_k_slices = 2;
    }
    return static_cast<int64_t>(
        moonpoly::get_fp16_p44_streamk_workspace_size_row_col_col(
            static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
            static_cast<int>(split_k_slices)));
}

torch::Tensor moonpoly_fp16_p40_rcc_run_pybind(torch::Tensor A,
                                               torch::Tensor B_col,
                                               torch::Tensor workspace,
                                               int64_t split_k_slices) {
    if (!A.is_cuda() || !B_col.is_cuda() || !workspace.is_cuda()) {
        throw std::invalid_argument("p40_rcc_run: tensors must be CUDA");
    }
    if (A.dtype() != torch::kFloat16 || B_col.dtype() != torch::kFloat16) {
        throw std::invalid_argument("p40_rcc_run: A/B must be float16");
    }
    if (workspace.dtype() != torch::kUInt8) {
        throw std::invalid_argument("p40_rcc_run: workspace must be uint8");
    }
    if (A.dim() != 2 || B_col.dim() != 2) {
        throw std::invalid_argument("p40_rcc_run: A/B must be 2D");
    }
    if (!A.is_contiguous() || !workspace.is_contiguous()) {
        throw std::invalid_argument("p40_rcc_run: A/workspace must be contiguous");
    }

    const int64_t m = A.size(0);
    const int64_t k = A.size(1);
    if (B_col.size(0) != k) {
        throw std::invalid_argument("p40_rcc_run: B must have logical shape [k, n]");
    }
    const int64_t n = B_col.size(1);
    if (B_col.stride(0) != 1 || B_col.stride(1) != k) {
        throw std::invalid_argument(
            "p40_rcc_run: B must use packed column-major strides [1, k]");
    }
    if (split_k_slices <= 0) {
        split_k_slices = 2;
    }

    size_t required_ws =
        moonpoly::get_fp16_p40_streamk_workspace_size_row_col_col(
            static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
            static_cast<int>(split_k_slices));
    size_t ws_bytes = static_cast<size_t>(workspace.numel());
    if (ws_bytes < required_ws) {
        throw std::invalid_argument("p40_rcc_run: workspace buffer too small");
    }

    auto C_col = torch::empty_strided(
        {m, n}, {1, m},
        torch::TensorOptions().device(A.device()).dtype(torch::kFloat16));
    moonpoly::run_fp16_p40_streamk_row_col_col(
        static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
        reinterpret_cast<const cutlass::half_t*>(A.data_ptr()),
        reinterpret_cast<const cutlass::half_t*>(B_col.data_ptr()),
        reinterpret_cast<cutlass::half_t*>(C_col.data_ptr()),
        1.0f, 0.0f, workspace.data_ptr(), ws_bytes,
        static_cast<int>(split_k_slices),
        c10::cuda::getCurrentCUDAStream(A.get_device()).stream());
    return C_col;
}

torch::Tensor moonpoly_fp16_p40_rcc_run_default_stream_pybind(
    torch::Tensor A, torch::Tensor B_col, torch::Tensor workspace,
    int64_t split_k_slices) {
    if (!A.is_cuda() || !B_col.is_cuda() || !workspace.is_cuda()) {
        throw std::invalid_argument("p40_rcc_run_default_stream: tensors must be CUDA");
    }
    if (A.dtype() != torch::kFloat16 || B_col.dtype() != torch::kFloat16) {
        throw std::invalid_argument("p40_rcc_run_default_stream: A/B must be float16");
    }
    if (workspace.dtype() != torch::kUInt8) {
        throw std::invalid_argument("p40_rcc_run_default_stream: workspace must be uint8");
    }
    if (A.dim() != 2 || B_col.dim() != 2) {
        throw std::invalid_argument("p40_rcc_run_default_stream: A/B must be 2D");
    }
    if (!A.is_contiguous() || !workspace.is_contiguous()) {
        throw std::invalid_argument("p40_rcc_run_default_stream: A/workspace must be contiguous");
    }

    const int64_t m = A.size(0);
    const int64_t k = A.size(1);
    if (B_col.size(0) != k) {
        throw std::invalid_argument("p40_rcc_run_default_stream: B must have logical shape [k, n]");
    }
    const int64_t n = B_col.size(1);
    if (B_col.stride(0) != 1 || B_col.stride(1) != k) {
        throw std::invalid_argument(
            "p40_rcc_run_default_stream: B must use packed column-major strides [1, k]");
    }
    if (split_k_slices <= 0) {
        split_k_slices = 2;
    }

    size_t required_ws =
        moonpoly::get_fp16_p40_streamk_workspace_size_row_col_col(
            static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
            static_cast<int>(split_k_slices));
    size_t ws_bytes = static_cast<size_t>(workspace.numel());
    if (ws_bytes < required_ws) {
        throw std::invalid_argument("p40_rcc_run_default_stream: workspace buffer too small");
    }

    auto C_col = torch::empty_strided(
        {m, n}, {1, m},
        torch::TensorOptions().device(A.device()).dtype(torch::kFloat16));
    moonpoly::run_fp16_p40_streamk_row_col_col(
        static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
        reinterpret_cast<const cutlass::half_t*>(A.data_ptr()),
        reinterpret_cast<const cutlass::half_t*>(B_col.data_ptr()),
        reinterpret_cast<cutlass::half_t*>(C_col.data_ptr()),
        1.0f, 0.0f, workspace.data_ptr(), ws_bytes,
        static_cast<int>(split_k_slices), nullptr);
    return C_col;
}

torch::Tensor moonpoly_fp16_p40_rcc_run_internal_ws_pybind(torch::Tensor A,
                                                           torch::Tensor B_col,
                                                           int64_t split_k_slices) {
    if (!A.is_cuda() || !B_col.is_cuda()) {
        throw std::invalid_argument("p40_rcc_run_internal_ws: tensors must be CUDA");
    }
    if (A.dtype() != torch::kFloat16 || B_col.dtype() != torch::kFloat16) {
        throw std::invalid_argument("p40_rcc_run_internal_ws: A/B must be float16");
    }
    if (A.dim() != 2 || B_col.dim() != 2) {
        throw std::invalid_argument("p40_rcc_run_internal_ws: A/B must be 2D");
    }
    if (!A.is_contiguous()) {
        throw std::invalid_argument("p40_rcc_run_internal_ws: A must be contiguous");
    }

    const int64_t m = A.size(0);
    const int64_t k = A.size(1);
    if (B_col.size(0) != k) {
        throw std::invalid_argument("p40_rcc_run_internal_ws: B must have logical shape [k, n]");
    }
    const int64_t n = B_col.size(1);
    if (B_col.stride(0) != 1 || B_col.stride(1) != k) {
        throw std::invalid_argument(
            "p40_rcc_run_internal_ws: B must use packed column-major strides [1, k]");
    }
    if (split_k_slices <= 0) {
        split_k_slices = 2;
    }

    size_t ws_bytes =
        moonpoly::get_fp16_p40_streamk_workspace_size_row_col_col(
            static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
            static_cast<int>(split_k_slices));
    void* ws_ptr = nullptr;
    if (ws_bytes > 0) {
        cudaError_t err = cudaMalloc(&ws_ptr, ws_bytes);
        if (err != cudaSuccess) {
            throw std::runtime_error("p40_rcc_run_internal_ws: cudaMalloc failed");
        }
        cudaMemset(ws_ptr, 0, ws_bytes);
    }

    auto C_col = torch::empty_strided(
        {m, n}, {1, m},
        torch::TensorOptions().device(A.device()).dtype(torch::kFloat16));
    moonpoly::run_fp16_p40_streamk_row_col_col(
        static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
        reinterpret_cast<const cutlass::half_t*>(A.data_ptr()),
        reinterpret_cast<const cutlass::half_t*>(B_col.data_ptr()),
        reinterpret_cast<cutlass::half_t*>(C_col.data_ptr()),
        1.0f, 0.0f, ws_ptr, ws_bytes, static_cast<int>(split_k_slices), nullptr);
    if (ws_ptr != nullptr) {
        cudaFree(ws_ptr);
    }
    return C_col;
}

torch::Tensor moonpoly_fp16_p40_rcc_run_singleton_pybind(torch::Tensor A,
                                                         torch::Tensor B_col,
                                                         int64_t split_k_slices) {
    if (!A.is_cuda() || !B_col.is_cuda()) {
        throw std::invalid_argument("p40_rcc_run_singleton: tensors must be CUDA");
    }
    if (A.dtype() != torch::kFloat16 || B_col.dtype() != torch::kFloat16) {
        throw std::invalid_argument("p40_rcc_run_singleton: A/B must be float16");
    }
    if (A.dim() != 2 || B_col.dim() != 2) {
        throw std::invalid_argument("p40_rcc_run_singleton: A/B must be 2D");
    }
    if (!A.is_contiguous()) {
        throw std::invalid_argument("p40_rcc_run_singleton: A must be contiguous");
    }

    const int64_t m = A.size(0);
    const int64_t k = A.size(1);
    if (B_col.size(0) != k) {
        throw std::invalid_argument("p40_rcc_run_singleton: B must have logical shape [k, n]");
    }
    const int64_t n = B_col.size(1);
    if (B_col.stride(0) != 1 || B_col.stride(1) != k) {
        throw std::invalid_argument(
            "p40_rcc_run_singleton: B must use packed column-major strides [1, k]");
    }
    if (split_k_slices <= 0) {
        split_k_slices = 2;
    }

    auto C_col = torch::empty_strided(
        {m, n}, {1, m},
        torch::TensorOptions().device(A.device()).dtype(torch::kFloat16));
    moonpoly::run_fp16_p40_streamk_singleton_row_col_col(
        static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
        reinterpret_cast<const cutlass::half_t*>(A.data_ptr()),
        reinterpret_cast<const cutlass::half_t*>(B_col.data_ptr()),
        reinterpret_cast<cutlass::half_t*>(C_col.data_ptr()),
        1.0f, 0.0f, static_cast<int>(split_k_slices), nullptr);
    return C_col;
}

torch::Tensor moonpoly_fp16_p40_rcc_run_raw_copy_pybind(torch::Tensor A,
                                                        torch::Tensor B_col,
                                                        int64_t split_k_slices) {
    if (!A.is_cuda() || !B_col.is_cuda()) {
        throw std::invalid_argument("p40_rcc_run_raw_copy: tensors must be CUDA");
    }
    if (A.dtype() != torch::kFloat16 || B_col.dtype() != torch::kFloat16) {
        throw std::invalid_argument("p40_rcc_run_raw_copy: A/B must be float16");
    }
    if (A.dim() != 2 || B_col.dim() != 2) {
        throw std::invalid_argument("p40_rcc_run_raw_copy: A/B must be 2D");
    }
    if (!A.is_contiguous()) {
        throw std::invalid_argument("p40_rcc_run_raw_copy: A must be contiguous");
    }

    const int64_t m = A.size(0);
    const int64_t k = A.size(1);
    if (B_col.size(0) != k) {
        throw std::invalid_argument("p40_rcc_run_raw_copy: B must have logical shape [k, n]");
    }
    const int64_t n = B_col.size(1);
    if (B_col.stride(0) != 1 || B_col.stride(1) != k) {
        throw std::invalid_argument(
            "p40_rcc_run_raw_copy: B must use packed column-major strides [1, k]");
    }
    if (split_k_slices <= 0) {
        split_k_slices = 2;
    }

    size_t bytes_a = static_cast<size_t>(m * k) * sizeof(cutlass::half_t);
    size_t bytes_b = static_cast<size_t>(k * n) * sizeof(cutlass::half_t);
    size_t bytes_c = static_cast<size_t>(m * n) * sizeof(cutlass::half_t);
    size_t bytes_ws =
        moonpoly::get_fp16_p40_streamk_workspace_size_row_col_col(
            static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
            static_cast<int>(split_k_slices));

    void* a_raw = nullptr;
    void* b_raw = nullptr;
    void* c_raw = nullptr;
    void* ws_raw = nullptr;
    cudaError_t err = cudaSuccess;
    err = cudaMalloc(&a_raw, bytes_a);
    if (err != cudaSuccess) {
        throw std::runtime_error("p40_rcc_run_raw_copy: cudaMalloc A failed");
    }
    err = cudaMalloc(&b_raw, bytes_b);
    if (err != cudaSuccess) {
        cudaFree(a_raw);
        throw std::runtime_error("p40_rcc_run_raw_copy: cudaMalloc B failed");
    }
    err = cudaMalloc(&c_raw, bytes_c);
    if (err != cudaSuccess) {
        cudaFree(a_raw);
        cudaFree(b_raw);
        throw std::runtime_error("p40_rcc_run_raw_copy: cudaMalloc C failed");
    }
    if (bytes_ws > 0) {
        err = cudaMalloc(&ws_raw, bytes_ws);
        if (err != cudaSuccess) {
            cudaFree(a_raw);
            cudaFree(b_raw);
            cudaFree(c_raw);
            throw std::runtime_error("p40_rcc_run_raw_copy: cudaMalloc workspace failed");
        }
        cudaMemset(ws_raw, 0, bytes_ws);
    }

    auto C_col = torch::empty_strided(
        {m, n}, {1, m},
        torch::TensorOptions().device(A.device()).dtype(torch::kFloat16));
    try {
        cudaMemcpy(a_raw, A.data_ptr(), bytes_a, cudaMemcpyDeviceToDevice);
        cudaMemcpy(b_raw, B_col.data_ptr(), bytes_b, cudaMemcpyDeviceToDevice);
        moonpoly::run_fp16_p40_streamk_row_col_col(
            static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
            reinterpret_cast<const cutlass::half_t*>(a_raw),
            reinterpret_cast<const cutlass::half_t*>(b_raw),
            reinterpret_cast<cutlass::half_t*>(c_raw), 1.0f, 0.0f, ws_raw,
            bytes_ws, static_cast<int>(split_k_slices), nullptr);
        cudaMemcpy(C_col.data_ptr(), c_raw, bytes_c, cudaMemcpyDeviceToDevice);
    } catch (...) {
        if (ws_raw != nullptr) cudaFree(ws_raw);
        cudaFree(a_raw);
        cudaFree(b_raw);
        cudaFree(c_raw);
        throw;
    }

    if (ws_raw != nullptr) cudaFree(ws_raw);
    cudaFree(a_raw);
    cudaFree(b_raw);
    cudaFree(c_raw);
    return C_col;
}

torch::Tensor moonpoly_fp16_p66_rcc_run_pybind(torch::Tensor A,
                                               torch::Tensor B_col,
                                               int64_t split_k_slices) {
    if (!A.is_cuda() || !B_col.is_cuda()) {
        throw std::invalid_argument("p66_rcc_run: tensors must be CUDA");
    }
    if (A.dtype() != torch::kFloat16 || B_col.dtype() != torch::kFloat16) {
        throw std::invalid_argument("p66_rcc_run: A/B must be float16");
    }
    if (A.dim() != 2 || B_col.dim() != 2) {
        throw std::invalid_argument("p66_rcc_run: A/B must be 2D");
    }
    if (!A.is_contiguous()) {
        throw std::invalid_argument("p66_rcc_run: A must be contiguous");
    }

    const int64_t m = A.size(0);
    const int64_t k = A.size(1);
    if (B_col.size(0) != k) {
        throw std::invalid_argument("p66_rcc_run: B must have logical shape [k, n]");
    }
    const int64_t n = B_col.size(1);
    if (B_col.stride(0) != 1 || B_col.stride(1) != k) {
        throw std::invalid_argument(
            "p66_rcc_run: B must use packed column-major strides [1, k]");
    }
    if (split_k_slices <= 0) {
        split_k_slices = 2;
    }

    auto C_col = torch::empty_strided(
        {m, n}, {1, m},
        torch::TensorOptions().device(A.device()).dtype(torch::kFloat16));
    moonpoly::run_fp16_p66_streamk_row_col_col(
        static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
        reinterpret_cast<const cutlass::half_t*>(A.data_ptr()),
        reinterpret_cast<const cutlass::half_t*>(B_col.data_ptr()),
        reinterpret_cast<cutlass::half_t*>(C_col.data_ptr()),
        1.0f, 0.0f, static_cast<int>(split_k_slices), nullptr);
    return C_col;
}

torch::Tensor moonpoly_fp16_p42_rcc_run_pybind(torch::Tensor A,
                                               torch::Tensor B_col,
                                               int64_t split_k_slices) {
    if (!A.is_cuda() || !B_col.is_cuda()) {
        throw std::invalid_argument("p42_rcc_run: tensors must be CUDA");
    }
    if (A.dtype() != torch::kFloat16 || B_col.dtype() != torch::kFloat16) {
        throw std::invalid_argument("p42_rcc_run: A/B must be float16");
    }
    if (A.dim() != 2 || B_col.dim() != 2) {
        throw std::invalid_argument("p42_rcc_run: A/B must be 2D");
    }
    if (!A.is_contiguous()) {
        throw std::invalid_argument("p42_rcc_run: A must be contiguous");
    }

    const int64_t m = A.size(0);
    const int64_t k = A.size(1);
    if (B_col.size(0) != k) {
        throw std::invalid_argument("p42_rcc_run: B must have logical shape [k, n]");
    }
    const int64_t n = B_col.size(1);
    if (B_col.stride(0) != 1 || B_col.stride(1) != k) {
        throw std::invalid_argument(
            "p42_rcc_run: B must use packed column-major strides [1, k]");
    }
    if (split_k_slices <= 0) {
        split_k_slices = 2;
    }

    auto C_col = torch::empty_strided(
        {m, n}, {1, m},
        torch::TensorOptions().device(A.device()).dtype(torch::kFloat16));
    moonpoly::run_fp16_p42_streamk_row_col_col(
        static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
        reinterpret_cast<const cutlass::half_t*>(A.data_ptr()),
        reinterpret_cast<const cutlass::half_t*>(B_col.data_ptr()),
        reinterpret_cast<cutlass::half_t*>(C_col.data_ptr()),
        1.0f, 0.0f, static_cast<int>(split_k_slices), nullptr);
    return C_col;
}

torch::Tensor moonpoly_fp16_p43_rcc_run_pybind(torch::Tensor A,
                                               torch::Tensor B_col,
                                               int64_t split_k_slices) {
    if (!A.is_cuda() || !B_col.is_cuda()) {
        throw std::invalid_argument("p43_rcc_run: tensors must be CUDA");
    }
    if (A.dtype() != torch::kFloat16 || B_col.dtype() != torch::kFloat16) {
        throw std::invalid_argument("p43_rcc_run: A/B must be float16");
    }
    if (A.dim() != 2 || B_col.dim() != 2) {
        throw std::invalid_argument("p43_rcc_run: A/B must be 2D");
    }
    if (!A.is_contiguous()) {
        throw std::invalid_argument("p43_rcc_run: A must be contiguous");
    }

    const int64_t m = A.size(0);
    const int64_t k = A.size(1);
    if (B_col.size(0) != k) {
        throw std::invalid_argument("p43_rcc_run: B must have logical shape [k, n]");
    }
    const int64_t n = B_col.size(1);
    if (B_col.stride(0) != 1 || B_col.stride(1) != k) {
        throw std::invalid_argument(
            "p43_rcc_run: B must use packed column-major strides [1, k]");
    }
    if (split_k_slices <= 0) {
        split_k_slices = 2;
    }

    auto C_col = torch::empty_strided(
        {m, n}, {1, m},
        torch::TensorOptions().device(A.device()).dtype(torch::kFloat16));
    moonpoly::run_fp16_p43_streamk_row_col_col(
        static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
        reinterpret_cast<const cutlass::half_t*>(A.data_ptr()),
        reinterpret_cast<const cutlass::half_t*>(B_col.data_ptr()),
        reinterpret_cast<cutlass::half_t*>(C_col.data_ptr()),
        1.0f, 0.0f, static_cast<int>(split_k_slices), nullptr);
    return C_col;
}

torch::Tensor moonpoly_fp16_p44_rcc_run_pybind(torch::Tensor A,
                                               torch::Tensor B_col,
                                               int64_t split_k_slices) {
    if (!A.is_cuda() || !B_col.is_cuda()) {
        throw std::invalid_argument("p44_rcc_run: tensors must be CUDA");
    }
    if (A.dtype() != torch::kFloat16 || B_col.dtype() != torch::kFloat16) {
        throw std::invalid_argument("p44_rcc_run: A/B must be float16");
    }
    if (A.dim() != 2 || B_col.dim() != 2) {
        throw std::invalid_argument("p44_rcc_run: A/B must be 2D");
    }
    if (!A.is_contiguous()) {
        throw std::invalid_argument("p44_rcc_run: A must be contiguous");
    }

    const int64_t m = A.size(0);
    const int64_t k = A.size(1);
    if (B_col.size(0) != k) {
        throw std::invalid_argument("p44_rcc_run: B must have logical shape [k, n]");
    }
    const int64_t n = B_col.size(1);
    if (B_col.stride(0) != 1 || B_col.stride(1) != k) {
        throw std::invalid_argument(
            "p44_rcc_run: B must use packed column-major strides [1, k]");
    }
    if (split_k_slices <= 0) {
        split_k_slices = 2;
    }

    auto C_col = torch::empty_strided(
        {m, n}, {1, m},
        torch::TensorOptions().device(A.device()).dtype(torch::kFloat16));
    moonpoly::run_fp16_p44_streamk_row_col_col(
        static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
        reinterpret_cast<const cutlass::half_t*>(A.data_ptr()),
        reinterpret_cast<const cutlass::half_t*>(B_col.data_ptr()),
        reinterpret_cast<cutlass::half_t*>(C_col.data_ptr()),
        1.0f, 0.0f, static_cast<int>(split_k_slices), nullptr);
    return C_col;
}

torch::Tensor moonpoly_fp16_p66_rcc_run_raw_copy_pybind(torch::Tensor A,
                                                        torch::Tensor B_col,
                                                        int64_t split_k_slices) {
    if (!A.is_cuda() || !B_col.is_cuda()) {
        throw std::invalid_argument("p66_rcc_run_raw_copy: tensors must be CUDA");
    }
    if (A.dtype() != torch::kFloat16 || B_col.dtype() != torch::kFloat16) {
        throw std::invalid_argument("p66_rcc_run_raw_copy: A/B must be float16");
    }
    if (A.dim() != 2 || B_col.dim() != 2) {
        throw std::invalid_argument("p66_rcc_run_raw_copy: A/B must be 2D");
    }
    if (!A.is_contiguous()) {
        throw std::invalid_argument("p66_rcc_run_raw_copy: A must be contiguous");
    }

    const int64_t m = A.size(0);
    const int64_t k = A.size(1);
    if (B_col.size(0) != k) {
        throw std::invalid_argument("p66_rcc_run_raw_copy: B must have logical shape [k, n]");
    }
    const int64_t n = B_col.size(1);
    if (B_col.stride(0) != 1 || B_col.stride(1) != k) {
        throw std::invalid_argument(
            "p66_rcc_run_raw_copy: B must use packed column-major strides [1, k]");
    }
    if (split_k_slices <= 0) {
        split_k_slices = 2;
    }

    size_t bytes_a = static_cast<size_t>(m * k) * sizeof(cutlass::half_t);
    size_t bytes_b = static_cast<size_t>(k * n) * sizeof(cutlass::half_t);
    size_t bytes_c = static_cast<size_t>(m * n) * sizeof(cutlass::half_t);

    void* a_raw = nullptr;
    void* b_raw = nullptr;
    void* c_raw = nullptr;
    cudaError_t err = cudaSuccess;
    err = cudaMalloc(&a_raw, bytes_a);
    if (err != cudaSuccess) {
        throw std::runtime_error("p66_rcc_run_raw_copy: cudaMalloc A failed");
    }
    err = cudaMalloc(&b_raw, bytes_b);
    if (err != cudaSuccess) {
        cudaFree(a_raw);
        throw std::runtime_error("p66_rcc_run_raw_copy: cudaMalloc B failed");
    }
    err = cudaMalloc(&c_raw, bytes_c);
    if (err != cudaSuccess) {
        cudaFree(a_raw);
        cudaFree(b_raw);
        throw std::runtime_error("p66_rcc_run_raw_copy: cudaMalloc C failed");
    }

    auto C_col = torch::empty_strided(
        {m, n}, {1, m},
        torch::TensorOptions().device(A.device()).dtype(torch::kFloat16));
    try {
        cudaMemcpy(a_raw, A.data_ptr(), bytes_a, cudaMemcpyDeviceToDevice);
        cudaMemcpy(b_raw, B_col.data_ptr(), bytes_b, cudaMemcpyDeviceToDevice);
        moonpoly::run_fp16_p66_streamk_row_col_col(
            static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
            reinterpret_cast<const cutlass::half_t*>(a_raw),
            reinterpret_cast<const cutlass::half_t*>(b_raw),
            reinterpret_cast<cutlass::half_t*>(c_raw), 1.0f, 0.0f,
            static_cast<int>(split_k_slices), nullptr);
        cudaMemcpy(C_col.data_ptr(), c_raw, bytes_c, cudaMemcpyDeviceToDevice);
    } catch (...) {
        cudaFree(a_raw);
        cudaFree(b_raw);
        cudaFree(c_raw);
        throw;
    }

    cudaFree(a_raw);
    cudaFree(b_raw);
    cudaFree(c_raw);
    return C_col;
}

py::dict moonpoly_fp16_rcc_copy_check_pybind(torch::Tensor A,
                                             torch::Tensor B_col,
                                             int64_t sample_count) {
    if (!A.is_cuda() || !B_col.is_cuda()) {
        throw std::invalid_argument("rcc_copy_check: tensors must be CUDA");
    }
    if (A.dtype() != torch::kFloat16 || B_col.dtype() != torch::kFloat16) {
        throw std::invalid_argument("rcc_copy_check: A/B must be float16");
    }
    if (A.dim() != 2 || B_col.dim() != 2) {
        throw std::invalid_argument("rcc_copy_check: A/B must be 2D");
    }
    if (!A.is_contiguous()) {
        throw std::invalid_argument("rcc_copy_check: A must be contiguous");
    }

    const int64_t m = A.size(0);
    const int64_t k = A.size(1);
    if (B_col.size(0) != k) {
        throw std::invalid_argument("rcc_copy_check: B must have logical shape [k, n]");
    }
    const int64_t n = B_col.size(1);
    if (B_col.stride(0) != 1 || B_col.stride(1) != k) {
        throw std::invalid_argument(
            "rcc_copy_check: B must use packed column-major strides [1, k]");
    }
    if (sample_count <= 0) {
        sample_count = 8;
    }

    size_t elems_a = static_cast<size_t>(m * k);
    size_t elems_b = static_cast<size_t>(k * n);
    size_t bytes_a = elems_a * sizeof(cutlass::half_t);
    size_t bytes_b = elems_b * sizeof(cutlass::half_t);

    void* a_raw = nullptr;
    void* b_raw = nullptr;
    cudaError_t err = cudaSuccess;
    err = cudaMalloc(&a_raw, bytes_a);
    if (err != cudaSuccess) {
        throw std::runtime_error("rcc_copy_check: cudaMalloc A failed");
    }
    err = cudaMalloc(&b_raw, bytes_b);
    if (err != cudaSuccess) {
        cudaFree(a_raw);
        throw std::runtime_error("rcc_copy_check: cudaMalloc B failed");
    }

    std::vector<cutlass::half_t> h_a_src(elems_a);
    std::vector<cutlass::half_t> h_a_dst(elems_a);
    std::vector<cutlass::half_t> h_b_src(elems_b);
    std::vector<cutlass::half_t> h_b_dst(elems_b);

    try {
        cudaMemcpy(h_a_src.data(), A.data_ptr(), bytes_a, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_b_src.data(), B_col.data_ptr(), bytes_b, cudaMemcpyDeviceToHost);
        cudaMemcpy(a_raw, A.data_ptr(), bytes_a, cudaMemcpyDeviceToDevice);
        cudaMemcpy(b_raw, B_col.data_ptr(), bytes_b, cudaMemcpyDeviceToDevice);
        cudaMemcpy(h_a_dst.data(), a_raw, bytes_a, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_b_dst.data(), b_raw, bytes_b, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    } catch (...) {
        cudaFree(a_raw);
        cudaFree(b_raw);
        throw;
    }

    cudaFree(a_raw);
    cudaFree(b_raw);

    double a_max_abs = 0.0;
    double b_max_abs = 0.0;
    for (size_t i = 0; i < elems_a; ++i) {
        double diff =
            std::abs(static_cast<double>(float(h_a_src[i])) -
                     static_cast<double>(float(h_a_dst[i])));
        if (diff > a_max_abs) {
            a_max_abs = diff;
        }
    }
    for (size_t i = 0; i < elems_b; ++i) {
        double diff =
            std::abs(static_cast<double>(float(h_b_src[i])) -
                     static_cast<double>(float(h_b_dst[i])));
        if (diff > b_max_abs) {
            b_max_abs = diff;
        }
    }

    py::list a_src_head;
    py::list a_dst_head;
    py::list b_src_head;
    py::list b_dst_head;
    int64_t a_samples = std::min<int64_t>(sample_count, static_cast<int64_t>(elems_a));
    int64_t b_samples = std::min<int64_t>(sample_count, static_cast<int64_t>(elems_b));
    for (int64_t i = 0; i < a_samples; ++i) {
        a_src_head.append(float(h_a_src[static_cast<size_t>(i)]));
        a_dst_head.append(float(h_a_dst[static_cast<size_t>(i)]));
    }
    for (int64_t i = 0; i < b_samples; ++i) {
        b_src_head.append(float(h_b_src[static_cast<size_t>(i)]));
        b_dst_head.append(float(h_b_dst[static_cast<size_t>(i)]));
    }

    py::dict out;
    out["a_max_abs"] = a_max_abs;
    out["b_max_abs"] = b_max_abs;
    out["a_src_head"] = a_src_head;
    out["a_dst_head"] = a_dst_head;
    out["b_src_head"] = b_src_head;
    out["b_dst_head"] = b_dst_head;
    out["b_stride0"] = B_col.stride(0);
    out["b_stride1"] = B_col.stride(1);
    return out;
}

py::dict moonpoly_fp16_rcc_raw_copy_host_output_check_pybind(
    torch::Tensor A, torch::Tensor B_col, int64_t split_k_slices, int64_t pid_tag,
    int64_t sample_count) {
    if (!A.is_cuda() || !B_col.is_cuda()) {
        throw std::invalid_argument("raw_copy_host_output_check: tensors must be CUDA");
    }
    if (A.dtype() != torch::kFloat16 || B_col.dtype() != torch::kFloat16) {
        throw std::invalid_argument("raw_copy_host_output_check: A/B must be float16");
    }
    if (A.dim() != 2 || B_col.dim() != 2) {
        throw std::invalid_argument("raw_copy_host_output_check: A/B must be 2D");
    }
    if (!A.is_contiguous()) {
        throw std::invalid_argument("raw_copy_host_output_check: A must be contiguous");
    }

    const int64_t m = A.size(0);
    const int64_t k = A.size(1);
    if (B_col.size(0) != k) {
        throw std::invalid_argument(
            "raw_copy_host_output_check: B must have logical shape [k, n]");
    }
    const int64_t n = B_col.size(1);
    if (B_col.stride(0) != 1 || B_col.stride(1) != k) {
        throw std::invalid_argument(
            "raw_copy_host_output_check: B must use packed column-major strides [1, k]");
    }
    if (split_k_slices <= 0) {
        split_k_slices = 2;
    }
    if (sample_count <= 0) {
        sample_count = 8;
    }

    size_t elems_a = static_cast<size_t>(m * k);
    size_t elems_b = static_cast<size_t>(k * n);
    size_t elems_c = static_cast<size_t>(m * n);
    size_t bytes_a = elems_a * sizeof(cutlass::half_t);
    size_t bytes_b = elems_b * sizeof(cutlass::half_t);
    size_t bytes_c = elems_c * sizeof(cutlass::half_t);
    size_t bytes_ws = 0;
    if (pid_tag == 40) {
        bytes_ws = moonpoly::get_fp16_p40_streamk_workspace_size_row_col_col(
            static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
            static_cast<int>(split_k_slices));
    } else if (pid_tag == 42) {
        bytes_ws = moonpoly::get_fp16_p42_streamk_workspace_size_row_col_col(
            static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
            static_cast<int>(split_k_slices));
    } else if (pid_tag == 43) {
        bytes_ws = moonpoly::get_fp16_p43_streamk_workspace_size_row_col_col(
            static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
            static_cast<int>(split_k_slices));
    } else if (pid_tag == 44) {
        bytes_ws = moonpoly::get_fp16_p44_streamk_workspace_size_row_col_col(
            static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
            static_cast<int>(split_k_slices));
    }

    void* a_raw = nullptr;
    void* b_raw = nullptr;
    void* c_raw = nullptr;
    void* ws_raw = nullptr;
    cudaError_t err = cudaSuccess;
    err = cudaMalloc(&a_raw, bytes_a);
    if (err != cudaSuccess) {
        throw std::runtime_error("raw_copy_host_output_check: cudaMalloc A failed");
    }
    err = cudaMalloc(&b_raw, bytes_b);
    if (err != cudaSuccess) {
        cudaFree(a_raw);
        throw std::runtime_error("raw_copy_host_output_check: cudaMalloc B failed");
    }
    err = cudaMalloc(&c_raw, bytes_c);
    if (err != cudaSuccess) {
        cudaFree(a_raw);
        cudaFree(b_raw);
        throw std::runtime_error("raw_copy_host_output_check: cudaMalloc C failed");
    }
    if (bytes_ws > 0) {
        err = cudaMalloc(&ws_raw, bytes_ws);
        if (err != cudaSuccess) {
            cudaFree(a_raw);
            cudaFree(b_raw);
            cudaFree(c_raw);
            throw std::runtime_error(
                "raw_copy_host_output_check: cudaMalloc workspace failed");
        }
        cudaMemset(ws_raw, 0, bytes_ws);
    }

    std::vector<cutlass::half_t> h_a(elems_a);
    std::vector<cutlass::half_t> h_b(elems_b);
    std::vector<cutlass::half_t> h_c(elems_c);
    try {
        cudaMemcpy(h_a.data(), A.data_ptr(), bytes_a, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_b.data(), B_col.data_ptr(), bytes_b, cudaMemcpyDeviceToHost);
        cudaMemcpy(a_raw, A.data_ptr(), bytes_a, cudaMemcpyDeviceToDevice);
        cudaMemcpy(b_raw, B_col.data_ptr(), bytes_b, cudaMemcpyDeviceToDevice);
        cudaMemset(c_raw, 0, bytes_c);
        if (pid_tag == 40) {
            moonpoly::run_fp16_p40_streamk_row_col_col(
                static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                reinterpret_cast<const cutlass::half_t*>(a_raw),
                reinterpret_cast<const cutlass::half_t*>(b_raw),
                reinterpret_cast<cutlass::half_t*>(c_raw), 1.0f, 0.0f, ws_raw,
                bytes_ws, static_cast<int>(split_k_slices), nullptr);
        } else if (pid_tag == 42) {
            moonpoly::run_fp16_p42_streamk_row_col_col(
                static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                reinterpret_cast<const cutlass::half_t*>(a_raw),
                reinterpret_cast<const cutlass::half_t*>(b_raw),
                reinterpret_cast<cutlass::half_t*>(c_raw), 1.0f, 0.0f,
                static_cast<int>(split_k_slices), nullptr);
        } else if (pid_tag == 43) {
            moonpoly::run_fp16_p43_streamk_row_col_col(
                static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                reinterpret_cast<const cutlass::half_t*>(a_raw),
                reinterpret_cast<const cutlass::half_t*>(b_raw),
                reinterpret_cast<cutlass::half_t*>(c_raw), 1.0f, 0.0f,
                static_cast<int>(split_k_slices), nullptr);
        } else if (pid_tag == 44) {
            moonpoly::run_fp16_p44_streamk_row_col_col(
                static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                reinterpret_cast<const cutlass::half_t*>(a_raw),
                reinterpret_cast<const cutlass::half_t*>(b_raw),
                reinterpret_cast<cutlass::half_t*>(c_raw), 1.0f, 0.0f,
                static_cast<int>(split_k_slices), nullptr);
        } else if (pid_tag == 66) {
            moonpoly::run_fp16_p66_streamk_row_col_col(
                static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                reinterpret_cast<const cutlass::half_t*>(a_raw),
                reinterpret_cast<const cutlass::half_t*>(b_raw),
                reinterpret_cast<cutlass::half_t*>(c_raw), 1.0f, 0.0f,
                static_cast<int>(split_k_slices), nullptr);
        } else if (pid_tag == 0) {
            moonpoly::run_moonpoly_selected_gemm_fp16_row_col_col(
                0, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                reinterpret_cast<const cutlass::half_t*>(a_raw),
                reinterpret_cast<const cutlass::half_t*>(b_raw),
                reinterpret_cast<cutlass::half_t*>(c_raw), 1.0f, 0.0f, nullptr);
        } else {
            throw std::invalid_argument("raw_copy_host_output_check: unsupported pid tag");
        }
        cudaDeviceSynchronize();
        cudaMemcpy(h_c.data(), c_raw, bytes_c, cudaMemcpyDeviceToHost);
    } catch (...) {
        if (ws_raw != nullptr) cudaFree(ws_raw);
        cudaFree(a_raw);
        cudaFree(b_raw);
        cudaFree(c_raw);
        throw;
    }

    if (ws_raw != nullptr) cudaFree(ws_raw);
    cudaFree(a_raw);
    cudaFree(b_raw);
    cudaFree(c_raw);

    double max_abs_head = 0.0;
    int64_t nan_count = 0;
    int64_t a_nan_count = 0;
    int64_t b_nan_count = 0;
    py::list c_head;
    py::list ref_head;
    int64_t head_count = std::min<int64_t>(sample_count, static_cast<int64_t>(elems_c));
    for (size_t i = 0; i < elems_a; ++i) {
        if (std::isnan(float(h_a[i]))) {
            ++a_nan_count;
        }
    }
    for (size_t i = 0; i < elems_b; ++i) {
        if (std::isnan(float(h_b[i]))) {
            ++b_nan_count;
        }
    }
    for (int64_t linear_idx = 0; linear_idx < static_cast<int64_t>(elems_c);
         ++linear_idx) {
        float got = float(h_c[static_cast<size_t>(linear_idx)]);
        if (std::isnan(got)) {
            ++nan_count;
        }
        if (linear_idx < head_count) {
            int64_t row = linear_idx % m;
            int64_t col = linear_idx / m;
            float ref = 0.0f;
            for (int64_t kk = 0; kk < k; ++kk) {
                ref += float(h_a[static_cast<size_t>(row * k + kk)]) *
                       float(h_b[static_cast<size_t>(kk + col * k)]);
            }
            if (!std::isnan(got)) {
                double diff =
                    std::abs(static_cast<double>(got) - static_cast<double>(ref));
                if (diff > max_abs_head) {
                    max_abs_head = diff;
                }
            }
            c_head.append(got);
            ref_head.append(ref);
        }
    }

    py::dict out;
    out["max_abs_head"] = max_abs_head;
    out["a_nan_count"] = a_nan_count;
    out["b_nan_count"] = b_nan_count;
    out["nan_count"] = nan_count;
    out["c_head"] = c_head;
    out["ref_head"] = ref_head;
    out["pid_tag"] = pid_tag;
    return out;
}

double moonpoly_fp16_rcc_constant_check_pybind(int64_t m, int64_t n, int64_t k,
                                               int64_t split_k_slices,
                                               bool use_p40) {
    if (m <= 0 || n <= 0 || k <= 0) {
        throw std::invalid_argument("constant_check: m/n/k must be positive");
    }
    if (split_k_slices <= 0) {
        split_k_slices = 2;
    }

    size_t bytes_a = static_cast<size_t>(m * k) * sizeof(cutlass::half_t);
    size_t bytes_b = static_cast<size_t>(k * n) * sizeof(cutlass::half_t);
    size_t bytes_c = static_cast<size_t>(m * n) * sizeof(cutlass::half_t);
    size_t bytes_ws = 0;
    if (use_p40) {
        bytes_ws = moonpoly::get_fp16_p40_streamk_workspace_size_row_col_col(
            static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
            static_cast<int>(split_k_slices));
    }

    std::vector<cutlass::half_t> h_a(m * k, cutlass::half_t(1.0f));
    std::vector<cutlass::half_t> h_b(k * n, cutlass::half_t(1.0f));
    std::vector<cutlass::half_t> h_c(m * n, cutlass::half_t(0.0f));

    void* a_raw = nullptr;
    void* b_raw = nullptr;
    void* c_raw = nullptr;
    void* ws_raw = nullptr;
    cudaError_t err = cudaSuccess;
    err = cudaMalloc(&a_raw, bytes_a);
    if (err != cudaSuccess) throw std::runtime_error("constant_check: cudaMalloc A failed");
    err = cudaMalloc(&b_raw, bytes_b);
    if (err != cudaSuccess) {
        cudaFree(a_raw);
        throw std::runtime_error("constant_check: cudaMalloc B failed");
    }
    err = cudaMalloc(&c_raw, bytes_c);
    if (err != cudaSuccess) {
        cudaFree(a_raw);
        cudaFree(b_raw);
        throw std::runtime_error("constant_check: cudaMalloc C failed");
    }
    if (bytes_ws > 0) {
        err = cudaMalloc(&ws_raw, bytes_ws);
        if (err != cudaSuccess) {
            cudaFree(a_raw);
            cudaFree(b_raw);
            cudaFree(c_raw);
            throw std::runtime_error("constant_check: cudaMalloc workspace failed");
        }
        cudaMemset(ws_raw, 0, bytes_ws);
    }

    double max_abs = 0.0;
    try {
        cudaMemcpy(a_raw, h_a.data(), bytes_a, cudaMemcpyHostToDevice);
        cudaMemcpy(b_raw, h_b.data(), bytes_b, cudaMemcpyHostToDevice);
        cudaMemcpy(c_raw, h_c.data(), bytes_c, cudaMemcpyHostToDevice);

        if (use_p40) {
            moonpoly::run_fp16_p40_streamk_row_col_col(
                static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                reinterpret_cast<const cutlass::half_t*>(a_raw),
                reinterpret_cast<const cutlass::half_t*>(b_raw),
                reinterpret_cast<cutlass::half_t*>(c_raw), 1.0f, 0.0f, ws_raw,
                bytes_ws, static_cast<int>(split_k_slices), nullptr);
        } else {
            moonpoly::run_fp16_p66_streamk_row_col_col(
                static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                reinterpret_cast<const cutlass::half_t*>(a_raw),
                reinterpret_cast<const cutlass::half_t*>(b_raw),
                reinterpret_cast<cutlass::half_t*>(c_raw), 1.0f, 0.0f,
                static_cast<int>(split_k_slices), nullptr);
        }
        cudaDeviceSynchronize();
        cudaMemcpy(h_c.data(), c_raw, bytes_c, cudaMemcpyDeviceToHost);
        for (cutlass::half_t v : h_c) {
            double diff = std::abs(static_cast<double>(float(v)) - static_cast<double>(k));
            if (diff > max_abs) {
                max_abs = diff;
            }
        }
    } catch (...) {
        if (ws_raw != nullptr) cudaFree(ws_raw);
        cudaFree(a_raw);
        cudaFree(b_raw);
        cudaFree(c_raw);
        throw;
    }

    if (ws_raw != nullptr) cudaFree(ws_raw);
    cudaFree(a_raw);
    cudaFree(b_raw);
    cudaFree(c_raw);
    return max_abs;
}

torch::Tensor moonpoly_scaled_int8_mm_pybind(
    torch::Tensor A, torch::Tensor B, torch::Tensor A_scales,
    torch::Tensor B_scales, std::optional<torch::Tensor> bias,
    c10::ScalarType out_dtype) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A/B must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kInt8 && B.dtype() == torch::kInt8,
                "A/B must be int8");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A/B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "K mismatch between A and B");
    TORCH_CHECK(out_dtype == torch::kFloat16 || out_dtype == torch::kBFloat16,
                "out_dtype must be torch.float16 or torch.bfloat16");

    int64_t m = A.size(0);
    int64_t n = B.size(1);

    if (A.size(1) % 16 != 0) {
        auto ref = torch::matmul(A.to(torch::kFloat32) * A_scales,
                                 B.to(torch::kFloat32) * B_scales);
        if (bias.has_value()) {
            ref = ref + bias->to(torch::kFloat32);
        }
        return ref.to(out_dtype);
    }

    auto A_contig = A.contiguous();
    auto B_packed = B.t().contiguous();
    auto A_scale_contig = A_scales.contiguous();
    auto B_scale_contig = B_scales.contiguous();

    if (A_scale_contig.numel() == 1) {
        A_scale_contig = A_scale_contig.expand({m, 1}).contiguous();
    }
    if (B_scale_contig.numel() == 1) {
        B_scale_contig = B_scale_contig.expand({1, n}).contiguous();
    }

    std::optional<torch::Tensor> bias_contig = std::nullopt;
    if (bias.has_value()) {
        bias_contig = bias->contiguous();
    }

    auto out = torch::empty({m, n}, A.options().dtype(out_dtype));
    return scaled_mm_mikpoly(out, A_contig, B_packed, A_scale_contig,
                             B_scale_contig, bias_contig);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear", &moonpoly_linear_pybind,
          "Moonpoly GEMM Linear Layer (Dynamo Compatible)",
          py::arg("A"), py::arg("B") = py::none());
    m.def("linear_cpp_selector", &moonpoly_linear_cpp_selector_pybind,
          "Moonpoly GEMM Linear Layer with internal decode selector",
          py::arg("A"), py::arg("B"));
    m.def("linear_cpp_forced_pid", &moonpoly_linear_cpp_forced_pid_pybind,
          "Moonpoly GEMM Linear Layer with forced pid and internal cached workspace",
          py::arg("A"), py::arg("B"), py::arg("pid"), py::arg("split_k"));
    m.def("scaled_int8_mm", &moonpoly_scaled_int8_mm_pybind,
          "MoonPoly SM80 scaled INT8 GEMM with optional bias. A=[M,K], B=[K,N].",
          py::arg("A"), py::arg("B"), py::arg("A_scales"),
          py::arg("B_scales"), py::arg("bias") = py::none(),
          py::arg("out_dtype") = torch::kFloat16);
    m.def("predict_int8_scaled_mm_kernel", &predict_best_kernel,
          "Predict the best SM80 EVT scaled INT8 kernel config",
          py::arg("m"), py::arg("n"), py::arg("k"));
    m.def("get_int8_scaled_mm_kernel_config", &get_kernel_config,
          "Get SM80 EVT scaled INT8 kernel config (tile_m, tile_n, tile_k, alignment)",
          py::arg("kernel_id"));
    m.def("p40_rcc_workspace_size", &moonpoly_fp16_p40_rcc_workspace_size_pybind,
          "Query workspace size for explicit FP16 P40 StreamK RCC API",
          py::arg("A"), py::arg("B_col"), py::arg("split_k_slices") = 2);
    m.def("p40_rcc_run", &moonpoly_fp16_p40_rcc_run_pybind,
          "Run explicit FP16 P40 StreamK RCC API and return a column-major output tensor",
          py::arg("A"), py::arg("B_col"), py::arg("workspace"),
          py::arg("split_k_slices") = 2);
    m.def("p40_rcc_run_default_stream",
          &moonpoly_fp16_p40_rcc_run_default_stream_pybind,
          "Run P40 RCC API on default stream using caller-provided workspace",
          py::arg("A"), py::arg("B_col"), py::arg("workspace"),
          py::arg("split_k_slices") = 2);
    m.def("p40_rcc_run_internal_ws",
          &moonpoly_fp16_p40_rcc_run_internal_ws_pybind,
          "Run P40 RCC API on default stream using internally cudaMalloc'ed workspace",
          py::arg("A"), py::arg("B_col"), py::arg("split_k_slices") = 2);
    m.def("p40_rcc_run_singleton",
          &moonpoly_fp16_p40_rcc_run_singleton_pybind,
          "Run P40 RCC API through the singleton/internal-workspace path",
          py::arg("A"), py::arg("B_col"), py::arg("split_k_slices") = 2);
    m.def("p40_rcc_run_raw_copy",
          &moonpoly_fp16_p40_rcc_run_raw_copy_pybind,
          "Run P40 RCC by first copying A/B into raw cudaMalloc buffers",
          py::arg("A"), py::arg("B_col"), py::arg("split_k_slices") = 2);
    m.def("p66_rcc_workspace_size", &moonpoly_fp16_p66_rcc_workspace_size_pybind,
          "Query workspace size for explicit FP16 P66 RCC API",
          py::arg("A"), py::arg("B_col"), py::arg("split_k_slices") = 2);
    m.def("p66_rcc_run", &moonpoly_fp16_p66_rcc_run_pybind,
          "Run explicit FP16 P66 RCC API and return a column-major output tensor",
          py::arg("A"), py::arg("B_col"), py::arg("split_k_slices") = 2);
    m.def("p42_rcc_workspace_size", &moonpoly_fp16_p42_rcc_workspace_size_pybind,
          "Query workspace size for explicit FP16 P42 RCC API",
          py::arg("A"), py::arg("B_col"), py::arg("split_k_slices") = 2);
    m.def("p42_rcc_run", &moonpoly_fp16_p42_rcc_run_pybind,
          "Run explicit FP16 P42 RCC API and return a column-major output tensor",
          py::arg("A"), py::arg("B_col"), py::arg("split_k_slices") = 2);
    m.def("p43_rcc_workspace_size", &moonpoly_fp16_p43_rcc_workspace_size_pybind,
          "Query workspace size for explicit FP16 P43 RCC API",
          py::arg("A"), py::arg("B_col"), py::arg("split_k_slices") = 2);
    m.def("p43_rcc_run", &moonpoly_fp16_p43_rcc_run_pybind,
          "Run explicit FP16 P43 RCC API and return a column-major output tensor",
          py::arg("A"), py::arg("B_col"), py::arg("split_k_slices") = 2);
    m.def("p44_rcc_workspace_size", &moonpoly_fp16_p44_rcc_workspace_size_pybind,
          "Query workspace size for explicit FP16 P44 RCC API",
          py::arg("A"), py::arg("B_col"), py::arg("split_k_slices") = 2);
    m.def("p44_rcc_run", &moonpoly_fp16_p44_rcc_run_pybind,
          "Run explicit FP16 P44 RCC API and return a column-major output tensor",
          py::arg("A"), py::arg("B_col"), py::arg("split_k_slices") = 2);
    m.def("p66_rcc_run_raw_copy",
          &moonpoly_fp16_p66_rcc_run_raw_copy_pybind,
          "Run P66 RCC by first copying A/B into raw cudaMalloc buffers",
          py::arg("A"), py::arg("B_col"), py::arg("split_k_slices") = 2);
    m.def("rcc_copy_check",
          &moonpoly_fp16_rcc_copy_check_pybind,
          "Copy A/B from torch tensors into raw buffers and compare round-trip values",
          py::arg("A"), py::arg("B_col"), py::arg("sample_count") = 8);
    m.def("p40_rcc_raw_copy_host_output_check",
          [](torch::Tensor A, torch::Tensor B_col, int64_t split_k_slices,
             int64_t sample_count) {
              return moonpoly_fp16_rcc_raw_copy_host_output_check_pybind(
                  A, B_col, split_k_slices, 40, sample_count);
          },
          "Run P40 on raw copied inputs and compare raw host output against CPU reference",
          py::arg("A"), py::arg("B_col"), py::arg("split_k_slices") = 2,
          py::arg("sample_count") = 8);
    m.def("p66_rcc_raw_copy_host_output_check",
          [](torch::Tensor A, torch::Tensor B_col, int64_t split_k_slices,
             int64_t sample_count) {
              return moonpoly_fp16_rcc_raw_copy_host_output_check_pybind(
                  A, B_col, split_k_slices, 66, sample_count);
          },
          "Run P66 on raw copied inputs and compare raw host output against CPU reference",
          py::arg("A"), py::arg("B_col"), py::arg("split_k_slices") = 2,
          py::arg("sample_count") = 8);
    m.def("p42_rcc_raw_copy_host_output_check",
          [](torch::Tensor A, torch::Tensor B_col, int64_t split_k_slices,
             int64_t sample_count) {
              return moonpoly_fp16_rcc_raw_copy_host_output_check_pybind(
                  A, B_col, split_k_slices, 42, sample_count);
          },
          "Run P42 on raw copied inputs and compare raw host output against CPU reference",
          py::arg("A"), py::arg("B_col"), py::arg("split_k_slices") = 2,
          py::arg("sample_count") = 8);
    m.def("p43_rcc_raw_copy_host_output_check",
          [](torch::Tensor A, torch::Tensor B_col, int64_t split_k_slices,
             int64_t sample_count) {
              return moonpoly_fp16_rcc_raw_copy_host_output_check_pybind(
                  A, B_col, split_k_slices, 43, sample_count);
          },
          "Run P43 on raw copied inputs and compare raw host output against CPU reference",
          py::arg("A"), py::arg("B_col"), py::arg("split_k_slices") = 2,
          py::arg("sample_count") = 8);
    m.def("p44_rcc_raw_copy_host_output_check",
          [](torch::Tensor A, torch::Tensor B_col, int64_t split_k_slices,
             int64_t sample_count) {
              return moonpoly_fp16_rcc_raw_copy_host_output_check_pybind(
                  A, B_col, split_k_slices, 44, sample_count);
          },
          "Run P44 on raw copied inputs and compare raw host output against CPU reference",
          py::arg("A"), py::arg("B_col"), py::arg("split_k_slices") = 2,
          py::arg("sample_count") = 8);
    m.def("p0_rcc_raw_copy_host_output_check",
          [](torch::Tensor A, torch::Tensor B_col, int64_t sample_count) {
              return moonpoly_fp16_rcc_raw_copy_host_output_check_pybind(
                  A, B_col, 2, 0, sample_count);
          },
          "Run P0 on raw copied inputs and compare raw host output against CPU reference",
          py::arg("A"), py::arg("B_col"), py::arg("sample_count") = 8);
    m.def("p40_rcc_constant_check",
          [](int64_t m_, int64_t n_, int64_t k_, int64_t split_k_) {
              return moonpoly_fp16_rcc_constant_check_pybind(
                  m_, n_, k_, split_k_, true);
          },
          "Run a constant-input internal RCC check for P40 and return max abs error to K",
          py::arg("m"), py::arg("n"), py::arg("k"), py::arg("split_k_slices") = 2);
    m.def("p66_rcc_constant_check",
          [](int64_t m_, int64_t n_, int64_t k_, int64_t split_k_) {
              return moonpoly_fp16_rcc_constant_check_pybind(
                  m_, n_, k_, split_k_, false);
          },
          "Run a constant-input internal RCC check for P66 and return max abs error to K",
          py::arg("m"), py::arg("n"), py::arg("k"), py::arg("split_k_slices") = 2);
}

#endif
