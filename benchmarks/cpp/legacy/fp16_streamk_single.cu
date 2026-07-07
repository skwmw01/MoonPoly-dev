#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "cuda_runtime.h"
#include "cublas_v2.h" // For cuBLAS
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"

// =====================================================================================
// Helper Macros for Error Checking
// =====================================================================================
#define CUDA_CHECK(status)                                                      \
  {                                                                             \
    cudaError_t error = status;                                                 \
    if (error != cudaSuccess) {                                                 \
      std::cerr << "CUDA Error at line " << __LINE__ << ": "                    \
                << cudaGetErrorString(error) << std::endl;                      \
      exit(EXIT_FAILURE);                                                       \
    }                                                                           \
  }

#define CUTLASS_CHECK(status)                                                   \
  {                                                                             \
    cutlass::Status error = status;                                             \
    if (error != cutlass::Status::kSuccess) {                                   \
      std::cerr << "CUTLASS Error at line " << __LINE__ << ": "                 \
                << cutlass::cutlassGetStatusString(error) << std::endl;         \
      exit(EXIT_FAILURE);                                                       \
    }                                                                           \
  }

#define CUBLAS_CHECK(status)                                                    \
  {                                                                             \
    cublasStatus_t error = status;                                              \
    if (error != CUBLAS_STATUS_SUCCESS) {                                       \
      std::cerr << "cuBLAS Error at line " << __LINE__ << std::endl;             \
      exit(EXIT_FAILURE);                                                       \
    }                                                                           \
  }

// =====================================================================================
// CUTLASS Stream-K GEMM Implementation
// =====================================================================================
struct CUTLASS_STREAMK_GEMM {
public:
    // Kernel Configuration: A known-good configuration for Ampere FP16 Tensor Core GEMM
    using ElementA = cutlass::half_t;
    using LayoutA = cutlass::layout::RowMajor;
    using ElementB = cutlass::half_t;
    using LayoutB = cutlass::layout::ColumnMajor;
    using ElementC = cutlass::half_t;
    using LayoutC = cutlass::layout::ColumnMajor;
    using ElementAccumulator = float;

    using ArchTag = cutlass::arch::Sm80;
    using OperatorClass = cutlass::arch::OpClassTensorOp;
    using ThreadblockSwizzle = cutlass::gemm::threadblock::ThreadblockSwizzleStreamK;


    // using ThreadblockShape = cutlass::gemm::GemmShape<64, 32, 32>;
    // using WarpShape = cutlass::gemm::GemmShape<32, 32, 32>;

    // using ThreadblockShape = cutlass::gemm::GemmShape<64, 32, 64>;
    // using WarpShape = cutlass::gemm::GemmShape<64, 16, 64>;

    // using ThreadblockShape = cutlass::gemm::GemmShape<16, 24, 32>;
    // using WarpShape = cutlass::gemm::GemmShape<16, 24, 32>;

    using ThreadblockShape = cutlass::gemm::GemmShape<64, 24, 32>;
    using WarpShape = cutlass::gemm::GemmShape<32, 24, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
    // static constexpr int NumStages = 10;
    // static constexpr int NumStages = 6;
    static constexpr int NumStages = 10;
    static constexpr int AlignmentA = 8;
    static constexpr int AlignmentB = 8;
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementC,
        8, // ElementsPerAccess
        ElementAccumulator,
        ElementAccumulator>;


    // using ThreadblockShape = cutlass::gemm::GemmShape<64, 72, 64>;
    // using WarpShape = cutlass::gemm::GemmShape<64, 24, 64>;
    // using InstructionShape =  cutlass::gemm::GemmShape<16, 8, 16>;
    // static constexpr int NumStages = 6;
    // static constexpr int AlignmentA = 8;
    // static constexpr int AlignmentB = 8;
    // using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    //     ElementC,
    //     1, // ElementsPerAccess
    //     ElementAccumulator,
    //     ElementAccumulator>;

    using Gemm = cutlass::gemm::device::GemmUniversal<
        ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
        ElementAccumulator, OperatorClass, ArchTag, ThreadblockShape,
        WarpShape, InstructionShape, EpilogueOp, ThreadblockSwizzle,
        NumStages, AlignmentA, AlignmentB
    >;

private:
    // 成员变量，用于保持初始化后的状态
    Gemm gemm_op;
    cutlass::device_memory::allocation<uint8_t> workspace;
    bool is_initialized;

    CUTLASS_STREAMK_GEMM() : is_initialized(false) {}
    ~CUTLASS_STREAMK_GEMM() {}

public:
    CUTLASS_STREAMK_GEMM(const CUTLASS_STREAMK_GEMM &) = delete;
    CUTLASS_STREAMK_GEMM &operator=(const CUTLASS_STREAMK_GEMM &) = delete;

    static CUTLASS_STREAMK_GEMM &get_instance() {
        static CUTLASS_STREAMK_GEMM instance;
        return instance;
    }

    void initialize(int m, int n, int k, int split_k_slices,
                    ElementA *ptr_A, ElementB *ptr_B, ElementC *ptr_C,
                    float alpha, float beta) {

        cutlass::gemm::GemmCoord problem_size(m, n, k);
        auto gemm_mode = cutlass::gemm::GemmUniversalMode::kGemm;

        std::cout << "CUTLASS problem size: M=" << m << ", N=" << n << ", K=" << k
                  << ", split_k=" << split_k_slices << std::endl;
        std::cout << "CUTLASS stride A: " << LayoutA::packed({m, k}).stride(0) << std::endl;
        std::cout << "CUTLASS stride B: " << LayoutB::packed({k, n}).stride(0) << std::endl;
        std::cout << "CUTLASS stride C: " << LayoutC::packed({m, n}).stride(0) << std::endl;

        typename Gemm::Arguments arguments(
            gemm_mode,
            problem_size,
            split_k_slices,
            {ElementAccumulator(alpha), ElementAccumulator(beta)},
            ptr_A, ptr_B, ptr_C, ptr_C,
            0, 0, 0, 0,
            LayoutA::packed({m, k}).stride(0),
            LayoutB::packed({k, n}).stride(0),
            LayoutC::packed({m, n}).stride(0),
            LayoutC::packed({m, n}).stride(0)
        );

        size_t workspace_size = gemm_op.get_workspace_size(arguments);
        std::cout << "CUTLASS workspace size: " << workspace_size << " bytes" << std::endl;
        workspace.reset(workspace_size);

        cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
        if (status != cutlass::Status::kSuccess) {
            std::cout << "CUTLASS initialization failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
        }
        CUTLASS_CHECK(status);

        is_initialized = true;
    }

    // 新的执行方法 (可以在循环中多次调用)
    void execute() {
        if (!is_initialized) {
            std::cerr << "Error: CUTLASS GEMM must be initialized before execution." << std::endl;
            exit(EXIT_FAILURE);
        }
        cutlass::Status status = gemm_op.run();
        CUTLASS_CHECK(status);
    }
};

// =====================================================================================
// cuBLAS GEMM Implementation
// =====================================================================================
void run_cublas_gemm(cublasHandle_t handle, int m, int n, int k,
                     cutlass::half_t *ptr_A, cutlass::half_t *ptr_B, cutlass::half_t *ptr_C,
                     float alpha, float beta) {

    // CUTLASS 配置: A(RowMajor m×k), B(ColumnMajor k×n), C(ColumnMajor m×n)
    // cuBLAS 假设所有矩阵都是 ColumnMajor 存储
    // 要匹配 CUTLASS 的 RowMajor A，需要转置 A
    // C = A * B，其中 A 是 RowMajor，B 和 C 都是 ColumnMajor
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T, // transa: 转置 A 以匹配 RowMajor -> ColumnMajor
        CUBLAS_OP_N, // transb: B 已经是 ColumnMajor，不需要转置
        m, n, k,
        &alpha,
        ptr_A, CUDA_R_16F, k, // A: 作为 k×m 的 ColumnMajor 传入，lda = k
        ptr_B, CUDA_R_16F, k, // B: k×n ColumnMajor, ldb = k
        &beta,
        ptr_C, CUDA_R_16F, m, // C: m×n ColumnMajor, ldc = m
        CUDA_R_32F, // computeType
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));

}


// =====================================================================================
// Result Comparison Function
// =====================================================================================
bool compare_results(cutlass::half_t* result_cutlass, cutlass::half_t* result_cublas,
                    int m, int n, float tolerance = 1e-3f) {
    using ElementC = cutlass::half_t;
    using LayoutC = cutlass::layout::ColumnMajor;

    cutlass::HostTensor<ElementC, LayoutC> h_cutlass({m, n});
    cutlass::HostTensor<ElementC, LayoutC> h_cublas({m, n});

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_cutlass.host_data(), result_cutlass,
                         m * n * sizeof(cutlass::half_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_cublas.host_data(), result_cublas,
                         m * n * sizeof(cutlass::half_t), cudaMemcpyDeviceToHost));

    float max_diff = 0.0f;
    float max_rel_diff = 0.0f;
    int error_count = 0;
    int nan_count = 0;
    int inf_count = 0;

    for (int i = 0; i < m * n; ++i) {
        float val_cutlass = float(h_cutlass.host_data()[i]);
        float val_cublas = float(h_cublas.host_data()[i]);

        if (std::isnan(val_cutlass) || std::isnan(val_cublas)) {
            nan_count++;
            error_count++;
            if (error_count <= 10) {
                std::cout << "NaN at index " << i << ": CUTLASS=" << val_cutlass
                          << ", cuBLAS=" << val_cublas << std::endl;
            }
            continue;
        }
        if (std::isinf(val_cutlass) || std::isinf(val_cublas)) {
            inf_count++;
            error_count++;
            if (error_count <= 10) {
                std::cout << "Inf at index " << i << ": CUTLASS=" << val_cutlass
                          << ", cuBLAS=" << val_cublas << std::endl;
            }
            continue;
        }

        float abs_diff = std::abs(val_cutlass - val_cublas);
        float rel_diff = (val_cublas != 0.0f) ? abs_diff / std::abs(val_cublas) : abs_diff;

        max_diff = std::max(max_diff, abs_diff);
        max_rel_diff = std::max(max_rel_diff, rel_diff);

        if (abs_diff > tolerance && rel_diff > tolerance) {
            error_count++;
            if (error_count <= 10) { // Show first 10 errors
                std::cout << "Mismatch at index " << i << ": CUTLASS=" << val_cutlass
                         << ", cuBLAS=" << val_cublas << ", diff=" << abs_diff << std::endl;
            }
        }
    }

    std::cout << "Result Comparison:" << std::endl;
    std::cout << "  Max absolute difference: " << max_diff << std::endl;
    std::cout << "  Max relative difference: " << max_rel_diff << std::endl;
    std::cout << "  NaN values: " << nan_count << std::endl;
    std::cout << "  Inf values: " << inf_count << std::endl;
    std::cout << "  Errors (tolerance=" << tolerance << "): " << error_count << "/" << (m*n) << std::endl;

    // 显示前几个元素用于调试
    std::cout << "First 10 elements comparison:" << std::endl;
    for (int i = 0; i < std::min(10, m*n); ++i) {
        float val_cutlass = float(h_cutlass.host_data()[i]);
        float val_cublas = float(h_cublas.host_data()[i]);
        std::cout << "  [" << i << "] CUTLASS=" << val_cutlass
                  << ", cuBLAS=" << val_cublas
                  << ", diff=" << std::abs(val_cutlass - val_cublas) << std::endl;
    }

    // 分析索引72的矩阵位置 (ColumnMajor存储: C[i][j] = data[j*m + i])
    if (m * n > 72) {
        int row_72 = 72 % m;  // 行索引
        int col_72 = 72 / m;  // 列索引
        std::cout << "Index 72 corresponds to matrix position: row=" << row_72
                  << ", col=" << col_72 << " (in ColumnMajor layout)" << std::endl;
    }

    return error_count == 0;
}

// =====================================================================================
// Main Function
// =====================================================================================
int main(int argc, char **argv) {
    if (argc < 4 || argc > 7) {
        std::cerr << "用法: " << argv[0]
                  << " <M> <N> <K> [split_k=2] [warmup=5] [iters=20]"
                  << std::endl;
        return -1;
    }

    int M = std::stoi(argv[1]);
    int N = std::stoi(argv[2]);
    int K = std::stoi(argv[3]);
    int split_k_slices = (argc >= 5) ? std::stoi(argv[4]) : 2;
    int warmup_iterations = (argc >= 6) ? std::stoi(argv[5]) : 5;
    int profile_iterations = (argc >= 7) ? std::stoi(argv[6]) : 20;
    float alpha = 1.0f;
    float beta = 0.0f;

    // 硬件检查
    int device_id = 0;
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device_id));
    if (props.major < 8) {
        std::cerr << "Stream-K and this example require NVIDIA Ampere or newer GPU (SM80+)." << std::endl;
        return 0;
    }

    // 准备数据
    using ElementA = cutlass::half_t;
    using LayoutA = cutlass::layout::RowMajor;
    using ElementB = cutlass::half_t;
    using LayoutB = cutlass::layout::ColumnMajor;
    using ElementC = cutlass::half_t;
    using LayoutC = cutlass::layout::ColumnMajor;

    cutlass::HostTensor<ElementA, LayoutA> h_A({M, K});
    cutlass::HostTensor<ElementB, LayoutB> h_B({K, N});
    cutlass::HostTensor<ElementC, LayoutC> h_C({M, N});

    cutlass::reference::host::TensorFillRandomUniform(
        h_A.host_view(), 1, cutlass::half_t(4), cutlass::half_t(-4),
        0);
    cutlass::reference::host::TensorFillRandomUniform(
        h_B.host_view(), 1, cutlass::half_t(4), cutlass::half_t(-4),
        0);
    cutlass::reference::host::TensorFillRandomUniform(
        h_C.host_view(), 1, cutlass::half_t(4), cutlass::half_t(-4),
        0);
    // cutlass::reference::host::TensorFill(h_A.host_view(), cutlass::half_t(1.0f));
    // cutlass::reference::host::TensorFill(h_B.host_view(), cutlass::half_t(1.0f));
    // cutlass::reference::host::TensorFill(h_C.host_view(), cutlass::half_t(0.0f));

    cutlass::device_memory::allocation<ElementA> d_A(h_A.capacity());
    cutlass::device_memory::allocation<ElementB> d_B(h_B.capacity());
    cutlass::device_memory::allocation<ElementC> d_C_cutlass(h_C.capacity());
    cutlass::device_memory::allocation<ElementC> d_C_cublas(h_C.capacity());
    d_A.copy_from_host(h_A.host_data());
    d_B.copy_from_host(h_B.host_data());
    d_C_cutlass.copy_from_host(h_C.host_data());
    d_C_cublas.copy_from_host(h_C.host_data());

    // 性能测试参数（默认 warmup=5, profile=20）

    std::cout << "====================================================================\n";
    std::cout << "Problem: M=" << M << ", N=" << N << ", K=" << K
              << ", Layout=row-col-col, DataType=FP16" << std::endl;
    std::cout << "CUTLASS Stream-K slices: " << split_k_slices << std::endl;
    std::cout << "====================================================================\n\n";

    // 计时器
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --- 测试 CUTLASS ---
    std::cout << "Profiling CUTLASS Stream-K..." << std::endl;

    // 调用一次 initialize() 进行设置
    auto& cutlass_runner = CUTLASS_STREAMK_GEMM::get_instance();
    cutlass_runner.initialize(M, N, K, split_k_slices, d_A.get(), d_B.get(), d_C_cutlass.get(), alpha, beta);

    // 预热
    for (int i = 0; i < warmup_iterations; ++i) {
        cutlass_runner.execute();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // 正式计时
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < profile_iterations; ++i) {
        cutlass_runner.execute(); // 只对 execute() 计时
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float cutlass_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&cutlass_ms, start, stop));
    double cutlass_avg_ms = cutlass_ms / profile_iterations;
    double cutlass_tflops = (2.0 * M * N * K) / (cutlass_avg_ms / 1000.0) / 1.0e12;

    std::cout << "  Average Time: " << cutlass_avg_ms << " ms" << std::endl;
    std::cout << "  Performance:  " << cutlass_tflops << " TFLOP/s" << std::endl;

    // 在测试cuBLAS之前，重置cuBLAS的输出矩阵为零
    d_C_cublas.copy_from_host(h_C.host_data());

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    // --- 测试 cuBLAS ---
    std::cout << "\nProfiling cuBLAS..." << std::endl;

    for (int i = 0; i < warmup_iterations; ++i) {
        run_cublas_gemm(handle, M, N, K, d_A.get(), d_B.get(), d_C_cublas.get(), alpha, beta);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < profile_iterations; ++i) {
        run_cublas_gemm(handle, M, N, K, d_A.get(), d_B.get(), d_C_cublas.get(), alpha, beta);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float cublas_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&cublas_ms, start, stop));
    double cublas_avg_ms = cublas_ms / profile_iterations;
    double cublas_tflops = (2.0 * M * N * K) / (cublas_avg_ms / 1000.0) / 1.0e12;

    std::cout << "  Average Time: " << cublas_avg_ms << " ms" << std::endl;
    std::cout << "  Performance:  " << cublas_tflops << " TFLOP/s" << std::endl;

    // --- 正确性验证 ---
    std::cout << "\n====================================================================\n";
    std::cout << "Correctness Verification:" << std::endl;
    bool results_match = compare_results(d_C_cutlass.get(), d_C_cublas.get(), M, N);
    if (results_match) {
        std::cout << "✓ CUTLASS results match cuBLAS reference (within tolerance)" << std::endl;
    } else {
        std::cout << "✗ CUTLASS results do NOT match cuBLAS reference!" << std::endl;
    }

    // --- 性能对比 ---
    std::cout << "\n====================================================================\n";
    std::cout << "Performance Comparison:" << std::endl;
    std::cout << "  Speedup of CUTLASS over cuBLAS: " << cublas_avg_ms / cutlass_avg_ms << "x" << std::endl;
    std::cout << "====================================================================\n";

    // 清理
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return results_match ? 0 : 1;
}
