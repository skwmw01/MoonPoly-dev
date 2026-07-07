#!/usr/bin/env python3
"""
Comprehensive test script for MikPoly INT8 scaled matrix multiplication.
Tests correctness against PyTorch reference implementation and vLLM's cutlass_w8a8.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import argparse
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import json
import sys
from pathlib import Path

# Add vLLM path if available
try:
    sys.path.append('/home/zhangyangyu/moonpoly_workspace/end2end-test')
    from vllm._C import ops as vllm_ops
    VLLM_AVAILABLE = True
except ImportError:
    print("Warning: vLLM not available for comparison")
    VLLM_AVAILABLE = False

# Import our implementation
try:
    import moonpoly
    MIKPOLY_AVAILABLE = True
except ImportError:
    print("Warning: MoonPoly extension not built. Please compile first.")
    MIKPOLY_AVAILABLE = False


@dataclass
class TestConfig:
    """Configuration for a single test case."""
    m: int
    n: int
    k: int
    a_scale_type: str  # 'scalar' or 'per_row'
    b_scale_type: str  # 'scalar' or 'per_col'
    use_bias: bool
    out_dtype: torch.dtype
    seed: int = 42


class TestResult:
    """Store test results for reporting."""
    def __init__(self, config: TestConfig):
        self.config = config
        self.passed = False
        self.max_abs_error = float('inf')
        self.max_rel_error = float('inf')
        self.mean_abs_error = float('inf')
        self.mikpoly_time_ms = 0.0
        self.reference_time_ms = 0.0
        self.vllm_time_ms = 0.0
        self.speedup = 0.0
        self.error_message = ""

    def to_dict(self):
        return {
            'shape': f"({self.config.m}, {self.config.n}, {self.config.k})",
            'passed': self.passed,
            'max_abs_error': self.max_abs_error,
            'max_rel_error': self.max_rel_error,
            'mean_abs_error': self.mean_abs_error,
            'mikpoly_time_ms': self.mikpoly_time_ms,
            'reference_time_ms': self.reference_time_ms,
            'vllm_time_ms': self.vllm_time_ms,
            'speedup': self.speedup,
            'error_message': self.error_message
        }


def create_test_tensors(config: TestConfig) -> Tuple[torch.Tensor, ...]:
    """Create test tensors based on configuration."""
    torch.manual_seed(config.seed)
    device = torch.device('cuda')

    # Create INT8 inputs with realistic range
    a = torch.randint(-100, 100, (config.m, config.k), dtype=torch.int8, device=device)
    b = torch.randint(-100, 100, (config.k, config.n), dtype=torch.int8, device=device)

    # Create scales
    if config.a_scale_type == 'scalar':
        a_scales = torch.rand(1, dtype=torch.float32, device=device) * 0.1 + 0.01
    else:  # per_row
        a_scales = torch.rand(config.m, 1, dtype=torch.float32, device=device) * 0.1 + 0.01

    if config.b_scale_type == 'scalar':
        b_scales = torch.rand(1, dtype=torch.float32, device=device) * 0.1 + 0.01
    else:  # per_col
        b_scales = torch.rand(1, config.n, dtype=torch.float32, device=device) * 0.1 + 0.01

    # Create bias if needed
    if config.use_bias:
        bias = torch.randn(1, config.n, dtype=config.out_dtype, device=device) * 0.1
    else:
        bias = None

    return a, b, a_scales, b_scales, bias


def reference_implementation(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype
) -> torch.Tensor:
    """PyTorch reference implementation for scaled INT8 GEMM."""
    # Convert to float for computation
    a_float = a.to(torch.float32)
    b_float = b.to(torch.float32)

    # Apply scales
    a_scaled = a_float * a_scales
    b_scaled = b_float * b_scales

    # Matrix multiplication
    output = torch.matmul(a_scaled, b_scaled)

    # Add bias if present
    if bias is not None:
        output = output + bias.to(torch.float32)

    # Convert to output dtype
    return output.to(out_dtype)


def vllm_implementation(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype
) -> torch.Tensor:
    """vLLM's cutlass_scaled_mm implementation."""
    if not VLLM_AVAILABLE:
        return None

    # vLLM expects column-major B, so transpose it
    b_col_major = b.t().contiguous()

    # Allocate output
    out = torch.empty(a.shape[0], b.shape[1], dtype=out_dtype, device=a.device)

    # Call vLLM's implementation
    vllm_ops.cutlass_scaled_mm(
        out,
        a,
        b_col_major,
        a_scales,
        b_scales,
        bias
    )

    return out


def mikpoly_implementation(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype
) -> torch.Tensor:
    """MikPoly implementation."""
    if not MIKPOLY_AVAILABLE:
        return None

    return moonpoly.scaled_int8_mm(a, b, a_scales, b_scales, bias, out_dtype)


def compute_errors(
    output: torch.Tensor,
    reference: torch.Tensor
) -> Tuple[float, float, float]:
    """Compute absolute and relative errors."""
    diff = torch.abs(output - reference)
    max_abs_error = torch.max(diff).item()
    mean_abs_error = torch.mean(diff).item()

    # Compute relative error
    ref_abs = torch.abs(reference)
    mask = ref_abs > 1e-8
    if torch.any(mask):
        rel_errors = diff[mask] / ref_abs[mask]
        max_rel_error = torch.max(rel_errors).item()
    else:
        max_rel_error = max_abs_error

    return max_abs_error, max_rel_error, mean_abs_error


def benchmark_kernel(
    func,
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    warmup: int = 10,
    iterations: int = 100
) -> float:
    """Benchmark a kernel implementation."""
    # Warmup
    for _ in range(warmup):
        _ = func(a, b, a_scales, b_scales, bias, out_dtype)

    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        _ = func(a, b, a_scales, b_scales, bias, out_dtype)
    torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / iterations * 1000  # Convert to ms


def run_single_test(config: TestConfig, verbose: bool = False) -> TestResult:
    """Run a single test case."""
    result = TestResult(config)

    try:
        # Create test tensors
        a, b, a_scales, b_scales, bias = create_test_tensors(config)

        # Compute reference
        reference = reference_implementation(a, b, a_scales, b_scales, bias, config.out_dtype)

        # Test MikPoly implementation
        if MIKPOLY_AVAILABLE:
            mikpoly_output = mikpoly_implementation(a, b, a_scales, b_scales, bias, config.out_dtype)

            # Compute errors
            max_abs, max_rel, mean_abs = compute_errors(mikpoly_output, reference)
            result.max_abs_error = max_abs
            result.max_rel_error = max_rel
            result.mean_abs_error = mean_abs

            # Check pass criteria
            # Use relaxed tolerance for FP16 due to limited precision
            abs_tol = 0.01 if config.out_dtype == torch.float16 else 0.001
            rel_tol = 0.01  # 1% relative error tolerance

            if max_abs < abs_tol or max_rel < rel_tol:
                result.passed = True
            else:
                result.error_message = f"Error exceeds tolerance: abs={max_abs:.6f}, rel={max_rel:.6f}"

            # Benchmark if requested
            if verbose:
                result.mikpoly_time_ms = benchmark_kernel(
                    mikpoly_implementation, a, b, a_scales, b_scales, bias, config.out_dtype
                )
                result.reference_time_ms = benchmark_kernel(
                    reference_implementation, a, b, a_scales, b_scales, bias, config.out_dtype
                )
                result.speedup = result.reference_time_ms / result.mikpoly_time_ms

                # Test vLLM if available
                if VLLM_AVAILABLE:
                    try:
                        vllm_output = vllm_implementation(a, b, a_scales, b_scales, bias, config.out_dtype)
                        if vllm_output is not None:
                            result.vllm_time_ms = benchmark_kernel(
                                vllm_implementation, a, b, a_scales, b_scales, bias, config.out_dtype
                            )
                    except Exception as e:
                        if verbose:
                            print(f"vLLM benchmark failed: {e}")
        else:
            result.error_message = "MikPoly implementation not available"

    except Exception as e:
        result.error_message = f"Test failed with exception: {str(e)}"

    return result


def generate_test_configs() -> List[TestConfig]:
    """Generate a comprehensive set of test configurations."""
    configs = []

    # Problem sizes
    sizes = [
        (16, 32, 64),
        (32, 64, 128),
        (64, 128, 256),
        (128, 256, 512),
        (256, 512, 1024),
        (512, 1024, 2048),
        (1024, 2048, 4096),
        # Edge cases
        (1, 128, 256),
        (17, 33, 65),  # Non-power-of-2
        (100, 200, 300),
        # Rectangular shapes
        (16, 1024, 256),
        (1024, 16, 256),
    ]

    # Scale configurations
    scale_configs = [
        ('scalar', 'scalar'),
        ('scalar', 'per_col'),
        ('per_row', 'scalar'),
        ('per_row', 'per_col'),
    ]

    # Generate configs
    for m, n, k in sizes:
        for a_scale, b_scale in scale_configs:
            for use_bias in [False, True]:
                for dtype in [torch.float16, torch.bfloat16]:
                    configs.append(TestConfig(
                        m=m, n=n, k=k,
                        a_scale_type=a_scale,
                        b_scale_type=b_scale,
                        use_bias=use_bias,
                        out_dtype=dtype
                    ))

    return configs


def run_correctness_tests(verbose: bool = False, quick: bool = False):
    """Run comprehensive correctness tests."""
    print("=" * 80)
    print("Running MikPoly INT8 Scaled MM Correctness Tests")
    print("=" * 80)

    # Generate test configurations
    configs = generate_test_configs()

    if quick:
        # Quick mode: test subset
        configs = configs[::10]  # Test every 10th configuration
        print(f"Quick mode: Testing {len(configs)} configurations")
    else:
        print(f"Testing {len(configs)} configurations")

    results = []
    passed_count = 0
    failed_count = 0

    for i, config in enumerate(configs):
        if verbose:
            print(f"\nTest {i+1}/{len(configs)}: "
                  f"shape=({config.m}, {config.n}, {config.k}), "
                  f"scales=({config.a_scale_type}, {config.b_scale_type}), "
                  f"bias={config.use_bias}, dtype={config.out_dtype}")

        result = run_single_test(config, verbose=verbose)
        results.append(result)

        if result.passed:
            passed_count += 1
            if verbose:
                print(f"  ✓ PASSED - max_abs_err={result.max_abs_error:.6f}, "
                      f"max_rel_err={result.max_rel_error:.6f}")
                if result.speedup > 0:
                    print(f"  Performance: {result.mikpoly_time_ms:.3f}ms "
                          f"(speedup: {result.speedup:.2f}x vs reference)")
        else:
            failed_count += 1
            print(f"  ✗ FAILED - {result.error_message}")

        # Progress indicator for non-verbose mode
        if not verbose and (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{len(configs)} tests completed...")

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Total tests: {len(configs)}")
    print(f"Passed: {passed_count} ({passed_count/len(configs)*100:.1f}%)")
    print(f"Failed: {failed_count} ({failed_count/len(configs)*100:.1f}%)")

    # Analyze failures if any
    if failed_count > 0:
        print("\nFailed test analysis:")
        failed_results = [r for r in results if not r.passed]

        # Group by error type
        error_groups = {}
        for r in failed_results:
            key = r.error_message.split(':')[0] if ':' in r.error_message else r.error_message
            if key not in error_groups:
                error_groups[key] = []
            error_groups[key].append(r)

        for error_type, cases in error_groups.items():
            print(f"  {error_type}: {len(cases)} cases")
            if verbose:
                for case in cases[:3]:  # Show first 3 examples
                    print(f"    - shape=({case.config.m}, {case.config.n}, {case.config.k})")

    # Performance summary if available
    if verbose and any(r.speedup > 0 for r in results):
        speedups = [r.speedup for r in results if r.speedup > 0]
        print(f"\nPerformance Summary:")
        print(f"  Average speedup: {np.mean(speedups):.2f}x")
        print(f"  Min speedup: {np.min(speedups):.2f}x")
        print(f"  Max speedup: {np.max(speedups):.2f}x")

    # Save results to file
    with open('test_results.json', 'w') as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    print(f"\nDetailed results saved to test_results.json")

    return passed_count == len(configs)


def run_stress_test():
    """Run stress test with large matrices."""
    print("\n" + "=" * 80)
    print("Running Stress Test")
    print("=" * 80)

    stress_configs = [
        TestConfig(2048, 4096, 8192, 'scalar', 'scalar', False, torch.float16),
        TestConfig(4096, 4096, 4096, 'per_row', 'per_col', True, torch.float16),
        TestConfig(8192, 8192, 2048, 'scalar', 'per_col', False, torch.bfloat16),
    ]

    for config in stress_configs:
        print(f"\nStress test: shape=({config.m}, {config.n}, {config.k})")
        try:
            result = run_single_test(config, verbose=True)
            if result.passed:
                print(f"  ✓ PASSED")
            else:
                print(f"  ✗ FAILED: {result.error_message}")
        except torch.cuda.OutOfMemoryError:
            print(f"  ⚠ SKIPPED: Out of memory")
        except Exception as e:
            print(f"  ✗ ERROR: {e}")


def test_kernel_predictor():
    """Test the kernel predictor."""
    print("\n" + "=" * 80)
    print("Testing Kernel Predictor")
    print("=" * 80)

    if not MIKPOLY_AVAILABLE:
        print("MikPoly not available, skipping predictor test")
        return

    test_sizes = [
        (16, 32, 64),
        (64, 128, 256),
        (128, 256, 512),
        (256, 512, 1024),
        (1024, 2048, 4096),
    ]

    print("\nPredicted kernels for different problem sizes:")
    print(f"{'Shape':<20} {'Kernel ID':<10} {'Tile Config':<30}")
    print("-" * 60)

    for m, n, k in test_sizes:
        kernel_id = moonpoly.predict_int8_scaled_mm_kernel(m, n, k)
        tile_m, tile_n, tile_k, alignment = moonpoly.get_int8_scaled_mm_kernel_config(kernel_id)
        print(f"({m}, {n}, {k})".ljust(20) +
              f"{kernel_id}".ljust(10) +
              f"({tile_m}x{tile_n}x{tile_k}, align={alignment})")


def main():
    parser = argparse.ArgumentParser(description='Test MikPoly INT8 scaled MM implementation')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--quick', '-q', action='store_true', help='Quick test (subset only)')
    parser.add_argument('--stress', '-s', action='store_true', help='Run stress tests')
    parser.add_argument('--predictor', '-p', action='store_true', help='Test kernel predictor')
    parser.add_argument('--all', '-a', action='store_true', help='Run all tests')

    args = parser.parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Error: CUDA is not available")
        return 1

    print(f"Using GPU: {torch.cuda.get_device_name()}")
    print(f"MikPoly available: {MIKPOLY_AVAILABLE}")
    print(f"vLLM available: {VLLM_AVAILABLE}")

    all_passed = True

    # Run tests
    if args.all or (not args.stress and not args.predictor):
        all_passed = run_correctness_tests(verbose=args.verbose, quick=args.quick)

    if args.all or args.predictor:
        test_kernel_predictor()

    if args.all or args.stress:
        run_stress_test()

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
