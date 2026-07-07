from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import warnings

project_name = "moonpoly"
_source_files = [
    "core/moonpoly_gemm.cu",
    "core/fp16/fp16_dispatch.cu",
    "generated/fp16/rcc/candidates.cu",
    "core/fp16/fp16_rcc.cu",
    "core/fp16/fp16_rcc_cost.cu",
    "core/fp32/fp32_dispatch.cu",
    "core/fp32/fp32_rcc.cu",
    "core/fp32/fp32_rcc_cost.cu",
    "core/int8/i8_dispatch.cu",
    "core/int8/i8_rcc.cu",
    "core/int8/i8_rcc_cost.cu",
    "core/int8/scaled_mm/scaled_mm_bindings.cpp",
    "core/int8/scaled_mm/scaled_mm_mikpoly_unified.cu",
]
source_files = [os.path.join(project_name, f) for f in _source_files]

current_dir = os.path.dirname(os.path.abspath(__file__))
cutlass_dir = os.environ.get("CUTLASS_DIR", os.path.join(current_dir, "3rdparty/cutlass"))
print (cutlass_dir)
if not os.path.isdir(cutlass_dir):
    raise Exception("CUTLASS not found. Set CUTLASS_DIR or checkout 3rdparty/cutlass.")

# Add Include directories
_cutlass_include_dirs = ["tools/util/include", "include"]
cutlass_include_dirs = [os.path.join(cutlass_dir, d) for d in _cutlass_include_dirs]
moonpoly_include_dirs = [
    os.path.join(current_dir, project_name, "include"),
    os.path.join(current_dir, project_name, "core"),
    os.path.join(current_dir, project_name, "core", "int8", "scaled_mm"),
    os.path.join(current_dir, project_name)
]
INCLUDE_DIRS = cutlass_include_dirs + moonpoly_include_dirs

# Set additional flags needed for compilation here
nvcc_flags=[
    "-O3",
    "-DNDEBUG",
    "-std=c++17",
    "--expt-extended-lambda",
    "--expt-relaxed-constexpr",
    "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
    "-DCUTLASS_ARCH_MMA_SM80_ENABLED=1",
    "--gpu-architecture=compute_80",
    "--gpu-code=sm_80,compute_80",
]
ld_flags=["-L/usr/local/lib -lcuda"]

setup(
    name='MoonPoly',
    version='0.1',
    ext_modules=[
        CUDAExtension(
                name="moonpoly",
                sources=source_files,
                include_dirs=INCLUDE_DIRS,
                extra_compile_args={'nvcc': nvcc_flags},
                # extra_cuda_cflags=[ '-DTORCH_USE_CUDA_DSA=1'],
                define_macros=[('PYTHON_BIND', '1')],
                libraries=ld_flags)
   ],
    cmdclass={
        'build_ext': BuildExtension
    })
