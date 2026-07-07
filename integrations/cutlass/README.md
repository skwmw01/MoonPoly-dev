# CUTLASS Integration

MoonPoly vendors CUTLASS under `3rdparty/cutlass` and carries a small
Pattern-2 compatibility patch for CUTLASS 4.1.

Vendored CUTLASS base commit:
`76c96b0be35cb263debe3e3d8418b80911a544ab` (`76c96b0b`, CUTLASS upstream).

The patch enables one fused launch to route CTAs to two different CUTLASS GEMM
micro-kernels. This is used by the Pattern-2 case study, where the output `N`
dimension is split into two regions and each region is handled by a different
micro-kernel.

Files touched by the patch:

- `cutlass/gemm/threadblock/threadblock_swizzle.h`
- `cutlass/gemm/kernel/gemm.h`
- `cutlass/gemm/device/gemm.h`
- `cutlass/gemm/device/twin_gemm.h`

The old CUTLASS 2.9 patch from the prototype is intentionally not kept here.
It overwrote full CUTLASS headers and does not apply cleanly to CUTLASS 4.1.
The current patch keeps the normal GEMM launch path unchanged and only uses
the flattened-grid offset path from `TwinGemm`.

To regenerate the patch record after editing the vendored CUTLASS files:

```bash
git diff -- 3rdparty/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h \
  3rdparty/cutlass/include/cutlass/gemm/kernel/gemm.h \
  3rdparty/cutlass/include/cutlass/gemm/device/gemm.h \
  > integrations/cutlass/cutlass_4_1_pattern2_twin_gemm.patch

git diff --no-index /dev/null \
  3rdparty/cutlass/include/cutlass/gemm/device/twin_gemm.h \
  >> integrations/cutlass/cutlass_4_1_pattern2_twin_gemm.patch || true
```
