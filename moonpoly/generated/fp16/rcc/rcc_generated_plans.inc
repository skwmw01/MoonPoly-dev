// Generated/curated runtime selector entries for direct FP16 RCC calls.
// Keep Linear(A[M,K], W[N,K]) decode shapes in llm_decode_selector.inc.
//
// P286 is a Qwen3 long-K split-K ablation kernel from qwen_10240_splitk_round7.
// It is valid for direct RCC (M,N,K)=(8,10240,5120), but not for the vLLM
// Linear mapping of Linear(8,10240,5120) -> RCC(10240,8,5120). Keep it behind
// MOONPOLY_ENABLE_QWEN_P286_RCC_ABLATION to prevent accidental serving use.
static const Fp16RccGeneratedPlanEntry kFp16RccGeneratedPlanEntries[] = {
    {8, 10240, 5120, 286, 4, true, 0.07137f},
};
