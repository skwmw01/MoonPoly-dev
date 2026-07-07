#include "cutlass/cutlass.h"
#include "cutlass/half.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/util/device_memory.h"
#include "moonpoly.cuh"
#include "candidates.cuh"

namespace moonpoly {
namespace fp16 {
namespace row_col_col {

#include "experiments/llama2_13b_pid_neighborhood_candidates.inc"
#include "experiments/merged_pid_neighborhood_round2_candidates.inc"
#include "experiments/merged_cutlass_param_sweep_batch1.inc"
#include "experiments/merged_cutlass_param_sweep_batch2.inc"
#include "experiments/merged_cutlass_focused_sweep_batch3.inc"
#include "experiments/merged_cutlass_focused_sweep_batch4.inc"
#include "experiments/merged_cutlass_focused_sweep_batch5.inc"
#include "experiments/qwen_longk_5120x25600_batch1.inc"
#include "experiments/qwen_longk_5120x25600_universal_batch2.inc"
#include "experiments/qwen_longk_5120x25600_universal_batch3.inc"
#include "experiments/qwen_bs8_51200_focused_round6.inc"
#include "experiments/llama_cublas_inferred_decode.inc"
#include "experiments/qwen_cublas_inferred_decode_main.inc"
#include "experiments/qwen_cublas_inferred_decode_longk.inc"
#include "experiments/qwen_10240_splitk_round7.inc"
#include "experiments/llama_cublas_inferred_main3.inc"
#include "experiments/llama_focused_round8.inc"
#include "experiments/llama_pid195_round9.inc"
#include "splitk_candidates.inc"

namespace {

constexpr int kPaperEliteFp16Pids[] = {
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    400, 407, 413, 425, 439,
};

}  // namespace

bool is_paper_elite_fp16_pid(int pid) {
  for (int elite_pid : kPaperEliteFp16Pids) {
    if (pid == elite_pid) {
      return true;
    }
  }
  return false;
}

int paper_elite_fp16_kernel_count() {
  return static_cast<int>(sizeof(kPaperEliteFp16Pids) / sizeof(kPaperEliteFp16Pids[0]));
}

const int *paper_elite_fp16_pids() {
  return kPaperEliteFp16Pids;
}

bool is_explicit_neighborhood_pid(int pid) {
  return (pid >= 90 && pid <= 205) || (pid >= 216 && pid <= 321) ||
         is_legacy_splitk_pid(pid);
}

int explicit_neighborhood_alignment(int pid) {
  if (is_legacy_splitk_pid(pid)) {
    return legacy_splitk_alignment(pid);
  }
  switch (pid) {
    case 90:
    case 91:
    case 92:
    case 93:
    case 94:
    case 95:
    case 96:
    case 97:
    case 98:
    case 99:
    case 100:
    case 101:
    case 102:
    case 103:
    case 104:
    case 105:
    case 106:
    case 107:
    case 108:
    case 109:
    case 110:
    case 111:
    case 112:
    case 113:
    case 120:
    case 121:
    case 122:
    case 123:
    case 124:
    case 125:
    case 132:
    case 133:
    case 134:
    case 135:
    case 136:
    case 137:
    case 144:
    case 145:
    case 146:
    case 147:
    case 149:
    case 150:
    case 151:
    case 153:
    case 154:
    case 156:
    case 157:
    case 159:
    case 160:
    case 162:
    case 164:
    case 165:
    case 166:
    case 167:
    case 169:
    case 170:
    case 171:
    case 172:
    case 174:
    case 175:
    case 176:
    case 177:
    case 179:
    case 180:
    case 181:
    case 182:
    case 184:
    case 185:
    case 186:
    case 187:
    case 188:
    case 190:
    case 191:
    case 192:
    case 193:
    case 195:
    case 196:
    case 197:
    case 199:
    case 200:
    case 201:
    case 202:
    case 204:
    case 205:
    case 216:
    case 217:
    case 218:
    case 219:
    case 220:
    case 221:
    case 222:
    case 223:
    case 224:
    case 226:
    case 227:
    case 228:
    case 229:
    case 230:
    case 231:
    case 233:
    case 234:
    case 235:
    case 236:
    case 237:
    case 238:
    case 239:
    case 262:
    case 263:
    case 264:
    case 265:
    case 266:
    case 267:
    case 270:
    case 271:
    case 272:
    case 273:
    case 274:
    case 275:
    case 278:
    case 279:
    case 280:
    case 281:
    case 282:
    case 284:
    case 285:
    case 286:
    case 287:
    case 288:
    case 289:
    case 291:
    case 292:
    case 293:
    case 294:
    case 295:
    case 296:
    case 297:
    case 298:
    case 299:
    case 300:
    case 301:
    case 302:
    case 303:
    case 304:
    case 305:
    case 307:
    case 308:
    case 309:
    case 311:
    case 312:
    case 314:
    case 315:
    case 316:
    case 317:
    case 318:
    case 320:
      return 8;
    case 114:
    case 115:
    case 116:
    case 117:
    case 118:
    case 119:
    case 126:
    case 127:
    case 128:
    case 129:
    case 130:
    case 131:
    case 138:
    case 139:
    case 140:
    case 141:
    case 142:
    case 143:
    case 148:
    case 152:
    case 155:
    case 158:
    case 161:
    case 163:
    case 168:
    case 173:
    case 178:
    case 183:
    case 189:
    case 194:
    case 198:
    case 203:
    case 225:
    case 232:
    case 268:
    case 269:
    case 276:
    case 277:
    case 283:
    case 290:
    case 306:
    case 310:
    case 313:
    case 319:
    case 321:
      return 4;
    default:
      return 1;
  }
}

SharedMem explicit_neighborhood_shared_mem(int pid) {
  if (is_legacy_splitk_pid(pid)) {
    return legacy_splitk_shared_mem(pid);
  }
  switch (pid) {
    case 90: return SharedMem(16, 64, 96);
    case 91: return SharedMem(16, 64, 64);
    case 92: return SharedMem(16, 64, 128);
    case 93: return SharedMem(16, 32, 96);
    case 94: return SharedMem(16, 64, 32);
    case 95: return SharedMem(16, 64, 32);
    case 96: return SharedMem(16, 64, 48);
    case 97: return SharedMem(16, 64, 16);
    case 98: return SharedMem(16, 64, 16);
    case 99: return SharedMem(16, 64, 16);
    case 100: return SharedMem(16, 32, 16);
    case 101: return SharedMem(16, 64, 40);
    case 102: return SharedMem(16, 64, 32);
    case 103: return SharedMem(16, 64, 48);
    case 104: return SharedMem(16, 64, 64);
    case 105: return SharedMem(16, 64, 80);
    case 106: return SharedMem(16, 32, 48);
    case 107: return SharedMem(16, 64, 64);
    case 108: return SharedMem(16, 64, 128);
    case 109: return SharedMem(16, 64, 96);
    case 110: return SharedMem(16, 64, 8);
    case 111: return SharedMem(16, 64, 8);
    case 112: return SharedMem(16, 64, 16);
    case 113: return SharedMem(16, 64, 16);
    case 114: return SharedMem(16, 64, 32);
    case 115: return SharedMem(16, 64, 48);
    case 116: return SharedMem(16, 64, 128);
    case 117: return SharedMem(16, 64, 40);
    case 118: return SharedMem(16, 64, 16);
    case 119: return SharedMem(16, 64, 32);
    case 120: return SharedMem(16, 64, 32);
    case 121: return SharedMem(16, 64, 48);
    case 122: return SharedMem(16, 64, 128);
    case 123: return SharedMem(16, 64, 40);
    case 124: return SharedMem(16, 64, 16);
    case 125: return SharedMem(16, 64, 32);
    case 126: return SharedMem(16, 64, 32);
    case 127: return SharedMem(16, 64, 48);
    case 128: return SharedMem(16, 64, 128);
    case 129: return SharedMem(16, 64, 40);
    case 130: return SharedMem(16, 64, 16);
    case 131: return SharedMem(16, 64, 32);
    case 132: return SharedMem(16, 64, 32);
    case 133: return SharedMem(16, 64, 48);
    case 134: return SharedMem(16, 64, 128);
    case 135: return SharedMem(16, 64, 40);
    case 136: return SharedMem(16, 64, 16);
    case 137: return SharedMem(16, 64, 32);
    case 138: return SharedMem(16, 64, 32);
    case 139: return SharedMem(16, 64, 48);
    case 140: return SharedMem(16, 64, 128);
    case 141: return SharedMem(16, 64, 40);
    case 142: return SharedMem(16, 64, 16);
    case 143: return SharedMem(16, 64, 32);
    case 144: return SharedMem(16, 64, 32);
    case 145: return SharedMem(16, 64, 48);
    case 146: return SharedMem(16, 64, 32);
    case 147: return SharedMem(16, 64, 32);
    case 148: return SharedMem(16, 64, 32);
    case 149: return SharedMem(16, 64, 48);
    case 150: return SharedMem(16, 64, 48);
    case 151: return SharedMem(16, 64, 48);
    case 152: return SharedMem(16, 64, 48);
    case 153: return SharedMem(16, 64, 128);
    case 154: return SharedMem(16, 64, 128);
    case 155: return SharedMem(16, 64, 128);
    case 156: return SharedMem(16, 64, 128);
    case 157: return SharedMem(16, 64, 128);
    case 158: return SharedMem(16, 64, 128);
    case 159: return SharedMem(16, 64, 40);
    case 160: return SharedMem(16, 64, 40);
    case 161: return SharedMem(16, 64, 40);
    case 162: return SharedMem(16, 64, 16);
    case 163: return SharedMem(16, 64, 16);
    case 164: return SharedMem(16, 64, 32);
    case 165: return SharedMem(16, 64, 32);
    case 166: return SharedMem(16, 64, 32);
    case 167: return SharedMem(16, 64, 32);
    case 168: return SharedMem(16, 64, 32);
    case 169: return SharedMem(16, 64, 48);
    case 170: return SharedMem(16, 64, 48);
    case 171: return SharedMem(16, 64, 48);
    case 172: return SharedMem(16, 64, 48);
    case 173: return SharedMem(16, 64, 48);
    case 174: return SharedMem(16, 64, 128);
    case 175: return SharedMem(16, 64, 128);
    case 176: return SharedMem(16, 64, 128);
    case 177: return SharedMem(16, 64, 128);
    case 178: return SharedMem(16, 64, 128);
    case 179: return SharedMem(16, 64, 128);
    case 180: return SharedMem(16, 64, 128);
    case 181: return SharedMem(16, 64, 128);
    case 182: return SharedMem(16, 64, 128);
    case 183: return SharedMem(16, 64, 128);
    case 184: return SharedMem(16, 64, 128);
    case 185: return SharedMem(16, 64, 128);
    case 186: return SharedMem(16, 64, 128);
    case 187: return SharedMem(16, 64, 128);
    case 188: return SharedMem(16, 64, 128);
    case 189: return SharedMem(16, 64, 128);
    case 190: return SharedMem(16, 64, 128);
    case 191: return SharedMem(16, 64, 128);
    case 192: return SharedMem(16, 64, 128);
    case 193: return SharedMem(16, 64, 128);
    case 194: return SharedMem(16, 64, 128);
    case 195: return SharedMem(16, 64, 48);
    case 196: return SharedMem(16, 64, 48);
    case 197: return SharedMem(16, 64, 48);
    case 198: return SharedMem(16, 64, 48);
    case 199: return SharedMem(16, 64, 16);
    case 200: return SharedMem(16, 64, 16);
    case 201: return SharedMem(16, 64, 16);
    case 202: return SharedMem(16, 64, 16);
    case 203: return SharedMem(16, 64, 16);
    case 204: return SharedMem(16, 64, 16);
    case 205: return SharedMem(16, 64, 16);
    case 216: return SharedMem(16, 64, 8);
    case 217: return SharedMem(16, 64, 8);
    case 218: return SharedMem(16, 64, 16);
    case 219: return SharedMem(16, 64, 16);
    case 220: return SharedMem(16, 64, 8);
    case 221: return SharedMem(16, 64, 8);
    case 222: return SharedMem(16, 64, 8);
    case 223: return SharedMem(16, 64, 8);
    case 224: return SharedMem(16, 64, 8);
    case 225: return SharedMem(16, 64, 8);
    case 226: return SharedMem(16, 64, 8);
    case 227: return SharedMem(16, 64, 8);
    case 228: return SharedMem(16, 64, 8);
    case 229: return SharedMem(16, 64, 8);
    case 230: return SharedMem(16, 64, 8);
    case 231: return SharedMem(16, 64, 8);
    case 232: return SharedMem(16, 64, 8);
    case 233: return SharedMem(16, 64, 8);
    case 234: return SharedMem(16, 64, 128);
    case 235: return SharedMem(16, 64, 128);
    case 236: return SharedMem(16, 64, 128);
    case 237: return SharedMem(16, 64, 128);
    case 238: return SharedMem(16, 64, 128);
    case 239: return SharedMem(16, 64, 128);
    case 262: return SharedMem(16, 64, 128);
    case 263: return SharedMem(16, 64, 128);
    case 264: return SharedMem(16, 64, 128);
    case 265: return SharedMem(16, 64, 128);
    case 266: return SharedMem(16, 64, 128);
    case 267: return SharedMem(16, 64, 128);
    case 268: return SharedMem(16, 64, 128);
    case 269: return SharedMem(16, 64, 128);
    case 270: return SharedMem(16, 64, 128);
    case 271: return SharedMem(16, 64, 128);
    case 272: return SharedMem(16, 64, 128);
    case 273: return SharedMem(16, 64, 128);
    case 274: return SharedMem(16, 64, 128);
    case 275: return SharedMem(16, 64, 128);
    case 276: return SharedMem(16, 64, 128);
    case 277: return SharedMem(16, 64, 128);
    case 278: return SharedMem(16, 64, 128);
    case 279: return SharedMem(16, 64, 128);
    case 280: return SharedMem(16, 64, 128);
    case 281: return SharedMem(16, 64, 128);
    case 282: return SharedMem(16, 64, 128);
    case 283: return SharedMem(16, 64, 128);
    case 284: return SharedMem(16, 64, 48);
    case 285: return SharedMem(16, 64, 48);
    case 286: return SharedMem(16, 64, 48);
    case 287: return SharedMem(16, 64, 32);
    case 288: return SharedMem(16, 64, 32);
    case 289: return SharedMem(16, 64, 32);
    case 290: return SharedMem(16, 64, 32);
    case 291: return SharedMem(16, 64, 48);
    case 292: return SharedMem(16, 64, 48);
    case 293: return SharedMem(16, 64, 48);
    case 294: return SharedMem(16, 64, 128);
    case 295: return SharedMem(16, 64, 128);
    case 296: return SharedMem(16, 64, 64);
    case 297: return SharedMem(16, 64, 64);
    case 298: return SharedMem(16, 64, 128);
    case 299: return SharedMem(16, 64, 64);
    case 300: return SharedMem(16, 64, 128);
    case 301: return SharedMem(16, 64, 64);
    case 302: return SharedMem(16, 64, 128);
    case 303: return SharedMem(16, 64, 128);
    case 304: return SharedMem(16, 64, 128);
    case 305: return SharedMem(16, 64, 128);
    case 306: return SharedMem(16, 64, 128);
    case 307: return SharedMem(16, 64, 48);
    case 308: return SharedMem(16, 64, 48);
    case 309: return SharedMem(16, 64, 48);
    case 310: return SharedMem(16, 64, 48);
    case 311: return SharedMem(16, 64, 16);
    case 312: return SharedMem(16, 64, 16);
    case 313: return SharedMem(16, 64, 16);
    case 314: return SharedMem(16, 64, 48);
    case 315: return SharedMem(16, 64, 48);
    case 316: return SharedMem(16, 64, 48);
    case 317: return SharedMem(16, 64, 48);
    case 318: return SharedMem(16, 64, 48);
    case 319: return SharedMem(16, 64, 48);
    case 320: return SharedMem(16, 64, 48);
    case 321: return SharedMem(16, 64, 48);
    default: return SharedMem();
  }
}

void run_explicit_neighborhood_gemm(
    int pid, int m, int n, int k, cutlass::half_t *A, cutlass::half_t *B,
    cutlass::half_t *C, float alpha, float beta, int split_k_slices,
    cudaStream_t stream) {
  if (is_legacy_splitk_pid(pid)) {
    run_legacy_splitk_gemm(pid, m, n, k, A, B, C, alpha, beta,
                           split_k_slices, stream);
    return;
  }
  switch (pid) {
    case 90: cutlass_gemm_P90(m, n, k, A, B, C, alpha, beta, stream); return;
    case 91: cutlass_gemm_P91(m, n, k, A, B, C, alpha, beta, stream); return;
    case 92: cutlass_gemm_P92(m, n, k, A, B, C, alpha, beta, stream); return;
    case 93: cutlass_gemm_P93(m, n, k, A, B, C, alpha, beta, stream); return;
    case 94: cutlass_gemm_P94(m, n, k, A, B, C, alpha, beta, stream); return;
    case 95: cutlass_gemm_P95(m, n, k, A, B, C, alpha, beta, stream); return;
    case 96: cutlass_gemm_P96(m, n, k, A, B, C, alpha, beta, stream); return;
    case 97: cutlass_gemm_P97(m, n, k, A, B, C, alpha, beta, stream); return;
    case 98: cutlass_gemm_P98(m, n, k, A, B, C, alpha, beta, stream); return;
    case 99: cutlass_gemm_P99(m, n, k, A, B, C, alpha, beta, stream); return;
    case 100: cutlass_gemm_P100(m, n, k, A, B, C, alpha, beta, stream); return;
    case 101: cutlass_gemm_P101(m, n, k, A, B, C, alpha, beta, stream); return;
    case 102: cutlass_gemm_P102(m, n, k, A, B, C, alpha, beta, stream); return;
    case 103: cutlass_gemm_P103(m, n, k, A, B, C, alpha, beta, stream); return;
    case 104: cutlass_gemm_P104(m, n, k, A, B, C, alpha, beta, stream); return;
    case 105: cutlass_gemm_P105(m, n, k, A, B, C, alpha, beta, stream); return;
    case 106: cutlass_gemm_P106(m, n, k, A, B, C, alpha, beta, stream); return;
    case 107: cutlass_gemm_P107(m, n, k, A, B, C, alpha, beta, stream); return;
    case 108: cutlass_gemm_P108(m, n, k, A, B, C, alpha, beta, stream); return;
    case 109: cutlass_gemm_P109(m, n, k, A, B, C, alpha, beta, stream); return;
    case 110: cutlass_gemm_P110(m, n, k, A, B, C, alpha, beta, stream); return;
    case 111: cutlass_gemm_P111(m, n, k, A, B, C, alpha, beta, stream); return;
    case 112: cutlass_gemm_P112(m, n, k, A, B, C, alpha, beta, stream); return;
    case 113: cutlass_gemm_P113(m, n, k, A, B, C, alpha, beta, stream); return;
    case 114: cutlass_gemm_P114(m, n, k, A, B, C, alpha, beta, stream); return;
    case 115: cutlass_gemm_P115(m, n, k, A, B, C, alpha, beta, stream); return;
    case 116: cutlass_gemm_P116(m, n, k, A, B, C, alpha, beta, stream); return;
    case 117: cutlass_gemm_P117(m, n, k, A, B, C, alpha, beta, stream); return;
    case 118: cutlass_gemm_P118(m, n, k, A, B, C, alpha, beta, stream); return;
    case 119: cutlass_gemm_P119(m, n, k, A, B, C, alpha, beta, stream); return;
    case 120: cutlass_gemm_P120(m, n, k, A, B, C, alpha, beta, stream); return;
    case 121: cutlass_gemm_P121(m, n, k, A, B, C, alpha, beta, stream); return;
    case 122: cutlass_gemm_P122(m, n, k, A, B, C, alpha, beta, stream); return;
    case 123: cutlass_gemm_P123(m, n, k, A, B, C, alpha, beta, stream); return;
    case 124: cutlass_gemm_P124(m, n, k, A, B, C, alpha, beta, stream); return;
    case 125: cutlass_gemm_P125(m, n, k, A, B, C, alpha, beta, stream); return;
    case 126: cutlass_gemm_P126(m, n, k, A, B, C, alpha, beta, stream); return;
    case 127: cutlass_gemm_P127(m, n, k, A, B, C, alpha, beta, stream); return;
    case 128: cutlass_gemm_P128(m, n, k, A, B, C, alpha, beta, stream); return;
    case 129: cutlass_gemm_P129(m, n, k, A, B, C, alpha, beta, stream); return;
    case 130: cutlass_gemm_P130(m, n, k, A, B, C, alpha, beta, stream); return;
    case 131: cutlass_gemm_P131(m, n, k, A, B, C, alpha, beta, stream); return;
    case 132: cutlass_gemm_P132(m, n, k, A, B, C, alpha, beta, stream); return;
    case 133: cutlass_gemm_P133(m, n, k, A, B, C, alpha, beta, stream); return;
    case 134: cutlass_gemm_P134(m, n, k, A, B, C, alpha, beta, stream); return;
    case 135: cutlass_gemm_P135(m, n, k, A, B, C, alpha, beta, stream); return;
    case 136: cutlass_gemm_P136(m, n, k, A, B, C, alpha, beta, stream); return;
    case 137: cutlass_gemm_P137(m, n, k, A, B, C, alpha, beta, stream); return;
    case 138: cutlass_gemm_P138(m, n, k, A, B, C, alpha, beta, stream); return;
    case 139: cutlass_gemm_P139(m, n, k, A, B, C, alpha, beta, stream); return;
    case 140: cutlass_gemm_P140(m, n, k, A, B, C, alpha, beta, stream); return;
    case 141: cutlass_gemm_P141(m, n, k, A, B, C, alpha, beta, stream); return;
    case 142: cutlass_gemm_P142(m, n, k, A, B, C, alpha, beta, stream); return;
    case 143: cutlass_gemm_P143(m, n, k, A, B, C, alpha, beta, stream); return;
    case 144: cutlass_gemm_P144(m, n, k, A, B, C, alpha, beta, stream); return;
    case 145: cutlass_gemm_P145(m, n, k, A, B, C, alpha, beta, stream); return;
    case 146: cutlass_gemm_P146(m, n, k, A, B, C, alpha, beta, stream); return;
    case 147: cutlass_gemm_P147(m, n, k, A, B, C, alpha, beta, stream); return;
    case 148: cutlass_gemm_P148(m, n, k, A, B, C, alpha, beta, stream); return;
    case 149: cutlass_gemm_P149(m, n, k, A, B, C, alpha, beta, stream); return;
    case 150: cutlass_gemm_P150(m, n, k, A, B, C, alpha, beta, stream); return;
    case 151: cutlass_gemm_P151(m, n, k, A, B, C, alpha, beta, stream); return;
    case 152: cutlass_gemm_P152(m, n, k, A, B, C, alpha, beta, stream); return;
    case 153: cutlass_gemm_P153(m, n, k, A, B, C, alpha, beta, stream); return;
    case 154: cutlass_gemm_P154(m, n, k, A, B, C, alpha, beta, stream); return;
    case 155: cutlass_gemm_P155(m, n, k, A, B, C, alpha, beta, stream); return;
    case 156: cutlass_gemm_P156(m, n, k, A, B, C, alpha, beta, stream); return;
    case 157: cutlass_gemm_P157(m, n, k, A, B, C, alpha, beta, stream); return;
    case 158: cutlass_gemm_P158(m, n, k, A, B, C, alpha, beta, stream); return;
    case 159: cutlass_gemm_P159(m, n, k, A, B, C, alpha, beta, stream); return;
    case 160: cutlass_gemm_P160(m, n, k, A, B, C, alpha, beta, stream); return;
    case 161: cutlass_gemm_P161(m, n, k, A, B, C, alpha, beta, stream); return;
    case 162: cutlass_gemm_P162(m, n, k, A, B, C, alpha, beta, stream); return;
    case 163: cutlass_gemm_P163(m, n, k, A, B, C, alpha, beta, stream); return;
    case 164: cutlass_gemm_P164(m, n, k, A, B, C, alpha, beta, stream); return;
    case 165: cutlass_gemm_P165(m, n, k, A, B, C, alpha, beta, stream); return;
    case 166: cutlass_gemm_P166(m, n, k, A, B, C, alpha, beta, stream); return;
    case 167: cutlass_gemm_P167(m, n, k, A, B, C, alpha, beta, stream); return;
    case 168: cutlass_gemm_P168(m, n, k, A, B, C, alpha, beta, stream); return;
    case 169: cutlass_gemm_P169(m, n, k, A, B, C, alpha, beta, stream); return;
    case 170: cutlass_gemm_P170(m, n, k, A, B, C, alpha, beta, stream); return;
    case 171: cutlass_gemm_P171(m, n, k, A, B, C, alpha, beta, stream); return;
    case 172: cutlass_gemm_P172(m, n, k, A, B, C, alpha, beta, stream); return;
    case 173: cutlass_gemm_P173(m, n, k, A, B, C, alpha, beta, stream); return;
    case 174: cutlass_gemm_P174(m, n, k, A, B, C, alpha, beta, stream); return;
    case 175: cutlass_gemm_P175(m, n, k, A, B, C, alpha, beta, stream); return;
    case 176: cutlass_gemm_P176(m, n, k, A, B, C, alpha, beta, stream); return;
    case 177: cutlass_gemm_P177(m, n, k, A, B, C, alpha, beta, stream); return;
    case 178: cutlass_gemm_P178(m, n, k, A, B, C, alpha, beta, stream); return;
    case 179: cutlass_gemm_P179(m, n, k, A, B, C, alpha, beta, stream); return;
    case 180: cutlass_gemm_P180(m, n, k, A, B, C, alpha, beta, stream); return;
    case 181: cutlass_gemm_P181(m, n, k, A, B, C, alpha, beta, stream); return;
    case 182: cutlass_gemm_P182(m, n, k, A, B, C, alpha, beta, stream); return;
    case 183: cutlass_gemm_P183(m, n, k, A, B, C, alpha, beta, stream); return;
    case 184: cutlass_gemm_P184(m, n, k, A, B, C, alpha, beta, stream); return;
    case 185: cutlass_gemm_P185(m, n, k, A, B, C, alpha, beta, stream); return;
    case 186: cutlass_gemm_P186(m, n, k, A, B, C, alpha, beta, stream); return;
    case 187: cutlass_gemm_P187(m, n, k, A, B, C, alpha, beta, stream); return;
    case 188: cutlass_gemm_P188(m, n, k, A, B, C, alpha, beta, stream); return;
    case 189: cutlass_gemm_P189(m, n, k, A, B, C, alpha, beta, stream); return;
    case 190: cutlass_gemm_P190(m, n, k, A, B, C, alpha, beta, stream); return;
    case 191: cutlass_gemm_P191(m, n, k, A, B, C, alpha, beta, stream); return;
    case 192: cutlass_gemm_P192(m, n, k, A, B, C, alpha, beta, stream); return;
    case 193: cutlass_gemm_P193(m, n, k, A, B, C, alpha, beta, stream); return;
    case 194: cutlass_gemm_P194(m, n, k, A, B, C, alpha, beta, stream); return;
    case 195: cutlass_gemm_P195(m, n, k, A, B, C, alpha, beta, stream); return;
    case 196: cutlass_gemm_P196(m, n, k, A, B, C, alpha, beta, stream); return;
    case 197: cutlass_gemm_P197(m, n, k, A, B, C, alpha, beta, stream); return;
    case 198: cutlass_gemm_P198(m, n, k, A, B, C, alpha, beta, stream); return;
    case 199: cutlass_gemm_P199(m, n, k, A, B, C, alpha, beta, stream); return;
    case 200: cutlass_gemm_P200(m, n, k, A, B, C, alpha, beta, stream); return;
    case 201: cutlass_gemm_P201(m, n, k, A, B, C, alpha, beta, stream); return;
    case 202: cutlass_gemm_P202(m, n, k, A, B, C, alpha, beta, stream); return;
    case 203: cutlass_gemm_P203(m, n, k, A, B, C, alpha, beta, stream); return;
    case 204: cutlass_gemm_P204(m, n, k, A, B, C, alpha, beta, stream); return;
    case 205: cutlass_gemm_P205(m, n, k, A, B, C, alpha, beta, stream); return;
    case 216: cutlass_gemm_P216(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 217: cutlass_gemm_P217(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 218: cutlass_gemm_P218(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 219: cutlass_gemm_P219(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 220: cutlass_gemm_P220(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 221: cutlass_gemm_P221(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 222: cutlass_gemm_P222(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 223: cutlass_gemm_P223(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 224: cutlass_gemm_P224(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 225: cutlass_gemm_P225(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 226: cutlass_gemm_P226(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 227: cutlass_gemm_P227(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 228: cutlass_gemm_P228(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 229: cutlass_gemm_P229(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 230: cutlass_gemm_P230(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 231: cutlass_gemm_P231(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 232: cutlass_gemm_P232(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 233: cutlass_gemm_P233(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 234: cutlass_gemm_P234(m, n, k, A, B, C, alpha, beta, stream); return;
    case 235: cutlass_gemm_P235(m, n, k, A, B, C, alpha, beta, stream); return;
    case 236: cutlass_gemm_P236(m, n, k, A, B, C, alpha, beta, stream); return;
    case 237: cutlass_gemm_P237(m, n, k, A, B, C, alpha, beta, stream); return;
    case 238: cutlass_gemm_P238(m, n, k, A, B, C, alpha, beta, stream); return;
    case 239: cutlass_gemm_P239(m, n, k, A, B, C, alpha, beta, stream); return;
    case 262: cutlass_gemm_P262(m, n, k, A, B, C, alpha, beta, stream); return;
    case 263: cutlass_gemm_P263(m, n, k, A, B, C, alpha, beta, stream); return;
    case 264: cutlass_gemm_P264(m, n, k, A, B, C, alpha, beta, stream); return;
    case 265: cutlass_gemm_P265(m, n, k, A, B, C, alpha, beta, stream); return;
    case 266: cutlass_gemm_P266(m, n, k, A, B, C, alpha, beta, stream); return;
    case 267: cutlass_gemm_P267(m, n, k, A, B, C, alpha, beta, stream); return;
    case 268: cutlass_gemm_P268(m, n, k, A, B, C, alpha, beta, stream); return;
    case 269: cutlass_gemm_P269(m, n, k, A, B, C, alpha, beta, stream); return;
    case 270: cutlass_gemm_P270(m, n, k, A, B, C, alpha, beta, stream); return;
    case 271: cutlass_gemm_P271(m, n, k, A, B, C, alpha, beta, stream); return;
    case 272: cutlass_gemm_P272(m, n, k, A, B, C, alpha, beta, stream); return;
    case 273: cutlass_gemm_P273(m, n, k, A, B, C, alpha, beta, stream); return;
    case 274: cutlass_gemm_P274(m, n, k, A, B, C, alpha, beta, stream); return;
    case 275: cutlass_gemm_P275(m, n, k, A, B, C, alpha, beta, stream); return;
    case 276: cutlass_gemm_P276(m, n, k, A, B, C, alpha, beta, stream); return;
    case 277: cutlass_gemm_P277(m, n, k, A, B, C, alpha, beta, stream); return;
    case 278: cutlass_gemm_P278(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 279: cutlass_gemm_P279(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 280: cutlass_gemm_P280(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 281: cutlass_gemm_P281(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 282: cutlass_gemm_P282(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 283: cutlass_gemm_P283(m, n, k, A, B, C, alpha, beta, split_k_slices, stream); return;
    case 284: cutlass_gemm_P284(m, n, k, A, B, C, alpha, beta, stream); return;
    case 285: cutlass_gemm_P285(m, n, k, A, B, C, alpha, beta, stream); return;
    case 286: cutlass_gemm_P286(m, n, k, A, B, C, alpha, beta, stream); return;
    case 287: cutlass_gemm_P287(m, n, k, A, B, C, alpha, beta, stream); return;
    case 288: cutlass_gemm_P288(m, n, k, A, B, C, alpha, beta, stream); return;
    case 289: cutlass_gemm_P289(m, n, k, A, B, C, alpha, beta, stream); return;
    case 290: cutlass_gemm_P290(m, n, k, A, B, C, alpha, beta, stream); return;
    case 291: cutlass_gemm_P291(m, n, k, A, B, C, alpha, beta, stream); return;
    case 292: cutlass_gemm_P292(m, n, k, A, B, C, alpha, beta, stream); return;
    case 293: cutlass_gemm_P293(m, n, k, A, B, C, alpha, beta, stream); return;
    case 294: cutlass_gemm_P294(m, n, k, A, B, C, alpha, beta, stream); return;
    case 295: cutlass_gemm_P295(m, n, k, A, B, C, alpha, beta, stream); return;
    case 296: cutlass_gemm_P296(m, n, k, A, B, C, alpha, beta, stream); return;
    case 297: cutlass_gemm_P297(m, n, k, A, B, C, alpha, beta, stream); return;
    case 298: cutlass_gemm_P298(m, n, k, A, B, C, alpha, beta, stream); return;
    case 299: cutlass_gemm_P299(m, n, k, A, B, C, alpha, beta, stream); return;
    case 300: cutlass_gemm_P300(m, n, k, A, B, C, alpha, beta, stream); return;
    case 301: cutlass_gemm_P301(m, n, k, A, B, C, alpha, beta, stream); return;
    case 302: cutlass_gemm_P302(m, n, k, A, B, C, alpha, beta, stream); return;
    case 303: cutlass_gemm_P303(m, n, k, A, B, C, alpha, beta, stream); return;
    case 304: cutlass_gemm_P304(m, n, k, A, B, C, alpha, beta, stream); return;
    case 305: cutlass_gemm_P305(m, n, k, A, B, C, alpha, beta, stream); return;
    case 306: cutlass_gemm_P306(m, n, k, A, B, C, alpha, beta, stream); return;
    case 307: cutlass_gemm_P307(m, n, k, A, B, C, alpha, beta, stream); return;
    case 308: cutlass_gemm_P308(m, n, k, A, B, C, alpha, beta, stream); return;
    case 309: cutlass_gemm_P309(m, n, k, A, B, C, alpha, beta, stream); return;
    case 310: cutlass_gemm_P310(m, n, k, A, B, C, alpha, beta, stream); return;
    case 311: cutlass_gemm_P311(m, n, k, A, B, C, alpha, beta, stream); return;
    case 312: cutlass_gemm_P312(m, n, k, A, B, C, alpha, beta, stream); return;
    case 313: cutlass_gemm_P313(m, n, k, A, B, C, alpha, beta, stream); return;
    case 314: cutlass_gemm_P314(m, n, k, A, B, C, alpha, beta, stream); return;
    case 315: cutlass_gemm_P315(m, n, k, A, B, C, alpha, beta, stream); return;
    case 316: cutlass_gemm_P316(m, n, k, A, B, C, alpha, beta, stream); return;
    case 317: cutlass_gemm_P317(m, n, k, A, B, C, alpha, beta, stream); return;
    case 318: cutlass_gemm_P318(m, n, k, A, B, C, alpha, beta, stream); return;
    case 319: cutlass_gemm_P319(m, n, k, A, B, C, alpha, beta, stream); return;
    case 320: cutlass_gemm_P320(m, n, k, A, B, C, alpha, beta, stream); return;
    case 321: cutlass_gemm_P321(m, n, k, A, B, C, alpha, beta, stream); return;
    default: return;
  }
}

}  // namespace row_col_col
}  // namespace fp16
}  // namespace moonpoly
