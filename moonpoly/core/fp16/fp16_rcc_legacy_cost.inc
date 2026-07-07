// Legacy FP16 handwritten piecewise cost functions.
// Included by fp16_rcc_cost.cu inside moonpoly::fp16::row_col_col.
// The default selector uses generated fitted costs; this block is the
// analytic fallback and compatibility path.

/*
   ============================ COST MODEL FUNCTIONS ===========================
 */

float gemm_k_func_P0(int X) {
  if (X >= 1 && X <= 1382) {
    return 0.0002469 * X + 0.006563;
    ;
  } else if (X >= 1383 && X <= 1449) {
    return 0.0002783 * X - 0.03698;
    ;
  } else {
    return 0.000248 * X + 0.005741;
    ;
  }
}

float gemm_k_func_P1(int X) {
  if (X >= 1 && X <= 862) {
    return 0.0003979 * X + 0.009904;
    ;
  } else if (X >= 863 && X <= 1100) {
    return 0.0004021 * X + 0.005983;
    ;
  } else if (X >= 1101 && X <= 3875) {
    return 0.0003993 * X + 0.008812;
    ;
  } else if (X >= 3876 && X <= 4057) {
    return 0.000395 * X + 0.02581;
    ;
  } else {
    return 0.0003991 * X + 0.00938;
    ;
  }
}

float gemm_k_func_P2(int X) {
  if (X >= 1 && X <= 769) {
    return 0.0004167 * X + 0.008072;
    ;
  } else if (X >= 770 && X <= 2032) {
    return 0.0004179 * X + 0.005729;
    ;
  } else if (X >= 2033 && X <= 3073) {
    return 0.0004167 * X + 0.00875;
    ;
  } else if (X >= 3074 && X <= 4121) {
    return 0.0004167 * X + 0.008762;
    ;
  } else {
    return 0.0004167 * X + 0.00884;
    ;
  }
}

float gemm_k_func_P3(int X) {
  if (X >= 1 && X <= 1025) {
    return 0.0002985 * X + 0.006937;
    ;
  } else if (X >= 1026 && X <= 3073) {
    return 0.0002985 * X + 0.007863;
    ;
  } else if (X >= 3074 && X <= 4095) {
    return 0.0002986 * X + 0.007249;
    ;
  } else if (X >= 4096 && X <= 4124) {
    return 0.0004503 * X - 0.6179;
    ;
  } else {
    return 0.0002984 * X + 0.007996;
    ;
  }
}

float gemm_k_func_P4(int X) {
  if (X >= 1 && X <= 734) {
    return 0.0001531 * X + 0.006246;
    ;
  } else if (X >= 735 && X <= 1784) {
    return 0.0001531 * X + 0.006234;
    ;
  } else if (X >= 1785 && X <= 3565) {
    return 0.0001543 * X + 0.003449;
    ;
  } else if (X >= 3566 && X <= 3577) {
    return 0.0001569 * X - 0.005434;
    ;
  } else {
    return 0.000154 * X + 0.005544;
    ;
  }
}

float gemm_k_func_P5(int X) {
  if (X >= 1 && X <= 507) {
    return 0.0000845 * X + 0.005965;
    ;
  } else if (X >= 508 && X <= 1056) {
    return 0.0000845 * X + 0.005967;
    ;
  } else if (X >= 1057 && X <= 1718) {
    return 0.0000845 * X + 0.005976;
    ;
  } else if (X >= 1719 && X <= 3496) {
    return 0.0000845 * X + 0.005935;
    ;
  } else {
    return 0.0000845 * X + 0.005883;
    ;
  }
}

float gemm_k_func_P6(int X) {
  if (X >= 1 && X <= 246) {
    return 0.0001458 * X + 0.006215;
    ;
  } else if (X >= 247 && X <= 1208) {
    return 0.0001461 * X + 0.006165;
    ;
  } else if (X >= 1209 && X <= 2685) {
    return 0.0001461 * X + 0.006171;
    ;
  } else if (X >= 2686 && X <= 4150) {
    return 0.0001461 * X + 0.006154;
    ;
  } else {
    return 0.0001483 * X - 0.003713;
    ;
  }
}

float gemm_k_func_P7(int X) {
  if (X >= 1 && X <= 383) {
    return 0.002361 * X + 0.01202;
    ;
  } else if (X >= 384 && X <= 831) {
    return 0.002765 * X - 0.0563;
    ;
  } else if (X >= 832 && X <= 977) {
    return 0.003246 * X - 0.4331;
    ;
  } else if (X >= 978 && X <= 4096) {
    return 0.002773 * X + 0.0129;
    ;
  } else if (X >= 4097 && X <= 4126) {
    return 0.001042 * X + 7.15;
    ;
  } else {
    return 0.00277 * X + 0.02435;
    ;
  }
}

float gemm_k_func_P8(int X) {
  if (X >= 1 && X <= 2110) {
    return 0.0001788 * X + 0.006045;
    ;
  } else if (X >= 2111 && X <= 2209) {
    return 0.0009336 * X - 1.59;
    ;
  } else if (X >= 2210 && X <= 2306) {
    return 0.000222 * X - 0.02384;
    ;
  } else if (X >= 2307 && X <= 2623) {
    return 0.0003031 * X - 0.2024;
    ;
  } else if (X >= 2624 && X <= 4298) {
    return 0.0002246 * X + 0.004217;
    ;
  } else if (X >= 4299 && X <= 4375) {
    return 0.001241 * X - 4.369;
    ;
  } else if (X >= 4376 && X <= 4455) {
    return 0.0007599 * X - 2.27;
    ;
  } else if (X >= 4456 && X <= 4729) {
    return 0.0002846 * X - 0.1548;
    ;
  } else if (X >= 4730 && X <= 4859) {
    return 0.0005323 * X - 1.325;
    ;
  } else {
    return 0.000318 * X - 0.2846;
    ;
  }
}

float gemm_k_func_P9(int X) {
  if (X >= 1 && X <= 512) {
    return 0.0005459 * X + 0.005948;
    ;
  } else if (X >= 513 && X <= 1792) {
    return 0.0005464 * X + 0.005532;
    ;
  } else if (X >= 1793 && X <= 1984) {
    return 0.0006627 * X - 0.1866;
    ;
  } else if (X >= 1985 && X <= 2032) {
    return 0.005023 * X - 8.7;
    ;
  } else if (X >= 2033 && X <= 2510) {
    return 0.00114 * X - 0.8137;
    ;
  } else if (X >= 2511 && X <= 4031) {
    return 0.0008139 * X - 0.01354;
    ;
  } else if (X >= 4032 && X <= 4098) {
    return 0.004424 * X - 14.53;
    ;
  } else if (X >= 4099 && X <= 4104) {
    return 0.0000289 * X + 3.435;
    ;
  } else if (X >= 4105 && X <= 4259) {
    return 0.002644 * X - 7.257;
    ;
  } else {
    return 0.0009897 * X - 0.2404;
    ;
  }
}

float gemm_k_func_P10(int X) {
  if (X >= 1 && X <= 1534) {
    return 0.0009453 * X + 0.006416;
    ;
  } else if (X >= 1535 && X <= 1583) {
    return 0.004164 * X - 4.889;
    ;
  } else if (X >= 1584 && X <= 1849) {
    return 0.0015 * X - 0.6815;
    ;
  } else if (X >= 1850 && X <= 2058) {
    return 0.001932 * X - 1.468;
    ;
  } else if (X >= 2059 && X <= 3329) {
    return 0.001217 * X - 0.02395;
    ;
  } else if (X >= 3330 && X <= 3644) {
    return 0.002593 * X - 4.482;
    ;
  } else if (X >= 3645 && X <= 3856) {
    return 0.001534 * X - 0.647;
    ;
  } else {
    return 0.001356 * X + 0.03642;
    ;
  }
}

float gemm_k_func_P11(int X) {
  if (X >= 1 && X <= 884) {
    return 0.0009024 * X + 0.00746;
    ;
  } else if (X >= 885 && X <= 906) {
    return 0.002301 * X - 1.194;
    ;
  } else if (X >= 907 && X <= 960) {
    return 0.001302 * X - 0.2978;
    ;
  } else if (X >= 961 && X <= 1136) {
    return 0.001949 * X - 0.854;
    ;
  } else if (X >= 1137 && X <= 1987) {
    return 0.001215 * X - 0.02468;
    ;
  } else if (X >= 1988 && X <= 2049) {
    return 0.003063 * X - 3.675;
    ;
  } else if (X >= 2050 && X <= 2179) {
    return 0.002221 * X - 1.96;
    ;
  } else if (X >= 2180 && X <= 3291) {
    return 0.001323 * X - 0.006933;
    ;
  } else if (X >= 3292 && X <= 4896) {
    return 0.001316 * X + 0.01183;
    ;
  } else {
    return 0.001366 * X - 0.2397;
    ;
  }
}

float gemm_k_func_P12(int X) {
  if (X >= 1 && X <= 379) {
    return 0.001121 * X + 0.007355;
    ;
  } else if (X >= 380 && X <= 768) {
    return 0.001122 * X + 0.0071;
    ;
  } else if (X >= 769 && X <= 940) {
    return 0.001825 * X - 0.475;
    ;
  } else if (X >= 941 && X <= 1667) {
    return 0.001349 * X - 0.02298;
    ;
  } else if (X >= 1668 && X <= 1725) {
    return 0.004284 * X - 4.928;
    ;
  } else if (X >= 1726 && X <= 1781) {
    return 0.002175 * X - 1.279;
    ;
  } else if (X >= 1782 && X <= 2864) {
    return 0.001477 * X - 0.02406;
    ;
  } else if (X >= 2865 && X <= 4720) {
    return 0.001466 * X + 0.00295;
    ;
  } else if (X >= 4721 && X <= 4772) {
    return 0.002026 * X - 2.663;
    ;
  } else {
    return 0.001441 * X + 0.1229;
    ;
  }
}

float gemm_k_func_P13(int X) {
  if (X >= 1 && X <= 561) {
    return 0.0001238 * X + 0.006115;
    ;
  } else if (X >= 562 && X <= 1475) {
    return 0.0001238 * X + 0.006125;
    ;
  } else if (X >= 1476 && X <= 2751) {
    return 0.0001238 * X + 0.006056;
    ;
  } else if (X >= 2752 && X <= 4102) {
    return 0.0001238 * X + 0.006009;
    ;
  } else {
    return 0.0001237 * X + 0.006348;
    ;
  }
}

float gemm_k_func_P14(int X) {
  if (X >= 1 && X <= 1039) {
    return 0.0003906 * X + 0.009576;
    ;
  } else if (X >= 1040 && X <= 1229) {
    return 0.0004087 * X - 0.006424;
    ;
  } else if (X >= 1230 && X <= 1297) {
    return 0.00111 * X - 0.8618;
    ;
  } else if (X >= 1298 && X <= 1439) {
    return 0.0005771 * X - 0.1762;
    ;
  } else if (X >= 1440 && X <= 2387) {
    return 0.0004635 * X - 0.00459;
    ;
  } else if (X >= 2388 && X <= 2507) {
    return 0.0004681 * X - 0.00473;
    ;
  } else if (X >= 2508 && X <= 2657) {
    return 0.001327 * X - 2.129;
    ;
  } else if (X >= 2658 && X <= 2749) {
    return 0.0006837 * X - 0.4473;
    ;
  } else if (X >= 2750 && X <= 4145) {
    return 0.0005253 * X - 0.006643;
    ;
  } else {
    return 0.0005216 * X + 0.004656;
    ;
  }
}

float gemm_k_func_P15(int X) {
  if (X >= 1 && X <= 487) {
    return 0.0005308 * X + 0.008827;
    ;
  } else if (X >= 488 && X <= 1543) {
    return 0.0005302 * X + 0.009356;
    ;
  } else if (X >= 1544 && X <= 1584) {
    return 0.001345 * X - 1.244;
    ;
  } else if (X >= 1585 && X <= 2885) {
    return 0.000565 * X + 0.003116;
    ;
  } else {
    return 0.0005628 * X + 0.008749;
    ;
  }
}

float gemm_k_func_P16(int X) {
  if (X >= 1 && X <= 14) {
    return 0.0002154 * X + 0.006948;
    ;
  } else if (X >= 15 && X <= 1968) {
    return 0.0002411 * X + 0.006629;
    ;
  } else if (X >= 1969 && X <= 4435) {
    return 0.0002411 * X + 0.005493;
    ;
  } else {
    return 0.0002407 * X + 0.008612;
    ;
  }
}

float gemm_k_func_P17(int X) {
  if (X >= 1 && X <= 532) {
    return 0.000467 * X + 0.008107;
    ;
  } else if (X >= 533 && X <= 642) {
    return 0.0005393 * X - 0.02574;
    ;
  } else if (X >= 643 && X <= 1072) {
    return 0.0004873 * X + 0.005965;
    ;
  } else if (X >= 1073 && X <= 1125) {
    return 0.001399 * X - 0.9656;
    ;
  } else if (X >= 1126 && X <= 1279) {
    return 0.0006799 * X - 0.169;
    ;
  } else if (X >= 1280 && X <= 4108) {
    return 0.000542 * X + 0.003448;
    ;
  } else if (X >= 4109 && X <= 4141) {
    return 0.0003679 * X + 0.7216;
    ;
  } else {
    return 0.0005423 * X + 0.002846;
    ;
  }
}

float gemm_k_func_P18(int X) {
  if (X >= 1 && X <= 527) {
    return 0.0002225 * X + 0.006581;
    ;
  } else if (X >= 528 && X <= 1039) {
    return 0.000223 * X + 0.006152;
    ;
  } else if (X >= 1040 && X <= 3904) {
    return 0.000223 * X + 0.00601;
    ;
  } else if (X >= 3905 && X <= 4096) {
    return 0.0002738 * X - 0.1921;
    ;
  } else if (X >= 4097 && X <= 4138) {
    return 0.0001104 * X + 0.477;
    ;
  } else if (X >= 4139 && X <= 4413) {
    return 0.0002916 * X - 0.2729;
    ;
  } else {
    return 0.000228 * X + 0.007439;
    ;
  }
}

float gemm_k_func_P19(int X) {
  if (X >= 1 && X <= 1089) {
    return 0.0002723 * X + 0.006601;
    ;
  } else if (X >= 1090 && X <= 2207) {
    return 0.0002768 * X + 0.001862;
    ;
  } else if (X >= 2208 && X <= 2263) {
    return 0.0008935 * X - 1.352;
    ;
  } else if (X >= 2264 && X <= 2449) {
    return 0.0004043 * X - 0.2463;
    ;
  } else if (X >= 2450 && X <= 2848) {
    return 0.0003113 * X - 0.0177;
    ;
  } else {
    return 0.0003016 * X + 0.00894;
    ;
  }
}

float gemm_k_func_P20(int X) {
  if (X >= 1 && X <= 1944) {
    return 0.0003832 * X + 0.008944;
    ;
  } else {
    return 0.0003838 * X + 0.009948;
    ;
  }
}

float gemm_k_func_P21(int X) {
  if (X >= 1 && X <= 2614) {
    return 0.0003488 * X + 0.006232;
    ;
  } else if (X >= 2615 && X <= 4009) {
    return 0.0003571 * X - 0.01873;
    ;
  } else {
    return 0.0003675 * X - 0.06003;
    ;
  }
}

float gemm_k_func_P22(int X) {
  if (X >= 1 && X <= 1280) {
    return 0.0003113 * X + 0.008524;
    ;
  } else if (X >= 1281 && X <= 1349) {
    return 0.0003616 * X - 0.03085;
    ;
  } else if (X >= 1350 && X <= 1365) {
    return 0.00196 * X - 2.184;
    ;
  } else if (X >= 1366 && X <= 1647) {
    return 0.0006252 * X - 0.3165;
    ;
  } else if (X >= 1648 && X <= 1711) {
    return 0.0006739 * X - 0.4165;
    ;
  } else if (X >= 1712 && X <= 2845) {
    return 0.0004643 * X - 0.06206;
    ;
  } else if (X >= 2846 && X <= 3325) {
    return 0.0005844 * X - 0.2939;
    ;
  } else if (X >= 3326 && X <= 4131) {
    return 0.0005049 * X - 0.04191;
    ;
  } else if (X >= 4132 && X <= 4297) {
    return 0.0004565 * X + 0.151;
    ;
  } else {
    return 0.0004918 * X + 0.003529;
    ;
  }
}

float gemm_k_func_P23(int X) {
  if (X >= 1 && X <= 261) {
    return 0.0007421 * X + 0.01168;
    ;
  } else if (X >= 262 && X <= 1030) {
    return 0.0007398 * X + 0.01183;
    ;
  } else if (X >= 1031 && X <= 3091) {
    return 0.0007386 * X + 0.01271;
    ;
  } else if (X >= 3092 && X <= 4096) {
    return 0.0007404 * X + 0.006244;
    ;
  } else {
    return 0.0007366 * X + 0.02145;
    ;
  }
}

float gemm_k_func_P24(int X) {
  if (X >= 1 && X <= 1476) {
    return 0.0003227 * X + 0.005689;
    ;
  } else if (X >= 1477 && X <= 3072) {
    return 0.000323 * X + 0.004998;
    ;
  } else if (X >= 3073 && X <= 3280) {
    return 0.0005549 * X - 0.1344;
    ;
  } else if (X >= 3281 && X <= 3421) {
    return 0.001257 * X - 2.431;
    ;
  } else if (X >= 3422 && X <= 3610) {
    return 0.00167 * X - 3.84;
    ;
  } else if (X >= 3611 && X <= 3617) {
    return -0.001524 * X + 7.667;
    ;
  } else if (X >= 3618 && X <= 3885) {
    return 0.0007037 * X - 0.3883;
    ;
  } else if (X >= 3886 && X <= 4108) {
    return 0.001094 * X - 1.907;
    ;
  } else if (X >= 4109 && X <= 4737) {
    return 0.0006299 * X - 0.01416;
    ;
  } else {
    return 0.0006155 * X + 0.05444;
    ;
  }
}

float gemm_k_func_P25(int X) {
  if (X >= 1 && X <= 953) {
    return 0.0001417 * X + 0.005967;
    ;
  } else if (X >= 954 && X <= 3284) {
    return 0.0001417 * X + 0.005972;
    ;
  } else {
    return 0.0001417 * X + 0.005965;
    ;
  }
}

float gemm_k_func_P26(int X) {
  if (X >= 1 && X <= 1788) {
    return 0.0002583 * X + 0.006141;
    ;
  } else if (X >= 1789 && X <= 1888) {
    return 0.0002826 * X - 0.03364;
    ;
  } else if (X >= 1889 && X <= 1926) {
    return 0.003045 * X - 5.222;
    ;
  } else if (X >= 1927 && X <= 2054) {
    return 0.0004629 * X - 0.2577;
    ;
  } else if (X >= 2055 && X <= 2276) {
    return 0.0006499 * X - 0.6413;
    ;
  } else if (X >= 2277 && X <= 3887) {
    return 0.0003748 * X - 0.01745;
    ;
  } else if (X >= 3888 && X <= 4098) {
    return 0.0007921 * X - 1.623;
    ;
  } else if (X >= 4099 && X <= 4108) {
    return -0.0000279 * X + 1.69;
    ;
  } else if (X >= 4109 && X <= 4365) {
    return 0.0009992 * X - 2.501;
    ;
  } else {
    return 0.0004262 * X - 0.004853;
    ;
  }
}

float gemm_k_func_P27(int X) {
  if (X >= 1 && X <= 751) {
    return 0.0003996 * X + 0.006259;
    ;
  } else if (X >= 752 && X <= 1277) {
    return 0.0003993 * X + 0.006457;
    ;
  } else if (X >= 1278 && X <= 2087) {
    return 0.0003983 * X + 0.00839;
    ;
  } else if (X >= 2088 && X <= 2691) {
    return 0.0004468 * X - 0.09496;
    ;
  } else if (X >= 2692 && X <= 4095) {
    return 0.0004522 * X - 0.1239;
    ;
  } else if (X >= 4096 && X <= 4185) {
    return 0.001345 * X - 3.728;
    ;
  } else if (X >= 4186 && X <= 4869) {
    return 0.0004536 * X - 0.02323;
    ;
  } else if (X >= 4870 && X <= 4940) {
    return 0.004259 * X - 18.37;
    ;
  } else if (X >= 4941 && X <= 4963) {
    return 0.001206 * X - 3.28;
    ;
  } else {
    return 0.001303 * X - 3.755;
    ;
  }
}

float gemm_k_func_P28(int X) {
  if (X >= 1 && X <= 1542) {
    return 0.0000861 * X + 0.005885;
    ;
  } else if (X >= 1543 && X <= 3393) {
    return 0.0000862 * X + 0.00585;
    ;
  } else if (X >= 3394 && X <= 3932) {
    return 0.0000942 * X - 0.02077;
    ;
  } else if (X >= 3933 && X <= 4152) {
    return 0.0000877 * X + 0.006263;
    ;
  } else if (X >= 4153 && X <= 4883) {
    return 0.0000884 * X + 0.003741;
    ;
  } else {
    return 0.0000868 * X + 0.01117;
    ;
  }
}

float gemm_k_func_P29(int X) {
  if (X >= 1 && X <= 1942) {
    return 0.0003698 * X + 0.005775;
    ;
  } else if (X >= 1943 && X <= 2704) {
    return 0.0003682 * X + 0.009736;
    ;
  } else if (X >= 2705 && X <= 3904) {
    return 0.00037 * X + 0.004825;
    ;
  } else if (X >= 3905 && X <= 4321) {
    return 0.0003686 * X + 0.01184;
    ;
  } else {
    return 0.0003687 * X + 0.01045;
    ;
  }
}

float gemm_k_func_P30(int X) {
  if (X >= 1 && X <= 256) {
    return 0.0008424 * X + 0.01302;
    ;
  } else if (X >= 257 && X <= 515) {
    return 0.0008396 * X + 0.01381;
    ;
  } else if (X >= 516 && X <= 1030) {
    return 0.0008455 * X + 0.01075;
    ;
  } else if (X >= 1031 && X <= 2048) {
    return 0.0008428 * X + 0.01318;
    ;
  } else if (X >= 2049 && X <= 4106) {
    return 0.000842 * X + 0.01503;
    ;
  } else {
    return 0.0008412 * X + 0.01808;
    ;
  }
}

float gemm_k_func_P31(int X) {
  if (X >= 1 && X <= 2560) {
    return 0.001169 * X + 0.01001;
    ;
  } else {
    return 0.001169 * X + 0.01099;
    ;
  }
}

float gemm_k_func_P32(int X) {
  if (X >= 1 && X <= 247) {
    return 0.0004293 * X + 0.006444;
    ;
  } else if (X >= 248 && X <= 511) {
    return 0.0004264 * X + 0.007546;
    ;
  } else if (X >= 512 && X <= 2065) {
    return 0.0004294 * X + 0.006726;
    ;
  } else if (X >= 2066 && X <= 3597) {
    return 0.0004299 * X + 0.006206;
    ;
  } else if (X >= 3598 && X <= 3967) {
    return 0.0004864 * X - 0.1859;
    ;
  } else if (X >= 3968 && X <= 4407) {
    return 0.0004414 * X - 0.007716;
    ;
  } else {
    return 0.0004367 * X + 0.01419;
    ;
  }
}

float gemm_k_func_P33(int X) {
  if (X >= 1 && X <= 280) {
    return 0.002559 * X + 0.009536;
    ;
  } else if (X >= 281 && X <= 512) {
    return 0.002562 * X + 0.007047;
    ;
  } else if (X >= 513 && X <= 1104) {
    return 0.00309 * X - 0.115;
    ;
  } else if (X >= 1105 && X <= 1216) {
    return 0.005932 * X - 3.211;
    ;
  } else {
    return 0.003274 * X + 0.009833;
    ;
  }
}

float gemm_k_func_P34(int X) {
  if (X >= 1 && X <= 512) {
    return 0.0002244 * X + 0.007168;
    ;
  } else if (X >= 513 && X <= 1042) {
    return 0.0002248 * X + 0.006802;
    ;
  } else if (X >= 1043 && X <= 2071) {
    return 0.000228 * X + 0.002209;
    ;
  } else if (X >= 2072 && X <= 4096) {
    return 0.0002269 * X + 0.004172;
    ;
  } else if (X >= 4097 && X <= 4495) {
    return 0.000264 * X - 0.1411;
    ;
  } else {
    return 0.0002314 * X + 0.005792;
    ;
  }
}

float gemm_k_func_P35(int X) {
  if (X >= 1 && X <= 2402) {
    return 0.0003768 * X + 0.005656;
    ;
  } else if (X >= 2403 && X <= 3573) {
    return 0.0003767 * X + 0.006835;
    ;
  } else if (X >= 3574 && X <= 4088) {
    return 0.0003769 * X + 0.006475;
    ;
  } else if (X >= 4089 && X <= 4122) {
    return 0.0003088 * X + 0.2834;
    ;
  } else if (X >= 4123 && X <= 4460) {
    return 0.0003774 * X + 0.002837;
    ;
  } else {
    return 0.0003771 * X + 0.004172;
    ;
  }
}

float gemm_k_func_P36(int X) {
  if (X >= 1 && X <= 527) {
    return 0.0003718 * X + 0.005589;
    ;
  } else if (X >= 528 && X <= 1413) {
    return 0.0003719 * X + 0.005682;
    ;
  } else if (X >= 1414 && X <= 2729) {
    return 0.0003717 * X + 0.005985;
    ;
  } else if (X >= 2730 && X <= 2945) {
    return 0.0003472 * X + 0.09641;
    ;
  } else if (X >= 2946 && X <= 4129) {
    return 0.0003909 * X - 0.03368;
    ;
  } else if (X >= 4130 && X <= 4863) {
    return 0.0004038 * X - 0.06229;
    ;
  } else if (X >= 4864 && X <= 4944) {
    return 0.006107 * X - 27.61;
    ;
  } else {
    return 0.0014 * X - 4.368;
    ;
  }
}

float gemm_k_func_P37(int X) {
  if (X >= 1 && X <= 527) {
    return 0.0002574 * X + 0.00686;
    ;
  } else if (X >= 528 && X <= 1413) {
    return 0.0002574 * X + 0.00684;
    ;
  } else if (X >= 1414 && X <= 2729) {
    return 0.0002604 * X + 0.002252;
    ;
  } else if (X >= 2730 && X <= 2945) {
    return 0.0002596 * X + 0.003718;
    ;
  } else if (X >= 2946 && X <= 4129) {
    return 0.0003923 * X - 0.4046;
    ;
  } else if (X >= 4130 && X <= 4863) {
    return 0.000287 * X + 0.007347;
    ;
  } else if (X >= 4864 && X <= 4944) {
    return 0.0002876 * X + 0.001975;
    ;
  } else {
    return 0.0002886 * X - 0.001699;
    ;
  }
}

float gemm_k_func_P38(int X) {
  if (X >= 1 && X <= 269) {
    return 0.0004739 * X + 0.005928;
    ;
  } else if (X >= 270 && X <= 512) {
    return 0.0004749 * X + 0.005933;
    ;
  } else if (X >= 513 && X <= 2045) {
    return 0.0004762 * X + 0.004272;
    ;
  } else if (X >= 2046 && X <= 4143) {
    return 0.0004746 * X + 0.007113;
    ;
  } else if (X >= 4144 && X <= 4681) {
    return 0.000467 * X + 0.04103;
    ;
  } else {
    return 0.0004822 * X - 0.03054;
    ;
  }
}

float gemm_k_func_P39(int X) {
  if (X >= 1 && X <= 2046) {
    return 0.0003643 * X + 0.008597;
    ;
  } else if (X >= 2047 && X <= 2061) {
    return 0.001622 * X - 2.552;
    ;
  } else if (X >= 2062 && X <= 2432) {
    return 0.0004312 * X - 0.06762;
    ;
  } else if (X >= 2433 && X <= 2462) {
    return 0.007023 * X - 15.99;
    ;
  } else if (X >= 2463 && X <= 2560) {
    return 0.001877 * X - 3.328;
    ;
  } else if (X >= 2561 && X <= 2699) {
    return 0.002397 * X - 4.799;
    ;
  } else if (X >= 2700 && X <= 3073) {
    return 0.0008205 * X - 0.5324;
    ;
  } else if (X >= 3074 && X <= 3584) {
    return 0.0006469 * X - 0.01136;
    ;
  } else if (X >= 3585 && X <= 4993) {
    return 0.0006484 * X + 0.006143;
    ;
  } else {
    return 0.002724 * X - 10.32;
    ;
  }
}

float gemm_k_func_P40(int X) {
  if (X >= 1 && X <= 1476) {
    return 0.0003227 * X + 0.005689;
    ;
  } else if (X >= 1477 && X <= 3072) {
    return 0.000323 * X + 0.004998;
    ;
  } else if (X >= 3073 && X <= 3280) {
    return 0.0005549 * X - 0.1344;
    ;
  } else if (X >= 3281 && X <= 3421) {
    return 0.001257 * X - 2.431;
    ;
  } else if (X >= 3422 && X <= 3610) {
    return 0.00167 * X - 3.84;
    ;
  } else if (X >= 3611 && X <= 3617) {
    return -0.001524 * X + 7.667;
    ;
  } else if (X >= 3618 && X <= 3885) {
    return 0.0007037 * X - 0.3883;
    ;
  } else if (X >= 3886 && X <= 4108) {
    return 0.001094 * X - 1.907;
    ;
  } else if (X >= 4109 && X <= 4737) {
    return 0.0006299 * X - 0.01416;
    ;
  } else {
    return 0.0006155 * X + 0.05444;
    ;
  }
}

float gemm_reduction_func_P40(int X) {
  if (X >= 1 && X <= 256) {
    return 0.0008424 * X + 0.01302;
    ;
  } else if (X >= 257 && X <= 515) {
    return 0.0008396 * X + 0.01381;
    ;
  } else if (X >= 516 && X <= 1030) {
    return 0.0008455 * X + 0.01075;
    ;
  } else if (X >= 1031 && X <= 2048) {
    return 0.0008428 * X + 0.01318;
    ;
  } else if (X >= 2049 && X <= 4106) {
    return 0.000842 * X + 0.01503;
    ;
  } else {
    return 0.0008412 * X + 0.01808;
    ;
  }
}

float gemm_reduction_func_P41(int X) {
  if (X >= 1 && X <= 256) {
    return 0.0008424 * X + 0.01302;
    ;
  } else if (X >= 257 && X <= 515) {
    return 0.0008396 * X + 0.01381;
    ;
  } else if (X >= 516 && X <= 1030) {
    return 0.0008455 * X + 0.01075;
    ;
  } else if (X >= 1031 && X <= 2048) {
    return 0.0008428 * X + 0.01318;
    ;
  } else if (X >= 2049 && X <= 4106) {
    return 0.000842 * X + 0.01503;
    ;
  } else {
    return 0.0008412 * X + 0.01808;
    ;
  }
}

float gemm_reduction_func_P42(int X) {
  if (X >= 1 && X <= 2614) {
    return 0.0003488 * X + 0.006232;
    ;
  } else if (X >= 2615 && X <= 4009) {
    return 0.0003571 * X - 0.01873;
    ;
  } else {
    return 0.0003675 * X - 0.06003;
    ;
  }
}

float gemm_reduction_func_P43(int X) {
  if (X >= 1 && X <= 247) {
    return 0.0004293 * X + 0.006444;
    ;
  } else if (X >= 248 && X <= 511) {
    return 0.0004264 * X + 0.007546;
    ;
  } else if (X >= 512 && X <= 2065) {
    return 0.0004294 * X + 0.006726;
    ;
  } else if (X >= 2066 && X <= 3597) {
    return 0.0004299 * X + 0.006206;
    ;
  } else if (X >= 3598 && X <= 3967) {
    return 0.0004864 * X - 0.1859;
    ;
  } else if (X >= 3968 && X <= 4407) {
    return 0.0004414 * X - 0.007716;
    ;
  } else {
    return 0.0004367 * X + 0.01419;
    ;
  }

}

float (*GEMM_K_FUNC[num_gemm_primary])(int) = {
    gemm_k_func_P0,  gemm_k_func_P1,  gemm_k_func_P2,  gemm_k_func_P3,
    gemm_k_func_P4,  gemm_k_func_P5,  gemm_k_func_P6,  gemm_k_func_P7,
    gemm_k_func_P8,  gemm_k_func_P9,  gemm_k_func_P10, gemm_k_func_P11,
    gemm_k_func_P12, gemm_k_func_P13, gemm_k_func_P14, gemm_k_func_P15,
    gemm_k_func_P16, gemm_k_func_P17, gemm_k_func_P18, gemm_k_func_P19,
    gemm_k_func_P20, gemm_k_func_P21, gemm_k_func_P22, gemm_k_func_P23,
    gemm_k_func_P24, gemm_k_func_P25, gemm_k_func_P26, gemm_k_func_P27,
    gemm_k_func_P28, gemm_k_func_P29, gemm_k_func_P30, gemm_k_func_P31,
    gemm_k_func_P32, gemm_k_func_P33, gemm_k_func_P34, gemm_k_func_P35,
    gemm_k_func_P36, gemm_k_func_P37, gemm_k_func_P38, gemm_k_func_P39,
    gemm_k_func_P40};
