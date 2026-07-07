add_executable(test_fp16_row_col_col test_fp16_row_col_col.cu)
target_link_libraries(test_fp16_row_col_col PUBLIC moonpoly_gemm -lcublas)
add_test(NAME moonpoly.fp16_rcc COMMAND $<TARGET_FILE:test_fp16_row_col_col>)
set_tests_properties(moonpoly.fp16_rcc PROPERTIES LABELS "correctness;gpu;fp16" TIMEOUT 60)

add_executable(test_fp16_pattern2_twin_gemm test_fp16_pattern2_twin_gemm.cu)
target_link_libraries(test_fp16_pattern2_twin_gemm PUBLIC moonpoly_gemm -lcublas)
add_test(NAME moonpoly.fp16_pattern2 COMMAND $<TARGET_FILE:test_fp16_pattern2_twin_gemm>)
set_tests_properties(moonpoly.fp16_pattern2 PROPERTIES LABELS "correctness;gpu;fp16;pattern2" TIMEOUT 60)
