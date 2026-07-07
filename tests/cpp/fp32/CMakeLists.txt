add_executable(test_fp32_row_col_col test_fp32_row_col_col.cu)
target_link_libraries(test_fp32_row_col_col PUBLIC moonpoly_gemm -lcublas)
add_test(NAME moonpoly.fp32_rcc COMMAND $<TARGET_FILE:test_fp32_row_col_col>)
set_tests_properties(moonpoly.fp32_rcc PROPERTIES LABELS "correctness;gpu;fp32" TIMEOUT 60)
