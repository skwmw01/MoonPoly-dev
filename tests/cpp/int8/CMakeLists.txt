add_executable(test_int8_row_col_col test_int8_row_col_col.cu)
target_link_libraries(test_int8_row_col_col PUBLIC moonpoly_gemm -lcublas)
add_test(NAME moonpoly.int8_rcc COMMAND $<TARGET_FILE:test_int8_row_col_col>)
set_tests_properties(moonpoly.int8_rcc PROPERTIES LABELS "correctness;gpu;int8" TIMEOUT 60)
