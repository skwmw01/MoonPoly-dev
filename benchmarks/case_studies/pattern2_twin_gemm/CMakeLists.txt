set(PATTERN2_TARGETS
  moonpoly_case_pattern2_micro_kernel_a
  moonpoly_case_pattern2_micro_kernel_b
  moonpoly_case_pattern2_twin_gemm
)

add_executable(moonpoly_case_pattern2_micro_kernel_a micro_kernel_A.cu)
add_executable(moonpoly_case_pattern2_micro_kernel_b micro_kernel_B.cu)
add_executable(moonpoly_case_pattern2_twin_gemm pattern2_twin_gemm.cu)

foreach(target ${PATTERN2_TARGETS})
  target_link_libraries(${target} PRIVATE cublas)
endforeach()
