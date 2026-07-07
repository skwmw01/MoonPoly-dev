add_subdirectory(fp16)
add_subdirectory(fp32)
add_subdirectory(int8)

add_library(moonpoly_gemm SHARED moonpoly_gemm.cu)
set_property(TARGET moonpoly_gemm PROPERTY POSITION_INDEPENDENT_CODE ON 
             PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(moonpoly_gemm fp16_dispatch)
target_link_libraries(moonpoly_gemm i8_dispatch)
target_link_libraries(moonpoly_gemm fp32_dispatch)
