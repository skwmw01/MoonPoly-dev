# INT8 GEMM Libraries
# Row-Col-Col
add_library(i8_rcc_cost SHARED i8_rcc_cost.cu)
add_library(i8_rcc SHARED i8_rcc.cu)
add_library(i8_dispatch SHARED i8_dispatch.cu)

set_property(TARGET i8_rcc_cost
             PROPERTY POSITION_INDEPENDENT_CODE ON 
             PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_property(TARGET i8_rcc
             PROPERTY POSITION_INDEPENDENT_CODE ON
             PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(i8_rcc i8_rcc_cost)

set_property(TARGET i8_dispatch 
             PROPERTY POSITION_INDEPENDENT_CODE ON
             PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(i8_dispatch i8_rcc)