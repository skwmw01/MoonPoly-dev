# FLOAT32 GEMM Libraries
# Row-Col-Col
add_library(fp32_rcc SHARED fp32_rcc.cu)
add_library(fp32_rcc_cost SHARED fp32_rcc_cost.cu)
add_library(fp32_dispatch SHARED fp32_dispatch.cu)

set_property(TARGET fp32_rcc_cost
             PROPERTY POSITION_INDEPENDENT_CODE ON
             PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set_property(TARGET fp32_rcc 
             PROPERTY POSITION_INDEPENDENT_CODE ON
             PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(fp32_rcc fp32_rcc_cost)
             
set_property(TARGET fp32_dispatch 
             PROPERTY POSITION_INDEPENDENT_CODE ON
             PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(fp32_dispatch fp32_rcc)