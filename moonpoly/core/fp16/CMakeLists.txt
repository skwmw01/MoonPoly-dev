# FP 16 GEMM Libraries
# Row-Col-Col
add_library(fp16_rcc_cost SHARED fp16_rcc_cost.cu)
add_library(fp16_rcc SHARED fp16_rcc.cu ../../generated/fp16/rcc/candidates.cu)
add_library(fp16_dispatch SHARED fp16_dispatch.cu)

set_property(TARGET fp16_rcc_cost
             PROPERTY POSITION_INDEPENDENT_CODE ON 
             PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set_property(TARGET fp16_rcc
             PROPERTY POSITION_INDEPENDENT_CODE ON 
             PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(fp16_rcc fp16_rcc_cost)

set_property(TARGET fp16_dispatch
             PROPERTY POSITION_INDEPENDENT_CODE ON 
             PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(fp16_dispatch fp16_rcc)

# Col-Col-Col
# add_library(fp16_ccc SHARED)
