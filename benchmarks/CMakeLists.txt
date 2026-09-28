add_executable(rckangaroo_hip_benchmark EXCLUDE_FROM_ALL
    hip_kernel_a_benchmark.hip
)
target_link_libraries(rckangaroo_hip_benchmark PRIVATE
    RCKangaroo::hip_backend
)
set_target_properties(rckangaroo_hip_benchmark PROPERTIES
    HIP_ARCHITECTURES "${RCK_HIP_ARCHITECTURES}"
)
