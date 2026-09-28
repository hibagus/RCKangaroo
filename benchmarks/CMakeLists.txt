add_executable(rckangaroo_hip_benchmark EXCLUDE_FROM_ALL
    hip_kernel_a_benchmark.hip
)
target_link_libraries(rckangaroo_hip_benchmark PRIVATE
    RCKangaroo::hip_backend
)
set_target_properties(rckangaroo_hip_benchmark PROPERTIES
    HIP_ARCHITECTURES "${RCK_HIP_ARCHITECTURES}"
)

add_executable(rckangaroo_field_benchmark EXCLUDE_FROM_ALL
	field_ops_benchmark.hip
)
target_include_directories(rckangaroo_field_benchmark PRIVATE
	"${PROJECT_SOURCE_DIR}/src/hip"
)
target_link_libraries(rckangaroo_field_benchmark PRIVATE
	RCKangaroo::core
)
set_target_properties(rckangaroo_field_benchmark PROPERTIES
	HIP_ARCHITECTURES "${RCK_HIP_ARCHITECTURES}"
)

add_executable(rckangaroo_wave64_inverse_benchmark EXCLUDE_FROM_ALL
	wave64_inverse_benchmark.hip
)
target_include_directories(rckangaroo_wave64_inverse_benchmark PRIVATE
	"${PROJECT_SOURCE_DIR}/src/hip"
)
target_link_libraries(rckangaroo_wave64_inverse_benchmark PRIVATE
	RCKangaroo::core
)
set_target_properties(rckangaroo_wave64_inverse_benchmark PROPERTIES
	HIP_ARCHITECTURES "${RCK_HIP_ARCHITECTURES}"
)
