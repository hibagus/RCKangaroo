find_package(Boost REQUIRED)

add_executable(rckangaroo_unit_tests
    unit/cpu_ec_tests.cpp
)

target_link_libraries(rckangaroo_unit_tests PRIVATE
    RCKangaroo::core
    Boost::headers
)

if(MSVC)
    target_compile_options(rckangaroo_unit_tests PRIVATE /W4)
else()
    target_compile_options(rckangaroo_unit_tests PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

add_test(NAME rckangaroo.unit.cpu_ec COMMAND rckangaroo_unit_tests)

add_executable(rckangaroo_runtime_options_tests
    unit/runtime_options_tests.cpp
)

target_link_libraries(rckangaroo_runtime_options_tests PRIVATE
    RCKangaroo::core
)

if(MSVC)
    target_compile_options(rckangaroo_runtime_options_tests PRIVATE /W4)
else()
    target_compile_options(rckangaroo_runtime_options_tests PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

add_test(NAME rckangaroo.unit.runtime_options COMMAND rckangaroo_runtime_options_tests)

add_executable(rckangaroo_kernel_params_abi_tests
    unit/kernel_params_abi_tests.cpp
)

target_link_libraries(rckangaroo_kernel_params_abi_tests PRIVATE
    RCKangaroo::core
)

if(MSVC)
    target_compile_options(rckangaroo_kernel_params_abi_tests PRIVATE /W4)
else()
    target_compile_options(rckangaroo_kernel_params_abi_tests PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

add_test(NAME rckangaroo.unit.kernel_params_abi COMMAND rckangaroo_kernel_params_abi_tests)

if(RCK_BUILD_HIP)
    add_executable(rckangaroo_gpu_field_tests gpu/field_tests.hip)
    target_include_directories(rckangaroo_gpu_field_tests PRIVATE
        "${PROJECT_SOURCE_DIR}/src/hip"
    )
    target_link_libraries(rckangaroo_gpu_field_tests PRIVATE
        RCKangaroo::core
        Boost::headers
    )
    set_target_properties(rckangaroo_gpu_field_tests PROPERTIES
        HIP_ARCHITECTURES "${RCK_HIP_ARCHITECTURES}"
    )

    add_test(NAME rckangaroo.gpu.field COMMAND rckangaroo_gpu_field_tests)
    set_tests_properties(rckangaroo.gpu.field PROPERTIES LABELS "gpu;hip")

	add_executable(rckangaroo_gpu_field_variant_tests gpu/field_variant_tests.hip)
	target_include_directories(rckangaroo_gpu_field_variant_tests PRIVATE
		"${PROJECT_SOURCE_DIR}/src/hip"
	)
	target_link_libraries(rckangaroo_gpu_field_variant_tests PRIVATE
		RCKangaroo::core
		Boost::headers
	)
	set_target_properties(rckangaroo_gpu_field_variant_tests PROPERTIES
		HIP_ARCHITECTURES "${RCK_HIP_ARCHITECTURES}"
	)

	add_test(NAME rckangaroo.gpu.field_variants COMMAND rckangaroo_gpu_field_variant_tests)
	set_tests_properties(rckangaroo.gpu.field_variants PROPERTIES LABELS "gpu;hip")

	add_executable(rckangaroo_gpu_kernel_tests gpu/kernel_tests.hip)
	target_link_libraries(rckangaroo_gpu_kernel_tests PRIVATE
		RCKangaroo::hip_backend
	)
	set_target_properties(rckangaroo_gpu_kernel_tests PROPERTIES
		HIP_ARCHITECTURES "${RCK_HIP_ARCHITECTURES}"
	)

	add_test(NAME rckangaroo.gpu.kernels COMMAND rckangaroo_gpu_kernel_tests)
	set_tests_properties(rckangaroo.gpu.kernels PROPERTIES LABELS "gpu;hip")

	if(RCK_ENABLE_SLOW_GPU_TESTS)
		file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/end_to_end")

		add_test(NAME rckangaroo.gpu.end_to_end.private_2
			COMMAND $<TARGET_FILE:rckangaroo_hip>
				-gpu 0 -range 32 -dp 14 -start 0
				-pubkey 02C6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5
				--seed 1 --duration 120
		)
		set_tests_properties(rckangaroo.gpu.end_to_end.private_2 PROPERTIES
			LABELS "gpu;hip;slow;end-to-end"
			PASS_REGULAR_EXPRESSION "PRIVATE KEY: 0+2"
			RUN_SERIAL TRUE
			TIMEOUT 180
			WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/end_to_end"
		)

		add_test(NAME rckangaroo.gpu.end_to_end.offset_private_3
			COMMAND $<TARGET_FILE:rckangaroo_hip>
				-gpu 0 -range 32 -dp 14 -start 1
				-pubkey 02F9308A019258C31049344F85F89D5229B531C845836F99B08601F113BCE036F9
				--seed 1 --duration 120
		)
		set_tests_properties(rckangaroo.gpu.end_to_end.offset_private_3 PROPERTIES
			LABELS "gpu;hip;slow;end-to-end"
			PASS_REGULAR_EXPRESSION "PRIVATE KEY: 0+3"
			RUN_SERIAL TRUE
			TIMEOUT 180
			WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/end_to_end"
		)
	endif()
endif()
