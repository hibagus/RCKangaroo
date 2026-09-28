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

	add_executable(rckangaroo_gpu_kernel_tests gpu/kernel_tests.hip)
	target_link_libraries(rckangaroo_gpu_kernel_tests PRIVATE
		RCKangaroo::hip_backend
	)
	set_target_properties(rckangaroo_gpu_kernel_tests PROPERTIES
		HIP_ARCHITECTURES "${RCK_HIP_ARCHITECTURES}"
	)

	add_test(NAME rckangaroo.gpu.kernels COMMAND rckangaroo_gpu_kernel_tests)
	set_tests_properties(rckangaroo.gpu.kernels PROPERTIES LABELS "gpu;hip")
endif()
