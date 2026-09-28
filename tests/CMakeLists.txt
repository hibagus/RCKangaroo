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
