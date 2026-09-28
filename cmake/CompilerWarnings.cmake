function(rck_set_project_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4)
    else()
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:C,CXX>:-Wall>
            $<$<COMPILE_LANGUAGE:C,CXX>:-Wextra>
            $<$<COMPILE_LANGUAGE:C,CXX>:-Wpedantic>
            $<$<COMPILE_LANGUAGE:C,CXX>:-Wno-unknown-pragmas>
        )
    endif()
endfunction()
