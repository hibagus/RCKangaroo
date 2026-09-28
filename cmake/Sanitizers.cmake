option(RCK_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

function(rck_enable_sanitizers target)
    if(NOT RCK_ENABLE_SANITIZERS)
        return()
    endif()
    if(MSVC)
        message(WARNING "RCK_ENABLE_SANITIZERS is not configured for MSVC")
        return()
    endif()

    target_compile_options(${target} PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(${target} PUBLIC -fsanitize=address,undefined)
endfunction()
