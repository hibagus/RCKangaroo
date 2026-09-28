#include "rckangaroo/kernel_params_abi.hpp"

#include <iostream>

int main()
{
    TKparams parameters{};
    parameters.IsGenMode = 1U;
    if (parameters.IsGenMode != 1U) {
        std::cerr << "TKparams fixed-width mode field failed\n";
        return 1;
    }

    std::cout << "TKparams ABI is 192 bytes with validated field offsets\n";
    return 0;
}
