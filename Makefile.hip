# Build RCKangaroo for AMD CDNA3 (MI300X, gfx942) and CDNA4 (MI355X, gfx950).
#
#   make -f Makefile.hip                      # gfx942 (default)
#   make -f Makefile.hip OFFLOAD_ARCH=gfx950  # CDNA4
#   make -f Makefile.hip OFFLOAD_ARCH="gfx942;gfx950"
#   make -f Makefile.hip test                 # arithmetic differential test
#   make -f Makefile.hip isa                  # static codegen check
#   make -f Makefile.hip bench                # ISA instruction-rate benchmark
#
# The CUDA build (CMakeLists.txt) is untouched and still works.

ROCM_PATH ?= $(shell if [ -d /opt/rocm/core-10 ]; then echo /opt/rocm/core-10; else echo /opt/rocm; fi)
HIPCC     := $(ROCM_PATH)/bin/hipcc
CXX       ?= g++

OFFLOAD_ARCH ?= gfx942
BUILD_DIR    ?= build-hip
TARGET       := $(BUILD_DIR)/rckangaroo-cdna

ARCH_FLAGS := $(foreach a,$(subst ;, ,$(OFFLOAD_ARCH)),--offload-arch=$(a))

INCLUDES := -I. -Iinclude -Iinclude/cdna -I$(ROCM_PATH)/include

# -fno-strict-aliasing: the field arithmetic and the inverse both alias u64 and
#   u32 views of the same buffers, as the CUDA original does.
# -Wno-unknown-pragmas: defs.h carries an MSVC #pragma warning directive.
COMMON_FLAGS := -O3 -std=c++17 -fno-strict-aliasing -Wno-unknown-pragmas \
                -D__HIP_PLATFORM_AMD__ $(INCLUDES)

# Deliberately NOT used here:
#   -fgpu-rdc          blocks cross-function device inlining and buys nothing
#                      for a single .hip translation unit
#   -ffast-math        meaningless for integer-only device code
#   -munsafe-fp-atomics likewise
#
# -Rpass-analysis=kernel-resource-usage is how VGPR/AGPR/LDS/occupancy get
# reported per kernel; keep it, it is the main feedback channel for this port.
HIPCCFLAGS := $(COMMON_FLAGS) $(ARCH_FLAGS) -Rpass-analysis=kernel-resource-usage

# Ec.cpp and utils.cpp stay on g++: utils.cpp uses GCC's {att|intel} dialect
# alternative inline asm with a "Jc" multi-alternative constraint that clang's
# integrated assembler rejects, and Ec.cpp uses ~40 x86 _addcarry_u64/_umul128
# intrinsics.
CXXFLAGS := -O3 -std=c++17 -march=native -fno-strict-aliasing -Wno-unknown-pragmas \
            -D__HIP_PLATFORM_AMD__ $(INCLUDES)

LDFLAGS := -L$(ROCM_PATH)/lib -lamdhip64 -pthread

# CallCubin.cpp is excluded: it is the CUDA driver API path for the prebuilt
# NVIDIA SASS kernels and has no AMD counterpart.
CXX_SRCS := RCKangaroo.cpp GpuKang.cpp Ec.cpp utils.cpp
HIP_SRCS := src/hip/RCGpuCore.hip

CXX_OBJS := $(addprefix $(BUILD_DIR)/,$(CXX_SRCS:.cpp=.o))
HIP_OBJS := $(addprefix $(BUILD_DIR)/,$(HIP_SRCS:.hip=.o))
OBJS     := $(CXX_OBJS) $(HIP_OBJS)
DEPS     := $(OBJS:.o=.d)

.PHONY: all clean test isa bench info
all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(dir $@)
	$(HIPCC) $(ARCH_FLAGS) -o $@ $(OBJS) $(LDFLAGS)
	@echo "built $@ for [$(OFFLOAD_ARCH)]"

# -MMD -MP so editing a header actually triggers a rebuild; the upstream
# Makefile in the sibling AMDKangaroo tree does not, which leaves stale objects.
$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -MMD -MP -c -o $@ $<

$(BUILD_DIR)/%.o: %.hip
	@mkdir -p $(dir $@)
	$(HIPCC) $(HIPCCFLAGS) -MMD -MP -c -o $@ $<

-include $(DEPS)

test:
	@tests/run_diff.sh 10000000 0 $(firstword $(subst ;, ,$(OFFLOAD_ARCH)))

isa:
	@tests/isa_quality.sh $(firstword $(subst ;, ,$(OFFLOAD_ARCH)))

bench: $(BUILD_DIR)/isa_rate
	@$(BUILD_DIR)/isa_rate 0

$(BUILD_DIR)/isa_rate: bench/isa_rate.hip
	@mkdir -p $(dir $@)
	$(HIPCC) -O3 $(ARCH_FLAGS) -o $@ $<

info:
	@echo "ROCM_PATH    = $(ROCM_PATH)"
	@echo "OFFLOAD_ARCH = $(OFFLOAD_ARCH)"
	@echo "TARGET       = $(TARGET)"
	@$(HIPCC) --version | head -2

clean:
	rm -rf $(BUILD_DIR)
