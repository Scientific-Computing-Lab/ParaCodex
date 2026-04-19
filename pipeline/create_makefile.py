import sys
import subprocess
import os

def get_gpu_compute_capability():
    try:
        # Run nvidia-smi to get compute capability
        # Output format example: "8.9"
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            universal_newlines=True
        ).strip()
        # Remove decimal point (e.g., 8.9 -> 89) and prepend cc
        return f"cc{result.replace('.', '')}"
    except Exception:
        # Fallback if nvidia-smi fails or no GPU found
        return "cc80"

def get_cuda_arch_sm():
    """Return GPU SM arch string for clang++ (e.g., sm_89)."""
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            universal_newlines=True
        ).strip()
        # e.g. 8.9 -> sm_89
        return f"sm_{result.replace('.', '')}"
    except Exception:
        return "sm_80"

SYCL_COMPILER = "/opt/sycl/bin/clang++"
SYCL_CUDA_PATH = "/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/cuda/12.9"
SYCL_GCC_INSTALL_DIR = "/usr/lib/gcc/x86_64-linux-gnu/11"

def generate_makefile_content(api, ref_kernel_name):
    sm = get_gpu_compute_capability()
    
    # Set default CC for the header based on API
    if api == 'sycl':
        default_cc = SYCL_COMPILER
    elif api == 'hip':
        default_cc = 'hipcc'
    else:
        default_cc = 'nvc++'

    oneapi_env = "ONEAPI_DEVICE_SELECTOR=cuda:gpu " if api == 'sycl' else ""

    # Common Header
    content = f"""# Makefile for {ref_kernel_name} ({api} version)

#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = {default_cc}
OPTIMIZE  = yes
DEBUG     = no
DEVICE    = gpu
SM        = {sm}
LAUNCHER  =
GATE_ROOT ?= $(abspath ../..)
REF_DIR   ?= $(GATE_ROOT)/golden_labels/src/{ref_kernel_name}
REF_BIN   ?= $(REF_DIR)/main
REF_MAKE  ?= Makefile.nvc
VERIFY_CPPFLAGS ?= -DGATE_VERIFY

RUN_ARGS ?= <RUN_ARGS>

#===============================================================================
# Program name & source code list
#===============================================================================

program = <PROGRAM_NAME>

"""

    # Source files definition based on API
    if api == 'cuda' or api == 'hip' or api == 'sycl':
        ext = 'cu' if api == 'cuda' else ('hip.cpp' if api == 'hip' else 'dp.cpp')
        content += f"""{api.upper()}_SRCS := $(wildcard *.{ext})
CPP_SRCS  := $(wildcard *.cpp) $(wildcard */*.cpp)
C_SRCS    := $(wildcard *.c) $(wildcard */*.c)
C_SRCS    := $(filter-out utilities/%,$(C_SRCS))
SRCS      := $({api.upper()}_SRCS) $(CPP_SRCS) $(C_SRCS)
obj       := $({api.upper()}_SRCS:.{ext}=.o) $(CPP_SRCS:.cpp=.o) $(C_SRCS:.c=.o)
"""
    elif api == 'ocl':
        content += """SRCS      := $(wildcard *.c) $(wildcard */*.c)
SRCS      := $(filter-out utilities/%,$(SRCS))
obj       := $(SRCS:.c=.o)
KERNELS   := $(wildcard *.cl */*.cl)
"""
    else: # omp, serial
        content += """source = <SOURCE_FILES>

obj = $(source:.cpp=.o)
"""

    content += """
#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
"""
    # Flags based on API
    if api == 'cuda':
        content += "CFLAGS := $(EXTRA_CFLAGS) -O3 -std=c++14 -cuda -I. -I./utilities -I$(GATE_ROOT)/gate_sdk\n"
        content += "LDFLAGS = -lm\n"
    elif api == 'hip':
        content += "CC := hipcc\n"
        content += "CFLAGS := $(EXTRA_CFLAGS) -O3 -std=c++14 -I. -I./utilities -I$(GATE_ROOT)/gate_sdk\n"
        content += "LDFLAGS = -lm\n"
    elif api == 'sycl':
        cuda_arch = get_cuda_arch_sm()
        content += f"CC := {SYCL_COMPILER}\n"
        content += f"CFLAGS := $(EXTRA_CFLAGS) --cuda-path={SYCL_CUDA_PATH} --gcc-install-dir={SYCL_GCC_INSTALL_DIR} -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch={cuda_arch} -O3 -std=c++17 -I. -I./utilities -I$(GATE_ROOT)/gate_sdk\n"
        content += f"LDFLAGS = -lm\n"
        content += f"# Required: tell the SYCL runtime to use the NVIDIA CUDA backend\n"
        content += f"export ONEAPI_DEVICE_SELECTOR = cuda:gpu\n"
    elif api == 'ocl':
        content += "CFLAGS := $(EXTRA_CFLAGS) -O3 -std=c11 -I. -I./utilities -I$(GATE_ROOT)/gate_sdk\n"
        content += "LDFLAGS = -lOpenCL\n"
    else: # omp, serial
        content += "CFLAGS := $(EXTRA_CFLAGS) -std=c++14 -Wall -I$(GATE_ROOT)/gate_sdk\n"
        content += "LDFLAGS = \n"

    content += """
# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
  LDFLAGS  += -g
endif


# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif
"""
    
    # GPU/OMP specific flags
    if api == 'omp':
        content += """
ifeq ($(DEVICE),gpu)
  CFLAGS +=-Minfo -mp=gpu -gpu=$(SM)
else
  CFLAGS +=-qopenmp
endif
"""

    content += """#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

"""
    # Compilation rules
    if api == 'cuda':
        content += """%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -x c -c $< -o $@

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@
"""
    elif api == 'hip':
        content += """%.o: %.hip.cpp
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -x c -c $< -o $@

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@
"""
    elif api == 'sycl':
        content += """%.o: %.dp.cpp
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -x c -c $< -o $@

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@
"""
    elif api == 'ocl':
        content += """%.o: %.c
	$(CC) $(CFLAGS) -x c -c $< -o $@
"""
    else: # omp, serial
        content += """%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@
"""

    content += f"""
clean:
	rm -rf $(program) $(obj)

run: $(program)
	{oneapi_env}$(LAUNCHER) ./$(program) $(RUN_ARGS)

.PHONY: ref_build
ref_build:
	$(MAKE) -C $(REF_DIR) -f $(REF_MAKE) clean
	$(MAKE) -C $(REF_DIR) -f $(REF_MAKE) CC="$(CC)" OPTIMIZE="$(OPTIMIZE)" DEBUG="$(DEBUG)" DEVICE="$(DEVICE)" SM="$(SM)" CFLAGS="$(CFLAGS) -I$(GATE_ROOT)/gate_sdk"

.PHONY: verify_build
verify_build:
	$(MAKE) -f Makefile.nvc clean
	$(MAKE) -f Makefile.nvc EXTRA_CFLAGS="$(VERIFY_CPPFLAGS) $(EXTRA_CFLAGS)"

.PHONY: ref_verify_build
ref_verify_build:
	$(MAKE) -C $(REF_DIR) -f $(REF_MAKE) clean
	$(MAKE) -C $(REF_DIR) -f $(REF_MAKE) CC="$(CC)" OPTIMIZE="$(OPTIMIZE)" DEBUG="$(DEBUG)" DEVICE="$(DEVICE)" SM="$(SM)" CFLAGS="$(VERIFY_CPPFLAGS) $(CFLAGS) -I$(GATE_ROOT)/gate_sdk"

.PHONY: check-correctness
check-correctness: verify_build ref_verify_build
"""
    
    # Gate harness command
    prefix = "OMP_TARGET_OFFLOAD=MANDATORY " if api == 'omp' else ""
    content += f"\t{prefix}python3 $(GATE_ROOT)/gate_sdk/scripts/gate_harness.py $(REF_BIN) ./$(program) $(RUN_ARGS)\n"

    # Add check-kernel for OCL
    if api == 'ocl':
        content += """
.PHONY: check-kernel
check-kernel:
	@if [ -z "$(KERNELS)" ]; then echo "Error: No OpenCL .cl kernel found in $(CURDIR)"; exit 1; fi
"""
    return content

def generate_makefile(api, ref_kernel_name):
    print(generate_makefile_content(api, ref_kernel_name))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        # Defaults for testing if arguments missing
        print("Usage: python3 create_makefile.py <api> <ref_kernel_name>", file=sys.stderr)
        sys.exit(1)
    
    api_type = sys.argv[1] # omp, ocl, cuda, serial
    ref_name = sys.argv[2] # e.g., ace-serial
    
    generate_makefile(api_type, ref_name)
