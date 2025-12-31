#!/usr/bin/env python3
"""
Script to systematically translate serial codes to OpenMP using Codex CLI
and measure compilation success rate.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path to allow imports when running as script
if __name__ == "__main__":
    script_dir = Path(__file__).parent
    parent_dir = script_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

from typing import Optional, List

from optimize_codex import (
    copy_translated_file,
    optimize_translated_code_four_stage,
    resolve_kernel_file_name,
    test_optimized_compilation,
    _extract_relevant_nsys,
    _run_compare_and_optimize_steps,
)
from path_config import (
    build_codex_cli_cmd,
    default_hecbench_src,
    get_codex_workdir,
    get_make_cmd,
    get_make_cmd_str,
    set_codex_workdir,
)

CODEX_CLI_CMD = build_codex_cli_cmd()


def _refresh_codex_cli_cmd():
    global CODEX_CLI_CMD
    CODEX_CLI_CMD = build_codex_cli_cmd()


def _build_omp_translation_prompt(kernel_name, file_names, target_api, source_api, source_dir: Path):
    """Build prompt for OpenMP translation, handling multiple files.
    
    Args:
        kernel_name: Name of the kernel
        file_names: List of file names to create (can be a single string for backward compatibility)
        target_api: Target API (e.g., 'omp')
        source_api: Source API (e.g., 'serial')
        source_dir: Path to source directory
    """
    kernel_dir = get_codex_workdir() / 'data' / 'src' / f"{kernel_name}-{target_api}"
    source_dir = get_codex_workdir() / 'golden_labels' / 'src' / f"{kernel_name}-{source_api}"
    api_label = 'CUDA' if source_api == 'cuda' else 'serial' if source_api == 'serial' else source_api
    build_cmd_str = get_make_cmd_str(target_api, 'build')
    clean_cmd_str = get_make_cmd_str(target_api, 'clean')
    
    # Handle both single file (string) and multiple files (list) for backward compatibility
    if isinstance(file_names, str):
        file_names = [file_names]
    
    # Normalize file names
    normalized_files = [resolve_kernel_file_name(fn, target_api) for fn in file_names]
    file_listing = '\n'.join(f'- {name}' for name in normalized_files)
    file_list_str = ', '.join(normalized_files)
    
    return f'''# Loop Classification for GPU Offload - Analysis Phase

## Task
Analyze loops in `{source_dir}/` and produce `{kernel_dir}/analysis.md`. Copy source files unmodified to `{kernel_dir}/`.

**Files:** {file_listing}  
**Reference:** Check Makefile in `{kernel_dir}/` (do not modify)

## Process

### 0. COPY THE SOURCE FILES - {file_listing} TO THE KERNEL DIRECTORY {kernel_dir}

### 1. Find All Loops
```bash
# Find main compute loop
grep -n "for.*iter\|for.*it\|while\|main(" *.c *.cpp 2>/dev/null | head -50

# List all loop-containing functions
grep -n "for\s*(" *.c *.cpp 2>/dev/null | head -100
```

Prioritize functions called in main compute loop:
- Every iteration → CRITICAL/IMPORTANT
- Once at setup → SECONDARY/AVOID

### 2. Classify Priority
For each loop: `iterations × ops/iter = total work`

- **CRITICAL:** >50% runtime OR called every iteration with O(N) work
- **IMPORTANT:** 5-50% runtime OR called every iteration with small work
- **SECONDARY:** Called once at setup
- **AVOID:** Setup/IO/RNG OR <10K iterations

### 3. Determine Loop Type (Decision Tree)

```
Q0: Nested inside another loop? → Note parent
Q1: Writes A[idx[i]] with varying idx? → Type D (Histogram)
Q2: Reads A[i-1] or accumulates across iterations? → Type E (Recurrence - CPU only)
Q3: Stage loop where L+1 depends on L?
    - Scratch swap (tmp1↔tmp2)? → C1 (FFT/Butterfly)
    - Level traversal with stencil calls? → C2 (Multigrid)
Q4: Inner bound varies with outer index? → Type B (Sparse)
Q5: Accumulates to scalar? → Type F (Reduction)
Q6: Accesses neighbors? → Type G (Stencil)
Default → Type A (Dense)
```

**Special Case - Outer A + Inner E:**
When outer loop iterates over INDEPENDENT samples and inner has RNG:
- Mark outer as Type A (CRITICAL) - parallelizable with per-thread RNG
- Mark inner RNG as Type E - sequential WITHIN each thread
- Note: "RNG replicable: YES - each sample can compute its own seed"

### 4. Type Reference

| Type | Pattern | Parallelizable |
|------|---------|----------------|
| A | Dense, constant bounds | YES |
| B | Sparse (CSR), inner bound varies | Outer only |
| C1 | FFT/Butterfly, scratch swap | Outer only |
| C2 | Multigrid, hierarchical calls | Outer only |
| D | Histogram, indirect write | YES + atomic |
| E | Recurrence, loop-carried dep | NO |
| F | Reduction to scalar | YES + reduction |
| G | Stencil, neighbor access | YES |

### 5. Data Analysis
For each array:
- Definition: flat vs pointer-to-pointer
- Allocation: static vs dynamic
- Struct members accessed?
- Global variables used?

### 6. Flag Issues
- Variable bounds
- Reduction needed
- Atomic required
- Stage dependency
- RNG in loop
- <10K iterations

## Output: analysis.md

### Loop Nesting Structure
```
- outer_loop (line:X) Type A
  └── inner_loop_1 (line:Y) Type E
- standalone_loop (line:Z) Type A
```

### Loop Details
For each CRITICAL/IMPORTANT/SECONDARY loop:
```
## Loop: [function] at [file:line]
- **Iterations:** [count]
- **Type:** [A-H] - [reason]
- **Parent loop:** [none / line:X]
- **Contains:** [inner loops or none]
- **Dependencies:** [none / reduction:vars / stage / recurrence]
- **Nested bounds:** [constant / variable]
- **Private vars:** [list]
- **Arrays:** [name(R/W/RW)]
- **Issues:** [flags]
```

### Summary Table
| Function | Type | Priority | Parent | Iterations | Dependencies | Issues |
|----------|------|----------|--------|------------|--------------|--------|

### Data Details
- **Dominant compute loop:** [main timed loop]
- **Arrays swapped between functions?:** YES/NO
- **Scratch arrays?:** YES/NO
- **Mid-computation sync?:** YES/NO
- **RNG in timed loop?:** YES/NO (only if inside timer)

## Constraints
- Find all loops in functions called from main compute loop
- Document only - no pragmas or code modifications
- When uncertain between B and C, choose C
- Copy all source files unmodified to `{kernel_dir}/`
'''


def _build_cuda_translation_prompt(kernel_name, file_names, target_api, source_api, source_dir: Path):
    """Build prompt for parallel-to-parallel translation, handling multiple files.
    
    Args:
        kernel_name: Name of the kernel
        file_names: List of file names to create (can be a single string for backward compatibility)
        target_api: Target API (e.g., 'cuda', 'omp')
        source_api: Source API (e.g., 'omp', 'cuda')
        source_dir: Path to source directory
    """
    kernel_dir = get_codex_workdir() / 'data' / 'src' / f"{kernel_name}-{target_api}"
    api_label = 'OpenMP' if source_api == 'omp' else 'serial' if source_api == 'serial' else source_api
    build_cmd_str = get_make_cmd_str(target_api, 'build')
    clean_cmd_str = get_make_cmd_str(target_api, 'clean')
    run_cmd_str = get_make_cmd_str(target_api, 'run')
    
    # Handle both single file (string) and multiple files (list) for backward compatibility
    if isinstance(file_names, str):
        file_names = [file_names]
    
    # Normalize file names
    normalized_files = [resolve_kernel_file_name(fn, target_api) for fn in file_names]
    file_listing = '\n'.join(f'- {name}' for name in normalized_files)
    file_list_str = ', '.join(normalized_files)
    
    # Use golden labels directory for source files if available, otherwise fall back to passed source_dir
    golden_source_dir = get_codex_workdir() / 'golden_labels' / 'src' / f"{kernel_name}-{source_api}"
    if golden_source_dir.exists():
        source_dir = golden_source_dir
    
    return f'''# Loop Classification for OMP Migration - Analysis Phase

## Task
Analyze CUDA kernels in `{source_dir}/` and produce `{kernel_dir}/analysis.md`. Copy source files to `{kernel_dir}/` with suffix conversion (.cu → .c or .cpp).

**Files:** {file_listing}  
**Reference:** Check Makefile in `{kernel_dir}/` (do not modify)

## Process

### 0. COPY SOURCE FILES WITH SUFFIX CONVERSION
- Copy `{file_listing}` from `{source_dir}/` to `{kernel_dir}/`
- Convert suffixes: `.cu` → `.c` (for C code) or `.cpp` (for C++ code). You can inspecct the makefile in {kernel_dir}/ to see the expected file names.
- Get baseline output. Run {clean_cmd_str} and `{run_cmd_str} > baseline_output.txt 2>&1` in {source_dir}/. Copy the baseline output to {kernel_dir}/baseline_output.txt.
- Preserve all file content exactly - no code modifications
- Document mapping: `original.cu → converted.c` in analysis.md
- Convert header includes in {file_listing}. Make sure the code can be compiled with the converted files.

## Create Environment
**You need** to create an enviroment to run the code in {kernel_dir}.
That means:
- Create any header fles, util files, etc. that are needed to run the code.
- Create a Makefile called Makefile.nvc in {kernel_dir}/ that can be used to run the code. the compiler that needs to be used is nvc++.

### 1. Find All CUDA Kernels and Loops
```bash
# Find CUDA kernels
grep -n "__global__\|__device__" *.cu 2>/dev/null

# Find kernel launch sites
grep -n "<<<.*>>>" *.cu 2>/dev/null

# Find device loops (inside kernels)
grep -n "for\s*(" *.cu 2>/dev/null | head -100

# Find host loops calling kernels
grep -n "for.*iter\|for.*it\|while" *.cu 2>/dev/null | head -50
```

Prioritize by execution pattern:
- Kernel called every iteration → CRITICAL/IMPORTANT
- Kernel called once at setup → SECONDARY/AVOID
- Device loops inside kernels → analyze work per thread

### 2. Classify Priority
For each kernel/loop: `grid_size × block_size × device_iterations × ops = total work`

- **CRITICAL:** >50% runtime OR called every iteration with O(N) work
- **IMPORTANT:** 5-50% runtime OR called every iteration with small work
- **SECONDARY:** Called once at setup
- **AVOID:** Setup/IO/memory allocation OR <10K total threads

### 3. Determine Kernel/Loop Type (Decision Tree)

```
Q0: Is this a __global__ kernel or host loop? → Note context
Q1: Writes A[idx[i]] with varying idx (atomicAdd)? → Type D (Histogram)
Q2: Uses __syncthreads() or shared memory dependencies? → Type E (Block-level recurrence)
Q3: Multi-stage kernel pattern?
    - Separate kernels for stages with global sync? → C1 (FFT/Butterfly)
    - Hierarchical grid calls? → C2 (Multigrid)
Q4: Block/thread indexing varies with outer dimension? → Type B (Sparse)
Q5: Uses atomicAdd to scalar (reduction pattern)? → Type F (Reduction)
Q6: Accesses neighboring threads' data? → Type G (Stencil)
Default → Type A (Dense)
```

**CUDA-Specific Patterns:**
- **Kernel with thread loop:** Outer grid parallelism + inner device loop
  - Mark grid dimension as Type A (CRITICAL) - maps to OMP parallel
  - Mark device loop by standard classification
  - Note: "Grid-stride loop" if thread loops beyond block size

- **Atomic operations:** 
  - atomicAdd → requires OMP atomic/reduction
  - Race conditions → document carefully

- **Shared memory:**
  - __shared__ arrays → maps to OMP private/firstprivate
  - __syncthreads() → limited OMP equivalent, may need restructuring

### 4. Type Reference

| Type | CUDA Pattern | OMP Equivalent | Notes |
|------|--------------|----------------|-------|
| A | Dense kernel, regular grid | YES - parallel for | Direct map |
| B | Sparse (CSR), varying bounds | Outer only | Inner sequential |
| C1 | Multi-kernel, global sync | Outer only | Barrier between stages |
| C2 | Hierarchical grid | Outer only | Nested parallelism tricky |
| D | Histogram, atomicAdd | YES + atomic | Performance loss expected |
| E | __syncthreads, shared deps | NO | Requires restructuring |
| F | Reduction, atomicAdd scalar | YES + reduction | OMP reduction clause |
| G | Stencil, halo exchange | YES | Ghost zone handling |

### 5. CUDA-Specific Data Analysis
For each array:
- Memory type: __global__, __shared__, __constant__, host
- Transfer pattern: cudaMemcpy direction and frequency
- Allocation: cudaMalloc vs managed memory
- Device pointers vs host pointers
- Struct members on device?

CUDA constructs to document:
- Thread indexing: threadIdx, blockIdx, blockDim, gridDim
- Synchronization: __syncthreads(), kernel boundaries
- Memory access patterns: coalesced vs strided
- Atomic operations and their locations

### 6. Flag OMP Migration Issues
- __syncthreads() usage (no direct OMP equivalent)
- Shared memory dependencies (complex privatization)
- Atomics (performance penalty in OMP)
- Reduction patterns (may need manual implementation)
- <10K total threads (overhead concern)
- Dynamic parallelism (not in OMP)
- Warp-level primitives (no OMP equivalent)

## Output: analysis.md

### File Conversion Mapping
```
original.cu → converted.c
kernel_utils.cu → kernel_utils.cpp
```

### Kernel/Loop Nesting Structure
```
- host_loop (line:X) calls kernel1 
  └── kernel1<<<grid,block>>> (line:Y) Type A
      └── device_loop (line:Z) Type A
- kernel2<<<grid,block>>> (line:W) Type D
```

### Kernel/Loop Details
For each CRITICAL/IMPORTANT/SECONDARY kernel or loop:
```
## Kernel/Loop: [name] at [file:line]
- **Context:** [__global__ kernel / host loop / __device__ function]
- **Launch config:** [grid_size × block_size] or [iterations]
- **Total threads/iterations:** [count]
- **Type:** [A-G] - [reason]
- **Parent loop:** [none / line:X]
- **Contains:** [device loops or none]
- **Dependencies:** [none / atomicAdd / __syncthreads / reduction]
- **Shared memory:** [YES/NO - size and usage]
- **Thread indexing:** [pattern used]
- **Private vars:** [list]
- **Arrays:** [name(R/W/RW) - memory type]
- **OMP Migration Issues:** [flags]
```

### Summary Table
| Kernel/Function | Type | Priority | Context | Total Work | Dependencies | OMP Issues |
|-----------------|------|----------|---------|------------|--------------|------------|

### CUDA-Specific Details
- **Dominant compute kernel:** [main timed kernel]
- **Memory transfers in timed loop?:** YES/NO
- **Shared memory usage:** [total bytes, patterns]
- **Synchronization points:** [__syncthreads locations]
- **Atomic operations:** [locations and variables]
- **Reduction patterns:** [manual vs atomicAdd]

### OMP Migration Strategy Notes
- **Direct kernel → parallel for:** [list]
- **Requires restructuring:** [list with reasons]
- **Performance concerns:** [atomics, false sharing, etc.]
- **Data management:** [allocation changes needed]

## Constraints
- Find all kernels and loops called from main compute section
- Document CUDA-specific constructs for migration planning
- Copy all source files with suffix conversion (.cu → .c/.cpp)
- No code modifications - documentation only
- Identify __syncthreads() patterns (critical for OMP feasibility)
'''

def _build_translation_prompt(kernel_name, file_name, target_api, source_api, source_dir: Path):
    """Build translation prompt, handling both single file (str) and multiple files (list).
    
    Args:
        kernel_name: Name of the kernel
        file_name: Single file name (str) or list of file names (List[str])
        target_api: Target API (e.g., 'omp', 'cuda')
        source_api: Source API (e.g., 'serial', 'omp')
        source_dir: Path to source directory
    """
    # Handle both single file (string) and multiple files (list) for backward compatibility
    if isinstance(file_name, str):
        file_names = [file_name]
    else:
        file_names = file_name
    
    # Use serial-to-omp prompt when source is serial, parallel-to-parallel prompt otherwise
    if source_api == 'serial':
        return _build_omp_translation_prompt(kernel_name, file_names, target_api, source_api, source_dir)
    else:
        # Parallel-to-parallel translation (e.g., omp->cuda, cuda->omp, etc.)
        return _build_cuda_translation_prompt(kernel_name, file_names, target_api, source_api, source_dir)

def run_supervisor(kernel_name, target_api, hecbench_src, output_dir, file_name: Optional[str] | List[str] = None):
    """Invoke the supervisor agent for a specific kernel.
    Returns (success: bool, output: str)
    
    Args:
        file_name: Can be a single file name (str), list of file names, or None
    """
    try:
        supervisor_path = Path(__file__).parent / 'supervisor_codex.py'
        cmd = [
            'python3', str(supervisor_path),
            '--hecbench-src', str(hecbench_src),
            '--target-api', target_api,
            '--kernels', kernel_name,
            '--results-dir', output_dir,
            '--codex-workdir', str(get_codex_workdir()),
        ]
        if file_name:
            # Handle both single file (str) and list of files
            if isinstance(file_name, list):
                # For multiple files, pass the first one as hint (supervisor will use the list internally)
                if file_name:
                    cmd.extend(['--file-name', file_name[0]])
            else:
                cmd.extend(['--file-name', file_name])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        success = result.returncode == 0
        # Extract this kernel's status line if present; otherwise return full output
        output = (result.stdout or '') + '\n' + (result.stderr or '')
        return success, output
    except subprocess.TimeoutExpired:
        return False, 'Supervisor timed out'
    except Exception as e:
        return False, f'Supervisor error: {e}'

def parse_jsonl(file_path, source_api):
    """Parse the JSONL file and extract source codes."""
    source_codes = []
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                if data.get('parallel_api') == source_api:
                    source_codes.append({
                        'line_num': line_num,
                        'kernel_name': data.get('kernel_name', 'unknown'),
                        'code': data.get('code', {}),
                        'original_data': data
                    })
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    return source_codes

def run_codex_translation(kernel_name, file_name, target_api, source_api, source_dir: Path):
    """Use Codex CLI to translate target code to target API.

    Returns dict on success: { 'combined': stdout+stderr, 'summary': stdout }
    Returns None on failure.
    """
    prompt = _build_translation_prompt(kernel_name, file_name, target_api, source_api, source_dir)

    try:
        # Run Codex CLI in interactive mode with bypass sandbox to allow cd and make commands
        result = subprocess.run(
            CODEX_CLI_CMD + [prompt],
            capture_output=True,
            text=True,
            timeout=6000
        )  # 100 minute timeout
        
        if result.returncode == 0:
            combined = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
            return {
                'combined': combined.strip(),
                'summary': (result.stdout or "").strip()
            }
        else:
            print(f"Codex CLI error: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"Codex CLI timeout for kernel {kernel_name}")
        return None
    except Exception as e:
        print(f"Error running Codex CLI: {e}")
        return None

def run_codex_translation_with_retry(kernel_name, file_name, target_api, source_api, source_dir: Path, max_attempts=3, delay_seconds=1):
    """Retry Codex translation up to max_attempts times.

    Returns a dict {'combined': ..., 'summary': ...} on success, or None if all attempts fail.
    """
    for attempt in range(1, max_attempts + 1):
        print(f"Translating with Codex CLI (attempt {attempt}/{max_attempts})...")
        translated = run_codex_translation(kernel_name, file_name, target_api, source_api, source_dir)
        if translated:
            return translated
        print(f"Codex translation attempt {attempt} failed; retrying..." if attempt < max_attempts else "Codex translation attempts exhausted.")
        if attempt < max_attempts:
            time.sleep(delay_seconds)
    return None

def test_compilation(kernel_dir, target_api):
    """Test compilation by running make clean and make."""
    kernel_dir = Path(kernel_dir)
    clean_cmd = get_make_cmd(target_api, 'clean')
    build_cmd = get_make_cmd(target_api, 'build')

    try:
        subprocess.run(clean_cmd, capture_output=True, text=True, timeout=30, cwd=kernel_dir)
        make_result = subprocess.run(build_cmd, capture_output=True, text=True, timeout=120, cwd=kernel_dir)

        success = make_result.returncode == 0
        if not success:
            print("Make compilation failed.")
        return success, "" if success else make_result.stderr

    except subprocess.TimeoutExpired:
        print("Make compilation timeout")
        return False, "Make compilation timeout"
    except Exception as e:
        print(f"Error during make compilation: {e}")
        return False, str(e)

def save_translated_code(code_content, kernel_name, output_dir, target_api, compilation_result=None):
    """Save initial translation artifacts with requested naming.

    kernel_name is expected to be "<kernel>_<file_name>".
    We save the Codex transcript as initial_transcript.txt via save_phase_result elsewhere,
    and here we save <stem>_initial.py for convenience browsing.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Create subdirectory for this kernel/file
    kernel_dir = output_dir / kernel_name
    kernel_dir.mkdir(exist_ok=True)

    # Derive stem and extension from file name portion
    try:
        file_part = kernel_name.split('_', 1)[1]
    except Exception:
        file_part = kernel_name
    file_path = Path(file_part)
    stem = file_path.stem
    ext = file_path.suffix  # Get original extension (.c, .cpp, etc.)

    # Save requested artifact name: <stem>_initial<ext> (preserves original extension)
    output_file = kernel_dir / f"{stem}_initial{ext}"
    with open(output_file, 'w') as f:
        f.write(code_content)

    # Save compilation result in the kernel subdirectory if provided
    if compilation_result is not None:
        comp_result_file = kernel_dir / 'compilation_result.txt'
        with open(comp_result_file, 'w') as f:
            f.write(f"Kernel: {kernel_name}\n")
            f.write(f"Compilation Success: {compilation_result['success']}\n")
            if not compilation_result['success']:
                f.write(f"Error: {compilation_result['error_msg']}\n")
    
    return output_file

def save_phase_result(kernel_name, file_name, output_dir, phase, compilation_result, transcript, supervisor_output, transcript_summary=None, target_api=None):
    """Save phase results (transcript and compilation output) for each phase."""
    output_dir = Path(output_dir)
    # Use kernel name and target API for directory structure if available, otherwise fall back to old format
    if target_api:
        kernel_output_dir = output_dir / f"{kernel_name}-{target_api}"
    else:
        kernel_output_dir = output_dir / f"{kernel_name}_{file_name}"
    kernel_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save compilation result
    if compilation_result is not None:
        comp_file = kernel_output_dir / f"{phase}_compilation.txt"
        with open(comp_file, 'w') as f:
            f.write(f"Phase: {phase}\n")
            f.write(f"Kernel: {kernel_name}\n")
            f.write(f"File: {file_name}\n")
            f.write(f"Compilation Success: {compilation_result['success']}\n")
            if not compilation_result['success']:
                f.write(f"Error: {compilation_result['error_msg']}\n")
            else:
                f.write("Compilation successful\n")
    
    # Save transcript (standardized name)
    if transcript is not None:
        transcript_file = kernel_output_dir / f"{phase}_transcript.txt"
        with open(transcript_file, 'w') as f:
            f.write(transcript)
        # Save short summary alongside full transcript when provided
        if transcript_summary is not None:
            transcript_summary_file = kernel_output_dir / f"{phase}_transcript_summary.txt"
            with open(transcript_summary_file, 'w') as f:
                f.write(transcript_summary)
    
    # Save supervisor output (standardized name: *_output.txt)
    if supervisor_output is not None:
        supervisor_file = kernel_output_dir / f"{phase}_output.txt"
        with open(supervisor_file, 'w') as f:
            f.write(f"Phase: {phase}\n")
            f.write(f"Kernel: {kernel_name}\n")
            f.write(f"File: {file_name}\n")
            f.write("="*50 + "\n")
            f.write(supervisor_output)

def create_timestamped_directory(target_api):
    """Create a timestamped directory for saving all translated files."""
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # dir_name = f"translated_serial_{target_api}_{timestamp}"
    # output_dir = Path(dir_name)
    # output_dir.mkdir(exist_ok=True)
    # return output_dir
    return None

def restart_translation_with_supervisor(kernel_name, file_name, code_files, target_api, source_api, hecbench_src, output_dir, max_attempts=3):
    """
    Restart translation process when supervisor fails.
    Returns (success: bool, final_result: dict, attempts: int)
    
    Args:
        file_name: Can be a single file name (str) or list of file names (List[str]) for backward compatibility
        code_files: Dictionary of file_name -> code_content, or single code_content string for backward compatibility
    """
    # Handle both single file (string) and multiple files (list) for backward compatibility
    if isinstance(file_name, str):
        all_file_names = [file_name]
    else:
        all_file_names = file_name
    
    file_list_str = ', '.join(all_file_names)
    print(f"Starting restart process for {kernel_name} (files: {file_list_str}) (max {max_attempts} attempts)")
    
    source_dir = Path(hecbench_src) / f"{kernel_name}-{source_api}"
    
    for attempt in range(1, max_attempts + 1):
        print(f"\n--- Restart attempt {attempt}/{max_attempts} for {kernel_name} (files: {file_list_str}) ---")
        
        # Re-translate with Codex CLI - handle all files at once
        print("Re-translating with Codex CLI...")
        translated_outputs = run_codex_translation_with_retry(
            kernel_name, all_file_names, target_api, source_api, source_dir,
            max_attempts=1, delay_seconds=0  # Already in retry loop
        )
        
        if not translated_outputs:
            print(f"✗ Re-translation failed on attempt {attempt}")
            continue
        
        print("✓ Re-translation successful")
        
        # Test compilation
        kernel_dir = Path(hecbench_src) / f"{kernel_name}-{target_api}"
        if kernel_dir.exists():
            success, error_msg = test_compilation(kernel_dir, target_api)
            if not success:
                print(f"✗ Compilation failed on attempt {attempt}: {error_msg}")
                continue
            
            print("✓ Compilation successful")
            
            # Run supervisor - pass all file names
            print(f"Running supervisor correctness check (attempt {attempt})...")
            sup_ok, sup_out = run_supervisor(kernel_name, target_api, hecbench_src, output_dir, file_name=all_file_names)
            
            if sup_ok:
                print(f"✓ Supervisor PASS on attempt {attempt}")
                
                # Copy all files for this kernel after successful supervisor
                supervised_file_copy = copy_translated_file(
                    kernel_name, all_file_names, target_api, hecbench_src, output_dir, 'supervised'
                )
                if supervised_file_copy:
                    print(f"✓ Restart - Supervised version saved: {supervised_file_copy}")
                
                return True, {
                    'translated_code': translated_outputs.get('combined'),
                    'translated_summary': translated_outputs.get('summary'),
                    'compilation_success': True,
                    'supervisor_success': True,
                    'attempt': attempt
                }, attempt
            else:
                print(f"✗ Supervisor failed on attempt {attempt}. Output:\n{sup_out}")
        else:
            print(f"✗ Kernel directory not found: {kernel_dir}")
    
    print(f"✗ All {max_attempts} restart attempts failed for {kernel_name} (files: {file_list_str})")
    return False, {
        'translated_code': None,
        'compilation_success': False,
        'supervisor_success': False,
        'attempt': max_attempts
    }, max_attempts

def restart_optimization_with_supervisor(kernel_name, file_name, target_api, output_dir, hecbench_src, max_attempts=3, supervise_max_attempts=3, source_api='serial'):
    """
    Restart optimization process when supervisor fails after optimization.
    Always restores the initial translated code before re-optimizing.
    Returns (success: bool, final_result: dict, attempts: int)
    
    Args:
        file_name: Can be a single file name (str) or list of file names (List[str]) for backward compatibility
        source_api: Source API (e.g., 'serial', 'omp', 'cuda') - used to determine optimization prompts
    """
    # Handle both single file (string) and multiple files (list) for backward compatibility
    if isinstance(file_name, str):
        all_file_names = [file_name]
        primary_file_name = file_name
    else:
        all_file_names = file_name
        primary_file_name = all_file_names[0] if all_file_names else 'unknown'
    
    file_list_str = ', '.join(all_file_names)
    print(f"Starting optimization restart process for {kernel_name} (files: {file_list_str}) (max {max_attempts} attempts)")
    
    # Find and restore the initial translated code that passed supervisor check
    # Use kernel name and target API for directory structure
    kernel_output_dir = Path(output_dir) / f"{kernel_name}-{target_api}"
    
    # Look for the initial file copy that was saved after successful supervisor check
    # Check in the initial_correct subdirectory
    initial_snapshot_dir = kernel_output_dir / "initial_correct"
    if initial_snapshot_dir.exists():
        print(f"Found initial snapshot directory: {initial_snapshot_dir}")
    else:
        print("⚠ No initial snapshot directory found, will use current kernel directory state")
    
    for attempt in range(1, max_attempts + 1):
        print(f"\n--- Optimization restart attempt {attempt}/{max_attempts} for {kernel_name} (files: {file_list_str}) ---")
        
        # Restore initial translated code if available
        if initial_snapshot_dir.exists():
            kernel_dir = Path(hecbench_src) / f"{kernel_name}-{target_api}"
            for fn in all_file_names:
                resolved = resolve_kernel_file_name(fn, target_api)
                src_path = initial_snapshot_dir / resolved
                dest_path = kernel_dir / resolved
                if src_path.exists():
                    try:
                        shutil.copy2(src_path, dest_path)
                        print(f"✓ Restored initial translated code: {dest_path}")
                    except Exception as e:
                        print(f"⚠ Failed to restore {dest_path}: {e}")
                else:
                    print(f"⚠ Initial snapshot file not found: {src_path}")
        
        # Re-run optimization - pass all file names
        print("Re-running optimization...")
        optimization_result = optimize_translated_code_four_stage(
            kernel_name=kernel_name,
            file_name=all_file_names,  # Pass all files
            target_api=target_api,
            output_dir=output_dir,
            hecbench_src_dir=hecbench_src,
            max_attempts=3,  # Use default retry attempts for restart
            supervise_max_attempts=supervise_max_attempts,
            source_api=source_api  # Pass source_api for parallel-to-parallel optimization
        )
        
        if not optimization_result['success']:
            print(f"✗ Optimization failed on attempt {attempt}")
            continue
        
        print("✓ Optimization successful")
        
        # Run supervisor after optimization - pass all file names
        print(f"Running supervisor correctness check after optimization (attempt {attempt})...")
        sup_ok, sup_out = run_supervisor(kernel_name, target_api, hecbench_src, output_dir, file_name=all_file_names)
        
        if sup_ok:
            print(f"✓ Supervisor PASS after optimization on attempt {attempt}")
            
            # Copy all files for this kernel after successful optimization supervisor
            optimized_supervised_file_copy = copy_translated_file(
                kernel_name, all_file_names, target_api, hecbench_src, output_dir, 'optimized_supervised'
            )
            if optimized_supervised_file_copy:
                print(f"✓ Optimization restart - Optimized supervised version saved: {optimized_supervised_file_copy}")
            
            return True, {
                'optimization_result': optimization_result,
                'supervisor_success': True,
                'attempt': attempt
            }, attempt
        else:
            print(f"✗ Supervisor failed after optimization on attempt {attempt}. Output:\n{sup_out}")
    
    print(f"✗ All {max_attempts} optimization restart attempts failed for {kernel_name} (files: {file_list_str})")
    return False, {
        'optimization_result': None,
        'supervisor_success': False,
        'attempt': max_attempts
    }, max_attempts

def main():
    parser = argparse.ArgumentParser(description='Translate serial codes to OpenMP using Codex CLI')
    parser.add_argument('--input', default='/root/codex_baseline/pipeline/combined_serial_filenames.jsonl', help='Input JSONL file')
    parser.add_argument('--output-dir', default='/root/codex_baseline/pipeline/Tom_aug_kernels_nas', help='Output directory for translated codes (auto-generated if not specified)')
    parser.add_argument('--hecbench-src', default=None, help='HeCBench source directory')
    parser.add_argument('--save-failed', action='store_true', help='Save failed translations')
    parser.add_argument('--max-codes', type=int, help='Maximum number of codes to process')
    parser.add_argument('--no-timestamp-dir', action='store_true', help='Disable timestamped directory creation')
    parser.add_argument('--skip-compilation', action='store_true', help='Skip compilation testing')
    parser.add_argument('--compile-only', action='store_true', help='Only compile existing translated files without running translation')
    parser.add_argument('--target-api', default='omp', help='Target API to translate to (omp, cuda, hip)')
    parser.add_argument('--source-api', default='serial', help='Source API to translate from (omp, cuda, hip)')
    parser.add_argument('--optimize', action='store_true', help='Run 4-stage optimization pipeline with optional cycling')
    # Four-stage optimization pipeline options (always used when --optimize is set)
    parser.add_argument('--opt-cyclic', action='store_true', help='Enable cyclic repetition of 4-stage sequence until target met or max cycles')
    parser.add_argument('--opt-max-cycles', type=int, default=2, help='Maximum number of cycles when cyclic is enabled')
    parser.add_argument('--opt-target-speedup', type=float, help='Stop when speedup vs baseline >= this value')
    parser.add_argument('--opt-target-runtime-ms', type=float, help='Stop when runtime in ms <= this target')
    parser.add_argument('--opt-supervisor-steps', default='', help='Comma-separated steps after which to run supervisor, e.g., 2')
    parser.add_argument('--opt-single-step', type=int, choices=[1, 2], help='Run only this optimization step (1-2)')
    parser.add_argument('--opt-step1-prompt-file', help='Custom prompt file for step 1')
    parser.add_argument('--opt-step2-prompt-file', help='Custom prompt file for step 2')
    # Note: step3 and step4 arguments kept for backward compatibility but not used (only 2 steps now)
    parser.add_argument('--opt-step3-prompt-file', help='Custom prompt file for step 3 (deprecated, not used)')
    parser.add_argument('--opt-step4-prompt-file', help='Custom prompt file for step 4 (deprecated, not used)')
    parser.add_argument('--supervise', action='store_true', help='Run supervisor correctness check and auto-fix after successful compilation')
    parser.add_argument('--supervise-after-optimization', action='store_true', help='Run supervisor again after optimization')
    parser.add_argument('--supervise-max-attempts', type=int, default=3, help='Maximum number of restart attempts when supervisor fails')
    parser.add_argument('--opt-max-attempts', type=int, default=2, help='Maximum number of retry attempts per optimization step')
    parser.add_argument('--translate-max-attempts', type=int, default=1, help='Maximum number of retry attempts for Codex translation')
    parser.add_argument('--codex-workdir', default=None, help='Codex CLI working directory (defaults to CODEX_WORKDIR env or cuda_omp_workdir)')

    args = parser.parse_args()
    if args.codex_workdir:
        # Explicitly set the workdir to override any existing CODEX_WORKDIR env var
        resolved_workdir = set_codex_workdir(args.codex_workdir)
        print(f"Set CODEX_WORKDIR to: {resolved_workdir}")
        print(f"Environment CODEX_WORKDIR: {os.environ.get('CODEX_WORKDIR')}")
    _refresh_codex_cli_cmd()
    final_workdir = get_codex_workdir()
    print(f"Codex CLI command will use workdir: {final_workdir}")
    print(f"Codex CLI command: {' '.join(CODEX_CLI_CMD)}")

    # ------------------------------------------------------------------
    # Initial cleanup of kernel directories for this run.
    # This removes previously generated code files for the given
    # input JSONL, target API, and Codex workdir so we start fresh.
    # ------------------------------------------------------------------
    try:
        clean_script = Path("/root/codex_baseline/utils/clean_kernel_dirs.py")
        if clean_script.exists():
            base_path = final_workdir / "data" / "src"
            clean_cmd = [
                sys.executable,
                str(clean_script),
                "--base_path", str(base_path),
                "--api", args.target_api,
            ]
            print(f"Running kernel cleanup: {' '.join(clean_cmd)}")
            subprocess.run(clean_cmd, check=False)
        else:
            print(f"Warning: clean_kernel_dirs.py not found at {clean_script}")
    except Exception as e:
        print(f"Warning: kernel cleanup step failed: {e}")

    if not args.hecbench_src:
        args.hecbench_src = str(default_hecbench_src())
    
    # Create timestamped directory if not disabled
    if not args.no_timestamp_dir:
        timestamped_dir = create_timestamped_directory(args.target_api)
        print(f"Created timestamped directory: {timestamped_dir}")
    else:
        timestamped_dir = None
    
    # Use timestamped directory as default output if not specified
    if not args.output_dir:
        if timestamped_dir:
            args.output_dir = str(timestamped_dir)
        else:
            args.output_dir = 'translated_codes'

    # Ensure the chosen output directory exists
    output_dir_path = Path(args.output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    print("=== Serial to OpenMP Translation using Codex CLI ===")
    print(f"Input file: {args.input}")
    print(f"Output directory: {args.output_dir}")
    print(f"HeCBench source directory: {args.hecbench_src}")
    print(f"Codex workdir: {get_codex_workdir()}")
    print(f"Target API: {args.target_api}")
    print(f"Optimize (2-stage) enabled: {args.optimize}")
    
    # Handle compile-only mode
    if args.compile_only:
        print("\n=== COMPILE-ONLY MODE ===")
        print("Skipping translation phase, only testing compilation of kernels from input file...")
        
        # Parse the JSONL file to get kernel names
        source_codes = parse_jsonl(args.input, args.source_api)
        print(f"Found {len(source_codes)} kernels to compile")
        
        successful_compilations = 0
        failed_compilations = 0
        compilation_results = []
        
        for code_data in source_codes:
            kernel_name = code_data['kernel_name']
            kernel_dir = Path(args.hecbench_src) / f"{kernel_name}-{args.target_api}"
            
            if kernel_dir.exists():
                print(f"Testing compilation for {kernel_name}...")
                success, error_msg = test_compilation(kernel_dir, args.target_api)
                if success:
                    print(f"✓ {kernel_name} compiled successfully")
                    successful_compilations += 1
                else:
                    print(f"✗ {kernel_name} compilation failed: {error_msg}")
                    failed_compilations += 1
                
                compilation_results.append({
                    'kernel_name': kernel_name,
                    'compilation_success': success,
                    'error_msg': error_msg
                })
            else:
                print(f"⚠ {kernel_name}-omp directory not found, skipping")
        
        print("\n" + "="*50)
        print("COMPILATION SUMMARY")
        print("="*50)
        print(f"Kernels found: {len(source_codes)}")
        print(f"Successful compilations: {successful_compilations}")
        print(f"Failed compilations: {failed_compilations}")
        
        if successful_compilations + failed_compilations > 0:
            compilation_rate = (successful_compilations / (successful_compilations + failed_compilations)) * 100
            print(f"Compilation success rate: {compilation_rate:.1f}%")
        
        return
    
    # Parse the JSONL file
    print("\n1. Parsing JSONL file...")
    source_codes = parse_jsonl(args.input, args.source_api)
    print(f"Found {len(source_codes)} source codes")
    
    if args.max_codes:
        source_codes = source_codes[:args.max_codes]
        print(f"Processing first {len(source_codes)} codes")
    
    # Statistics
    total_codes = len(source_codes)
    successful_translations = 0
    failed_translations = 0
    
    results = []
    
    # Process each source code
    print("\n2. Translation Phase")
    print("="*30)
    
    compilation_results = []  # Track compilation results for summary
    optimization_results = []  # Track optimization results for summary
    
    for i, code_data in enumerate(source_codes, 1):
        kernel_name = code_data['kernel_name']
        print(f"\n--- Processing {i}/{total_codes}: {kernel_name} ---")
        
        # Extract the code files
        code_files = code_data['code']
        if not code_files:
            print(f"No code files found for {kernel_name}")
            continue
        
        # Collect all file names for this kernel (for passing to optimization/supervisor)
        all_file_names = list(code_files.keys())
        primary_file_name = all_file_names[0] if all_file_names else 'unknown'
        
        print(f"Processing kernel {kernel_name} with {len(all_file_names)} file(s): {', '.join(all_file_names)}")
        for file_name in all_file_names:
            code_content = code_files[file_name]
            print(f"  - {file_name}: {len(code_content)} characters")
        
        # Translate with Codex CLI (with retry) - ONCE for all files in this kernel
        source_dir = Path(args.hecbench_src) / f"{kernel_name}-{args.source_api}"
        translated_outputs = run_codex_translation_with_retry(
            kernel_name, all_file_names, args.target_api, args.source_api, source_dir,
            max_attempts=args.translate_max_attempts, delay_seconds=1
        )
        
        if translated_outputs:
            successful_translations += 1
            
            # Test compilation for this kernel.
            # For serial→parallel we actually run a make-based compile test.
            # For parallel↔parallel (e.g. omp↔cuda/hip) we SKIP the initial
            # compilation check here and treat it as successful, so the
            # pipeline can continue into supervisor/optimization which will
            # manage their own builds.
            compilation_result = None
            optimization_result = None
            
            if not args.skip_compilation:
                kernel_dir = Path(args.hecbench_src) / f"{kernel_name}-{args.target_api}"
                if kernel_dir.exists():
                    if args.source_api == 'serial':
                        # Serial → parallel: run the real compilation test
                        success, error_msg = test_compilation(kernel_dir, args.target_api)
                    else:
                        # Parallel ↔ parallel: skip initial compilation test
                        # and allow later phases (supervisor/optimization) to
                        # drive builds and handle failures.
                        print(f"Skipping initial compilation test for parallel-to-parallel translation of {kernel_name}")
                        success, error_msg = True, ""
                    compilation_result = {
                        'success': success,
                        'error_msg': error_msg
                    }
                    compilation_results.append({
                        'kernel_name': kernel_name,
                        'compilation_success': success,
                        'error_msg': error_msg,
                        'restart_attempts': compilation_result.get('restart_attempts', 0) if compilation_result else 0,
                        'restart_success': compilation_result.get('restart_success', False) if compilation_result else False
                    })
                    
                    # PHASE 1: Copy initial translated files after successful compilation
                    if success:
                        # Copy all files for this kernel
                        initial_file_copy = copy_translated_file(
                            kernel_name, all_file_names, args.target_api, args.hecbench_src, args.output_dir, 'initial'
                        )
                        if initial_file_copy:
                            print(f"✓ Phase 1 - Initial translation saved: {initial_file_copy}")
                        
                        # Save initial compilation result + transcripts (full + summary)
                        save_phase_result(
                            kernel_name, primary_file_name, args.output_dir, 'initial',
                            compilation_result,
                            translated_outputs.get('combined'),
                            None,
                            transcript_summary=translated_outputs.get('summary'),
                            target_api=args.target_api
                        )

                    # PHASE 2: Supervisor correctness pass (pre-optimization)
                    if success and args.supervise:
                        print(f"Running Phase 2 - Supervisor correctness check for {kernel_name}...")
                        # Retry supervisor according to --supervise-max-attempts
                        # Pass all file names to supervisor
                        sup_ok = False
                        sup_out = ''
                        for sup_attempt in range(1, args.supervise_max_attempts + 1):
                            print(f"Supervisor attempt {sup_attempt}/{args.supervise_max_attempts}...")
                            sup_ok, sup_out = run_supervisor(
                                kernel_name, args.target_api, args.hecbench_src, args.output_dir, file_name=all_file_names
                            )
                            if sup_ok:
                                break
                        if sup_ok:
                            print(f"✓ Phase 2 - Supervisor PASS for {kernel_name}")
                            
                            # Copy all files for this kernel (after supervisor modifications)
                            supervised_file_copy = copy_translated_file(
                                kernel_name, all_file_names, args.target_api, args.hecbench_src, args.output_dir, 'initial_supervised'
                            )
                            if supervised_file_copy:
                                print(f"✓ Phase 2 - Supervised version saved: {supervised_file_copy}")
                            
                            # Run compare_and_optimize_steps after successful supervisor
                            _run_compare_and_optimize_steps(kernel_name, primary_file_name, args.output_dir, args.target_api)
                            
                            # Save supervisor result with requested naming
                            save_phase_result(kernel_name, primary_file_name, args.output_dir, 'initial_supervised', 
                                            {'success': True, 'error_msg': ''}, None, sup_out, target_api=args.target_api)

                            # Also save the check-correctness output under the requested name by
                            # copying the supervisor-generated result into our standardized file.
                            try:
                                kernel_output_dir = Path(args.output_dir) / f"{kernel_name}-{args.target_api}"
                                sup_result_src = kernel_output_dir / "supervisor_result.txt"
                                if sup_result_src.exists():
                                    (kernel_output_dir / "initial_supervised_compilation.txt").write_text(sup_result_src.read_text())
                            except Exception:
                                pass

                            # Copy supervisor transcript under requested naming
                            try:
                                kernel_output_dir = Path(args.output_dir) / f"{kernel_name}-{args.target_api}"
                                sup_trans_src = kernel_output_dir / "supervisor_transcript.txt"
                                if sup_trans_src.exists():
                                    (kernel_output_dir / "initial_supervised_transcript.txt").write_text(sup_trans_src.read_text())
                            except Exception:
                                pass

                            # Run NCU after supervisor and save outputs for the next step to consume
                            kernel_dir = Path(args.hecbench_src) / f"{kernel_name}-{args.target_api}"
                            ok_ncu, ncu_out = test_optimized_compilation(kernel_dir, args.target_api, kernel_name=kernel_name)
                            try:
                                kernel_output_dir = Path(args.output_dir) / f"{kernel_name}-{args.target_api}"
                                (kernel_output_dir / "initial_supervised_ncu_output.txt").write_text(ncu_out or "")
                                rel = _extract_relevant_nsys(ncu_out or "")
                                (kernel_output_dir / "initial_supervised_ncu_relevant.txt").write_text(rel)
                            except Exception:
                                pass
                        else:
                            print(f"✗ Phase 2 - Supervisor failed for {kernel_name} after {args.supervise_max_attempts} attempts. Output:\n{sup_out}")
                            print(f"Starting restart process with {args.supervise_max_attempts} attempts...")
                            
                            # Restart translation process - pass all file names
                            # Note: restart_translation_with_supervisor needs to be updated to handle multiple files
                            # For now, we'll pass the primary file name but the function should handle all files
                            restart_success, restart_result, restart_attempts = restart_translation_with_supervisor(
                                kernel_name, all_file_names, code_files, args.target_api, args.source_api, 
                                args.hecbench_src, args.output_dir, args.supervise_max_attempts
                            )
                            
                            if restart_success:
                                print(f"✓ Restart successful after {restart_attempts} attempts")
                                # Update the compilation result to reflect restart success
                                compilation_result = {
                                    'success': True,
                                    'error_msg': '',
                                    'restart_attempts': restart_attempts,
                                    'restart_success': True
                                }
                                # Update the translated code with the successful restart result
                                translated_code = restart_result['translated_code']
                                
                                # Copy all files for this kernel (after successful restart)
                                supervised_file_copy = copy_translated_file(
                                    kernel_name, all_file_names, args.target_api, args.hecbench_src, args.output_dir, 'initial_supervised'
                                )
                                if supervised_file_copy:
                                    print(f"✓ Phase 2 - Supervised version saved after restart: {supervised_file_copy}")
                                
                                # Run compare_and_optimize_steps after successful restart
                                _run_compare_and_optimize_steps(kernel_name, primary_file_name, args.output_dir, args.target_api)
                                
                                # Save supervisor result
                                save_phase_result(
                                    kernel_name, primary_file_name, args.output_dir, 'initial_supervised',
                                    {'success': True, 'error_msg': ''},
                                    None,
                                    restart_result.get('translated_code', ''),
                                    target_api=args.target_api
                                )
                            else:
                                print(f"✗ All restart attempts failed for {kernel_name}")
                                compilation_result = {
                                    'success': False,
                                    'error_msg': f'All {args.supervise_max_attempts} restart attempts failed',
                                    'restart_attempts': restart_attempts,
                                    'restart_success': False
                                }
                                
                                # Save failed supervisor result
                                save_phase_result(kernel_name, primary_file_name, args.output_dir, 'supervised', 
                                                {'success': False, 'error_msg': f'All {args.supervise_max_attempts} restart attempts failed'}, None, sup_out, target_api=args.target_api)
                    
                    # PHASE 3: Run four-stage optimization if compilation was successful and optimization is enabled
                    if success and args.optimize:
                        print(f"Running Phase 3 - Four-stage Optimization for {kernel_name} (files: {', '.join(all_file_names)})...")
                        # Steps sequence (single-step override if provided)
                        if args.opt_single_step:
                            steps = [args.opt_single_step]
                        else:
                            steps = [1, 2]
                        try:
                            sup_steps = [int(s.strip()) for s in args.opt_supervisor_steps.split(',') if s.strip()]
                        except Exception:
                            sup_steps = []

                        # Load custom prompts
                        custom_prompts = {}
                        step_prompt_files = [args.opt_step1_prompt_file, args.opt_step2_prompt_file]
                        for idx, pf in enumerate(step_prompt_files, start=1):
                            if pf and Path(pf).exists():
                                with open(pf, 'r') as f:
                                    custom_prompts[idx] = f.read()

                        four_stage_result = optimize_translated_code_four_stage(
                            kernel_name=kernel_name,
                            file_name=all_file_names,  # Pass all files for this kernel
                            target_api=args.target_api,
                            output_dir=args.output_dir,
                            hecbench_src_dir=args.hecbench_src,
                            steps=steps,
                            custom_prompts=custom_prompts,
                            cyclic=args.opt_cyclic,
                            target_speedup=args.opt_target_speedup,
                            target_runtime_ms=args.opt_target_runtime_ms,
                            max_cycles=args.opt_max_cycles,
                            supervisor_steps=sup_steps,
                            max_attempts=args.opt_max_attempts,
                            supervise_max_attempts=args.supervise_max_attempts,
                            source_api=args.source_api,  # Pass source_api for parallel-to-parallel optimization
                        )

                        # Record result (use primary file name for reporting)
                        optimization_result = four_stage_result
                        optimization_results.append({
                            'kernel_name': kernel_name,
                            'file_name': primary_file_name,  # Use primary for reporting
                            'all_file_names': all_file_names,  # Track all files
                            'four_stage': True,
                            'optimization_success': four_stage_result.get('success', False),
                            'optimization_compilation_success': True if four_stage_result.get('success') else False,
                            'error_msg': four_stage_result.get('error_msg', ''),
                            'best_runtime_ms': four_stage_result.get('best_runtime_ms'),
                            'baseline_runtime_ms': four_stage_result.get('baseline_runtime_ms'),
                            'cycles': four_stage_result.get('cycles'),
                        })

                        # PHASE 4: Supervisor correctness pass after optimization
                        if optimization_result and optimization_result.get('success') and args.supervise_after_optimization:
                            print(f"Running Phase 4 - Supervisor correctness check after optimization for {kernel_name}...")
                            sup_ok, sup_out = run_supervisor(
                                kernel_name, args.target_api, args.hecbench_src, args.output_dir, file_name=all_file_names
                            )
                            if sup_ok:
                                print(f"✓ Phase 4 - Supervisor (post-opt) PASS for {kernel_name}")
                                
                                # Copy all files for this kernel (after final supervisor modifications)
                                optimized_supervised_file_copy = copy_translated_file(
                                    kernel_name, all_file_names, args.target_api, args.hecbench_src, args.output_dir, 'optimized_supervised'
                                )
                                if optimized_supervised_file_copy:
                                    print(f"✓ Phase 4 - Optimized supervised version saved: {optimized_supervised_file_copy}")
                                
                                # Run compare_and_optimize_steps after successful supervisor
                                _run_compare_and_optimize_steps(kernel_name, primary_file_name, args.output_dir, args.target_api)
                                
                                # Save optimized supervised phase results
                                save_phase_result(kernel_name, primary_file_name, args.output_dir, 'optimized_supervised', 
                                                {'success': True, 'error_msg': ''}, None, sup_out, target_api=args.target_api)
                            else:
                                print(f"✗ Phase 4 - Supervisor (post-opt) failed for {kernel_name}. Output:\n{sup_out}")
                                print(f"Starting optimization restart process with {args.supervise_max_attempts} attempts...")
                                
                                # Restart optimization process - pass all file names
                                opt_restart_success, opt_restart_result, opt_restart_attempts = restart_optimization_with_supervisor(
                                    kernel_name, all_file_names, args.target_api, args.output_dir,
                                    args.hecbench_src,
                                    max_attempts=args.supervise_max_attempts,
                                    supervise_max_attempts=args.supervise_max_attempts,
                                    source_api=args.source_api  # Pass source_api for parallel-to-parallel optimization
                                )
                                
                                if opt_restart_success:
                                    print(f"✓ Optimization restart successful after {opt_restart_attempts} attempts")
                                    # Update the optimization result with restart information
                                    optimization_result = opt_restart_result['optimization_result']
                                    optimization_result['restart_attempts'] = opt_restart_attempts
                                    optimization_result['restart_success'] = True
                                    
                                    # Copy all files for this kernel (after successful optimization restart)
                                    optimized_supervised_file_copy = copy_translated_file(
                                        kernel_name, all_file_names, args.target_api, args.hecbench_src, args.output_dir, 'optimized_supervised'
                                    )
                                    if optimized_supervised_file_copy:
                                        print(f"✓ Phase 4 - Optimized supervised version saved after restart: {optimized_supervised_file_copy}")
                                    
                                    # Run compare_and_optimize_steps after successful restart
                                    _run_compare_and_optimize_steps(kernel_name, primary_file_name, args.output_dir, args.target_api)
                                    
                                    # Save optimized supervised phase results
                                    save_phase_result(kernel_name, primary_file_name, args.output_dir, 'optimized_supervised', 
                                                    {'success': True, 'error_msg': ''}, None, opt_restart_result.get('optimization_result', {}).get('transcript', ''), target_api=args.target_api)
                                else:
                                    print(f"✗ All optimization restart attempts failed for {kernel_name}")
                                    optimization_result['restart_attempts'] = opt_restart_attempts
                                    optimization_result['restart_success'] = False
                                    optimization_result['success'] = False
                                    optimization_result['error_msg'] = f'All {args.supervise_max_attempts} optimization restart attempts failed'
                                    
                                    # Save failed optimized supervised phase results
                                    save_phase_result(kernel_name, primary_file_name, args.output_dir, 'optimized_supervised', 
                                                    {'success': False, 'error_msg': f'All {args.supervise_max_attempts} optimization restart attempts failed'}, None, sup_out, target_api=args.target_api)
            
            # Save generated source code - use unified directory name (kernel_name-target_api) for all files
            # This ensures all files for a kernel go into the same directory
            unified_kernel_name = f"{kernel_name}-{args.target_api}"
            for file_name in all_file_names:
                try:
                    kernel_dir = Path(args.hecbench_src) / f"{kernel_name}-{args.target_api}"
                    save_name = resolve_kernel_file_name(file_name, args.target_api)
                    candidate_path = kernel_dir / save_name
                    generated_code = candidate_path.read_text() if candidate_path.exists() else translated_outputs.get('combined')
                except Exception:
                    generated_code = translated_outputs.get('combined')

                # Save to unified directory - use unified_kernel_name as base, but include file_name in the saved filename
                # Modify the save to use unified directory but preserve file-specific naming
                output_dir_p = Path(args.output_dir)
                unified_dir = output_dir_p / unified_kernel_name
                unified_dir.mkdir(parents=True, exist_ok=True)
                
                # Preserve original file extension
                file_path = Path(file_name)
                stem = file_path.stem
                ext = file_path.suffix  # Get original extension (.c, .cpp, etc.)
                output_file = unified_dir / f"{stem}_initial{ext}"
                with open(output_file, 'w') as f:
                    f.write(generated_code)
                
                # Save compilation result once (only for first file to avoid duplicates)
                if file_name == primary_file_name and compilation_result is not None:
                    comp_result_file = unified_dir / 'compilation_result.txt'
                    with open(comp_result_file, 'w') as f:
                        f.write(f"Kernel: {kernel_name}\n")
                        f.write(f"Files: {', '.join(all_file_names)}\n")
                        f.write(f"Compilation Success: {compilation_result['success']}\n")
                        if not compilation_result['success']:
                            f.write(f"Error: {compilation_result['error_msg']}\n")
                
                print(f"Saved {file_name} to: {output_file}")
            
            if optimization_result and optimization_result.get('success'):
                print(f"Optimization results saved to: {optimization_result.get('optimized_file_copy', 'N/A')}")
                if optimization_result.get('initial_file_copy'):
                    print(f"Initial file copied to: {optimization_result['initial_file_copy']}")
                if optimization_result.get('optimized_file_copy'):
                    print(f"Optimized file copied to: {optimization_result['optimized_file_copy']}")
                
        else:
            print("✗ Translation failed")
            failed_translations += 1
            
            if args.save_failed:
                failed_dir = Path(args.output_dir) / 'failed'
                failed_dir.mkdir(exist_ok=True)
                # Use unified directory name for failed translations too
                unified_kernel_name = f"{kernel_name}-{args.target_api}"
                unified_failed_dir = failed_dir / unified_kernel_name
                unified_failed_dir.mkdir(parents=True, exist_ok=True)
                for file_name in all_file_names:
                    # Preserve original file extension
                    file_path = Path(file_name)
                    stem = file_path.stem
                    ext = file_path.suffix  # Get original extension (.c, .cpp, etc.)
                    output_file = unified_failed_dir / f"{stem}_failed{ext}"
                    with open(output_file, 'w') as f:
                        f.write("Translation failed")
                    print(f"Saved failed {file_name} to: {output_file}")
        
        # Store results for all files (one entry per file for backward compatibility)
        for file_name in all_file_names:
            results.append({
                'kernel_name': kernel_name,
                'file_name': file_name,
                'translation_success': translated_outputs is not None,
                'error_msg': "Translation failed" if translated_outputs is None else ""
            })
        
        # Small delay to avoid overwhelming the system
        time.sleep(1)
    
    # Calculate compilation statistics from inline results
    successful_compilations = sum(1 for result in compilation_results if result['compilation_success'])
    failed_compilations = len(compilation_results) - successful_compilations
    
    # Calculate restart statistics
    total_restart_attempts = sum(result.get('restart_attempts', 0) for result in compilation_results)
    successful_restarts = sum(1 for result in compilation_results if result.get('restart_success', False))
    
    # Calculate optimization statistics
    successful_optimizations = sum(1 for result in optimization_results if result['optimization_success'])
    failed_optimizations = len(optimization_results) - successful_optimizations
    successful_optimization_compilations = sum(1 for result in optimization_results if result['optimization_compilation_success'])
    
    # Calculate optimization restart statistics
    total_opt_restart_attempts = sum(result.get('restart_attempts', 0) for result in optimization_results)
    successful_opt_restarts = sum(1 for result in optimization_results if result.get('restart_success', False))
    
    # Print summary
    print("\n" + "="*50)
    print("TRANSLATION SUMMARY")
    print("="*50)
    print(f"Total codes processed: {total_codes}")
    print(f"Successful translations: {successful_translations}")
    print(f"Failed translations: {failed_translations}")
    print(f"Successful compilations: {successful_compilations}")
    print(f"Failed compilations: {failed_compilations}")
    
    if args.supervise and total_restart_attempts > 0:
        print(f"Total restart attempts: {total_restart_attempts}")
        print(f"Successful restarts: {successful_restarts}")
    
    if args.optimize:
        print(f"Successful optimizations: {successful_optimizations}")
        print(f"Failed optimizations: {failed_optimizations}")
        print(f"Successful optimization compilations: {successful_optimization_compilations}")
        
        if args.supervise_after_optimization and total_opt_restart_attempts > 0:
            print(f"Total optimization restart attempts: {total_opt_restart_attempts}")
            print(f"Successful optimization restarts: {successful_opt_restarts}")
    
    if total_codes > 0:
        translation_rate = (successful_translations / total_codes) * 100
        compilation_rate = (successful_compilations / total_codes) * 100
        print(f"\nTranslation success rate: {translation_rate:.1f}%")
        print(f"Compilation success rate: {compilation_rate:.1f}%")
        
        if args.optimize and len(optimization_results) > 0:
            optimization_rate = (successful_optimizations / len(optimization_results)) * 100
            optimization_compilation_rate = (successful_optimization_compilations / len(optimization_results)) * 100
            print(f"Optimization success rate: {optimization_rate:.1f}%")
            print(f"Optimization compilation success rate: {optimization_compilation_rate:.1f}%")
    
    # Save detailed results in parent directory
    results_file = Path(args.output_dir) / 'translation_results.json'
    # Merge best_runtime_ms from optimization_results into translation_results entries
    key_best_ms = {}
    for orow in optimization_results:
        if orow.get('optimization_success') and orow.get('best_runtime_ms') is not None:
            key = f"{orow.get('kernel_name')}|{orow.get('file_name')}"
            key_best_ms[key] = orow.get('best_runtime_ms')
    for row in results:
        key = f"{row.get('kernel_name')}|{row.get('file_name')}"
        if key in key_best_ms:
            row['best_runtime_ms'] = key_best_ms[key]

    with open(results_file, 'w') as f:
        summary_data = {
            'total_codes': total_codes,
            'successful_translations': successful_translations,
            'failed_translations': failed_translations,
            'successful_compilations': successful_compilations,
            'failed_compilations': failed_compilations,
            'translation_rate': translation_rate if total_codes > 0 else 0,
            'compilation_rate': compilation_rate if total_codes > 0 else 0,
            'total_restart_attempts': total_restart_attempts,
            'successful_restarts': successful_restarts
        }
        
        if args.optimize:
            summary_data.update({
                'successful_optimizations': successful_optimizations,
                'failed_optimizations': failed_optimizations,
                'successful_optimization_compilations': successful_optimization_compilations,
                'optimization_rate': optimization_rate if len(optimization_results) > 0 else 0,
                'optimization_compilation_rate': optimization_compilation_rate if len(optimization_results) > 0 else 0,
                'total_opt_restart_attempts': total_opt_restart_attempts,
                'successful_opt_restarts': successful_opt_restarts
            })
        
        json.dump({
            'summary': summary_data,
            'translation_results': results,
            'compilation_results': compilation_results,
            'optimization_results': optimization_results
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Save compilation summary in parent directory
    summary_file = Path(args.output_dir) / 'compilation_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("TRANSLATION SUMMARY\n")
        f.write("="*50 + "\n")
        f.write(f"Total codes processed: {total_codes}\n")
        f.write(f"Successful translations: {successful_translations}\n")
        f.write(f"Failed translations: {failed_translations}\n")
        f.write(f"Successful compilations: {successful_compilations}\n")
        f.write(f"Failed compilations: {failed_compilations}\n")
        
        if args.supervise and total_restart_attempts > 0:
            f.write(f"Total restart attempts: {total_restart_attempts}\n")
            f.write(f"Successful restarts: {successful_restarts}\n")
        
        if args.optimize:
            f.write(f"Successful optimizations: {successful_optimizations}\n")
            f.write(f"Failed optimizations: {failed_optimizations}\n")
            f.write(f"Successful optimization compilations: {successful_optimization_compilations}\n")
            
            if args.supervise_after_optimization and total_opt_restart_attempts > 0:
                f.write(f"Total optimization restart attempts: {total_opt_restart_attempts}\n")
                f.write(f"Successful optimization restarts: {successful_opt_restarts}\n")
        
        if total_codes > 0:
            f.write(f"\nTranslation success rate: {translation_rate:.1f}%\n")
            f.write(f"Compilation success rate: {compilation_rate:.1f}%\n")
            
            if args.optimize and len(optimization_results) > 0:
                f.write(f"Optimization success rate: {optimization_rate:.1f}%\n")
                f.write(f"Optimization compilation success rate: {optimization_compilation_rate:.1f}%\n")
    
    print(f"Compilation summary saved to: {summary_file}")
    
    # Print final directory information
    if timestamped_dir:
        print(f"\nAll translated files saved in: {timestamped_dir}")
        print(f"Each kernel has its own subdirectory with .txt files")
        print(f"Results and summary files are in the parent directory")

if __name__ == "__main__":
    main()