# Prompt Files Organization

This directory contains the modular prompt files used by the translation pipeline. Prompts are organized by source API, target API, and step.

## File Naming Convention

Prompts follow the naming pattern: `{source_api}_{target_api}_{step}.md`

- `source_api`: The source programming model (e.g., `serial`, `cuda`, `ocl`)
- `target_api`: The target programming model (e.g., `omp`, `cuda`, `ocl`)
- `step`: The pipeline step (`analysis`, `step1`, `step2`)

## Available Prompts

### Serial to OpenMP Translation

- `serial_omp_analysis.md` - Loop classification and analysis for serial code (3.9KB)
- `serial_omp_step1.md` - GPU offload with OpenMP implementation (12.6KB)
- `serial_omp_step2.md` - Performance tuning and optimization (7.4KB)

### CUDA to OpenMP Translation

- `cuda_omp_analysis.md` - Loop classification for CUDA code migration (7.0KB)
- `cuda_omp_step1.md` - CUDA to OpenMP migration implementation (9.3KB)
- `cuda_omp_step2.md` - Performance tuning for migrated code (6.3KB)

### CUDA to OpenCL Translation

- `cuda_ocl_analysis.md` - Analysis for CUDA to OpenCL migration (9.1KB)
- `cuda_ocl_step1.md` - CUDA to OpenCL implementation (8.7KB)
- `cuda_ocl_step2.md` - Performance tuning for OpenCL (9.9KB)

### OpenCL to CUDA Translation

- `ocl_cuda_analysis.md` - Analysis for OpenCL to CUDA migration (9.4KB)
- `ocl_cuda_step1.md` - OpenCL to CUDA implementation (13.8KB)
- `ocl_cuda_step2.md` - Performance tuning for CUDA (11.1KB)

## Usage

### API Name Normalization

The prompt loader automatically normalizes API names for consistency:
- `ocl` or `opencl` → `ocl` (in prompt filenames)
- `omp` → `omp` (unchanged)
- `cuda` → `cuda` (unchanged)

This means you can use either `ocl` or `opencl` when calling the prompt loader functions, and they will correctly find the `*_ocl_*.md` files.

### In Python Code

Use the `prompt_loader` module to load prompts:

```python
from prompt_loader import load_translation_prompt, load_optimization_prompt

# Load analysis/translation prompt
analysis_prompt = load_translation_prompt('serial', 'omp')

# Load optimization step prompts
step1_prompt = load_optimization_prompt('serial', 'omp', 1)
step2_prompt = load_optimization_prompt('serial', 'omp', 2)
```

### Template Variables

Prompts use Python `format()` style placeholders that must be replaced with actual values:

#### Analysis Prompts
- `{source_dir}` - Source code directory
- `{kernel_dir}` - Target kernel directory
- `{file_listing}` - List of files being translated

#### Optimization Step Prompts
- `{kernel_dir}` - Kernel working directory
- `{file_listing}` - List of files being optimized
- `{profile_log_path}` - Path to profiling output
- `{clean_cmd_str}` - Clean command string
- `{build_cmd_str}` - Build command string
- `{correctness_run_cmd}` - Correctness test command
- `{correctness_fallback_cmd}` - Fallback correctness test
- `{nsys_profile_cmd}` - Nsight Systems profiling command
- `{nsys_profile_fallback_cmd}` - Fallback profiling command
- `{kernel_name}` - Name of the kernel
- `{cwd}` - Current working directory

## Adding New Translation Pairs

To add a new source→target translation pair:

1. Create analysis prompt: `{source}_{target}_analysis.md`
2. Create step 1 prompt: `{source}_{target}_step1.md`
3. Create step 2 prompt: `{source}_{target}_step2.md`

The `prompt_loader` module will automatically find and load them based on the naming convention.

## Modifying Prompts

To modify a prompt:

1. Edit the corresponding `.md` file in this directory
2. Test the changes by running the pipeline
3. No code changes needed - prompts are loaded dynamically

## Source Documentation

The original consolidated prompt documentation files have been removed.
All prompts are now maintained as individual `.md` files in this directory.

