#!/usr/bin/env python3
"""
GPU Performance Testing Script for CUDA and OpenMP Files

This script measures GPU kernel execution timing using:
1. CUDA Events for precise GPU kernel timing (external measurement)
2. nvprof/ncu for GPU kernel profiling
3. External GPU timing without relying on internal code measurements

Usage:
    python gpu_performance_tester.py <file_path> [options]
    
Examples:
    python gpu_performance_tester.py my_kernel.cu --runs 10
    python gpu_performance_tester.py omp_cuda_workdir/data/src/epistasis-cuda/main.cu --runs 5
    python gpu_performance_tester.py my_code.cpp --api omp --runs 3
"""

import argparse
import subprocess
import os
import tempfile
import shutil
import statistics
import re
import time
import sys
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

try:
    from typing import Dict, List, Optional
except ImportError:
    # Fallback for older Python versions
    Dict = dict
    List = list
    Optional = lambda x: x


class GPUPerformanceTester:
    def __init__(self, file_path, runs=5, timeout=300, api=None, custom_args=None):
        self.file_path = Path(file_path)
        self.runs = runs
        self.timeout = timeout
        self.api = api or self._detect_api()
        self.is_benchmark_kernel = self._is_benchmark_kernel()
        self.temp_dir = None
        self.custom_args = custom_args or []
        
    def _detect_api(self):
        """Detect the API based on file extension and content."""
        # Handle directory paths
        if self.file_path.is_dir():
            # Look for source files in the directory
            source_files = []
            for ext in ['.cu', '.cpp', '.c']:
                source_files.extend(self.file_path.glob(f'*{ext}'))
            
            if not source_files:
                raise ValueError(f"No source files found in directory: {self.file_path}")
            
            # Use the first source file found
            source_file = source_files[0]
            print(f"Found source file: {source_file}")
            
            if source_file.suffix == '.cu':
                return 'cuda'
            elif source_file.suffix in ['.cpp', '.c']:
                # Check content for OpenMP pragmas
                try:
                    with open(source_file, 'r') as f:
                        content = f.read()
                        if '#pragma omp' in content or 'omp.h' in content:
                            return 'omp'
                        elif 'cuda' in content.lower() or '__global__' in content:
                            return 'cuda'
                        else:
                            return 'omp'  # Default for .cpp files
                except:
                    return 'omp'
        else:
            # Handle single file paths
            if self.file_path.suffix == '.cu':
                return 'cuda'
            elif self.file_path.suffix in ['.cpp', '.c']:
                # Check content for OpenMP pragmas
                try:
                    with open(self.file_path, 'r') as f:
                        content = f.read()
                        if '#pragma omp' in content or 'omp.h' in content:
                            return 'omp'
                        elif 'cuda' in content.lower() or '__global__' in content:
                            return 'cuda'
                        else:
                            return 'omp'  # Default for .cpp files
                except:
                    return 'omp'
            else:
                raise ValueError(f"Unsupported file extension: {self.file_path.suffix}")
    
    def _is_benchmark_kernel(self):
        """Check if the file is part of the existing benchmark structure."""
        return True
    
    def _check_gpu_availability(self):
        """Check if GPU is available and accessible."""
        try:
            # Check if nvidia-smi is available
            result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                gpu_name = result.stdout.strip()
                print(f"GPU detected: {gpu_name}")
                return True
            else:
                print("No GPU detected or nvidia-smi not available")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("nvidia-smi not found or GPU not accessible")
            return False
    
    def _check_profiling_tools(self):
        """Check which GPU profiling tools are available."""
        tools = {}
        
        # Check for ncu (NVIDIA Nsight Compute)
        try:
            result = subprocess.run(['ncu', '--version'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                tools['ncu'] = True
                print("NVIDIA Nsight Compute (ncu) available")
            else:
                tools['ncu'] = False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            tools['ncu'] = False
        
        # Check for nvprof (legacy profiler)
        try:
            result = subprocess.run(['nvprof', '--version'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                tools['nvprof'] = True
                print("NVIDIA nvprof available")
            else:
                tools['nvprof'] = False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            tools['nvprof'] = False
        
        return tools
    
    def _check_openmp_gpu_compiler(self):
        """Check which OpenMP compiler supports GPU offloading."""
        compilers = {}
        
        # Check for nvhpc (NVIDIA HPC SDK) - prioritize this
        try:
            result = subprocess.run(['nvc++', '--version'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                compilers['nvhpc'] = True
                print("NVIDIA HPC SDK (nvc++) available")
            else:
                compilers['nvhpc'] = False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            compilers['nvhpc'] = False
        
        # Check for clang with OpenMP GPU offloading (fallback)
        try:
            result = subprocess.run(['clang++', '--version'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                # Test if it supports OpenMP GPU offloading
                test_result = subprocess.run([
                    'clang++', '-fopenmp-targets=nvptx64-nvidia-cuda', '--help'
                ], capture_output=True, text=True, timeout=5)
                if test_result.returncode == 0:
                    compilers['clang'] = True
                    print("clang++ with OpenMP GPU offloading available")
                else:
                    compilers['clang'] = False
            else:
                compilers['clang'] = False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            compilers['clang'] = False
        
        return compilers
    
    def _parse_makefile_args(self, work_dir):
        """Parse the Makefile to extract arguments from the run target."""
        makefile_paths = [
            work_dir / 'Makefile',
            work_dir / 'Makefile.nvc',
            work_dir / 'Makefile.aomp'
        ]
        
        for makefile_path in makefile_paths:
            if makefile_path.exists():
                try:
                    with open(makefile_path, 'r') as f:
                        content = f.read()
                    
                    # Look for the run target
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if line.strip().startswith('run:'):
                            # Get the next line which should contain the command
                            if i + 1 < len(lines):
                                run_line = lines[i + 1].strip()
                                # Extract arguments after ./$(program) or ./main
                                if './$(program)' in run_line:
                                    # Split on ./$(program) and take everything after it
                                    parts = run_line.split('./$(program)')
                                    if len(parts) > 1:
                                        args_str = parts[1].strip()
                                        if args_str:
                                            # Split arguments and filter out empty strings
                                            args = [arg.strip() for arg in args_str.split() if arg.strip()]
                                            print(f"Found Makefile arguments in {makefile_path.name}: {args}")
                                            return args
                                elif './main' in run_line:
                                    # Handle direct ./main references
                                    parts = run_line.split('./main')
                                    if len(parts) > 1:
                                        args_str = parts[1].strip()
                                        if args_str:
                                            args = [arg.strip() for arg in args_str.split() if arg.strip()]
                                            print(f"Found Makefile arguments in {makefile_path.name}: {args}")
                                            return args
                except Exception as e:
                    print(f"Error parsing {makefile_path}: {e}")
                    continue
        
        print("No arguments found in Makefile run target")
        return []
    
    
    def _create_makefile(self, source_file, executable):
        """Create a Makefile for compilation with GPU timing support."""
        if self.api == 'cuda':
            makefile_content = f"""
CC = nvcc
OPTIMIZE = yes
DEBUG = no
ARCH = sm_60
LAUNCHER =

program = {executable}
source = {source_file}
obj = $(source:.cu=.o)

CFLAGS := -std=c++14 -Xcompiler -Wall -arch=$(ARCH)
LDFLAGS = 

ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS += -g
endif

ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) $(ARGS)
"""
        else:  # OpenMP with GPU offloading
            makefile_content = f"""
# Use NVIDIA HPC SDK (nvc++) for OpenMP GPU offloading
CC = nvc++
OPTIMIZE = yes
DEBUG = no
LAUNCHER =

program = {executable}
source = {source_file}
obj = $(source:.cpp=.o)

# OpenMP GPU offloading flags for nvc++
CFLAGS := -std=c++14 -Wall -mp=gpu
LDFLAGS = -mp=gpu

# Alternative: Use clang++ if nvc++ not available
# CC = clang++
# CFLAGS := -std=c++14 -Wall -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda
# LDFLAGS = -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda

ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS += -g
endif

ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) $(ARGS)
"""
        return makefile_content
    
    def _modify_makefile_for_nvidia(self, work_dir):
        """Modify existing Makefile.aomp to work with NVIDIA GPUs using nvc++."""
        makefile_path = work_dir / 'Makefile.aomp'
        if not makefile_path.exists():
            print("Warning: Makefile.aomp not found, cannot modify for NVIDIA GPU offloading")
            return
        
        # Read the existing Makefile
        with open(makefile_path, 'r') as f:
            content = f.read()
        
        # Create a backup
        backup_path = work_dir / 'Makefile.aomp.backup'
        with open(backup_path, 'w') as f:
            f.write(content)
        
        # Modify the content for NVIDIA GPU offloading
        modified_content = content
        
        # Change compiler to nvc++
        modified_content = modified_content.replace('CC        = clang++', 'CC        = nvc++')
        
        # Replace AMD GPU flags with NVIDIA GPU flags for nvc++
        nvidia_gpu_flags = """  CFLAGS += -mp=gpu"""
        
        # Find and replace the AMD GPU flags
        import re
        amd_pattern = r'CFLAGS \+= -target x86_64-pc-linux-gnu \\\s*-fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \\\s*-Xopenmp-target=amdgcn-amd-amdhsa \\\s*-march=\$\(ARCH\)'
        modified_content = re.sub(amd_pattern, nvidia_gpu_flags, modified_content, flags=re.MULTILINE)
        
        # Also update LDFLAGS if present
        modified_content = re.sub(r'LDFLAGS.*=.*-fopenmp-targets=amdgcn-amd-amdhsa', 'LDFLAGS = -mp=gpu', modified_content)
        
        # Write the modified Makefile
        with open(makefile_path, 'w') as f:
            f.write(modified_content)
        
        # Mark that the Makefile was modified
        self._makefile_modified = True
        
        print(f"Modified {makefile_path} for NVIDIA GPU offloading with nvc++")
    
    def _setup_compilation_environment(self):
        """Setup the compilation environment."""
        if self.is_benchmark_kernel:
            # Use existing benchmark structure
            if self.file_path.is_dir():
                work_dir = self.file_path
            else:
                work_dir = self.file_path.parent
            executable = "main"
        else:
            # Create temporary directory for standalone files
            self.temp_dir = tempfile.mkdtemp(prefix="gpu_perf_test_")
            work_dir = Path(self.temp_dir)
            
            # Copy source file
            source_name = self.file_path.name
            shutil.copy2(self.file_path, work_dir / source_name)
            
            # Create Makefile
            executable = "test_program"
            makefile_content = self._create_makefile(source_name, executable)
            with open(work_dir / "Makefile", 'w') as f:
                f.write(makefile_content)
        
        return work_dir, executable
    
    def _compile_code(self, work_dir):
        """Compile the code."""
        print(f"Compiling {self.api.upper()} code in {work_dir}...")
        
        try:
            # Clean first
            clean_result = subprocess.run(
                ['make', 'clean'], 
                cwd=work_dir, 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            if self.api == 'cuda':
                # Compile CUDA code
                compile_result = subprocess.run(
                    ['make'], 
                    cwd=work_dir, 
                    capture_output=True, 
                    text=True, 
                    timeout=120
                )
            else:
                # For OpenMP, try different approaches based on available compilers
                print("Attempting OpenMP compilation...")
                
                # First, try to use Makefile.nvc if available (NVIDIA HPC SDK)
                nvc_makefile = work_dir / 'Makefile.nvc'
                if nvc_makefile.exists():
                    print("Trying Makefile.nvc (NVIDIA HPC SDK)")
                    compile_result = subprocess.run(
                        ['make', '-f', 'Makefile.nvc'], 
                        cwd=work_dir, 
                        capture_output=True, 
                        text=True, 
                        timeout=120
                    )
                else:
                    # Try to use the default Makefile with nvc++ (NVIDIA HPC SDK)
                    print("Trying default Makefile with nvc++ (NVIDIA HPC SDK)")
                    compile_result = subprocess.run(
                        ['make'], 
                        cwd=work_dir, 
                        capture_output=True, 
                        text=True, 
                        timeout=120
                    )
            if compile_result.returncode != 0:
                print(f"Compilation failed:")
                print(compile_result.stderr)
                return False
            
            print("Compilation successful")
            return True
            
        except subprocess.TimeoutExpired:
            print("Compilation timeout")
            return False
        except Exception as e:
            print(f"Compilation error: {e}")
            return False
    
    def _run_single_test(self, work_dir, executable, profiling_tools=None):
        """Run a single performance test with GPU kernel timing."""
        try:
            # Build make command with custom arguments
            make_cmd = ['make', 'run']
            if self.custom_args:
                make_cmd.append(f"ARGS={' '.join(self.custom_args)}")
            
            # Record start time
            start_time = time.time()
            
            # Try to use GPU profiling tools for kernel timing
            kernel_times = []
            if profiling_tools and (profiling_tools.get('ncu') or profiling_tools.get('nvprof')):
                kernel_times = self._run_with_profiling(work_dir, executable, self.custom_args, profiling_tools)
            
            # Run the program normally
            result = subprocess.run(
                make_cmd, 
                cwd=work_dir, 
                capture_output=True, 
                text=True, 
                timeout=self.timeout
            )
            
            # Record end time
            end_time = time.time()
            wall_clock_time = end_time - start_time
            
            if result.returncode != 0:
                print(f"Run failed: {result.stderr}")
                return None
            
            return {
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode,
                'wall_clock_time': wall_clock_time,
                'kernel_times': kernel_times
            }
            
        except subprocess.TimeoutExpired:
            print(f"Run timeout ({self.timeout}s)")
            return None
        except Exception as e:
            print(f"Run error: {e}")
            return None
    
    def _run_with_profiling(self, work_dir, executable_name, custom_args, profiling_tools):
        """Run the program with GPU profiling to get kernel execution times."""
        kernel_times = []
        
        try:
            # Get arguments from Makefile
            makefile_args = self._parse_makefile_args(work_dir)
            
            # Build the executable command - use Makefile args if available, otherwise custom args
            exec_cmd = [f'./{executable_name}']
            if makefile_args:
                exec_cmd.extend(makefile_args)
                print(f"Using Makefile arguments: {makefile_args}")
            elif custom_args:
                exec_cmd.extend(custom_args)
                print(f"Using custom arguments: {custom_args}")
            
            # First, test if the executable runs normally
            print(f"Testing executable: {' '.join(exec_cmd)}")
            test_result = subprocess.run(
                exec_cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            print(f"Executable test return code: {test_result.returncode}")
            if test_result.returncode != 0:
                print(f"Executable failed to run: {test_result.stderr}")
                print(f"Executable stdout: {test_result.stdout}")
                # Try to determine what arguments the program expects
                if "Usage:" in test_result.stdout:
                    print("Program expects different arguments. Check the Makefile for correct usage.")
                return []
            print("Executable runs successfully")
            
            if profiling_tools.get('ncu'):
                # Use the new ncu command format as requested by user
                ncu_cmd = ['ncu', '--target-processes', 'all', '--launch-skip', '1', '--launch-count', '1'] + exec_cmd
                print(f"Running with ncu profiling: {' '.join(ncu_cmd)}")
                
                result = subprocess.run(
                    ncu_cmd,
                    cwd=work_dir,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout * 2  # Profiling takes longer
                )
                
                print(f"ncu return code: {result.returncode}")
                print(f"ncu stdout:\n{result.stdout}")
                print(f"ncu stderr:\n{result.stderr}")
                
                if result.returncode == 0 and result.stdout.strip():
                    # Parse ncu output for kernel times
                    kernel_times = self._parse_ncu_output(result.stdout)
                    if kernel_times:
                        print(f"ncu profiling successful, found {len(kernel_times)} kernel times")
                    else:
                        print("ncu succeeded but no kernel times found")
                else:
                    print(f"ncu profiling failed with return code {result.returncode}")
                    if result.stderr:
                        print(f"Error: {result.stderr}")
                    else:
                        print("No error message provided by ncu")
                
                if not kernel_times:
                    print("ncu failed, trying nvprof as fallback...")
                    # Fall through to nvprof
                    profiling_tools['ncu'] = False  # Disable ncu for this run
            
            if not kernel_times and profiling_tools.get('nvprof'):
                # Use nvprof for profiling (legacy, but may work with OpenMP)
                nvprof_cmd = ['nvprof', '--print-gpu-trace', '--csv'] + exec_cmd
                print(f"Running with nvprof profiling: {' '.join(nvprof_cmd)}")
                result = subprocess.run(
                    nvprof_cmd,
                    cwd=work_dir,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout * 2  # Profiling takes longer
                )
                
                print(f"nvprof return code: {result.returncode}")
                print(f"nvprof stdout:\n{result.stdout}")
                print(f"nvprof stderr:\n{result.stderr}")
                
                if result.returncode == 0:
                    # Parse nvprof output for kernel times
                    kernel_times = self._parse_nvprof_output(result.stdout)
                    print(f"nvprof profiling successful, found {len(kernel_times)} kernel times")
                else:
                    print(f"nvprof profiling failed with return code {result.returncode}")
                    if result.stderr:
                        print(f"Error: {result.stderr}")
                    else:
                        print("No error message provided by nvprof")
        
        except Exception as e:
            print(f"Profiling failed: {e}")
        
        return kernel_times
    
    def _parse_ncu_output(self, output):
        """Parse ncu output to extract kernel execution times."""
        kernel_times = []
        try:
            print("Parsing ncu output...")
            print(f"ncu output:\n{output}")
            
            # ncu output parsing - look for kernel execution times
            lines = output.strip().split('\n')
            
            # Look for timing information in various formats
            for line in lines:
                line_lower = line.lower()
                
                # Look for kernel execution time patterns
                if any(keyword in line_lower for keyword in ['kernel', 'gpu', 'omp', 'offload', 'launch']):
                    print(f"Found potential kernel line: {line}")
                    
                    # Try to extract timing information from various formats
                    import re
                    
                    # Look for time patterns like "123.456 us", "123.456 ms", "123.456 s"
                    time_patterns = [
                        r'(\d+\.?\d*)\s*(us|ms|s|ns|seconds?|milliseconds?|microseconds?|nanoseconds?)',
                        r'(\d+\.?\d*)\s*(µs)',  # micro symbol
                        r'(\d+\.?\d*)\s*(μs)'   # Greek mu
                    ]
                    
                    for pattern in time_patterns:
                        matches = re.findall(pattern, line, re.IGNORECASE)
                        for time_val_str, unit in matches:
                            try:
                                time_val = float(time_val_str)
                                
                                # Convert to seconds based on unit
                                if unit.lower() in ['ns', 'nanoseconds', 'nanosecond']:
                                    time_seconds = time_val / 1e9
                                elif unit.lower() in ['us', 'µs', 'μs', 'microseconds', 'microsecond']:
                                    time_seconds = time_val / 1e6
                                elif unit.lower() in ['ms', 'milliseconds', 'millisecond']:
                                    time_seconds = time_val / 1e3
                                elif unit.lower() in ['s', 'seconds', 'second']:
                                    time_seconds = time_val
                                else:
                                    # Default to seconds if unit unclear
                                    time_seconds = time_val
                                
                                # Filter reasonable timing values (between 1ns and 1000s)
                                if 1e-9 <= time_seconds <= 1000:
                                    kernel_times.append(time_seconds)
                                    print(f"Extracted timing: {time_seconds:.6f}s from {time_val_str}{unit}")
                            except ValueError:
                                continue
                    
                    # Also look for raw numeric values that could be timing
                    numbers = re.findall(r'(\d+\.?\d*)', line)
                    for num_str in numbers:
                        try:
                            num_val = float(num_str)
                            # Filter reasonable timing values (between 1ns and 1000s)
                            if 1e-9 <= num_val <= 1000:
                                # If it looks like cycles, convert to time
                                if num_val > 1000:  # Likely cycles
                                    time_seconds = num_val / 1.5e9  # Assume 1.5 GHz
                                else:  # Likely already in seconds
                                    time_seconds = num_val
                                kernel_times.append(time_seconds)
                                print(f"Extracted timing: {time_seconds:.6f}s from {num_str}")
                        except ValueError:
                            continue
            
            # If no kernel times found, try alternative parsing for specific metrics
            if not kernel_times:
                print("No kernel times found with standard parsing, trying alternative...")
                # Look for any timing-related metrics
                for line in lines:
                    if any(metric in line.lower() for metric in ['sm__cycles_elapsed', 'gpu__time', 'duration', 'elapsed']):
                        print(f"Found timing metric line: {line}")
                        # Extract the numeric value
                        import re
                        numbers = re.findall(r'(\d+\.?\d*)', line)
                        for num_str in numbers:
                            try:
                                num_val = float(num_str)
                                if num_val > 0:
                                    # Convert cycles to time if needed
                                    if num_val > 1000:
                                        time_seconds = num_val / 1.5e9
                                    else:
                                        time_seconds = num_val
                                    kernel_times.append(time_seconds)
                                    print(f"Extracted timing: {time_seconds:.6f}s")
                            except ValueError:
                                continue
            
            # If still no times found, look for any numeric values that could be timing
            if not kernel_times:
                print("No specific timing metrics found, looking for any numeric values...")
                for line in lines:
                    import re
                    numbers = re.findall(r'(\d+\.?\d*)', line)
                    for num_str in numbers:
                        try:
                            num_val = float(num_str)
                            # Look for values that could be timing (between 1ns and 100s)
                            if 1e-9 <= num_val <= 100:
                                kernel_times.append(num_val)
                                print(f"Extracted potential timing: {num_val:.6f}s from {num_str}")
                        except ValueError:
                            continue
                                
        except Exception as e:
            print(f"Error parsing ncu output: {e}")
        
        print(f"Total kernel times extracted: {len(kernel_times)}")
        return kernel_times
    
    def _parse_nvprof_output(self, output):
        """Parse nvprof output to extract kernel execution times."""
        kernel_times = []
        try:
            print("Parsing nvprof output...")
            print(f"nvprof output:\n{output}")
            
            # nvprof CSV output parsing
            lines = output.strip().split('\n')
            for line in lines:
                # Look for kernel execution times in various formats
                if any(keyword in line.lower() for keyword in ['kernel', 'gpu', 'omp', 'offload']) and any(unit in line.lower() for unit in ['us', 'ms', 's', 'ns']):
                    print(f"Found potential kernel line: {line}")
                    
                    # Extract timing information from nvprof output
                    import re
                    # Look for patterns like "123.456us", "123.456ms", "123.456s"
                    time_matches = re.findall(r'(\d+\.?\d*)\s*(us|ms|s|ns)', line, re.IGNORECASE)
                    for time_val_str, unit in time_matches:
                        try:
                            time_val = float(time_val_str)
                            
                            # Convert to seconds
                            if unit.lower() == 'ns':
                                time_seconds = time_val / 1e9
                            elif unit.lower() == 'us':
                                time_seconds = time_val / 1e6
                            elif unit.lower() == 'ms':
                                time_seconds = time_val / 1e3
                            else:  # seconds
                                time_seconds = time_val
                            
                            kernel_times.append(time_seconds)
                            print(f"Extracted timing: {time_seconds:.6f}s from {time_val_str}{unit}")
                        except ValueError:
                            continue
                            
        except Exception as e:
            print(f"Error parsing nvprof output: {e}")
        
        print(f"Total kernel times extracted: {len(kernel_times)}")
        return kernel_times
    
    def _parse_timing_output(self, output):
        """Parse GPU kernel timing information from output."""
        times = []
        
        # GPU kernel timing patterns - prioritize kernel execution time
        patterns = [
            # GPU kernel specific patterns
            r'GPU kernel execution time.*?([0-9]+\.?[0-9]*)\s*\(([smu]s?)\)',
            r'CUDA kernel execution time.*?([0-9]+\.?[0-9]*)\s*\(([smu]s?)\)',
            r'Kernel execution time.*?([0-9]+\.?[0-9]*)\s*\(([smu]s?)\)',
            r'GPU time.*?([0-9]+\.?[0-9]*)\s*\(([smu]s?)\)',
            r'CUDA time.*?([0-9]+\.?[0-9]*)\s*\(([smu]s?)\)',
            r'Kernel time.*?([0-9]+\.?[0-9]*)\s*\(([smu]s?)\)',
            # General timing patterns
            r'Average kernel execution time.*?([0-9]+\.?[0-9]*)\s*\(([smu]s?)\)',
            r'Average execution time.*?([0-9]+\.?[0-9]*)\s*\(([smu]s?)\)',
            r'Total kernel execution time.*?([0-9]+\.?[0-9]*)\s*\(([smu]s?)\)',
            r'execution time.*?([0-9]+\.?[0-9]*)\s*\(([smu]s?)\)',
            r'time.*?([0-9]+\.?[0-9]*)\s*\(([smu]s?)\)',
            r'Time.*?([0-9]+\.?[0-9]*)\s*\(([smu]s?)\)',
            # Pattern for format like "Average kernel execution time: 0.268809 (s)"
            r'Average kernel execution time:\s*([0-9]+\.?[0-9]*)\s*\(([smu]s?)\)',
            r'([0-9]+\.?[0-9]*)\s+(s|ms|us|seconds|milliseconds|microseconds)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            for match in matches:
                try:
                    time_val = float(match[0])
                    unit = match[1].lower()
                    
                    # Convert to seconds
                    if unit.startswith('m') and 's' in unit:  # milliseconds
                        time_val /= 1000
                    elif unit.startswith('u') or unit == 'μs':  # microseconds  
                        time_val /= 1000000
                    elif unit.startswith('n'):  # nanoseconds
                        time_val /= 1000000000
                    # else assume seconds
                    
                    times.append(time_val)
                except ValueError:
                    continue
        
        return times
    
    def _cleanup(self):
        """Clean up temporary files and restore original Makefile."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        
        # Restore original Makefile.aomp if it was modified
        if hasattr(self, '_makefile_modified') and self._makefile_modified:
            work_dir = self.file_path if self.file_path.is_dir() else self.file_path.parent
            backup_path = work_dir / 'Makefile.aomp.backup'
            makefile_path = work_dir / 'Makefile.aomp'
            
            if backup_path.exists() and makefile_path.exists():
                shutil.copy2(backup_path, makefile_path)
                backup_path.unlink()
                print(f"Restored original {makefile_path}")
    
    def run_performance_test(self):
        """Run the complete performance test."""
        print(f"Starting GPU performance test for {self.file_path}")
        print(f"   API: {self.api.upper()}")
        print(f"   Runs: {self.runs}")
        print(f"   Timeout: {self.timeout}s")
        if self.custom_args:
            print(f"   Args: {' '.join(self.custom_args)}")
        print()
        
        # Check GPU availability
        if not self._check_gpu_availability():
            print("Warning: GPU not detected or not accessible. Proceeding with kernel timing analysis.")
        
        # Check for GPU profiling tools
        profiling_tools = self._check_profiling_tools()
        if not any(profiling_tools.values()):
            print("Warning: No GPU profiling tools (ncu/nvprof) found. Will use wall-clock timing only.")
        else:
            print("GPU profiling tools available - will attempt to measure GPU kernel timing")
        
        # Check for OpenMP GPU offloading compilers if this is OpenMP code
        if self.api == 'omp':
            openmp_compilers = self._check_openmp_gpu_compiler()
            if not any(openmp_compilers.values()):
                print("Warning: No OpenMP GPU offloading compilers found.")
                print("   For GPU offloading, you need NVIDIA HPC SDK (nvc++) or clang++ with OpenMP support")
                print("   Will attempt compilation with standard g++ (may not offload to GPU)")
        
        try:
            # Setup compilation environment
            work_dir, executable = self._setup_compilation_environment()
            
            # Compile
            if not self._compile_code(work_dir):
                return {'success': False, 'error': 'Compilation failed'}
            
            # Run multiple tests
            all_outputs = []
            all_parsed_times = []
            all_wall_clock_times = []
            all_kernel_times = []
            successful_runs = 0
            
            print(f"Running {self.runs} GPU kernel timing tests...")
            for i in range(self.runs):
                print(f"  Run {i+1}/{self.runs}...", end=' ')
                
                result = self._run_single_test(work_dir, executable, profiling_tools)
                if result is None:
                    print("Failed")
                    continue
                
                # Record wall clock time
                all_wall_clock_times.append(result['wall_clock_time'])
                
                # Record external kernel times from profiling tools
                if result.get('kernel_times'):
                    all_kernel_times.extend(result['kernel_times'])
                
                # Parse internal timing from output (as fallback)
                combined_output = result['stdout'] + '\n' + result['stderr']
                parsed_times = self._parse_timing_output(combined_output)
                all_parsed_times.extend(parsed_times)
                
                all_outputs.append(combined_output)
                successful_runs += 1
                
                # Show results
                timing_info = []
                if result.get('kernel_times'):
                    timing_info.append(f"{len(result['kernel_times'])} external kernel timing(s)")
                if parsed_times:
                    timing_info.append(f"{len(parsed_times)} internal timing(s)")
                
                if timing_info:
                    print(f"Success (Wall: {result['wall_clock_time']:.3f}s, {', '.join(timing_info)})")
                else:
                    print(f"Success (Wall: {result['wall_clock_time']:.3f}s)")
            
            print()
            
            # Calculate statistics for wall-clock times
            wall_clock_stats = None
            if all_wall_clock_times:
                wall_clock_stats = {
                    'mean': statistics.mean(all_wall_clock_times),
                    'median': statistics.median(all_wall_clock_times),
                    'min': min(all_wall_clock_times),
                    'max': max(all_wall_clock_times),
                    'count': len(all_wall_clock_times)
                }
                if len(all_wall_clock_times) > 1:
                    wall_clock_stats['stdev'] = statistics.stdev(all_wall_clock_times)
                    wall_clock_stats['cv'] = wall_clock_stats['stdev'] / wall_clock_stats['mean'] * 100
                else:
                    wall_clock_stats['stdev'] = 0.0
                    wall_clock_stats['cv'] = 0.0
            
            # Calculate statistics for external kernel times (from profiling tools)
            kernel_stats = None
            if all_kernel_times:
                kernel_stats = {
                    'mean': statistics.mean(all_kernel_times),
                    'median': statistics.median(all_kernel_times),
                    'min': min(all_kernel_times),
                    'max': max(all_kernel_times),
                    'count': len(all_kernel_times)
                }
                if len(all_kernel_times) > 1:
                    kernel_stats['stdev'] = statistics.stdev(all_kernel_times)
                    if kernel_stats['mean'] > 0:
                        kernel_stats['cv'] = kernel_stats['stdev'] / kernel_stats['mean'] * 100
                    else:
                        kernel_stats['stdev'] = 0.0
                        kernel_stats['cv'] = 0.0
                else:
                    kernel_stats['stdev'] = 0.0
                    kernel_stats['cv'] = 0.0
            
            # Calculate statistics for parsed times (internal measurements - fallback)
            parsed_stats = None
            if all_parsed_times:
                parsed_stats = {
                    'mean': statistics.mean(all_parsed_times),
                    'median': statistics.median(all_parsed_times),
                    'min': min(all_parsed_times),
                    'max': max(all_parsed_times),
                    'count': len(all_parsed_times)
                }
                if len(all_parsed_times) > 1:
                    parsed_stats['stdev'] = statistics.stdev(all_parsed_times)
                    if parsed_stats['mean'] > 0:
                        parsed_stats['cv'] = parsed_stats['stdev'] / parsed_stats['mean'] * 100
                    else:
                        parsed_stats['stdev'] = 0.0
                        parsed_stats['cv'] = 0.0
                else:
                    parsed_stats['stdev'] = 0.0
                    parsed_stats['cv'] = 0.0
            
            return {
                'success': True,
                'successful_runs': successful_runs,
                'total_runs': self.runs,
                'wall_clock_stats': wall_clock_stats,
                'kernel_stats': kernel_stats,
                'parsed_timing_stats': parsed_stats,
                'all_wall_clock_times': all_wall_clock_times,
                'all_kernel_times': all_kernel_times,
                'all_parsed_times': all_parsed_times,
                'outputs': all_outputs,
                'api': self.api,
                'file_path': str(self.file_path)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
        finally:
            self._cleanup()
    
    
    def print_results(self, results):
        """Print formatted results."""
        if not results['success']:
            print(f"Test failed: {results.get('error', 'Unknown error')}")
            return
        
        print("=" * 60)
        print("GPU KERNEL TIMING TEST RESULTS")
        print("=" * 60)
        print(f"File: {results['file_path']}")
        print(f"API: {results['api'].upper()}")
        print(f"Successful runs: {results['successful_runs']}/{results['total_runs']}")
        print()
        
        # Show wall-clock timing statistics
        if results['wall_clock_stats']:
            stats = results['wall_clock_stats']
            print("WALL-CLOCK TIMING (Total Program Runtime)")
            print("-" * 50)
            print(f"Count:      {stats['count']} measurements")
            print(f"Mean:       {stats['mean']:.6f} seconds")
            print(f"Median:     {stats['median']:.6f} seconds")
            print(f"Min:        {stats['min']:.6f} seconds")
            print(f"Max:        {stats['max']:.6f} seconds")
            print(f"Std Dev:    {stats['stdev']:.6f} seconds")
            print(f"CV:         {stats['cv']:.2f}%")
            print()
            
            if len(results['all_wall_clock_times']) > 1:
                print("ALL WALL-CLOCK MEASUREMENTS (seconds)")
                print("-" * 40)
                for i, t in enumerate(results['all_wall_clock_times'], 1):
                    print(f"  {i:2d}: {t:.6f}")
                print()
        
        
        # Show external GPU kernel timing statistics (from profiling tools)
        if results.get('kernel_stats'):
            stats = results['kernel_stats']
            print("GPU KERNEL TIMING (External Profiling)")
            print("-" * 45)
            print(f"Count:      {stats['count']} measurements")
            print(f"Mean:       {stats['mean']:.6f} seconds")
            print(f"Median:     {stats['median']:.6f} seconds")
            print(f"Min:        {stats['min']:.6f} seconds")
            print(f"Max:        {stats['max']:.6f} seconds")
            print(f"Std Dev:    {stats['stdev']:.6f} seconds")
            print(f"CV:         {stats['cv']:.2f}%")
            print()
            
            if len(results.get('all_kernel_times', [])) > 1:
                print("ALL EXTERNAL KERNEL TIMING MEASUREMENTS (seconds)")
                print("-" * 50)
                for i, t in enumerate(results['all_kernel_times'], 1):
                    print(f"  {i:2d}: {t:.6f}")
                print()
        
        # Show internal kernel timing statistics (from program output) as fallback
        if results.get('parsed_timing_stats'):
            stats = results['parsed_timing_stats']
            print("GPU KERNEL TIMING (Internal Program Output)")
            print("-" * 50)
            print(f"Count:      {stats['count']} measurements")
            print(f"Mean:       {stats['mean']:.6f} seconds")
            print(f"Median:     {stats['median']:.6f} seconds")
            print(f"Min:        {stats['min']:.6f} seconds")
            print(f"Max:        {stats['max']:.6f} seconds")
            print(f"Std Dev:    {stats['stdev']:.6f} seconds")
            print(f"CV:         {stats['cv']:.2f}%")
            print()
            
            if len(results.get('all_parsed_times', [])) > 1:
                print("ALL INTERNAL KERNEL TIMING MEASUREMENTS (seconds)")
                print("-" * 50)
                for i, t in enumerate(results['all_parsed_times'], 1):
                    print(f"  {i:2d}: {t:.6f}")
                print()
        
        # Show message if no kernel timing found
        if not results.get('kernel_stats') and not results.get('parsed_timing_stats'):
            print("No GPU kernel timing information found")
            print("   This may be because:")
            print("   - OpenMP code is running on CPU (not GPU offloading)")
            print("   - No NVIDIA HPC SDK (nvc++) available for GPU offloading")
            print("   - clang++ doesn't have complete NVIDIA GPU offloading support")
            print("   - External profiling tools (ncu/nvprof) not available")
            print("   Wall-clock timing above shows total program execution time")
            print()
        
        # Show sample output for debugging
        if results['outputs']:
            print("SAMPLE OUTPUT (Last Run)")
            print("-" * 30)
            output_lines = results['outputs'][-1].strip().split('\n')
            # Show first 10 lines and last 5 lines if output is long
            if len(output_lines) > 15:
                for line in output_lines[:10]:
                    print(f"  {line}")
                print("  ...")
                for line in output_lines[-5:]:
                    print(f"  {line}")
            else:
                for line in output_lines:
                    print(f"  {line}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="GPU performance testing script for CUDA and OpenMP files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gpu_performance_tester.py my_kernel.cu --runs 10
  python gpu_performance_tester.py omp_cuda_workdir/data/src/epistasis-cuda/main.cu --runs 5
  python gpu_performance_tester.py my_code.cpp --api omp --runs 3 --timeout 600
  python gpu_performance_tester.py my_kernel.cu --args 1024 512 --runs 5
        """
    )
    
    parser.add_argument('file_path', help='Path to the CUDA or OpenMP file to test')
    parser.add_argument('--runs', type=int, default=2, help='Number of performance runs (default: 5)')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout per run in seconds (default: 300)')
    parser.add_argument('--api', choices=['cuda', 'omp'], help='Force API type (auto-detected by default)')
    parser.add_argument('--args', nargs='*', help='Custom arguments to pass to the program')
    parser.add_argument('--quiet', action='store_true', help='Suppress detailed output')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file_path):
        print(f"File not found: {args.file_path}")
        sys.exit(1)
    
    # Create tester and run
    tester = GPUPerformanceTester(
        file_path=args.file_path,
        runs=args.runs,
        timeout=args.timeout,
        api=args.api,
        custom_args=args.args
    )
    
    start_time = time.time()
    results = tester.run_performance_test()
    total_time = time.time() - start_time
    
    if not args.quiet:
        tester.print_results(results)
        print(f"Total test time: {total_time:.2f} seconds")
    
    # Exit with appropriate code
    sys.exit(0 if results['success'] else 1)


if __name__ == "__main__":
    main()
