#!/usr/bin/env python3
"""
CPU Performance Testing Script for CUDA and OpenMP Files

This script measures CPU time instead of wall clock time using:
1. CPU time measurement via time.process_time() or time.clock()
2. System time measurement via time.perf_counter()
3. Parsing internal timing from program output

Usage:
    python cpu_performance_tester.py <file_path> [options]
    
Examples:
    python cpu_performance_tester.py my_kernel.cu --runs 10
    python cpu_performance_tester.py omp_cuda_workdir/data/src/epistasis-cuda/main.cu --runs 5
    python cpu_performance_tester.py my_code.cpp --api omp --runs 3
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
import resource
from pathlib import Path
from typing import List, Dict, Optional, Tuple

try:
    from typing import Dict, List, Optional
except ImportError:
    # Fallback for older Python versions
    Dict = dict
    List = list
    Optional = lambda x: x


class CPUPerformanceTester:
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
    
    def _create_makefile(self, source_file, executable):
        """Create a Makefile for compilation."""
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
        else:  # OpenMP
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

# Alternative: Use g++ if nvc++ not available
# CC = g++
# CFLAGS := -std=c++14 -Wall -fopenmp
# LDFLAGS = -fopenmp

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
    
    def _setup_compilation_environment(self):
        """Setup the compilation environment."""
        if self.is_benchmark_kernel:
            # Use existing benchmark structure
            work_dir = self.file_path.parent
            executable = "main"
        else:
            # Create temporary directory for standalone files
            self.temp_dir = tempfile.mkdtemp(prefix="cpu_perf_test_")
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
            # Compile
                compile_result = subprocess.run(
                    ['make'], 
                    cwd=work_dir, 
                    capture_output=True, 
                    text=True, 
                    timeout=120
                )
            else:
                # Try to use Makefile.nvc if available (NVIDIA HPC SDK)
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
    
    def _get_cpu_times(self):
        """Get current CPU times using resource module."""
        try:
            # Get process times (user + system time)
            usage = resource.getrusage(resource.RUSAGE_SELF)
            return {
                'user_time': usage.ru_utime,      # User CPU time
                'system_time': usage.ru_stime,    # System CPU time
                'total_cpu_time': usage.ru_utime + usage.ru_stime
            }
        except:
            # Fallback to process_time if resource module fails
            try:
                cpu_time = time.process_time()
                return {
                    'user_time': cpu_time,
                    'system_time': 0.0,
                    'total_cpu_time': cpu_time
                }
            except:
                return {
                    'user_time': 0.0,
                    'system_time': 0.0,
                    'total_cpu_time': 0.0
                }
    
    def _run_single_test(self, work_dir):
        """Run a single performance test with CPU timing."""
        try:
            # Get initial CPU times
            initial_times = self._get_cpu_times()
            
            # Build make command with custom arguments
            make_cmd = ['make', 'run']
            if self.custom_args:
                make_cmd.append(f"ARGS={' '.join(self.custom_args)}")
            
            # Run the program
            result = subprocess.run(
                make_cmd, 
                cwd=work_dir, 
                capture_output=True, 
                text=True, 
                timeout=self.timeout
            )
            
            # Get final CPU times
            final_times = self._get_cpu_times()
            
            # Calculate CPU time differences
            cpu_times = {
                'user_time': final_times['user_time'] - initial_times['user_time'],
                'system_time': final_times['system_time'] - initial_times['system_time'],
                'total_cpu_time': final_times['total_cpu_time'] - initial_times['total_cpu_time']
            }
            
            if result.returncode != 0:
                print(f"Run failed: {result.stderr}")
                return None
            
            return {
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode,
                'cpu_times': cpu_times
            }
            
        except subprocess.TimeoutExpired:
            print(f"Run timeout ({self.timeout}s)")
            return None
        except Exception as e:
            print(f"Run error: {e}")
            return None
    
    def _parse_timing_output(self, output):
        """Parse timing information from output."""
        times = []
        
        # Common timing patterns found in the benchmark suite
        patterns = [
            r'Average kernel execution time.*?([0-9]+\.?[0-9]*)\s*\(([smu]s?)\)',  # Various time units
            r'Average execution time.*?([0-9]+\.?[0-9]*)\s*\(([smu]s?)\)',
            r'Total kernel execution time.*?([0-9]+\.?[0-9]*)\s*\(([smu]s?)\)',
            r'execution time.*?([0-9]+\.?[0-9]*)\s*\(([smu]s?)\)',
            r'time.*?([0-9]+\.?[0-9]*)\s*\(([smu]s?)\)',
            r'Time.*?([0-9]+\.?[0-9]*)\s*\(([smu]s?)\)',
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
        """Clean up temporary files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def run_performance_test(self):
        """Run the complete performance test."""
        print(f"Starting CPU performance test for {self.file_path}")
        print(f"   API: {self.api.upper()}")
        print(f"   Runs: {self.runs}")
        print(f"   Timeout: {self.timeout}s")
        if self.custom_args:
            print(f"   Args: {' '.join(self.custom_args)}")
        print()
        
        try:
            # Setup compilation environment
            work_dir, executable = self._setup_compilation_environment()
            
            # Compile
            if not self._compile_code(work_dir):
                return {'success': False, 'error': 'Compilation failed'}
            
            # Run multiple tests
            all_outputs = []
            all_parsed_times = []
            all_cpu_times = []
            successful_runs = 0
            
            print(f"Running {self.runs} CPU performance tests...")
            for i in range(self.runs):
                print(f"  Run {i+1}/{self.runs}...", end=' ')
                
                result = self._run_single_test(work_dir)
                if result is None:
                    print("Failed")
                    continue
                
                # Record CPU times
                all_cpu_times.append(result['cpu_times'])
                
                # Parse timing from output (optional internal measurements)
                combined_output = result['stdout'] + '\n' + result['stderr']
                parsed_times = self._parse_timing_output(combined_output)
                
                all_outputs.append(combined_output)
                successful_runs += 1
                
                if parsed_times:
                    all_parsed_times.extend(parsed_times)
                    print(f"Success (CPU: {result['cpu_times']['total_cpu_time']:.3f}s, {len(parsed_times)} internal timing(s))")
                else:
                    print(f"Success (CPU: {result['cpu_times']['total_cpu_time']:.3f}s)")
            
            print()
            
            # Calculate statistics for CPU times
            cpu_stats = None
            if all_cpu_times:
                # Extract total CPU times for statistics
                total_cpu_times = [t['total_cpu_time'] for t in all_cpu_times]
                user_times = [t['user_time'] for t in all_cpu_times]
                system_times = [t['system_time'] for t in all_cpu_times]
                
                cpu_stats = {
                    'total_cpu': {
                        'mean': statistics.mean(total_cpu_times),
                        'median': statistics.median(total_cpu_times),
                        'min': min(total_cpu_times),
                        'max': max(total_cpu_times),
                        'count': len(total_cpu_times)
                    },
                    'user_cpu': {
                        'mean': statistics.mean(user_times),
                        'median': statistics.median(user_times),
                        'min': min(user_times),
                        'max': max(user_times),
                        'count': len(user_times)
                    },
                    'system_cpu': {
                        'mean': statistics.mean(system_times),
                        'median': statistics.median(system_times),
                        'min': min(system_times),
                        'max': max(system_times),
                        'count': len(system_times)
                    }
                }
                
                # Add standard deviation and coefficient of variation
                for cpu_type in ['total_cpu', 'user_cpu', 'system_cpu']:
                    times_list = total_cpu_times if cpu_type == 'total_cpu' else (user_times if cpu_type == 'user_cpu' else system_times)
                    if len(times_list) > 1:
                        cpu_stats[cpu_type]['stdev'] = statistics.stdev(times_list)
                        cpu_stats[cpu_type]['cv'] = cpu_stats[cpu_type]['stdev'] / cpu_stats[cpu_type]['mean'] * 100
                    else:
                        cpu_stats[cpu_type]['stdev'] = 0.0
                        cpu_stats[cpu_type]['cv'] = 0.0
            
            # Calculate statistics for parsed times (if any)
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
                    parsed_stats['cv'] = parsed_stats['stdev'] / parsed_stats['mean'] * 100
                else:
                    parsed_stats['stdev'] = 0.0
                    parsed_stats['cv'] = 0.0
            
            return {
                'success': True,
                'successful_runs': successful_runs,
                'total_runs': self.runs,
                'cpu_stats': cpu_stats,
                'parsed_timing_stats': parsed_stats,
                'all_cpu_times': all_cpu_times,
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
        print("CPU PERFORMANCE TEST RESULTS")
        print("=" * 60)
        print(f"File: {results['file_path']}")
        print(f"API: {results['api'].upper()}")
        print(f"Successful runs: {results['successful_runs']}/{results['total_runs']}")
        print()
        
        # Show CPU timing statistics
        if results['cpu_stats']:
            stats = results['cpu_stats']
            
            # Total CPU time
            total_stats = stats['total_cpu']
            print("TOTAL CPU TIME")
            print("-" * 20)
            print(f"Count:      {total_stats['count']} measurements")
            print(f"Mean:       {total_stats['mean']:.6f} seconds")
            print(f"Median:     {total_stats['median']:.6f} seconds")
            print(f"Min:        {total_stats['min']:.6f} seconds")
            print(f"Max:        {total_stats['max']:.6f} seconds")
            print(f"Std Dev:    {total_stats['stdev']:.6f} seconds")
            print(f"CV:         {total_stats['cv']:.2f}%")
            print()
            
            # User CPU time
            user_stats = stats['user_cpu']
            print("USER CPU TIME")
            print("-" * 20)
            print(f"Count:      {user_stats['count']} measurements")
            print(f"Mean:       {user_stats['mean']:.6f} seconds")
            print(f"Median:     {user_stats['median']:.6f} seconds")
            print(f"Min:        {user_stats['min']:.6f} seconds")
            print(f"Max:        {user_stats['max']:.6f} seconds")
            print(f"Std Dev:    {user_stats['stdev']:.6f} seconds")
            print(f"CV:         {user_stats['cv']:.2f}%")
            print()
            
            # System CPU time
            system_stats = stats['system_cpu']
            print("SYSTEM CPU TIME")
            print("-" * 20)
            print(f"Count:      {system_stats['count']} measurements")
            print(f"Mean:       {system_stats['mean']:.6f} seconds")
            print(f"Median:     {system_stats['median']:.6f} seconds")
            print(f"Min:        {system_stats['min']:.6f} seconds")
            print(f"Max:        {system_stats['max']:.6f} seconds")
            print(f"Std Dev:    {system_stats['stdev']:.6f} seconds")
            print(f"CV:         {system_stats['cv']:.2f}%")
            print()
            
            if len(results['all_cpu_times']) > 1:
                print("ALL CPU TIME MEASUREMENTS (seconds)")
                print("-" * 40)
                for i, t in enumerate(results['all_cpu_times'], 1):
                    print(f"  {i:2d}: Total={t['total_cpu_time']:.6f}, User={t['user_time']:.6f}, System={t['system_time']:.6f}")
                print()
        
        # Show parsed timing statistics if available (internal measurements)
        if results['parsed_timing_stats']:
            stats = results['parsed_timing_stats']
            print("INTERNAL TIMING (From Program Output)")
            print("-" * 45)
            print(f"Count:      {stats['count']} measurements")
            print(f"Mean:       {stats['mean']:.6f} seconds")
            print(f"Median:     {stats['median']:.6f} seconds")
            print(f"Min:        {stats['min']:.6f} seconds")
            print(f"Max:        {stats['max']:.6f} seconds")
            print(f"Std Dev:    {stats['stdev']:.6f} seconds")
            print(f"CV:         {stats['cv']:.2f}%")
            print()
            
            if len(results['all_parsed_times']) > 1:
                print("ALL INTERNAL MEASUREMENTS (seconds)")
                print("-" * 40)
                for i, t in enumerate(results['all_parsed_times'], 1):
                    print(f"  {i:2d}: {t:.6f}")
                print()
        else:
            print("No internal timing information found in program output")
            print("   (This is normal - CPU timing is more reliable anyway)")
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
        description="CPU performance testing script for CUDA and OpenMP files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cpu_performance_tester.py my_kernel.cu --runs 10
  python cpu_performance_tester.py omp_cuda_workdir/data/src/epistasis-cuda/main.cu --runs 5
  python cpu_performance_tester.py my_code.cpp --api omp --runs 3 --timeout 600
  python cpu_performance_tester.py my_kernel.cu --args 1024 512 --runs 5
        """
    )
    
    parser.add_argument('file_path', help='Path to the CUDA or OpenMP file to test')
    parser.add_argument('--runs', type=int, default=5, help='Number of performance runs (default: 5)')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout per run in seconds (default: 300)')
    parser.add_argument('--api', choices=['cuda', 'omp'], help='Force API type (auto-detected by default)')
    parser.add_argument('--args', nargs='*', help='Custom arguments to pass to the program')
    parser.add_argument('--quiet', action='store_true', help='Suppress detailed output')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file_path):
        print(f"File not found: {args.file_path}")
        sys.exit(1)
    
    # Create tester and run
    tester = CPUPerformanceTester(
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
