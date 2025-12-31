#!/usr/bin/env python3
"""
Performance Evaluation Script for Translated OMP Code

This script evaluates the performance of translated OMP code by:
1. For each subdirectory in the provided path, copying main_initial.cpp and main_optimized.cpp
2. Renaming them to main.cpp in the corresponding {kernel_name}-omp directory
3. Running the performance tester on each file
4. Saving results in an organized way

Usage:
    python performance_eval.py <translated_code_path>
    
Example:
    python performance_eval.py /home/erel.kaplan/codex_baseline/cuda_omp_workdir/translated_serial_omp_20250906_113840_gpt_5_2nd_try
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
try:
    from typing import Dict, List, Optional
except ImportError:
    # Fallback for older Python versions
    Dict = dict
    List = list
    Optional = lambda x: x


class PerformanceEvaluator:
    def __init__(self, translated_code_path, base_src_path=None, test_original=False, test_only_original=False, original_src_path=None, use_cpu_timing=False, use_gpu_timing=False):
        self.translated_code_path = Path(translated_code_path)
        self.base_src_path = Path(base_src_path) if base_src_path else Path("/root/codex_baseline/cuda_omp_workdir/data/src")
        self.use_cpu_timing = use_cpu_timing
        self.use_gpu_timing = use_gpu_timing
        
        if use_gpu_timing:
            self.performance_tester_path = Path("/root/codex_baseline/cuda_omp_workdir/gpu_performance_tester.py")
        elif use_cpu_timing:
            self.performance_tester_path = Path("/root/codex_baseline/cuda_omp_workdir/cpu_performance_tester.py")
        else:
            self.performance_tester_path = Path("/root/codex_baseline/cuda_omp_workdir/performance_tester.py")
        
        self.test_original = test_original
        self.test_only_original = test_only_original
        self.original_src_path = Path(original_src_path) if original_src_path else Path("/root/codex_baseline/data_backup/src")
        self.results = {}
        
        # Validate paths
        if not self.translated_code_path.exists():
            raise ValueError(f"Translated code path does not exist: {self.translated_code_path}")
        
        if not self.base_src_path.exists():
            raise ValueError(f"Base source path does not exist: {self.base_src_path}")
        
        if not self.performance_tester_path.exists():
            tester_type = "CPU performance tester" if self.use_cpu_timing else "performance tester"
            raise ValueError(f"{tester_type} not found: {self.performance_tester_path}")
        
        if (self.test_original or self.test_only_original) and not self.original_src_path.exists():
            raise ValueError(f"Original source path does not exist: {self.original_src_path}")
        
        if self.test_only_original and self.test_original:
            raise ValueError("Cannot use both --test-original and --test-only-original at the same time")
        
        if self.use_cpu_timing and self.use_gpu_timing:
            raise ValueError("Cannot use both --use-cpu-timing and --use-gpu-timing at the same time")
    
    def get_kernel_name_from_subdir(self, subdir_name):
        """Extract kernel name from subdirectory name (e.g., 'atomicCost_main.cpp' -> 'atomicCost')"""
        # Remove '_main.cpp' suffix if present
        if subdir_name.endswith('_main.cpp'):
            return subdir_name[:-9]  # Remove '_main.cpp'
        return subdir_name
    
    def find_target_directory(self, kernel_name):
        """Find the target {kernel_name}-omp directory in the base source path"""
        target_dir = self.base_src_path / f"{kernel_name}-omp"
        if target_dir.exists():
            return target_dir
        return None
    
    def copy_and_rename_file(self, source_file, target_dir):
        """Copy source file to target directory and rename it to main.cpp"""
        try:
            target_file = target_dir / "main.cpp"
            
            # Backup existing main.cpp if it exists
            if target_file.exists():
                backup_file = target_dir / "main.cpp.backup"
                shutil.copy2(target_file, backup_file)
                print(f"  Backed up existing main.cpp to main.cpp.backup")
            
            # Copy and rename the file
            shutil.copy2(source_file, target_file)
            print(f"  Copied {source_file.name} to {target_file}")
            return True
            
        except Exception as e:
            print(f"  Error copying {source_file} to {target_dir}: {e}")
            return False
    
    def run_performance_test(self, target_dir, test_name):
        """Run performance test on the file in target directory"""
        main_cpp_path = target_dir / "main.cpp"
        
        if not main_cpp_path.exists():
            return {
                'success': False,
                'error': f"main.cpp not found in {target_dir}",
                'output': '',
                'stderr': ''
            }
        
        try:
            print(f"  Running performance test on {test_name}...")
            
            # Run the performance tester
            result = subprocess.run(
                [sys.executable, str(self.performance_tester_path), str(main_cpp_path)],
                capture_output=True,
                text=True,
                timeout=6000  # 10 minute timeout
            )
            
            return {
                'success': result.returncode == 0,
                'returncode': result.returncode,
                'output': result.stdout,
                'stderr': result.stderr,
                'test_name': test_name
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Performance test timed out',
                'output': '',
                'stderr': 'Test timed out after 10 minutes'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Error running performance test: {e}",
                'output': '',
                'stderr': str(e)
            }
    
    def restore_backup(self, target_dir):
        """Restore the backup main.cpp file if it exists"""
        backup_file = target_dir / "main.cpp.backup"
        main_file = target_dir / "main.cpp"
        
        if backup_file.exists():
            try:
                shutil.copy2(backup_file, main_file)
                backup_file.unlink()  # Remove backup file
                print(f"  Restored original main.cpp from backup")
                return True
            except Exception as e:
                print(f"  Error restoring backup: {e}")
                return False
        return True
    
    def test_original_code(self, kernel_name, target_dir):
        """Test the original code from data_backup"""
        if not (self.test_original or self.test_only_original):
            return None
            
        print(f"  Testing original version...")
        
        # Find original code directory
        original_dir = self.original_src_path / f"{kernel_name}-omp"
        original_main = original_dir / "main.cpp"
        
        if not original_main.exists():
            print(f"  Warning: Original main.cpp not found in {original_dir}")
            return {
                'success': False,
                'error': f"Original main.cpp not found in {original_dir}",
                'output': '',
                'stderr': ''
            }
        
        # Copy original main.cpp to target directory
        if self.copy_and_rename_file(original_main, target_dir):
            original_result = self.run_performance_test(target_dir, "original")
            print(f"  Original test: {'SUCCESS' if original_result['success'] else 'FAILED'}")
            return original_result
        else:
            return {
                'success': False,
                'error': 'Failed to copy original main.cpp'
            }
    
    def evaluate_kernel(self, subdir_path):
        """Evaluate performance for a single kernel"""
        kernel_name = self.get_kernel_name_from_subdir(subdir_path.name)
        print(f"\nEvaluating kernel: {kernel_name}")
        
        # Find target directory
        target_dir = self.find_target_directory(kernel_name)
        if not target_dir:
            print(f"  Warning: Target directory {kernel_name}-omp not found, skipping")
            return {
                'kernel_name': kernel_name,
                'success': False,
                'error': f"Target directory {kernel_name}-omp not found"
            }
        
        print(f"  Target directory: {target_dir}")
        
        # Check for required files
        main_initial = subdir_path / "main_initial.cpp"
        main_optimized = subdir_path / "main_optimized.cpp"
        
        if not main_initial.exists():
            print(f"  Error: main_initial.cpp not found in {subdir_path}")
            return {
                'kernel_name': kernel_name,
                'success': False,
                'error': 'main_initial.cpp not found'
            }
        
        if not main_optimized.exists():
            print(f"  Error: main_optimized.cpp not found in {subdir_path}")
            return {
                'kernel_name': kernel_name,
                'success': False,
                'error': 'main_optimized.cpp not found'
            }
        
        results = {
            'kernel_name': kernel_name,
            'target_dir': str(target_dir),
            'original_test': None,
            'initial_test': None,
            'optimized_test': None,
            'success': True
        }
        
        # Test original version (if requested)
        if self.test_original or self.test_only_original:
            original_result = self.test_original_code(kernel_name, target_dir)
            results['original_test'] = original_result
        
        # Test initial and optimized versions (skip if test_only_original is True)
        if not self.test_only_original:
            # Test initial version
            print(f"  Testing initial version...")
            if self.copy_and_rename_file(main_initial, target_dir):
                initial_result = self.run_performance_test(target_dir, "initial")
                results['initial_test'] = initial_result
                print(f"  Initial test: {'SUCCESS' if initial_result['success'] else 'FAILED'}")
            else:
                results['initial_test'] = {
                    'success': False,
                    'error': 'Failed to copy main_initial.cpp'
                }
                results['success'] = False
            
            # Test optimized version
            print(f"  Testing optimized version...")
            if self.copy_and_rename_file(main_optimized, target_dir):
                optimized_result = self.run_performance_test(target_dir, "optimized")
                results['optimized_test'] = optimized_result
                print(f"  Optimized test: {'SUCCESS' if optimized_result['success'] else 'FAILED'}")
            else:
                results['optimized_test'] = {
                    'success': False,
                    'error': 'Failed to copy main_optimized.cpp'
                }
                results['success'] = False
        
        # Restore original main.cpp
        self.restore_backup(target_dir)
        
        return results
    
    def save_results(self, output_file):
        """Save evaluation results to a file"""
        try:
            with open(output_file, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("PERFORMANCE EVALUATION RESULTS\n")
                f.write("=" * 80 + "\n")
                f.write(f"Translated Code Path: {self.translated_code_path}\n")
                f.write(f"Evaluation Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Kernels Evaluated: {len(self.results)}\n")
                f.write("\n")
                
                successful_kernels = sum(1 for r in self.results.values() if r.get('success', False))
                f.write(f"Successful Evaluations: {successful_kernels}/{len(self.results)}\n")
                f.write("\n")
                
                for kernel_name, result in self.results.items():
                    f.write("-" * 60 + "\n")
                    f.write(f"KERNEL: {kernel_name}\n")
                    f.write("-" * 60 + "\n")
                    
                    if not result.get('success', False):
                        f.write(f"Status: FAILED\n")
                        f.write(f"Error: {result.get('error', 'Unknown error')}\n\n")
                        continue
                    
                    f.write(f"Status: SUCCESS\n")
                    f.write(f"Target Directory: {result.get('target_dir', 'N/A')}\n\n")
                    
                    # Initial test results
                    initial_test = result.get('initial_test')
                    if initial_test:
                        f.write("INITIAL VERSION TEST:\n")
                        f.write(f"  Success: {'YES' if initial_test['success'] else 'NO'}\n")
                        if not initial_test['success']:
                            f.write(f"  Error: {initial_test.get('error', 'Unknown error')}\n")
                        f.write(f"  Output:\n{initial_test.get('output', 'No output')}\n")
                        if initial_test.get('stderr'):
                            f.write(f"  Stderr:\n{initial_test['stderr']}\n")
                        f.write("\n")
                    
                    # Optimized test results
                    optimized_test = result.get('optimized_test')
                    if optimized_test:
                        f.write("OPTIMIZED VERSION TEST:\n")
                        f.write(f"  Success: {'YES' if optimized_test['success'] else 'NO'}\n")
                        if not optimized_test['success']:
                            f.write(f"  Error: {optimized_test.get('error', 'Unknown error')}\n")
                        f.write(f"  Output:\n{optimized_test.get('output', 'No output')}\n")
                        if optimized_test.get('stderr'):
                            f.write(f"  Stderr:\n{optimized_test['stderr']}\n")
                        f.write("\n")
                
                f.write("=" * 80 + "\n")
                f.write("END OF RESULTS\n")
                f.write("=" * 80 + "\n")
            
            # Only print the save message for the first save or final save
            if len(self.results) == 1 or len(self.results) == len([d for d in self.translated_code_path.iterdir() if d.is_dir()]):
                print(f"\nResults saved to: {output_file}")
            
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def save_kernel_results(self, output_dir, result):
        """Save results for a single kernel to its own subdirectory"""
        try:
            kernel_name = result['kernel_name']
            kernel_dir = output_dir / kernel_name
            kernel_dir.mkdir(exist_ok=True)
            
            # Save original test results (if available)
            original_test = result.get('original_test')
            if original_test:
                original_file = kernel_dir / "original_results.txt"
                with open(original_file, 'w') as f:
                    f.write("=" * 60 + "\n")
                    f.write(f"ORIGINAL VERSION TEST RESULTS - {kernel_name}\n")
                    f.write("=" * 60 + "\n")
                    f.write(f"Success: {'YES' if original_test['success'] else 'NO'}\n")
                    if not original_test['success']:
                        f.write(f"Error: {original_test.get('error', 'Unknown error')}\n")
                    f.write(f"Return Code: {original_test.get('returncode', 'N/A')}\n")
                    f.write(f"Test Name: {original_test.get('test_name', 'N/A')}\n\n")
                    f.write("OUTPUT:\n")
                    f.write("-" * 20 + "\n")
                    f.write(original_test.get('output', 'No output') + "\n")
                    if original_test.get('stderr'):
                        f.write("\nSTDERR:\n")
                        f.write("-" * 20 + "\n")
                        f.write(original_test['stderr'] + "\n")
                    f.write("\n" + "=" * 60 + "\n")
            
            # Save initial test results
            initial_test = result.get('initial_test')
            if initial_test:
                initial_file = kernel_dir / "initial_results.txt"
                with open(initial_file, 'w') as f:
                    f.write("=" * 60 + "\n")
                    f.write(f"INITIAL VERSION TEST RESULTS - {kernel_name}\n")
                    f.write("=" * 60 + "\n")
                    f.write(f"Success: {'YES' if initial_test['success'] else 'NO'}\n")
                    if not initial_test['success']:
                        f.write(f"Error: {initial_test.get('error', 'Unknown error')}\n")
                    f.write(f"Return Code: {initial_test.get('returncode', 'N/A')}\n")
                    f.write(f"Test Name: {initial_test.get('test_name', 'N/A')}\n\n")
                    f.write("OUTPUT:\n")
                    f.write("-" * 20 + "\n")
                    f.write(initial_test.get('output', 'No output') + "\n")
                    if initial_test.get('stderr'):
                        f.write("\nSTDERR:\n")
                        f.write("-" * 20 + "\n")
                        f.write(initial_test['stderr'] + "\n")
                    f.write("\n" + "=" * 60 + "\n")
            
            # Save optimized test results
            optimized_test = result.get('optimized_test')
            if optimized_test:
                optimized_file = kernel_dir / "optimized_results.txt"
                with open(optimized_file, 'w') as f:
                    f.write("=" * 60 + "\n")
                    f.write(f"OPTIMIZED VERSION TEST RESULTS - {kernel_name}\n")
                    f.write("=" * 60 + "\n")
                    f.write(f"Success: {'YES' if optimized_test['success'] else 'NO'}\n")
                    if not optimized_test['success']:
                        f.write(f"Error: {optimized_test.get('error', 'Unknown error')}\n")
                    f.write(f"Return Code: {optimized_test.get('returncode', 'N/A')}\n")
                    f.write(f"Test Name: {optimized_test.get('test_name', 'N/A')}\n\n")
                    f.write("OUTPUT:\n")
                    f.write("-" * 20 + "\n")
                    f.write(optimized_test.get('output', 'No output') + "\n")
                    if optimized_test.get('stderr'):
                        f.write("\nSTDERR:\n")
                        f.write("-" * 20 + "\n")
                        f.write(optimized_test['stderr'] + "\n")
                    f.write("\n" + "=" * 60 + "\n")
            
            # Save error results if kernel failed
            if not result.get('success', False):
                error_file = kernel_dir / "error_results.txt"
                with open(error_file, 'w') as f:
                    f.write("=" * 60 + "\n")
                    f.write(f"KERNEL EVALUATION ERROR - {kernel_name}\n")
                    f.write("=" * 60 + "\n")
                    f.write(f"Error: {result.get('error', 'Unknown error')}\n")
                    f.write(f"Target Directory: {result.get('target_dir', 'N/A')}\n")
                    f.write("\n" + "=" * 60 + "\n")
            
        except Exception as e:
            print(f"Error saving kernel results for {result.get('kernel_name', 'unknown')}: {e}")
    
    def run_evaluation(self):
        """Run the complete performance evaluation"""
        if self.use_gpu_timing:
            tester_type = "GPU performance tester"
        elif self.use_cpu_timing:
            tester_type = "CPU performance tester"
        else:
            tester_type = "Wall clock performance tester"
        print(f"Starting performance evaluation...")
        print(f"Translated code path: {self.translated_code_path}")
        print(f"Base source path: {self.base_src_path}")
        print(f"Performance tester: {tester_type} ({self.performance_tester_path})")
        
        # Get all subdirectories
        subdirs = [d for d in self.translated_code_path.iterdir() if d.is_dir()]
        
        if not subdirs:
            print("No subdirectories found in the translated code path")
            return
        
        print(f"Found {len(subdirs)} subdirectories to evaluate")
        
        # Generate output directory
        dir_name = self.translated_code_path.name
        if self.use_gpu_timing:
            output_dir = self.translated_code_path.parent / f"{dir_name}_gpu_eval_performance"
        elif self.use_cpu_timing:
            output_dir = self.translated_code_path.parent / f"{dir_name}_cpu_eval_performance"
        else:
            output_dir = self.translated_code_path.parent / f"{dir_name}_eval_performance"
        
        # Create output directory
        output_dir.mkdir(exist_ok=True)
        print(f"Results will be saved to: {output_dir}")
        
        # Evaluate each kernel
        for i, subdir in enumerate(subdirs):
            try:
                result = self.evaluate_kernel(subdir)
                self.results[result['kernel_name']] = result
                
                # Save results after each kernel for fault tolerance
                print(f"  Saving results after kernel {i+1}/{len(subdirs)}...")
                self.save_kernel_results(output_dir, result)
                
            except Exception as e:
                print(f"Error evaluating {subdir.name}: {e}")
                error_result = {
                    'kernel_name': subdir.name,
                    'success': False,
                    'error': str(e)
                }
                self.results[subdir.name] = error_result
                # Still save results even if there was an error
                self.save_kernel_results(output_dir, error_result)
        
        # Print summary
        print(f"\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        successful = sum(1 for r in self.results.values() if r.get('success', False))
        print(f"Total kernels: {len(self.results)}")
        print(f"Successful evaluations: {successful}")
        print(f"Failed evaluations: {len(self.results) - successful}")
        print(f"Results saved to: {output_dir}")
        print(f"Each kernel has its own subdirectory with:")
        if self.test_original or self.test_only_original:
            print(f"  - original_results.txt (if original test was run)")
        if not self.test_only_original:
            print(f"  - initial_results.txt (if initial test was run)")
            print(f"  - optimized_results.txt (if optimized test was run)")
        print(f"  - error_results.txt (if kernel evaluation failed)")


def main():
    parser = argparse.ArgumentParser(
        description="Performance evaluation script for translated OMP code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python performance_eval.py /path/to/translated_serial_omp_20250906_113840_gpt_5_2nd_try
  python performance_eval.py /path/to/translated_code --base-src /custom/src/path
  python performance_eval.py /path/to/translated_code --test-original
  python performance_eval.py /path/to/translated_code --test-only-original
  python performance_eval.py /path/to/translated_code --use-cpu-timing
  python performance_eval.py /path/to/translated_code --use-gpu-timing
  python performance_eval.py /path/to/translated_code --test-original --use-cpu-timing
  python performance_eval.py /path/to/translated_code --test-original --use-gpu-timing
  python performance_eval.py /path/to/translated_code --test-original --original-src /custom/original/path
        """
    )
    
    parser.add_argument('translated_code_path', 
                       help='Path to the directory containing translated OMP code')
    parser.add_argument('--base-src', 
                       help='Base source path containing {kernel_name}-omp directories (default: /root/codex_baseline/cuda_omp_workdir/data/src)')
    parser.add_argument('--test-original', action='store_true',
                       help='Also test the original code from data_backup/src')
    parser.add_argument('--test-only-original', action='store_true',
                       help='Test only the original code (skip translated versions)')
    parser.add_argument('--use-cpu-timing', action='store_true',
                       help='Use CPU performance tester instead of wall clock performance tester')
    parser.add_argument('--use-gpu-timing', action='store_true',
                       help='Use GPU performance tester instead of wall clock performance tester')
    parser.add_argument('--original-src', 
                       help='Path to original source code (default: /root/codex_baseline/data_backup/src)')
    
    args = parser.parse_args()
    
    try:
        evaluator = PerformanceEvaluator(
            translated_code_path=args.translated_code_path,
            base_src_path=args.base_src,
            test_original=args.test_original,
            test_only_original=args.test_only_original,
            original_src_path=args.original_src,
            use_cpu_timing=args.use_cpu_timing,
            use_gpu_timing=args.use_gpu_timing
        )
        evaluator.run_evaluation()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
