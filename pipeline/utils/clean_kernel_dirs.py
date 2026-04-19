#!/usr/bin/env python3
"""
Script to clean code files from kernel directories based on a JSONL file.
Deletes all files listed in the JSONL plus cleanup files (.md, .txt, profile files, etc.)
"""

import json
import os
import subprocess
import argparse
from typing import Dict, Set


def delete_files_by_pattern(kernel_dir: str, pattern_func) -> list:
    """Delete files matching a pattern and return list of deleted filenames."""
    deleted = []
    if not os.path.exists(kernel_dir):
        return deleted
    
    for file in os.listdir(kernel_dir):
        if pattern_func(file):
            file_path = os.path.join(kernel_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
                deleted.append(file)
    return deleted


def clean_kernel_directories(jsonl_file: str, base_path: str):
    """Clean kernel directories based on JSONL file.
    
    Args:
        jsonl_file: Path to JSONL file with kernel information.
                   Each line should contain: {"kernel_name": ..., "parallel_api": ..., "code": {...}}
        base_path:  Base path (e.g., /root/codex_baseline/cuda_ocl_workdir)
                   The script will look in {base_path}/data/src for kernel directories.
    """
    # Construct the src path
    src_path = os.path.join(base_path, 'data', 'src')
    
    if not os.path.exists(src_path):
        print(f'Error: Directory {src_path} does not exist')
        return
    
    # Read JSONL file to get kernels and their files
    kernels_data = {}
    
    with open(jsonl_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            kernel_name = data.get('kernel_name', 'unknown')
            api = data.get('parallel_api', 'unknown')
            code_files = data.get('code', {})
            
            # Store kernel data with API
            if kernel_name not in kernels_data:
                kernels_data[kernel_name] = {'api': api, 'files': set()}
            kernels_data[kernel_name]['files'].update(code_files.keys())
    
    print(f'Found {len(kernels_data)} kernels to clean:')
    for kernel, info in kernels_data.items():
        print(f'  - {kernel}-{info["api"]}: {sorted(info["files"])}')
    
    # Clean each kernel directory
    cleaned_count = 0
    
    for kernel_name, info in kernels_data.items():
        api = info['api']
        code_files = info['files']
        kernel_dir = os.path.join(src_path, f'{kernel_name}-{api}')
        
        if not os.path.exists(kernel_dir):
            print(f'\n⚠ Directory {kernel_dir} does not exist')
            continue
        
        print(f'\nCleaning {kernel_dir}...')
        
        # Run make clean first
        original_dir = os.getcwd()
        try:
            os.chdir(kernel_dir)
            print(f'    - Running make clean...')
            clean_result = subprocess.run(['make', '-f', 'Makefile.nvc', 'clean'], 
                                       capture_output=True, text=True, timeout=30)
            if clean_result.returncode == 0:
                print(f'    ✓ make clean successful')
            else:
                print(f'    ⚠ make clean failed: {clean_result.stderr}')
        except subprocess.TimeoutExpired:
            print(f'    ⚠ make clean timeout')
        except Exception as e:
            print(f'    ⚠ Error running make clean: {e}')
        finally:
            os.chdir(original_dir)
        
        # Delete code files from JSONL
        deleted_code_files = []
        for code_file in code_files:
            file_path = os.path.join(kernel_dir, code_file)
            if os.path.exists(file_path):
                os.remove(file_path)
                deleted_code_files.append(code_file)
                print(f'    - Deleted: {code_file}')
            else:
                print(f'    - File not found: {code_file}')
        
        # Delete cleanup files
        md_files = delete_files_by_pattern(kernel_dir, lambda f: f.endswith('.md'))
        txt_files = delete_files_by_pattern(kernel_dir, lambda f: f.endswith('.txt'))
        profile_files = delete_files_by_pattern(kernel_dir, lambda f: f.startswith('profile'))
        bak_files = delete_files_by_pattern(kernel_dir, lambda f: f.endswith('.bak') or f.endswith('.backup') or 'backup' in f)
        log_files = delete_files_by_pattern(kernel_dir, lambda f: f.endswith('.log'))
        cl_files = delete_files_by_pattern(kernel_dir, lambda f: f.endswith('.cl'))
        nsys_files = delete_files_by_pattern(kernel_dir, lambda f: f.startswith('nsys'))
        
        # Print summary
        if deleted_code_files:
            print(f'    ✓ Deleted {len(deleted_code_files)} code files')
        if md_files:
            print(f'    ✓ Deleted {len(md_files)} .md files: {md_files}')
        if txt_files:
            print(f'    ✓ Deleted {len(txt_files)} .txt files: {txt_files}')
        if profile_files:
            print(f'    ✓ Deleted {len(profile_files)} profile files: {profile_files}')
        if bak_files:
            print(f'    ✓ Deleted {len(bak_files)} backup files: {bak_files}')
        if log_files:
            print(f'    ✓ Deleted {len(log_files)} .log files: {log_files}')
        
        cleaned_count += 1
    
    print(f'\n✓ Cleaned {cleaned_count} kernel directories')


def main():
    parser = argparse.ArgumentParser(
        description='Clean kernel code files from directories based on JSONL file',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'jsonl_file',
        help='Path to JSONL file with kernel information'
    )
    parser.add_argument(
        'base_path',
        help='Base path (e.g., /root/codex_baseline/cuda_ocl_workdir). Script will look in {base_path}/data/src'
    )
    args = parser.parse_args()
    
    clean_kernel_directories(args.jsonl_file, args.base_path)


if __name__ == "__main__":
    main()