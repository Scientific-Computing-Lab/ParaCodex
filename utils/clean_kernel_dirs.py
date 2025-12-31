#!/usr/bin/env python3
"""
Script to clean only the code files from {kernel_name}-omp directories that appear in the JSONL.
Keeps Makefiles, LICENSE files, etc. but removes the main code files.
"""

from cgitb import text
import json
import os
import shutil
import subprocess
import argparse
from typing import List


def clean_kernel_code_files(input_files: List[str], base_path, api):
    """Clean only the code files from {kernel_name}-{api} directories that appear in one or more JSONL files.

    Args:
        input_files: List of JSONL paths. Each JSONL line must contain
                     {"kernel_name": ..., "code": {...}} entries.
        base_path:   Base path that contains the {kernel_name}-{api} directories.
        api:         "omp" or "cuda" (controls extension mapping).
    """
    # Read the filtered JSONL files to get the list of kernels and their code files
    kernels_and_files = {}
    
    for input_file in input_files:
        with open(input_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                kernel_name = data.get('kernel_name', 'unknown')
                code_files = data.get('code', {})
                # Merge code files from multiple JSONLs
                existing = kernels_and_files.get(kernel_name, set())
                existing.update(code_files.keys())
                kernels_and_files[kernel_name] = existing
    
    # Normalize sets to sorted lists for printing/processing
    kernels_and_files = {k: sorted(list(v)) for k, v in kernels_and_files.items()}
    
    print(f'Found {len(kernels_and_files)} kernels to clean:')
    for kernel, files in kernels_and_files.items():
        print(f'  - {kernel}: {files}')
    
    # Clean each kernel directory
    cleaned_count = 0
    
    for kernel, code_files in kernels_and_files.items():
        kernel_dir = os.path.join(base_path, f'{kernel}-{api}')
        
        if os.path.exists(kernel_dir):
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
            
            # Delete only the code files that appear in JSONL
            deleted_files = []
            for code_file in code_files:
                if api == 'cuda':
                    code_file = code_file.replace('.cpp', '.cu')
                elif api == 'omp':
                    code_file = code_file.replace('.cu', '.cpp')
                else:
                    raise ValueError(f'Invalid API: {api}')

                file_path = os.path.join(kernel_dir, code_file)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    deleted_files.append(code_file)
                    print(f'    - Deleted: {code_file}')
                else:
                    print(f'    - File not found: {code_file}')
            
            # Delete any .md files in the directory
            md_files_deleted = []
            for file in os.listdir(kernel_dir):
                if file.endswith('.md'):
                    file_path = os.path.join(kernel_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        md_files_deleted.append(file)
                        print(f'    - Deleted: {file}')

            profile_files_deleted = []
            for file in os.listdir(kernel_dir):
                if file.startswith('profile'):
                    file_path = os.path.join(kernel_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        profile_files_deleted.append(file)
                        print(f'    - Deleted: {file}')

            text_files_deleted = []
            for file in os.listdir(kernel_dir):
                if file.endswith('.txt'):
                    file_path = os.path.join(kernel_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        text_files_deleted.append(file)
                        print(f'    - Deleted: {file}')

            bak_files_deleted = []
            for file in os.listdir(kernel_dir):
                if file.endswith('.bak') or file.endswith('.backup') or "backup" in file:
                    file_path = os.path.join(kernel_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        bak_files_deleted.append(file)
                        print(f'    - Deleted: {file}')

            log_files_deleted = []
            for file in os.listdir(kernel_dir):
                if file.endswith('.log'):
                    file_path = os.path.join(kernel_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        log_files_deleted.append(file)
                        print(f'    - Deleted: {file}')

            

            if text_files_deleted:
                print(f'    - Deleted {len(text_files_deleted)} text files: {text_files_deleted}')
            
            if md_files_deleted:
                print(f'    - Deleted {len(md_files_deleted)} .md files: {md_files_deleted}')
            
            if profile_files_deleted:
                print(f'    - Deleted {len(profile_files_deleted)} profile files: {profile_files_deleted}')
            
            if bak_files_deleted:
                print(f'    - Deleted {len(bak_files_deleted)} backup files: {bak_files_deleted}')
            
            if deleted_files:
                print(f'  ✓ Deleted {len(deleted_files)} code files from {kernel_dir}')
                cleaned_count += 1
            else:
                print(f'  ✓ No code files found to delete in {kernel_dir}')
                cleaned_count += 1
        else:
            print(f'  ⚠ Directory {kernel_dir} does not exist')
    
    print(f'\nCleaned {cleaned_count} kernel directories')


def main():
    parser = argparse.ArgumentParser(description='Clean kernel code files', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--input',
        nargs='+',
        default=['../pipeline/combined_serial_filenames.jsonl'],
        help='One or more input JSONL files (relative to the current directory)'
    )
    parser.add_argument('--base_path', default='/root/codex_baseline/serial_omp_nas_workdir/data/src', help='Base path to the kernel directories')
    parser.add_argument('--api', default='serial', help='API to clean')
    args = parser.parse_args()
    
    clean_kernel_code_files(args.input, args.base_path, args.api)

if __name__ == "__main__":
    main()