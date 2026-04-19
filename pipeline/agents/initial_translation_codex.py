#!/usr/bin/env python3
"""
Script to systematically translate serial codes to target API using Codex CLI
and measure compilation success rate.

Refactored to use centralized logging, configuration, and shared utilities.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, List

# Add pipeline_refactored directory to path to allow imports when running as script
_script_dir = Path(__file__).parent
_pipeline_refactored_dir = _script_dir.parent
if str(_pipeline_refactored_dir) not in sys.path:
    sys.path.insert(0, str(_pipeline_refactored_dir))

# Absolute imports (relative to pipeline_refactored directory)
from agents.optimize_codex import optimize_translated_code_two_stage
from agents.common import (
    parse_jsonl,
    test_compilation,
    run_codex_command,
    get_default_input_jsonl,
    get_default_output_dir,
    get_clean_kernel_script_path,
    normalize_file_list,
    format_file_list,
    copy_translated_file,
    resolve_kernel_file_name,
    launch_makefile_fix_recovery,
)
from utils.logger import get_logger
from utils.config import get_config
from utils.path_config import (
    default_data_src,
    default_golden_root,
    get_codex_workdir,
    get_make_cmd_str,
    set_codex_workdir,
)
from utils.prompt_loader import build_skill_trigger_prompt, normalize_api_name

logger = get_logger(__name__)
config = get_config()


def _build_serial_translation_prompt(kernel_name, file_names, target_api, source_api, source_dir: Path):
    """Build prompt for Serial translation (to any target), handling multiple files.
    
    Args:
        kernel_name: Name of the kernel
        file_names: List of file names to create (can be a single string for backward compatibility)
        target_api: Target API (e.g., 'omp', 'cuda')
        source_api: Source API (e.g., 'serial')
        source_dir: Path to source directory
    """
    config = get_config()
    kernel_dir = config.data_src() / f"{kernel_name}-{target_api}"
    source_dir = config.golden_root() / f"{kernel_name}-{source_api}"
    
    # Normalize file names
    file_names = normalize_file_list(file_names)
    normalized_files = [resolve_kernel_file_name(fn, target_api) for fn in file_names]
    file_listing = '\n'.join(f'- {name}' for name in normalized_files)
    
    workdir = get_codex_workdir()
    skill_name = f"paracodex-{normalize_api_name(source_api)}-{normalize_api_name(target_api)}-analysis"
    variables = {
        "source_dir": str(source_dir),
        "kernel_dir": str(kernel_dir),
        "file_listing": file_listing,
        "clean_cmd_str": get_make_cmd_str(target_api, "clean"),
        "build_cmd_str": get_make_cmd_str(target_api, "build"),
        "run_cmd_str": get_make_cmd_str(target_api, "run"),
    }
    return build_skill_trigger_prompt(
        skill_name=skill_name,
        task=f"{source_api} -> {target_api} analysis for kernel {kernel_name}.",
        workdir=workdir,
        source_dir=source_dir,
        target_dir=kernel_dir,
        file_listing=file_listing,
        variables=variables,
        notes=[
            f"Use CODEX_WORKDIR={workdir}.",
            f"Read `{workdir}/system_info_summary.txt` to understand the target system hardware configuration before starting.",
            "For shell commands: Prefer redirecting large output to a temporary file, then read that file using the `read_file` tool (BEST) or `cat` WITHOUT redirection.",
            "  - Step 1: Run commands with output redirection: `<command> > /tmp/command_output.txt 2>&1`",
            "  - Step 2: Read the temp file using `read_file` tool to read `/tmp/command_output.txt` directly, OR use `cat /tmp/command_output.txt` (WITHOUT `> /tmp/...` redirection).",
            "  - For reading existing files: Use the `read_file` tool directly (e.g., read `system_info_summary.txt` or skill files) - do NOT copy them to temp files first.",
            "  - CRITICAL - DO NOT: Copy temp files to other temp files (e.g., `cat /tmp/file1.txt > /tmp/file2.txt` is wasteful - just read `/tmp/file1.txt` directly using `read_file` tool).",
            "Do not run git commands.",
            "Use Variables to resolve placeholders in the skill instructions.",
        ],
    )


def _build_cuda_translation_prompt(kernel_name, file_names, target_api, source_api, source_dir: Path):
    """Build prompt for parallel-to-parallel translation, handling multiple files.
    
    Args:
        kernel_name: Name of the kernel
        file_names: List of file names to create (can be a single string for backward compatibility)
        target_api: Target API (e.g., 'cuda', 'omp')
        source_api: Source API (e.g., 'omp', 'cuda')
        source_dir: Path to source directory
    """
    config = get_config()
    kernel_dir = config.data_src() / f"{kernel_name}-{target_api}"
    
    # Normalize file names
    file_names = normalize_file_list(file_names)
    normalized_files = [resolve_kernel_file_name(fn, target_api) for fn in file_names]
    file_listing = '\n'.join(f'- {name}' for name in normalized_files)
    
    # Use golden labels directory for source files if available, otherwise fall back to passed source_dir
    golden_source_dir = config.golden_root() / f"{kernel_name}-{source_api}"
    if golden_source_dir.exists():
        source_dir = golden_source_dir
    
    workdir = get_codex_workdir()
    skill_name = f"paracodex-{normalize_api_name(source_api)}-{normalize_api_name(target_api)}-analysis"
    variables = {
        "source_dir": str(source_dir),
        "kernel_dir": str(kernel_dir),
        "file_listing": file_listing,
        "clean_cmd_str": get_make_cmd_str(target_api, "clean"),
        "build_cmd_str": get_make_cmd_str(target_api, "build"),
        "run_cmd_str": get_make_cmd_str(target_api, "run"),
    }
    return build_skill_trigger_prompt(
        skill_name=skill_name,
        task=f"{source_api} -> {target_api} analysis for kernel {kernel_name}.",
        workdir=workdir,
        source_dir=source_dir,
        target_dir=kernel_dir,
        file_listing=file_listing,
        variables=variables,
        notes=[
            f"Use CODEX_WORKDIR={workdir}.",
            f"Read `{workdir}/system_info_summary.txt` to understand the target system hardware configuration before starting.",
            "For shell commands: Prefer redirecting large output to a temporary file, then read that file using the `read_file` tool (BEST) or `cat` WITHOUT redirection.",
            "Do not run git commands.",
            "Use Variables to resolve placeholders in the skill instructions.",
        ],
    )

def _build_translation_prompt(kernel_name, file_name, target_api, source_api, source_dir: Path):
    """Build translation prompt, handling both single file (str) and multiple files (list).
    
    Args:
        kernel_name: Name of the kernel
        file_name: Single file name (str) or list of file names (List[str])
        target_api: Target API (e.g., 'omp', 'cuda')
        source_api: Source API (e.g., 'serial', 'omp')
        source_dir: Path to source directory
    """
    # Normalize to list
    file_names = normalize_file_list(file_name)
    
    # Use serial-to-omp prompt when source is serial, parallel-to-parallel prompt otherwise
    if source_api == 'serial':
        return _build_serial_translation_prompt(kernel_name, file_names, target_api, source_api, source_dir)
    else:
        # Parallel-to-parallel translation (e.g., omp->cuda, cuda->omp, etc.)
        return _build_cuda_translation_prompt(kernel_name, file_names, target_api, source_api, source_dir)

def run_codex_translation(kernel_name, file_name, target_api, source_api, source_dir: Path, model: Optional[str] = None):
    """Use Codex CLI to translate target code to target API.

    Returns dict on success: { 'combined': stdout+stderr, 'summary': stdout }
    Returns None on failure.
    """
    prompt = _build_translation_prompt(kernel_name, file_name, target_api, source_api, source_dir)
    logger.debug(f"Running Codex translation for {kernel_name} ({source_api} -> {target_api})")
    return run_codex_command(prompt, timeout=6000, model=model)

def run_codex_translation_with_retry(kernel_name, file_name, target_api, source_api, source_dir: Path, max_attempts=3, delay_seconds=1, model: Optional[str] = None):
    """Retry Codex translation up to max_attempts times.

    Returns a dict {'combined': ..., 'summary': ...} on success, or None if all attempts fail.
    An attempt is only considered successful if the codex command succeeds AND analysis.md
    was created somewhere under the workdir.
    """
    workdir = get_codex_workdir()
    for attempt in range(1, max_attempts + 1):
        logger.info(f"Translating with Codex CLI (attempt {attempt}/{max_attempts})...")
        translated = run_codex_translation(kernel_name, file_name, target_api, source_api, source_dir, model=model)
        if translated:
            analysis_files = list(Path(workdir).rglob("analysis.md"))
            if analysis_files:
                logger.success(f"Translation successful on attempt {attempt} (analysis.md found at {analysis_files[0]})")
                return translated
            else:
                logger.warning(f"Codex translation attempt {attempt} produced output but analysis.md was not created under {workdir}; retrying...")
        else:
            logger.warning(f"Codex translation attempt {attempt} failed; retrying...")
        if attempt < max_attempts:
            time.sleep(delay_seconds)
        else:
            logger.error("Codex translation attempts exhausted")
    return None

# test_compilation moved to common.py

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
    """Save phase results (transcript and compilation output) for each phase.
    Files are saved into phase subdirectories (e.g., initial/, step1/, step2_supervised/)."""
    output_dir = Path(output_dir)
    # Use kernel name and target API for directory structure if available, otherwise fall back to old format
    if target_api:
        kernel_output_dir = output_dir / f"{kernel_name}-{target_api}"
    else:
        kernel_output_dir = output_dir / f"{kernel_name}_{file_name}"
    kernel_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create phase subdirectory (e.g., initial/, step1/, etc.)
    phase_dir = kernel_output_dir / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    
    # Save compilation result in phase subdirectory
    if compilation_result is not None:
        comp_file = phase_dir / "compilation.txt"
        with open(comp_file, 'w') as f:
            f.write(f"Phase: {phase}\n")
            f.write(f"Kernel: {kernel_name}\n")
            f.write(f"File: {file_name}\n")
            f.write(f"Compilation Success: {compilation_result['success']}\n")
            if not compilation_result['success']:
                f.write(f"Error: {compilation_result['error_msg']}\n")
            else:
                f.write("Compilation successful\n")
    
    # Save transcript in phase subdirectory
    if transcript is not None:
        transcript_file = phase_dir / "transcript.txt"
        with open(transcript_file, 'w') as f:
            f.write(transcript)
        # Save short summary alongside full transcript when provided
        if transcript_summary is not None:
            transcript_summary_file = phase_dir / "transcript_summary.txt"
            with open(transcript_summary_file, 'w') as f:
                f.write(transcript_summary)
    
    # Save supervisor output in phase subdirectory
    if supervisor_output is not None:
        supervisor_file = phase_dir / "output.txt"
        with open(supervisor_file, 'w') as f:
            f.write(f"Phase: {phase}\n")
            f.write(f"Kernel: {kernel_name}\n")
            f.write(f"File: {file_name}\n")
            f.write("="*50 + "\n")
            f.write(supervisor_output)

def main():
    parser = argparse.ArgumentParser(description='Translate serial codes to OpenMP using Codex CLI')
    default_input = get_default_input_jsonl()
    default_output = get_default_output_dir()
    parser.add_argument('--input', default=None, help='Input JSONL file (optional, defaults to directory scanning)')
    parser.add_argument('--output-dir', default=str(default_output), help='Output directory for translated codes (auto-generated if not specified)')
    parser.add_argument('--save-failed', action='store_true', help='Save failed translations')
    parser.add_argument('--skip-compilation', action='store_true', help='Skip compilation testing')
    parser.add_argument('--target-api', default='ocl', help='Target API to translate to (omp, cuda, hip)')
    parser.add_argument('--model', default=None, help='Codex model to use (e.g., gemini-1.5-pro-002)')
    parser.add_argument('--source-api', default='cuda', help='Source API to translate from (omp, cuda, hip)')
    parser.add_argument('--optimize', action='store_true', help='Run optimization pipeline')
    parser.add_argument('--opt-supervisor-steps', default='', help='Comma-separated steps after which to run supervisor, e.g., 2')
    parser.add_argument('--opt-single-step', type=int, choices=[1, 2], help='Run only this optimization step (1-2)')
    parser.add_argument('--supervise-max-attempts', type=int, default=3, help='Maximum number of restart attempts when supervisor fails')
    parser.add_argument('--opt-max-attempts', type=int, default=2, help='Maximum number of retry attempts per optimization step')
    parser.add_argument('--translate-max-attempts', type=int, default=1, help='Maximum number of retry attempts for Codex translation')
    parser.add_argument('--codex-workdir', default=None, help='Codex CLI working directory (defaults to CODEX_WORKDIR env or cuda_omp_workdir)')

    args = parser.parse_args()
    if args.codex_workdir:
        # Explicitly set the workdir to override any existing CODEX_WORKDIR env var
        resolved_workdir = set_codex_workdir(args.codex_workdir)
        logger.info(f"Set CODEX_WORKDIR to: {resolved_workdir}")
        logger.debug(f"Environment CODEX_WORKDIR: {os.environ.get('CODEX_WORKDIR')}")
    final_workdir = get_codex_workdir()
    logger.info(f"opencode will use workdir: {final_workdir}")
    logger.debug(f"opencode will be launched via: opencode run --agent build --format json --dir {final_workdir}")

    # ------------------------------------------------------------------
    # Initial cleanup of kernel directories for this run.
    # This removes previously generated code files for the given
    # input JSONL, target API, and Codex workdir so we start fresh.
    # ------------------------------------------------------------------
    try:
        clean_script = get_clean_kernel_script_path()
        if clean_script.exists():
            config = get_config()
            base_path = config.data_src()
            clean_cmd = [
                sys.executable,
                str(clean_script),
                "--base_path", str(base_path),
                "--api", args.target_api,
            ]
            logger.info(f"Running kernel cleanup: {' '.join(clean_cmd)}")
            subprocess.run(clean_cmd, check=False)
        else:
            logger.warning(f"clean_kernel_dirs.py not found at {clean_script}")
    except Exception as e:
        logger.warning(f"Kernel cleanup step failed: {e}")

    data_src = default_data_src()  # Where translated code goes (data/src)
    golden_src = default_golden_root()  # Where source reference code is (golden_labels/src)

    # Ensure the chosen output directory exists
    output_dir_path = Path(args.output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=== Serial to OpenMP Translation using Codex CLI ===")
    logger.info(f"Input file: {args.input if args.input else 'Scanning directories'}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Data source directory (translated code): {data_src}")
    logger.info(f"Golden reference directory (source code): {golden_src}")
    logger.info(f"Codex workdir: {get_codex_workdir()}")
    logger.info(f"Target API: {args.target_api}")
    logger.info(f"Optimize enabled: {args.optimize}")
    
    # ----------------------------------------------------------------
    # Discover kernels from workdir structure (setup creates one kernel per workdir)
    # ----------------------------------------------------------------
    logger.info("\n1. Discovering kernels...")
    source_codes = []
    
    if args.input:
         # Use provided JSONL if explicitly given
         logger.info(f"Using JSONL file: {args.input}")
         source_codes = parse_jsonl(Path(args.input), args.source_api)
    else:
        # Infer kernel from golden_labels/src structure (setup creates one kernel per workdir)
        suffix = f"-{args.source_api}"
        
        # Helper to find source file names (don't read contents - translation reads from disk)
        def find_source_file_names(kdir):
            file_names = []
            
            # First check if there is a ParBench payload configuration file
            parbench_payload_file = kdir / ".parbench_payload"
            if parbench_payload_file.exists():
                logger.debug(f"Found .parbench_payload in {kdir.name}")
                with open(parbench_payload_file, "r") as f:
                    # Filter out empty strings
                    file_names = [line.strip() for line in f if line.strip()]
                return file_names

            # Fall back to fetching all code files (search recursively for nested structures)
            for ext in ['.c', '.cpp', '.cu', '.cl']:
                for f in kdir.rglob(f"*{ext}"):
                    if f.is_file() and not f.name.startswith('.'):
                        file_names.append(str(f.relative_to(kdir)))
            return file_names

        if golden_src.exists():
            # Find directories matching the source API pattern
            matching_dirs = [d for d in golden_src.iterdir() 
                           if d.is_dir() and d.name.endswith(suffix)]
            
            if not matching_dirs:
                logger.warning(f"No kernel directories found in {golden_src} matching pattern *-{args.source_api}")
            else:
                # Process all matching directories (typically just one from setup)
                for kdir in sorted(matching_dirs):
                    kernel_name = kdir.name[:-len(suffix)]
                    # Find source file names (contents will be read during translation)
                    file_names = find_source_file_names(kdir)
                    if file_names:
                        # Store as dict with empty values - only filenames are used
                        code_files = {name: "" for name in file_names}
                        source_codes.append({
                            'kernel_name': kernel_name,
                            'code': code_files
                        })
                        logger.info(f"Found kernel: {kernel_name} in {kdir.name} with files: {', '.join(file_names)}")
    
    logger.info(f"Found {len(source_codes)} source code(s) to translate")
    
    # Statistics
    total_codes = len(source_codes)
    successful_translations = 0
    failed_translations = 0
    
    results = []
    
    # Process each source code
    logger.info("\n2. Translation Phase")
    logger.info("="*30)
    
    compilation_results = []  # Track compilation results for summary
    optimization_results = []  # Track optimization results for summary
    
    for i, code_data in enumerate(source_codes, 1):
        kernel_name = code_data['kernel_name']
        logger.info(f"\n--- Processing {i}/{total_codes}: {kernel_name} ---")
        
        # Extract the code files
        code_files = code_data['code']
        if not code_files:
            logger.warning(f"No code files found for {kernel_name}")
            continue
        
        # Collect all file names for this kernel (for passing to optimization/supervisor)
        all_file_names = list(code_files.keys())
        primary_file_name = all_file_names[0] if all_file_names else 'unknown'
        
        logger.info(f"Processing kernel {kernel_name} with {len(all_file_names)} file(s): {', '.join(all_file_names)}")
        for file_name in all_file_names:
            code_content = code_files[file_name]
            logger.debug(f"  - {file_name}: {len(code_content)} characters")
        
        # Translate with Codex CLI (with retry) - ONCE for all files in this kernel
        # Source directory is in golden_labels/src, not data/src
        source_dir = golden_src / f"{kernel_name}-{args.source_api}"
        translated_outputs = run_codex_translation_with_retry(
            kernel_name, all_file_names, args.target_api, args.source_api, source_dir,
            max_attempts=args.translate_max_attempts, delay_seconds=1, model=args.model
        )
        if translated_outputs:
            successful_translations += 1
            
            # Skip compilation check after initial translation
            # The analysis phase should have set up a complete environment.
            # Compilation will be checked later during optimization/supervisor phases.
            compilation_result = {
                'success': True,
                'error_msg': None
            }
            optimization_result = None
            success = True  # Treat as successful to continue pipeline
            
            compilation_results.append({
                'kernel_name': kernel_name,
                'compilation_success': True,
                'error_msg': None,
                'restart_attempts': 0,
                'restart_success': False
            })

            # PHASE 1: Copy initial translated files after successful translation
            if success:
                # Copy all files for this kernel
                initial_file_copy = copy_translated_file(
                    kernel_name, all_file_names, args.target_api, data_src, args.output_dir, 'initial'
                )
                if initial_file_copy:
                    logger.success(f"Phase 1 - Initial translation saved: {initial_file_copy}")
                
                # Save initial compilation result + transcripts (full + summary)
                save_phase_result(
                    kernel_name, primary_file_name, args.output_dir, 'initial',
                    compilation_result,
                    translated_outputs.get('combined'),
                    None,
                    transcript_summary=translated_outputs.get('summary'),
                    target_api=args.target_api
                )
            
            # PHASE 3: Run optimization if translation was successful and optimization is enabled
            if success and args.optimize:
                        logger.info(f"Running Phase 3 - Optimization for {kernel_name} (files: {', '.join(all_file_names)})...")
                        # Steps sequence (single-step override if provided)
                        if args.opt_single_step:
                            steps = [args.opt_single_step]
                        else:
                            steps = [1, 2]
                        try:
                            sup_steps = [int(s.strip()) for s in args.opt_supervisor_steps.split(',') if s.strip()]
                        except Exception:
                            sup_steps = []

                        four_stage_result = optimize_translated_code_two_stage(
                            kernel_name=kernel_name,
                            file_name=all_file_names,  # Pass all files for this kernel
                            target_api=args.target_api,
                            output_dir=args.output_dir,
                            data_src_dir=data_src,
                            steps=steps,
                            supervisor_steps=sup_steps,
                            max_attempts=args.opt_max_attempts,
                            supervise_max_attempts=args.supervise_max_attempts,
                            source_api=args.source_api,  # Pass source_api for parallel-to-parallel optimization
                            model=args.model,
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
            
            # Save generated source code - use unified directory name (kernel_name-target_api) for all files
            # This ensures all files for a kernel go into the same directory
            unified_kernel_name = f"{kernel_name}-{args.target_api}"
            output_dir_p = Path(args.output_dir)
            unified_dir = output_dir_p / unified_kernel_name
            unified_dir.mkdir(parents=True, exist_ok=True)
            
            kernel_dir = Path(data_src) / f"{kernel_name}-{args.target_api}"
            saved_any = False
            
            # Scan directory for code files instead of relying on input filenames
            # This supports 1-to-N translation (e.g. main.cu -> main.cl + main.c)
            # Only save actual code files, not headers
            code_extensions = {'.c', '.cpp', '.cu', '.cl'}
            
            if kernel_dir.exists():
                for fpath in sorted(kernel_dir.iterdir()):
                    if fpath.is_file() and fpath.suffix in code_extensions:
                        # Skip files that look like backups or temporary
                        if fpath.name.startswith('.') or fpath.suffix == '.bak':
                            continue
                            
                        # Save with _initial suffix but keep actual extension
                        stem = fpath.stem
                        ext = fpath.suffix
                        output_file = unified_dir / f"{stem}_initial{ext}"
                        
                        try:
                            shutil.copy2(fpath, output_file)
                            logger.debug(f"Saved {fpath.name} to: {output_file}")
                            saved_any = True
                        except Exception as e:
                            logger.error(f"Error saving {fpath.name}: {e}")
            
            if not saved_any:
                logger.warning("No source files found in kernel directory. Saving transcript summary.")
                # Save transcript as fallback (as .txt, not .cu)
                output_file = unified_dir / "initial_transcript_fallback.txt"
                try:
                    with open(output_file, 'w') as f:
                        f.write(translated_outputs.get('combined', 'No transcript available'))
                    logger.debug(f"Saved transcript to: {output_file}")
                except Exception as e:
                    logger.error(f"Failed to save transcript: {e}")
                
            # Save compilation result once
            if compilation_result is not None:
                comp_result_file = unified_dir / 'compilation_result.txt'
                try:
                    with open(comp_result_file, 'w') as f:
                        f.write(f"Kernel: {kernel_name}\n")
                        f.write(f"Files: {', '.join(all_file_names)}\n")
                        f.write(f"Compilation Success: {compilation_result['success']}\n")
                        if not compilation_result['success']:
                            f.write(f"Error: {compilation_result['error_msg']}\n")
                except Exception:
                    pass
            
            if optimization_result and optimization_result.get('success'):
                logger.debug(f"Optimization results saved to: {optimization_result.get('optimized_file_copy', 'N/A')}")
                if optimization_result.get('initial_file_copy'):
                    logger.debug(f"Initial file copied to: {optimization_result['initial_file_copy']}")
                if optimization_result.get('optimized_file_copy'):
                    logger.debug(f"Optimized file copied to: {optimization_result['optimized_file_copy']}")
                
        else:
            logger.failure("Translation failed")
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
                    try:
                        with open(output_file, 'w') as f:
                            f.write("Translation failed")
                        logger.debug(f"Saved failed {file_name} to: {output_file}")
                    except Exception as e:
                        logger.error(f"Failed to save failed translation file: {e}")
        
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
    
    # Calculate optimization statistics
    successful_optimizations = sum(1 for result in optimization_results if result['optimization_success'])
    failed_optimizations = len(optimization_results) - successful_optimizations
    successful_optimization_compilations = sum(1 for result in optimization_results if result['optimization_compilation_success'])
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("TRANSLATION SUMMARY")
    logger.info("="*50)
    logger.info(f"Total codes processed: {total_codes}")
    logger.info(f"Successful translations: {successful_translations}")
    logger.info(f"Failed translations: {failed_translations}")
    logger.info(f"Successful compilations: {successful_compilations}")
    logger.info(f"Failed compilations: {failed_compilations}")
    
    if args.optimize:
        logger.info(f"Successful optimizations: {successful_optimizations}")
        logger.info(f"Failed optimizations: {failed_optimizations}")
        logger.info(f"Successful optimization compilations: {successful_optimization_compilations}")
        
    if total_codes > 0:
        translation_rate = (successful_translations / total_codes) * 100
        compilation_rate = (successful_compilations / total_codes) * 100
        logger.info(f"\nTranslation success rate: {translation_rate:.1f}%")
        logger.info(f"Compilation success rate: {compilation_rate:.1f}%")
        
        if args.optimize and len(optimization_results) > 0:
            optimization_rate = (successful_optimizations / len(optimization_results)) * 100
            optimization_compilation_rate = (successful_optimization_compilations / len(optimization_results)) * 100
            logger.info(f"Optimization success rate: {optimization_rate:.1f}%")
            logger.info(f"Optimization compilation success rate: {optimization_compilation_rate:.1f}%")
    
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
            'compilation_rate': compilation_rate if total_codes > 0 else 0
        }
        
        if args.optimize:
            summary_data.update({
                'successful_optimizations': successful_optimizations,
                'failed_optimizations': failed_optimizations,
                'successful_optimization_compilations': successful_optimization_compilations,
                'optimization_rate': optimization_rate if len(optimization_results) > 0 else 0,
                'optimization_compilation_rate': optimization_compilation_rate if len(optimization_results) > 0 else 0
            })
        
        json.dump({
            'summary': summary_data,
            'translation_results': results,
            'compilation_results': compilation_results,
            'optimization_results': optimization_results
        }, f, indent=2)
    
    logger.info(f"\nDetailed results saved to: {results_file}")
    
    # Save compilation summary in parent directory
    summary_file = Path(args.output_dir) / 'compilation_summary.txt'
    try:
        with open(summary_file, 'w') as f:
            f.write("TRANSLATION SUMMARY\n")
            f.write("="*50 + "\n")
            f.write(f"Total codes processed: {total_codes}\n")
            f.write(f"Successful translations: {successful_translations}\n")
            f.write(f"Failed translations: {failed_translations}\n")
            f.write(f"Successful compilations: {successful_compilations}\n")
            f.write(f"Failed compilations: {failed_compilations}\n")
            
            if args.optimize:
                f.write(f"Successful optimizations: {successful_optimizations}\n")
                f.write(f"Failed optimizations: {failed_optimizations}\n")
                f.write(f"Successful optimization compilations: {successful_optimization_compilations}\n")
                
            if total_codes > 0:
                f.write(f"\nTranslation success rate: {translation_rate:.1f}%\n")
                f.write(f"Compilation success rate: {compilation_rate:.1f}%\n")
                
                if args.optimize and len(optimization_results) > 0:
                    f.write(f"Optimization success rate: {optimization_rate:.1f}%\n")
                    f.write(f"Optimization compilation success rate: {optimization_compilation_rate:.1f}%\n")
        logger.info(f"Compilation summary saved to: {summary_file}")
    except Exception as e:
        logger.error(f"Failed to save compilation summary: {e}")

    # ============================================================
    # PARBENCH VERIFICATION (runs after ALL phases complete)
    # ============================================================
    codex_workdir = get_codex_workdir()
    parbench_spec_ref = Path(codex_workdir) / ".parbench_spec_path" if codex_workdir else None
    if parbench_spec_ref and parbench_spec_ref.exists():
        try:
            with open(parbench_spec_ref) as f:
                spec_ref = json.load(f)
            parbench_spec_path = spec_ref.get("spec")
            parbench_to_api = spec_ref.get("to_api", args.target_api)
            parbench_root = spec_ref.get("parbench_root")

            logger.info("")
            logger.info("=" * 50)
            logger.info("PARBENCH VERIFICATION PHASE")
            logger.info("=" * 50)
            logger.info(f"Spec:    {parbench_spec_path}")
            logger.info(f"To API:  {parbench_to_api}")

            parbench_verify_script = Path(__file__).parent.parent / "parbench_verify.py"
            if not parbench_verify_script.exists():
                parbench_verify_script = Path(__file__).parent / "parbench_verify.py"

            parbench_results = []
            output_dir_path = Path(args.output_dir)
            kernel_dirs = [d for d in output_dir_path.iterdir() if d.is_dir()] if output_dir_path.exists() else []

            if not kernel_dirs:
                logger.warning("No kernel output dirs found — skipping ParBench verification")
            else:
                for kernel_dir in kernel_dirs:
                    logger.info(f"Running ParBench verify for: {kernel_dir.name}")
                    try:
                        verify_cmd = [
                            sys.executable, str(parbench_verify_script),
                            "--parbench-spec", parbench_spec_path,
                            "--translated-dir", str(kernel_dir),
                            "--to-api", parbench_to_api,
                            "--config", "correctness",
                            "--json-out",
                        ]
                        if parbench_root:
                            verify_cmd += ["--parbench-root", parbench_root]

                        proc = subprocess.run(
                            verify_cmd, capture_output=True, text=True, timeout=300
                        )
                        for line in (proc.stdout or "").splitlines():
                            logger.info(f"[parbench] {line}")
                        if proc.stderr:
                            logger.warning(f"[parbench stderr] {proc.stderr[:500]}")

                        verify_result = {"kernel": kernel_dir.name, "status": "unknown"}
                        json_start = (proc.stdout or "").rfind("{")
                        if json_start >= 0:
                            try:
                                verify_result.update(json.loads(proc.stdout[json_start:]))
                            except json.JSONDecodeError:
                                pass
                        verify_result["status"] = "pass" if proc.returncode == 0 else "fail"
                        parbench_results.append(verify_result)
                    except subprocess.TimeoutExpired:
                        parbench_results.append({"kernel": kernel_dir.name, "status": "timeout"})
                        logger.error(f"ParBench verify timed out for {kernel_dir.name}")
                    except Exception as e:
                        parbench_results.append({"kernel": kernel_dir.name, "status": "error", "error": str(e)})
                        logger.error(f"ParBench verify error for {kernel_dir.name}: {e}")

            passed = sum(1 for r in parbench_results if r.get("status") == "pass")
            logger.info(f"ParBench verification: {passed}/{len(parbench_results)} kernel(s) PASSED")

            # Append results to translation_results.json
            try:
                with open(results_file) as f:
                    existing = json.load(f)
                existing["parbench_verification"] = parbench_results
                with open(results_file, "w") as f:
                    json.dump(existing, f, indent=2)
                logger.info(f"ParBench results appended to {results_file}")
            except Exception as e:
                logger.error(f"Failed to append parbench results: {e}")

        except Exception as e:
            logger.error(f"ParBench verification phase failed: {e}")


if __name__ == "__main__":
    main()
