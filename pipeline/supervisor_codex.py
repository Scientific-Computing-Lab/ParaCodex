#!/usr/bin/env python3
"""
Supervisor agent that enforces numerical correctness of translated and optimized code.

Responsibilities:
- Iterate kernels under data/src/*-(omp|cuda)
- Always run the configured clean command (defaults to `make -f Makefile.nvc clean`) then invoke correctness checks
  - OpenMP: default `make -f Makefile.nvc check-correctness`
  - CUDA: default `make -f Makefile.nvc run` when no check target exists
- If correctness fails or compilation fails, use Codex CLI to repair the code in-place
- Repeat up to a max number of attempts per kernel/file

Notes:
- Does not modify makefiles
- Uses the existing golden-label serial reference through the make targets
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from optimize_codex import copy_translated_file, resolve_kernel_file_name, _run_compare_and_optimize_steps
from path_config import (
    NAS_IDENTIFIER,
    build_codex_cli_cmd,
    default_golden_root,
    default_hecbench_src,
    default_jsonl,
    get_gate_sdk_dir,
    get_make_cmd,
    get_make_cmd_str,
    set_codex_workdir,
)


def run_make(kernel_dir: Path, target_api: str, goal: str, extra_make_vars: Optional[Dict[str, str]] = None) -> Tuple[bool, str]:
    """Run make according to the goal.

    goal: one of ["compile", "check"]. Always runs make clean first.
    Returns (success, combined_output)
    """
    try:
        # make clean
        clean = subprocess.run(
            get_make_cmd(target_api, "clean"), cwd=kernel_dir, capture_output=True, text=True, timeout=60
        )
    except Exception:
        # Proceed even if clean fails; gather no output
        pass

    try:
        cmd = get_make_cmd(target_api, "build" if goal == "compile" else "check")

        # Append extra make variables (e.g., REF_DIR override) without spaces inside values
        if extra_make_vars:
            for k, v in extra_make_vars.items():
                cmd.append(f"{k}={v}")

        proc = subprocess.run(
            cmd, cwd=kernel_dir, capture_output=True, text=True, timeout=1800
        )
        ok = proc.returncode == 0
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        return ok, out
    except subprocess.TimeoutExpired:
        return False, f"Timeout executing {' '.join(cmd)}"
    except Exception as e:
        return False, str(e)


def infer_source_filename(kernel_dir: Path, target_api: str, original_name_hint: Optional[str]) -> Optional[str]:
    """Infer the primary source file to repair.

    If original_name_hint is provided, prefer it with appropriate extension adjustments.
    Otherwise, pick main.cpp for OMP and main.cu for CUDA if present, else the first *.cpp/cu found.
    """
    if original_name_hint:
        normalized = Path(resolve_kernel_file_name(original_name_hint, target_api)).name
        candidate = kernel_dir / normalized
        if candidate.exists():
            return normalized

    if target_api == "omp":
        for preferred in ("main.cpp", "main.c"):
            if (kernel_dir / preferred).exists():
                return preferred
        cand_cpp = list(kernel_dir.glob("*.cpp"))
        if cand_cpp:
            return cand_cpp[0].name
        cand_c = list(kernel_dir.glob("*.c"))
        if cand_c:
            return cand_c[0].name
        return None
    else:
        preferred = "main.cu"
        if (kernel_dir / preferred).exists():
            return preferred
        cand = list(kernel_dir.glob("*.cu"))
        return cand[0].name if cand else None


def build_supervise_session_prompt(
    kernel_dir: Path, file_name: str | List[str], target_api: str, golden_file: Path
) -> str:
    # Handle both single file (string) and multiple files (list) for backward compatibility
    if isinstance(file_name, str):
        file_names = [file_name]
    else:
        file_names = file_name
    
    file_list_str = ', '.join(file_names)
    file_listing = '\n'.join(f'- {name}' for name in file_names)
    primary_file_name = file_names[0] if file_names else 'unknown'
    
    abs_gate_sdk = str(get_gate_sdk_dir())
    compile_cmd = get_make_cmd_str(target_api, "build")
    check_cmd = get_make_cmd_str(target_api, "check")
    clean_cmd = get_make_cmd_str(target_api, "clean")
    golden_file_str = str(golden_file)

    return f'''**Goal:** Your sole purpose is to ensure the candidate code at `{kernel_dir}` (file(s): {file_list_str}) is numerically identical to the golden reference at `{golden_file_str}`. You will achieve this by instrumenting both with `gate.h` macros and fixing any discrepancies in the candidate code.
**You must** keep the OpenMP GPU offloading and pragmas. You may change them but **DO NOT** fall back into CPU-only code.
**Context:**
- You are activated **after** an optimization step has modified the file(s): {file_list_str} in `{kernel_dir}`.
- gate macros are located in `{abs_gate_sdk}/gate.h`.
---

### Your Task (Step-by-Step Workflow)

1.  **Instrument Golden Reference (if needed):**
    * Ensure `{golden_file_str}` includes `#include "gate.h"`.
    * After the main computation, add `GATE_CHECKSUM_*` or `GATE_STATS_*` macros to the golden file to capture the final state of all primary result buffers. This is your "source of truth". *You should only need to do this once.*

2.  **Instrument Candidate Code:**
    * Ensure the candidate file(s) in `{kernel_dir}` include `#include "gate.h"`:
{file_listing}
    * Add the **exact same GATE macros** as the golden reference, observing the same variables. The metric names, data types, and sample counts (`n` for stats) must match perfectly.

3.  **Build and Run Check:**
    * From the `{kernel_dir}` directory, run the following commands in order:
        1.  `{clean_cmd}`
        2.  `make -f Makefile.nvc check-correctness`

4.  **Debug and Fix (Iterate if Needed):**
    * **If the check passes:** Your job is done. Stop and output the final, correct code in the file(s) in `{kernel_dir}`.
    * **If the check fails:**
        a. Analyze the failure output from the GATE check.
        b. Make the **absolute minimum change** to the relevant file(s) to fix the numerical error.
        c. Loop back to Step 3 (Build and Run Check). **Do not stop until the check passes.**

---

### Debugging Strategy

When a check fails, use this hierarchy of likely causes:

* **Data Mapping Errors (Most Common):** The error is almost certainly in an OpenMP `map` clause.
    * Is a variable that is read on the GPU mapped with `map(to: ...)`?
    * Is a variable that is written on the GPU and read back by the CPU mapped with `map(from: ...)` or `map(tofrom: ...)`?
    * Are the array sections correct? (e.g., `map(to: A[0:N])`).
* **Race Conditions:** If the previous step involved adding `collapse`, `nowait`, or changing loop structures, suspect a race condition. Ensure loop iterations are truly independent.
* **Reduction Errors:** Ensure any reduction variables (e.g., sums, max, min) are correctly declared in a `reduction(...)` clause.
* **Privatization:** Check that loop-local variables are correctly handled by OpenMP and are not causing state to leak between threads.

---

### Strict Rules

* **BEFORE** any time you you want to compile the code, you must run `{clean_cmd}` in `{kernel_dir}`.
* **DO NOT** perform any performance optimizations. Your only goal is correctness.
* **DO NOT** modify Makefiles, input data, or build commands.
* **DO NOT** change the golden reference file (`{golden_file_str}`) except to add `gate.h` and GATE macros.
* **ONLY** edit the candidate file(s) in `{kernel_dir}`:
{file_listing}
* **KEEP** OpenMP GPU offloading and pragmas. You may change them but **DO NOT** fall back into CPU-only code.

**Never / Forbidden**
- Run commands that read / write to files outside of your current working directory.

**Deliverable:**
- The final, corrected source code for the file(s) in `{kernel_dir}` that successfully passes the `make -f Makefile.nvc check-correctness`:
{file_listing}
'''


def run_codex_supervise_session(
    kernel_dir: Path, file_name: str | List[str], target_api: str, golden_file: Path
) -> Tuple[bool, str]:
    prompt = build_supervise_session_prompt(kernel_dir, file_name, target_api, golden_file)
    try:
        result = subprocess.run(
            build_codex_cli_cmd() + [prompt],
            capture_output=True,
            text=True,
            timeout=1800,
        )
    except subprocess.TimeoutExpired:
        return False, "Codex supervision session timed out"
    except Exception as e:
        return False, f"Codex supervision session error: {e}"

    out = (result.stdout or "") + "\n" + (result.stderr or "")
    return True, out


def ensure_correctness(kernel_dir: Path, target_api: str, original_name_hint: Optional[str] | List[str], golden_root: Path, results_dir: Optional[Path]) -> Dict[str, str]:
    """Attempt to compile and pass correctness; repair iteratively if needed.
    Returns a dict with status information.
    
    Args:
        original_name_hint: Can be a single file name (str), list of file names, or None
    """
    # Handle both single file (string) and multiple files (list) for backward compatibility
    if isinstance(original_name_hint, list):
        # Use first file for inference, but pass list to prompt
        file_name = infer_source_filename(kernel_dir, target_api, original_name_hint[0] if original_name_hint else None)
        file_names_for_prompt = original_name_hint
    else:
        file_name = infer_source_filename(kernel_dir, target_api, original_name_hint)
        file_names_for_prompt = [file_name] if file_name else []
    
    if not file_name:
        return {"kernel": kernel_dir.name, "status": "error", "message": "No source file found"}

    # Compute golden label path
    # Try multiple source API possibilities: serial (for serial->omp), cuda (for cuda->omp), omp (for omp->cuda)
    kernel_name = kernel_dir.name.replace(f"-{target_api}", "")
    
    # Try different source API suffixes in order of likelihood
    source_api_candidates = ["serial", "cuda", "omp"]
    golden_dir = None
    golden_file_path = None
    
    for source_api in source_api_candidates:
        candidate_golden_dir = golden_root / f"{kernel_name}-{source_api}"
        if not candidate_golden_dir.exists():
            continue
            
        golden_dir = candidate_golden_dir
        golden_candidates: List[Path] = []
        if original_name_hint:
            # Handle both single file (str) and list of files
            if isinstance(original_name_hint, list):
                # Use first file for golden reference lookup
                if original_name_hint:
                    hinted = Path(original_name_hint[0]).name
                    golden_candidates.append(golden_dir / hinted)
            else:
                hinted = Path(original_name_hint).name
                golden_candidates.append(golden_dir / hinted)
        golden_candidates.extend(
            [
                golden_dir / "main.cpp",
                golden_dir / "main.c",
                golden_dir / "main.cu",
            ]
        )
        golden_file_path = next((p for p in golden_candidates if p.exists()), None)
        if golden_file_path is None:
            alt_cpp = sorted(golden_dir.glob("*.cpp"))
            alt_c = sorted(golden_dir.glob("*.c"))
            alt_cu = sorted(golden_dir.glob("*.cu"))
            if alt_cpp:
                golden_file_path = alt_cpp[0]
            elif alt_c:
                golden_file_path = alt_c[0]
            elif alt_cu:
                golden_file_path = alt_cu[0]
        
        if golden_file_path and golden_file_path.exists():
            break
    
    if golden_file_path is None or not golden_file_path.exists():
        # If we found a directory but no file, use that directory in the error message
        if golden_dir:
            return {"kernel": kernel_name, "status": "error", "message": f"Golden source not found in {golden_dir}"}
        else:
            # Try to list what directories do exist for better error message
            possible_dirs = [golden_root / f"{kernel_name}-{api}" for api in source_api_candidates]
            existing_dirs = [str(d) for d in possible_dirs if d.exists()]
            if existing_dirs:
                return {"kernel": kernel_name, "status": "error", "message": f"Golden source directory not found. Tried: {', '.join([str(d) for d in possible_dirs])}. Existing: {', '.join(existing_dirs)}"}
            else:
                return {"kernel": kernel_name, "status": "error", "message": f"Golden source not found. Tried: {', '.join([str(d) for d in possible_dirs])}"}

    golden_backup = golden_file_path.with_suffix(golden_file_path.suffix + ".supervisor.bak")

    # Backup golden label file
    try:
        golden_backup.write_bytes(golden_file_path.read_bytes())
    except Exception as e:
        return {"kernel": kernel_name, "status": "error", "message": f"Golden backup failed: {e}"}

    # Determine candidate source path and create a pre-session backup for rollback on failure
    candidate_path = kernel_dir / file_name
    candidate_backup = kernel_dir / f"{file_name}.supervisor.bak"
    try:
        if candidate_path.exists():
            candidate_backup.write_bytes(candidate_path.read_bytes())
    except Exception:
        # Non-fatal: continue without candidate backup
        pass

    # Single Codex session: inject into golden and translated, build, run check, and fix if needed.
    # Pass the list of files to the prompt (or single file as list for consistency)
    print(f"[Supervisor] Launching Codex supervision for {kernel_name}/{file_name}...")
    ok_session, transcript = run_codex_supervise_session(
        kernel_dir, file_names_for_prompt if file_names_for_prompt else [file_name], target_api, golden_file_path
    )

    # After session, verify correctness once using our own make call with include path to ensure reproducibility
    abs_gate_sdk = str(get_gate_sdk_dir())
    compile_vars = {
        "EXTRA_CFLAGS": f"-I{abs_gate_sdk}"
    }
    compiles, _ = run_make(kernel_dir, target_api, goal="compile", extra_make_vars=compile_vars)

    ref_dir_override: Path = golden_dir
    if NAS_IDENTIFIER in str(kernel_dir):
        ref_dir_override = golden_dir.parent

    if not ref_dir_override.exists():
        ref_dir_override = golden_dir

    if (golden_dir / "Makefile.nvc").exists():
        ref_make_override = "Makefile.nvc"
    elif (golden_dir / "Makefile").exists():
        ref_make_override = "Makefile"
    else:
        ref_make_override = "Makefile.nvc"

    check_vars = {
        "REF_DIR": str(ref_dir_override),
        "REF_MAKE": ref_make_override,
        "EXTRA_CFLAGS": f"-I{abs_gate_sdk}"
    }
    ok_check, check_out = run_make(kernel_dir, target_api, goal="check", extra_make_vars=check_vars)

    # Save injected translated source, transcript, and supervisor result to results_dir if provided
    updated_translated_text = None
    try:
        updated_translated_text = (kernel_dir / file_name).read_text()
    except Exception:
        pass

    if results_dir and updated_translated_text:
        try:
            # Save in the main kernel directory structure
            subdir = results_dir / f"{kernel_name}-{target_api}"
            subdir.mkdir(parents=True, exist_ok=True)
            
            # Determine phase based on context - this will be set by the caller
            # For now, we'll save with a generic name that can be renamed by the caller
            injected_name = f"{Path(file_name).stem}_supervisor_injected{Path(file_name).suffix}"
            (subdir / injected_name).write_text(updated_translated_text)
            
            # Save supervisor transcript with generic name
            (subdir / "supervisor_transcript.txt").write_text(transcript)
            
            # Save the compilation and check-correctness output
            supervisor_result_content = f"Check-correctness output:\n{check_out if ok_check else 'FAILED'}"
            (subdir / "supervisor_result.txt").write_text(supervisor_result_content)
            
            # Also save compilation result
            compilation_result_content = f"Compilation Success: {compiles}\n"
            if not compiles:
                compilation_result_content += f"Compilation Error: {compiles}\n"
            (subdir / "supervisor_compilation_result.txt").write_text(compilation_result_content)
            
        except Exception:
            pass

    # Restore original golden file unconditionally (we don't keep golden instrumentation)
    if golden_backup.exists():
        try:
            golden_file_path.write_bytes(golden_backup.read_bytes())
            golden_backup.unlink()
        except Exception:
            pass

    # Write a standardized supervisor_check_output file in the kernel output dir when results_dir provided
    try:
        if results_dir:
            subdir = results_dir / f"{kernel_name}-{target_api}"
            subdir.mkdir(parents=True, exist_ok=True)
            (subdir / "supervisor_check_output.txt").write_text(check_out)
    except Exception:
        pass

    success = compiles and ok_check

    # Rollback candidate on failure; clean up backup on success
    try:
        if success:
            if candidate_backup.exists():
                candidate_backup.unlink()
        else:
            if candidate_backup.exists():
                candidate_path.write_bytes(candidate_backup.read_bytes())
                candidate_backup.unlink()
    except Exception:
        # Do not mask the primary status due to rollback issues
        pass

    # Run compare_and_optimize_steps after successful supervisor
    if success and results_dir:
        try:
            # Extract file_name from original_name_hint or infer it
            file_name = original_name_hint or infer_source_filename(kernel_dir, target_api, None)
            if file_name:
                output_dir_str = str(results_dir)
                _run_compare_and_optimize_steps(kernel_name, file_name, output_dir_str, target_api)
        except Exception as e:
            # Don't fail supervisor if compare_and_optimize_steps fails
            print(f"[WARN] compare_and_optimize_steps failed: {e}")

    if success:
        return {"kernel": kernel_name, "status": "pass"}
    else:
        return {"kernel": kernel_name, "status": "failed", "message": check_out if not ok_check else transcript}


def supervise_translated_code(kernel_name: str, file_name: str | List[str], target_api: str, output_dir: str, hecbench_src_dir: str, golden_root: str, phase: str = 'supervised') -> Dict[str, str]:
    """
    Library-style entry point mirroring optimize_codex.optimize_translated_code.
    Runs a single Codex session to inject macros and fix correctness, then saves artifacts
    in the same results directory structure used by translation/optimization.
    
    Args:
        kernel_name: Name of the kernel
        file_name: Name of the file(s) to supervise (can be a single string or list of strings)
        target_api: Target API ('omp' or 'cuda')
        output_dir: Output directory for results
        hecbench_src_dir: Directory containing source code
        golden_root: Root directory for golden reference files
        phase: The phase name ('supervised' or 'optimized_supervised') for proper file naming
    """
    # Handle both single file (string) and multiple files (list) for backward compatibility
    if isinstance(file_name, str):
        file_names = [file_name]
        primary_file_name = file_name
    else:
        file_names = file_name
        primary_file_name = file_names[0] if file_names else 'unknown'
    
    file_list_str = ', '.join(file_names) if len(file_names) > 1 else file_names[0]
    print(f"Starting supervision for {kernel_name}/{file_list_str} (phase: {phase})...")

    # Paths
    kernel_dir = Path(hecbench_src_dir) / f"{kernel_name}-{target_api}"
    results_dir = Path(output_dir)
    golden_root_path = Path(golden_root)

    # Run supervision (single Codex session) and save outputs
    # Pass the list of files to ensure_correctness so the prompt can show all files
    res = ensure_correctness(kernel_dir, target_api, file_names, golden_root_path, results_dir)

    # Copy the current state of the kernel directory file after supervisor modifications
    try:
        injected_copy = copy_translated_file(
            kernel_name, primary_file_name, target_api, hecbench_src_dir, output_dir, phase
        )
        if injected_copy:
            print(f"✓ Supervisor - {phase} version saved: {injected_copy}")
    except Exception:
        injected_copy = None

    success = res.get('status') == 'pass'
    return {
        'success': success,
        'error_msg': '' if success else res.get('message', ''),
        'injected_file_copy': injected_copy,
        'transcript': res.get('message', '') if not success else ''
    }


def load_kernels_from_jsonl(jsonl_path: Path, source_api: str) -> List[Dict]:
    items: List[Dict] = []
    try:
        with open(jsonl_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if data.get("parallel_api") == source_api:
                        items.append(data)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return items


def main():
    parser = argparse.ArgumentParser(description="Supervisor agent to enforce correctness on kernels with GATE macro injection")
    parser.add_argument("--hecbench-src", default=None, help="Kernel source root")
    parser.add_argument("--target-api", choices=["omp", "cuda"], default="omp", help="Target API of kernels to check")
    parser.add_argument("--from-jsonl", default=None, help="JSONL to infer kernel names and original filenames")
    parser.add_argument("--kernels", nargs="*", help="Specific kernel names to check (e.g., entropy)")
    parser.add_argument("--golden-root", default=None, help="Root of golden-label serial sources")
    parser.add_argument("--results-dir", help="Results directory where injected sources and transcripts will be saved")
    parser.add_argument("--codex-workdir", default=None, help="Override Codex workdir (defaults to CODEX_WORKDIR env or cuda_omp_workdir)")
    parser.add_argument("--file-name", default=None, help="Original source filename hint when supervising a single kernel")
    args = parser.parse_args()

    if args.codex_workdir:
        set_codex_workdir(args.codex_workdir)

    src_root = Path(args.hecbench_src) if args.hecbench_src else default_hecbench_src()
    if not src_root.exists():
        print(f"Error: source root not found: {src_root}")
        sys.exit(1)

    # Map kernel -> original file hints from JSONL when available
    # Support both single file (str) and multiple files (list) for backward compatibility
    name_to_file_hints: Dict[str, str | List[str]] = {}
    if args.from_jsonl:
        jsonl_path = Path(args.from_jsonl)
    else:
        # Try combined_serial_filenames.jsonl first (has all files), fall back to default
        script_dir = Path(__file__).parent
        combined_jsonl = script_dir / "combined_serial_filenames.jsonl"
        if combined_jsonl.exists():
            jsonl_path = combined_jsonl
        else:
            jsonl_path = default_jsonl()
    entries = load_kernels_from_jsonl(jsonl_path, source_api="serial")
    for e in entries:
        kname = e.get("kernel_name")
        code_map = e.get("code", {}) or {}
      
        # Collect all files for this kernel
        file_list = list(code_map.keys())
        if len(file_list) > 1:
            name_to_file_hints[kname] = file_list
        elif file_list:
            name_to_file_hints[kname] = file_list[0]

    # Discover kernel dirs
    suffix = "-" + args.target_api
    if args.kernels:
        kernel_dirs = [src_root / f"{k}{suffix}" for k in args.kernels]
    else:
        kernel_dirs = [d for d in src_root.iterdir() if d.is_dir() and d.name.endswith(suffix)]

    if args.file_name and args.kernels and len(args.kernels) == 1:
        # If a single file name is provided via CLI, check if JSONL has all files for this kernel
        kernel_name = args.kernels[0]
        if kernel_name in name_to_file_hints:
            # If we already have data from JSONL, prefer it (especially if it's a list)
            existing_hint = name_to_file_hints[kernel_name]
            if isinstance(existing_hint, list):
                # JSONL has multiple files - use the list, ignore the single --file-name
                hint = existing_hint
            else:
                # JSONL only had one file, use the provided file name (might be same or different)
                hint = args.file_name
        else:
            # Kernel not found in JSONL, use the provided file name
            hint = args.file_name
        name_to_file_hints[kernel_name] = hint

    results = []
    golden_root = Path(args.golden_root) if args.golden_root else default_golden_root()
    results_dir = Path(args.results_dir) if args.results_dir else None
    for kdir in sorted(kernel_dirs):
        kname = kdir.name.replace(suffix, "")
        hint = name_to_file_hints.get(kname)
        print(f"\n=== Supervising kernel: {kname} ({args.target_api}) ===")
        print(f"Directory: {kdir}")
        # ensure_correctness can handle both single file (str) and list of files
        res = ensure_correctness(kdir, args.target_api, hint, golden_root, results_dir)
        results.append(res)
        status = res.get("status")
        if status == "pass":
            print(f"✓ Correctness PASS for {kname}")
        else:
            print(f"✗ {kname}: {status} -> {res.get('message','')}")

    # Summary
    passed = sum(1 for r in results if r.get("status") == "pass")
    total = len(results)
    print("\n" + "="*50)
    print("SUPERVISOR SUMMARY")
    print("="*50)
    print(f"Total kernels processed: {total}")
    print(f"Passed correctness: {passed}")
    print(f"Failed/other: {total - passed}")

    # Non-zero exit if any failed
    sys.exit(0 if passed == total else 2)


if __name__ == "__main__":
    main()


