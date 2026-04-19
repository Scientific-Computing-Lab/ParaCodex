#!/usr/bin/env python3
"""
parbench_verify.py — Run the ParBench harness on translated code.

Usage:
    python parbench_verify.py \\
        --parbench-spec /path/to/parbench/specs/hecbench-gaussian-cuda.json \\
        --translated-dir /path/to/output/gaussian-cuda-sycl \\
        --to-api sycl \\
        --config correctness

Strategy:
    1. Find the *target* spec (same kernel, target API) in the same specs/ folder.
    2. Read all support files and verification_only files from the original source dir.
    3. Create a tmpdir sandbox and populate it:
         - support files  (copied from original src)
         - prompt_payload files (copied from --translated-dir, replacing originals)
    4. Write a patched version of the target spec pointing build.working_directory
       and provenance.source_path at the sandbox (via a temp override JSON).
    5. Run `python -m harness verify <patched_spec> --project-root <parbench_root>`.
    6. Parse and return the result as JSON to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from utils import parbench_utils


PARBENCH_ROOT_DEFAULT = Path("/root/codex_baseline/parbench")


def find_target_spec(source_spec_path: Path, to_api: str) -> Path | None:
    """Find the spec file for the same kernel but target API."""
    specs_dir = source_spec_path.parent
    # Source spec name pattern: <suite>-<kernel>-<source_api>.json
    stem = source_spec_path.stem  # e.g. hecbench-gaussian-cuda
    # Replace the last segment (source api) with target api
    parts = stem.rsplit("-", 1)
    if len(parts) != 2:
        return None
    base = parts[0]  # e.g. hecbench-gaussian
    target_spec = specs_dir / f"{base}-{to_api}.json"
    return target_spec if target_spec.exists() else None


def resolve_source_dir(spec: dict, parbench_root: Path) -> Path | None:
    """Resolve the actual source directory from the spec using parbench_utils."""
    resolved = parbench_utils.resolve_source_dir(spec)
    return Path(resolved) if resolved else None


# Source code extensions to inject when doing a direct (no-target-spec) copy
_SOURCE_EXTENSIONS = {'.cpp', '.c', '.cc', '.cxx', '.cu', '.cuh', '.cl', '.h', '.hpp', '.hip', '.sycl'}
# Files to always skip when copying from translated_dir
_SKIP_NAMES = {'Makefile.nvc', 'Makefile'}  # Makefile handled separately
_SKIP_SUFFIXES = {'.md', '.txt', '.log', '.json', '.bak', '.py'}


def build_sandbox(
    source_spec: dict,
    target_spec: dict,
    translated_dir: Path,
    parbench_root: Path,
    target_spec_found: bool = True,
) -> Path:
    """
    Create a tmpdir sandbox:
      - Copy all support + verification_only files from original source dir.
      - Copy prompt_payload files from translated_dir (overrides originals).
    Returns the sandbox directory path.
    """
    original_src = resolve_source_dir(source_spec, parbench_root)
    target_src = resolve_source_dir(target_spec, parbench_root)

    sandbox = Path(tempfile.mkdtemp(prefix="parbench_verify_sandbox_"))

    # Step 1: copy support files from source (original CUDA dir) — gives us reference.h, LICENSE etc.
    src_to_copy = target_src if (target_spec_found and target_src and target_src.exists()) else original_src
    if src_to_copy and src_to_copy.exists():
        for fpath in src_to_copy.iterdir():
            if fpath.is_file():
                shutil.copy2(fpath, sandbox / fpath.name)

    if target_spec_found:
        # Step 2a: overwrite prompt_payload files with translations using spec's file names
        target_payload_files = target_spec.get("files", {}).get("prompt_payload", [])
        for fname in target_payload_files:
            # Try to find the translated file in translated_dir
            candidates = list(translated_dir.glob(f"*{fname}*")) + list(translated_dir.glob(fname))
            candidates = [c for c in candidates if not c.name.endswith(".bak")]

            if not candidates:
                stem = Path(fname).stem
                candidates = list(translated_dir.glob(f"{stem}*"))
                candidates = [c for c in candidates if not c.name.endswith(".bak")]

            if candidates:
                preferred = sorted(candidates, key=lambda p: (
                    0 if "optimized" in p.name else
                    1 if "initial" in p.name else 2
                ))
                shutil.copy2(preferred[0], sandbox / fname)
                print(f"[sandbox] Injected translated file: {preferred[0].name} → {fname}")
            else:
                print(f"[sandbox] WARNING: No translated file found for '{fname}' in {translated_dir}")
    else:
        # Step 2b: no target spec — copy all source files from translated_dir using their actual names
        # This preserves the correct file names (e.g. main.cpp, kernel.cl) instead of mapping
        # them to CUDA names (main.cu, kernel.h) from the source spec.
        for fpath in sorted(translated_dir.iterdir()):
            if not fpath.is_file():
                continue
            if fpath.name in _SKIP_NAMES:
                continue
            if fpath.suffix.lower() in _SKIP_SUFFIXES:
                continue
            if fpath.suffix.lower() in _SOURCE_EXTENSIONS or fpath.name.startswith('reference'):
                shutil.copy2(fpath, sandbox / fpath.name)
                print(f"[sandbox] Injected translated file: {fpath.name}")

    # Step 3: Copy the pipeline-generated Makefile to override the golden one
    pipeline_makefile = translated_dir / "Makefile.nvc"
    if pipeline_makefile.exists():
        shutil.copy2(pipeline_makefile, sandbox / "Makefile")
        print("[sandbox] Injected pipeline Makefile.nvc as Makefile")
    else:
        print("[sandbox] WARNING: Pipeline Makefile.nvc not found in translated directory")

    return sandbox


def patch_spec_for_sandbox(target_spec: dict, sandbox: Path, parbench_root: Path) -> Path:
    """
    Write a patched copy of the target spec where:
      - provenance.source_path  → sandbox (relative to a fake downloads_root)
      - build.working_directory → sandbox (same)
    Also writes a temporary config/paths.json so the harness resolves to sandbox.

    Returns the path to the patched spec JSON.
    """
    patched = dict(target_spec)

    # We'll set repo_root to sandbox's parent and source_path to sandbox's name
    # so that downloads_root / repo_root / source_path → sandbox
    # Simplest: set both repo_root = "" and source_path = str(sandbox)
    # But harness joins paths, so set downloads_root to sandbox.parent, repo_root="", source_path=sandbox.name

    sandbox_parent = str(sandbox.parent)
    sandbox_name = sandbox.name

    patched_provenance = dict(target_spec.get("provenance", {}))
    patched_provenance["repo_root"] = ""
    patched_provenance["source_path"] = sandbox_name
    patched["provenance"] = patched_provenance

    patched_build = dict(target_spec.get("build", {}))
    patched_build["working_directory"] = sandbox_name
    patched["build"] = patched_build

    # Remove _resolved if present (shouldn't be, but just in case)
    patched.pop("_resolved", None)

    # Write patched spec
    spec_path = sandbox / "_patched_spec.json"
    with open(spec_path, "w") as f:
        json.dump(patched, f, indent=2)

    # Write a local config/paths.json pointing downloads_root to sandbox.parent
    local_config_dir = sandbox / "config"
    local_config_dir.mkdir(exist_ok=True)
    with open(local_config_dir / "paths.json", "w") as f:
        json.dump({
            "project_root": str(sandbox),
            "downloads_root": sandbox_parent,
            "hecbench_root": sandbox_parent,
        }, f)

    return spec_path


def run_harness_verify(
    spec_path: Path,
    project_root: Path,
    parbench_root: Path,
    config: str = "correctness",
    verbose: bool = False,
) -> dict:
    """Run `python -m harness verify` and parse the output."""
    cmd = [
        sys.executable, "-m", "harness",
        "--project-root", str(project_root),
        "--json",
        "verify",
        str(spec_path),
        "--config", config,
    ]
    if verbose:
        cmd.insert(cmd.index("verify"), "--verbose")

    print(f"[harness] Running: {' '.join(cmd)}")
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(parbench_root),
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": "Harness timed out after 600s"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

    # The harness prints human-readable text and then JSON (due to --json flag)
    stdout = proc.stdout
    stderr = proc.stderr

    print("[harness stdout]", stdout[:2000] if stdout else "(empty)")
    if stderr:
        print("[harness stderr]", stderr[:1000])

    # Try to parse the JSON result from stdout
    result = {
        "return_code": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
    }

    # The harness emits JSON after the human-readable summary
    # Find the JSON block (starts with '{')
    json_start = stdout.rfind("{")
    if json_start >= 0:
        try:
            harness_json = json.loads(stdout[json_start:])
            result.update(harness_json)
            result["status"] = "pass" if proc.returncode == 0 else "fail"
        except json.JSONDecodeError:
            result["status"] = "pass" if proc.returncode == 0 else "fail"
    else:
        result["status"] = "pass" if proc.returncode == 0 else "fail"

    return result


def verify(
    parbench_spec: str,
    translated_dir: str,
    to_api: str,
    config: str = "correctness",
    parbench_root: str | None = None,
    verbose: bool = False,
    keep_sandbox: bool = False,
) -> dict:
    """Main entry point for verification. Returns a result dict."""
    source_spec_path = Path(parbench_spec).resolve()
    translated_dir_path = Path(translated_dir).resolve()
    pb_root = Path(parbench_root).resolve() if parbench_root else PARBENCH_ROOT_DEFAULT

    if not source_spec_path.exists():
        return {"status": "error", "error": f"Source spec not found: {source_spec_path}"}
    if not translated_dir_path.exists():
        return {"status": "error", "error": f"Translated dir not found: {translated_dir_path}"}

    # Load source spec
    with open(source_spec_path) as f:
        source_spec = json.load(f)

    # Find target spec
    target_spec_path = find_target_spec(source_spec_path, to_api)
    target_spec_found = target_spec_path is not None
    if not target_spec_found:
        # No target spec: fall back to source spec, but patch it for the target API
        import re as _re
        print(f"[verify] No target spec found for '{to_api}'. Using source spec with build patched for target API.")
        target_spec = dict(source_spec)
        target_spec_path = source_spec_path
        _build = dict(target_spec.get('build', {}))
        # Strip CUDA-specific build flags (ARCH=sm_XX) that don't apply to other APIs
        _cmds = dict(_build.get('commands', {}))
        if _cmds.get('build'):
            _cmds['build'] = _re.sub(r'\s+ARCH=\S+', '', _cmds['build']).strip()
            if not _cmds['build']:
                _cmds['build'] = 'make'
        _build['commands'] = _cmds
        # Detect actual executable name from Makefile.nvc (it may differ from "main")
        _makefile = translated_dir_path / 'Makefile.nvc'
        if _makefile.exists():
            _mf_text = _makefile.read_text(errors='replace')
            _m = _re.search(r'^program\s*[=:?]+\s*(\S+)', _mf_text, _re.MULTILINE)
            if _m:
                _exe = _m.group(1).strip()
                _outputs = dict(_build.get('outputs', {}))
                _outputs['executable'] = _exe
                _build['outputs'] = _outputs
                print(f"[verify] Detected executable name from Makefile.nvc: '{_exe}'")
                # Also patch run.executable to use the real binary name
                _run = dict(target_spec.get('run', {}))
                _run['executable'] = f'./{_exe}'
                target_spec['run'] = _run
        target_spec['build'] = _build
    else:
        print(f"[verify] Using target spec: {target_spec_path.name}")
        with open(target_spec_path) as f:
            target_spec = json.load(f)

    # Build sandbox
    sandbox = build_sandbox(source_spec, target_spec, translated_dir_path, pb_root,
                            target_spec_found=target_spec_found)
    print(f"[verify] Sandbox created at: {sandbox}")

    # Write a local parbench-like structure for the harness
    # The harness needs: project_root/config/paths.json + the spec
    # We'll use sandbox itself as a minimal project root
    patched_spec_path = patch_spec_for_sandbox(target_spec, sandbox, pb_root)

    # Create a minimal manifest.jsonl in sandbox so harness `pairs` doesn't crash
    (sandbox / "manifest.jsonl").write_text("")

    try:
        result = run_harness_verify(
            patched_spec_path,
            project_root=sandbox,
            parbench_root=pb_root,
            config=config,
            verbose=verbose,
        )
    finally:
        if not keep_sandbox:
            shutil.rmtree(sandbox, ignore_errors=True)
        else:
            print(f"[verify] Sandbox kept at: {sandbox}")

    result["parbench_spec"] = str(source_spec_path)
    result["target_spec"] = str(target_spec_path)
    result["translated_dir"] = str(translated_dir_path)
    result["config"] = config
    result["to_api"] = to_api

    # Attach baseline for comparison if available
    baseline = target_spec.get("baseline_results", {})
    if baseline:
        result["baseline"] = baseline.get("configurations", {}).get(config)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Verify translated code against a ParBench spec using the harness."
    )
    parser.add_argument("--parbench-spec", required=True, help="Path to the SOURCE spec JSON (e.g. hecbench-gaussian-cuda.json)")
    parser.add_argument("--translated-dir", required=True, help="Directory containing translated files")
    parser.add_argument("--to-api", required=True, help="Target API (e.g. sycl, omp, cuda)")
    parser.add_argument("--config", default="correctness", choices=["correctness", "performance"], help="Run configuration")
    parser.add_argument("--parbench-root", default=None, help="Path to parbench root (default: /root/codex_baseline/parbench)")
    parser.add_argument("--keep-sandbox", action="store_true", help="Don't delete the sandbox tmp dir after running")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose harness output")
    parser.add_argument("--json-out", action="store_true", help="Print full JSON result to stdout")

    args = parser.parse_args()

    result = verify(
        parbench_spec=args.parbench_spec,
        translated_dir=args.translated_dir,
        to_api=args.to_api,
        config=args.config,
        parbench_root=args.parbench_root,
        verbose=args.verbose,
        keep_sandbox=args.keep_sandbox,
    )

    print("\n" + "="*60)
    print("PARBENCH VERIFICATION RESULT")
    print("="*60)
    status = result.get("status", "unknown").upper()
    print(f"Status:  {status}")
    print(f"Config:  {result.get('config', 'N/A')}")
    print(f"To API:  {result.get('to_api', 'N/A')}")

    if args.json_out:
        print("\n[JSON]\n" + json.dumps(result, indent=2, default=str))

    sys.exit(0 if result.get("status") == "pass" else 1)


if __name__ == "__main__":
    main()
