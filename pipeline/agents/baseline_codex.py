#!/usr/bin/env python3
"""
Baseline agent: translates and optimizes serial code to GPU-offloaded OpenMP
in a single session using the paracodex-baseline skill.

Unlike the multi-step pipeline (analysis → step1 → step2), this agent
invokes only one skill session per kernel, then optionally runs the supervisor
for correctness verification.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

# Add pipeline_refactored directory to path
_script_dir = Path(__file__).parent
_pipeline_refactored_dir = _script_dir.parent
if str(_pipeline_refactored_dir) not in sys.path:
    sys.path.insert(0, str(_pipeline_refactored_dir))

from agents.common import (
    copy_translated_file,
    normalize_file_list,
    run_codex_command,
    resolve_kernel_file_name,
)
from agents.supervisor_codex import (
    build_supervise_session_prompt,
    ensure_correctness,
)
from utils.logger import get_logger
from utils.config import get_config
from utils.path_config import (
    default_data_src,
    default_golden_root,
    get_codex_workdir,
    get_make_cmd_str,
    get_correctness_run_cmd_str,
    set_codex_workdir,
)
from utils.prompt_loader import build_skill_trigger_prompt

logger = get_logger(__name__)
config = get_config()

SKILL_NAME = "paracodex-baseline"


def _find_executable(kernel_dir: Path, kernel_name: str) -> str:
    """Infer the executable name from the Makefile or fall back to kernel_name."""
    makefile = kernel_dir / "Makefile.nvc"
    if makefile.exists():
        for line in makefile.read_text(errors="replace").splitlines():
            stripped = line.strip()
            if stripped.startswith("PROGRAM_NAME"):
                parts = stripped.split("=", 1)
                if len(parts) == 2:
                    name = parts[1].strip()
                    if name:
                        return name
    return kernel_name


def build_baseline_prompt(
    kernel_name: str,
    file_names: List[str],
    target_api: str,
    source_api: str,
    source_dir: Path,
    kernel_dir: Path,
) -> str:
    """Build the trigger prompt for the paracodex-baseline skill."""
    normalized_files = [resolve_kernel_file_name(fn, target_api) for fn in file_names]
    file_listing = "\n".join(f"- {name}" for name in normalized_files)

    workdir = get_codex_workdir()
    executable = _find_executable(kernel_dir, kernel_name)

    variables = {
        "source_dir": str(source_dir),
        "kernel_dir": str(kernel_dir),
        "file_listing": file_listing,
        "executable": executable,
        "target_api": target_api,
        "clean_cmd_str": get_make_cmd_str(target_api, "clean"),
        "build_cmd_str": get_make_cmd_str(target_api, "build"),
        "run_cmd": get_correctness_run_cmd_str(target_api),
    }

    return build_skill_trigger_prompt(
        skill_name=SKILL_NAME,
        task=f"Baseline GPU offload: {source_api} → {target_api} for kernel {kernel_name}.",
        workdir=workdir,
        source_dir=source_dir,
        target_dir=kernel_dir,
        file_listing=file_listing,
        variables=variables,
        notes=[
            f"Use CODEX_WORKDIR={workdir}.",
            "This is a SINGLE-SESSION baseline: translate, optimize, compile, and run in one pass.",
            "Do NOT use any external skill scripts or multi-step pipelines.",
            "For shell commands: redirect large output to a temp file, then read it.",
            "Do not run git commands.",
            "Use Variables to resolve placeholders in the skill instructions.",
        ],
    )


def run_baseline_for_kernel(
    kernel_name: str,
    file_names: List[str],
    target_api: str,
    source_api: str,
    data_src: Path,
    golden_src: Path,
    output_dir: Path,
    model: Optional[str] = None,
) -> dict:
    """Run the baseline skill session for a single kernel.

    Returns a status dict with keys: success, transcript, summary, error.
    """
    source_dir = golden_src / f"{kernel_name}-{source_api}"
    kernel_dir = data_src / f"{kernel_name}-{target_api}"

    if not source_dir.exists():
        msg = f"Source dir not found: {source_dir}"
        logger.error(msg)
        return {"success": False, "error": msg, "transcript": None, "summary": None}

    if not kernel_dir.exists():
        kernel_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[baseline] Running skill for kernel: {kernel_name}")
    logger.info(f"[baseline]   source: {source_dir}")
    logger.info(f"[baseline]   target: {kernel_dir}")
    logger.info(f"[baseline]   files:  {', '.join(file_names)}")

    prompt = build_baseline_prompt(
        kernel_name=kernel_name,
        file_names=file_names,
        target_api=target_api,
        source_api=source_api,
        source_dir=source_dir,
        kernel_dir=kernel_dir,
    )

    result = run_codex_command(prompt, timeout=6000, model=model)

    if result:
        logger.success(f"[baseline] Skill session completed for {kernel_name}")
        return {
            "success": True,
            "transcript": result.get("combined"),
            "summary": result.get("summary"),
            "error": None,
        }
    else:
        msg = "Baseline skill session returned no result (timeout or SDK error)"
        logger.error(f"[baseline] {msg}")
        return {"success": False, "error": msg, "transcript": None, "summary": None}


def save_baseline_artifacts(
    kernel_name: str,
    file_names: List[str],
    target_api: str,
    data_src: Path,
    output_dir: Path,
    result: dict,
):
    """Snapshot the kernel workdir into output_dir/<kernel>-<api>/baseline/ and save transcripts."""
    # Full directory snapshot (same as copy_translated_file used by the standard pipeline)
    copy_translated_file(kernel_name, file_names, target_api, data_src, output_dir, "baseline")

    # Save transcripts alongside the snapshot
    baseline_dir = output_dir / f"{kernel_name}-{target_api}" / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    if result.get("transcript"):
        (baseline_dir / "transcript.txt").write_text(result["transcript"])
    if result.get("summary"):
        (baseline_dir / "transcript_summary.txt").write_text(result["summary"])


def find_source_file_names(kdir: Path) -> List[str]:
    """Discover source files in a golden-label kernel directory."""
    # Check for ParBench payload
    payload_file = kdir / ".parbench_payload"
    if payload_file.exists():
        lines = [l.strip() for l in payload_file.read_text().splitlines() if l.strip()]
        if lines:
            return lines

    file_names = []
    for ext in [".c", ".cpp", ".cu", ".cl"]:
        for f in kdir.rglob(f"*{ext}"):
            if f.is_file() and not f.name.startswith("."):
                file_names.append(str(f.relative_to(kdir)))
    return file_names


def main():
    parser = argparse.ArgumentParser(
        description="Baseline GPU-offload agent: single-session translation + optimization."
    )
    parser.add_argument("--source-api", default="serial", help="Source API (e.g. serial)")
    parser.add_argument("--target-api", default="omp", help="Target API (e.g. omp)")
    parser.add_argument("--codex-workdir", default=None, help="Codex CLI working directory")
    parser.add_argument("--output-dir", default=None, help="Directory for saved artifacts")
    parser.add_argument("--model", default=None, help="Model in opencode provider/model format")
    parser.add_argument(
        "--supervise",
        action="store_true",
        help="Run supervisor agent after baseline to verify correctness",
    )
    parser.add_argument(
        "--supervise-max-attempts",
        type=int,
        default=3,
        help="Max repair attempts for supervisor",
    )

    args = parser.parse_args()

    if args.codex_workdir:
        resolved = set_codex_workdir(args.codex_workdir)
        logger.info(f"Set CODEX_WORKDIR to: {resolved}")

    workdir = get_codex_workdir()
    data_src = default_data_src()
    golden_src = default_golden_root()

    output_dir = Path(args.output_dir) if args.output_dir else workdir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== ParaCodex Baseline Agent ===")
    logger.info(f"Translation: {args.source_api} → {args.target_api}")
    logger.info(f"Workdir:     {workdir}")
    logger.info(f"Golden src:  {golden_src}")
    logger.info(f"Data src:    {data_src}")
    logger.info(f"Output dir:  {output_dir}")
    logger.info(f"Supervisor:  {'enabled' if args.supervise else 'disabled'}")
    logger.info("")

    # Discover kernels from golden_labels/src
    suffix = f"-{args.source_api}"
    source_codes = []

    if golden_src.exists():
        matching_dirs = [d for d in golden_src.iterdir() if d.is_dir() and d.name.endswith(suffix)]
        for kdir in sorted(matching_dirs):
            kernel_name = kdir.name[: -len(suffix)]
            file_names = find_source_file_names(kdir)
            if file_names:
                source_codes.append({"kernel_name": kernel_name, "file_names": file_names})
                logger.info(f"Found kernel: {kernel_name} — files: {', '.join(file_names)}")

    if not source_codes:
        logger.error(f"No kernels found in {golden_src} matching pattern *-{args.source_api}")
        sys.exit(1)

    total = len(source_codes)
    succeeded = 0
    failed = 0

    for i, entry in enumerate(source_codes, 1):
        kernel_name = entry["kernel_name"]
        file_names = entry["file_names"]

        logger.info(f"\n--- [{i}/{total}] {kernel_name} ---")

        # Run baseline skill session
        result = run_baseline_for_kernel(
            kernel_name=kernel_name,
            file_names=file_names,
            target_api=args.target_api,
            source_api=args.source_api,
            data_src=data_src,
            golden_src=golden_src,
            output_dir=output_dir,
            model=args.model,
        )

        # Save artifacts regardless of success
        save_baseline_artifacts(
            kernel_name=kernel_name,
            file_names=file_names,
            target_api=args.target_api,
            data_src=data_src,
            output_dir=output_dir,
            result=result,
        )

        if result["success"]:
            succeeded += 1
            logger.success(f"[baseline] {kernel_name}: session completed")

            # Optionally run supervisor for correctness verification
            if args.supervise:
                logger.info(f"[supervisor] Starting correctness verification for {kernel_name}...")
                kernel_dir = data_src / f"{kernel_name}-{args.target_api}"
                golden_root = default_golden_root()
                sup_result = ensure_correctness(
                    kernel_dir=kernel_dir,
                    target_api=args.target_api,
                    original_name_hint=file_names,
                    golden_root=golden_root,
                    results_dir=output_dir,
                    phase="baseline_supervised",
                )
                sup_status = sup_result.get("status", "unknown")
                logger.info(f"[supervisor] {kernel_name}: {sup_status}")
        else:
            failed += 1
            logger.error(f"[baseline] {kernel_name}: session failed — {result['error']}")

    logger.info("")
    logger.info("=== Baseline Pipeline Complete ===")
    logger.info(f"Total: {total}  Succeeded: {succeeded}  Failed: {failed}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
