#!/usr/bin/env python3

import os
import sys
import argparse
import shutil
import datetime
import subprocess
from pathlib import Path

# Add the current directory to sys.path so we can import local modules
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from utils import parbench_utils

def main():
    parser = argparse.ArgumentParser(
        description="Setup pipeline working directory.",
        epilog="When --parbench_spec is given, --source_directory and --from_api are derived automatically from the spec."
    )
    parser.add_argument("--source_directory", "-s", required=False, default=None, help="Path to the source directory (derived from spec when --parbench_spec is used)")
    parser.add_argument("--from_api", "-f", required=False, default=None, help="Source API, e.g. serial, cuda (derived from spec when --parbench_spec is used)")
    parser.add_argument("--to_api", "-t", required=True, help="Target API (e.g., cuda)")
    parser.add_argument("--model", "-m", default=None, help="Model to use (opencode format: provider/model, Codex format: bare name)")
    parser.add_argument("--engine", "-e", default=None, choices=["opencode", "codex"],
                        help="Agentic engine to use (default: codex)")
    parser.add_argument("--supervise", action="store_true",
                        help="Run supervisor agent after optimization to verify and repair correctness")
    parser.add_argument("--baseline", action="store_true",
                        help="Use baseline mode: single-session translation+optimization (no multi-step pipeline)")
    parser.add_argument("--parbench_spec", default=None, help="Path to a ParBench JSON spec file")
    parser.add_argument("--yes", "-y", action="store_true", help="Automatically confirm and run the command without prompting")
    
    args = parser.parse_args()

    # If --parbench_spec is provided, derive source_directory and from_api from the spec
    if args.parbench_spec:
        import json
        try:
            with open(args.parbench_spec, "r") as f:
                spec_data = json.load(f)
        except Exception as e:
            print(f"Error: Cannot read ParBench spec file '{args.parbench_spec}': {e}")
            sys.exit(1)

        # Derive from_api from the spec if not explicitly provided
        if not args.from_api:
            args.from_api = spec_data.get("identity", {}).get("parallel_api", "serial")
            print(f"ParBench spec: auto-detected source API as '{args.from_api}'")

        # Derive source_directory from the spec if not explicitly provided
        if not args.source_directory:
            spec_source_path = spec_data.get("provenance", {}).get("source_path")
            if spec_source_path:
                # Use parbench_utils for robust resolution
                args.source_directory = parbench_utils.resolve_source_dir(spec_data, args.parbench_spec)
                
                if not args.source_directory:
                    # Last resort fallback (e.g. for display)
                    spec_path = Path(args.parbench_spec).resolve()
                    args.source_directory = str(spec_path.parent.parent / spec_source_path)
                print(f"ParBench spec: auto-detected source directory as '{args.source_directory}'")
            else:
                print("Error: --parbench_spec provided but spec has no 'provenance.source_path'. Please also pass --source_directory.")
                sys.exit(1)
    else:
        # Without a spec, both source_directory and from_api are required
        if not args.source_directory or not args.from_api:
            parser.error("--source_directory/-s and --from_api/-f are required unless --parbench_spec is provided")

    source_dir = Path(args.source_directory).resolve()
    from_api = args.from_api
    to_api = args.to_api
    
    # Get PARACODEX_ROOT from env or default
    paracodex_root = os.environ.get("PARACODEX_ROOT")
    if not paracodex_root:
        paracodex_root = "/root/codex_baseline/pipeline_refactored"
    
    paracodex_root_path = Path(paracodex_root).resolve()
    
    # Validate source directory
    if not source_dir.exists() or not source_dir.is_dir():
        print(f"Error: Source directory '{source_dir}' does not exist or is not a directory.")
        sys.exit(1)

    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Create workdir path
    # or mimicking the bash script which used fixed /root/codex_baseline
    base_dir = Path(os.environ.get("CODEX_BASE_DIR", "/root/codex_baseline"))
    kernel_name = source_dir.name.lower()
    run_label = f"{kernel_name}_{from_api}_to_{to_api}_baseline" if args.baseline else f"{kernel_name}_{from_api}_to_{to_api}"
    workdir_name = f"{run_label}_workdir_{timestamp}"
    workdir = base_dir / workdir_name

    # Create output directory with same pattern (replace workdir -> output)
    output_dir_name = f"{run_label}_output_{timestamp}"
    output_dir = base_dir / output_dir_name
    
    # Determine the PROGRAM_NAME by looking for the main source file
    program_name = source_dir.name
    src_files = list(source_dir.rglob("*.cu")) + list(source_dir.rglob("*.c")) + list(source_dir.rglob("*.cpp"))
    # filter out utilities
    src_files = [f for f in src_files if f.name not in ['timer.c', 'timer.h', 'util.c', 'util.h']]
    if len(src_files) == 1:
        program_name = src_files[0].stem
    elif len(src_files) > 1:
        # If there's a file with the same name as the directory, use it
        for f in src_files:
            if f.stem.lower() == source_dir.name.lower():
                program_name = f.stem
                break
        else:
            # Fallback to the first one or 'main'
            program_name = "main" if "main.c" in [f.name for f in src_files] or "main.cpp" in [f.name for f in src_files] else src_files[0].stem
            
    print(f"Determined PROGRAM_NAME as: {program_name}")
    
    print(f"Creating workdir: {workdir}")
    try:
        workdir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Error creating workdir: {e}")
        sys.exit(1)
        
    # Copy gate_sdk
    gate_sdk_src = paracodex_root_path / "gate_sdk"
    if gate_sdk_src.exists() and gate_sdk_src.is_dir():
        print("Copying gate_sdk...")
        try:
            shutil.copytree(gate_sdk_src, workdir / "gate_sdk")
        except Exception as e:
            print(f"Error copying gate_sdk: {e}")
            sys.exit(1)
    else:
        print(f"Error: gate_sdk not found at {gate_sdk_src}")
        sys.exit(1)

    # Copy system_info_summary.txt
    sys_info_src = paracodex_root_path / "scripts" / "system_info_summary.txt"
    if sys_info_src.exists() and sys_info_src.is_file():
        print("Copying system_info_summary.txt...")
        try:
            shutil.copy2(sys_info_src, workdir / "system_info_summary.txt")
        except Exception as e:
            print(f"Error copying system_info_summary.txt: {e}")
            sys.exit(1)
    else:
        print(f"Error: system_info_summary.txt not found at {sys_info_src}")
        sys.exit(1)

    # Copy Agents.md
    agents_md_src = paracodex_root_path / "AGENTS.md"
    if agents_md_src.exists() and agents_md_src.is_file():
        print("Copying Agents.md...")
        try:
            shutil.copy2(agents_md_src, workdir / "Agents.md")
        except Exception as e:
            print(f"Error copying Agents.md: {e}")
            sys.exit(1)
    else:
        print(f"Error: Agents.md not found at {agents_md_src}")
        sys.exit(1)
     
    # Setup golden_labels/src
    print("Setting up golden_labels/src...")
    golden_labels_src_dir = workdir / "golden_labels" / "src"
    try:
        golden_labels_src_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy the source directory into golden_labels/src
        # This keeps the original structure and content
        dest_dir_name = f"{source_dir.name.lower()}-{from_api}"
        golden_dest_path = golden_labels_src_dir / dest_dir_name
        if golden_dest_path.exists():
            shutil.rmtree(golden_dest_path)
        shutil.copytree(source_dir, golden_dest_path)
        
        # Also generate a valid Makefile in golden_labels/src matching the SOURCE API
        # This is needed so that check-correctness (which calls ref_build) can properly build the reference code
        # Only generate if it doesn't already exist (may have been copied from source directory)
        
        # ParBench support: if a spec file was provided, dump the prompt payload so the translation agent knows what files to use
        if args.parbench_spec:
            try:
                import json
                with open(args.parbench_spec, "r") as f:
                    spec_data = json.load(f)
                payload_files = spec_data.get("files", {}).get("prompt_payload", [])
                if payload_files:
                    payload_path = golden_dest_path / ".parbench_payload"
                    with open(payload_path, "w") as f:
                        f.write("\n".join(payload_files))
                    print(f"Created ParBench payload file with {len(payload_files)} files at {payload_path}")
                else:
                    print(f"Warning: No prompt_payload found in ParBench spec: {args.parbench_spec}")
                
                # Write .parbench_spec_path at the workdir root so downstream verify can find the spec
                spec_ref_path = workdir / ".parbench_spec_path"
                with open(spec_ref_path, "w") as f:
                    json.dump({
                        "spec": str(Path(args.parbench_spec).resolve()),
                        "to_api": to_api,
                        "parbench_root": str(Path(args.parbench_spec).resolve().parent.parent),
                    }, f, indent=2)
                print(f"Created .parbench_spec_path at {spec_ref_path}")
            except Exception as e:
                print(f"Error reading ParBench spec {args.parbench_spec}: {e}")
        ref_makefile_name = "Makefile.nvc"
        golden_makefile_path = golden_dest_path / ref_makefile_name
        if not golden_makefile_path.exists():
            try:
                # We import here to ensure the script is available/findable
                sys.path.append(str(Path(__file__).parent))
                from create_makefile import generate_makefile_content
                
                # For the reference makefile, ref_kernel_name is essentially self-referential
                golden_makefile_content = generate_makefile_content(from_api, dest_dir_name)
                golden_makefile_content = golden_makefile_content.replace("<PROGRAM_NAME>", program_name)
                
                # Use RUN_ARGS from spec if available, otherwise clear it or set default
                run_args_val = "10"
                if parbench_spec_data:
                    # In some ParBench specs, we might find default args
                    pass
                
                golden_makefile_content = golden_makefile_content.replace("<RUN_ARGS>", run_args_val)
                
                with open(golden_makefile_path, "w") as f:
                    f.write(golden_makefile_content)
                print(f"Generated Reference Makefile at {golden_makefile_path} with PROGRAM_NAME={program_name}, RUN_ARGS={run_args_val}")
                
            except Exception as e:
                 print(f"Error generating Reference Makefile: {e}")
                 pass
        else:
            print(f"{ref_makefile_name} already exists at {golden_makefile_path}, skipping generation")

    except Exception as e:
        print(f"Error setting up golden_labels/src: {e}")
        sys.exit(1)

    # Setup data/src and generate Makefile there
    print("Preparing data/src...")
    data_src_dir = workdir / "data" / "src"
    try:
        data_src_dir.mkdir(parents=True, exist_ok=True)
        
        # Create the kernel directory in data/src to hold the Makefile
        # The agent will copy source files here later
        data_kernel_dir = data_src_dir / f"{source_dir.name.lower()}-{to_api}"
        data_kernel_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate Makefile using create_makefile.py logic
        try:
            # We import here to ensure the script is available/findable
            # Append current script directory to sys.path just in case
            sys.path.append(str(Path(__file__).parent))
            from create_makefile import generate_makefile_content
            
            # Note: We pass dest_dir_name (e.g. ace-serial-serial) as ref_kernel_name
            # This ensures REF_DIR points to the correct location in golden_labels
            makefile_content = generate_makefile_content(to_api, dest_dir_name)
            
            makefile_name = "Makefile.nvc"
            makefile_path = data_kernel_dir / makefile_name
            with open(makefile_path, "w") as f:
                f.write(makefile_content)
            print(f"Generated Makefile at {makefile_path}")
            
        except Exception as e:
            print(f"Error generating Makefile: {e}")
            pass
            
    except Exception as e:
        print(f"Error creating data/src: {e}")
        sys.exit(1)

    # Create output directory
    print(f"Creating output directory: {output_dir}")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directory: {e}")
        sys.exit(1)
    
    print("Setup complete.")
    print(f"Workdir: {workdir}")
    print(f"Output directory: {output_dir}")
    
    # Get the actual script path for execution
    if args.baseline:
        agent_script = paracodex_root_path / "agents" / "baseline_codex.py"
    else:
        agent_script = paracodex_root_path / "agents" / "initial_translation_codex.py"
    actual_script_path = str(agent_script)

    # Build the command
    if args.baseline:
        cmd = [
            sys.executable,
            actual_script_path,
            "--source-api", from_api,
            "--target-api", to_api,
            "--codex-workdir", str(workdir),
            "--output-dir", str(output_dir),
        ]
        if args.supervise:
            cmd.append("--supervise")
        if args.model:
            cmd.extend(["--model", args.model])
    else:
        cmd = [
            sys.executable,
            actual_script_path,
            "--source-api", from_api,
            "--target-api", to_api,
            "--codex-workdir", str(workdir),
            "--output-dir", str(output_dir),
            "--optimize"
        ]
        if args.model:
            cmd.extend(["--model", args.model])
        # Supervisor: run after optimization step 2 (the final step)
        if args.supervise:
            cmd.extend(["--opt-supervisor-steps", "2"])

    # Propagate engine choice to child processes via env var
    if args.engine:
        os.environ["PIPELINE_ENGINE"] = args.engine

    # For display, use $PARACODEX_ROOT if available, otherwise use the actual path
    if os.environ.get("PARACODEX_ROOT"):
        display_script_path = f"$PARACODEX_ROOT/agents/{'baseline_codex.py' if args.baseline else 'initial_translation_codex.py'}"
    else:
        display_script_path = actual_script_path

    print("\n" + "="*70)
    print("Next step: Run the translation script with:")
    print("="*70)
    if args.baseline:
        cmd_str = f"python {display_script_path} --source-api {from_api} --target-api {to_api} --codex-workdir {workdir} --output-dir {output_dir}"
        if args.supervise:
            cmd_str += " --supervise"
    else:
        cmd_str = f"python {display_script_path} --source-api {from_api} --target-api {to_api} --codex-workdir {workdir} --output-dir {output_dir} --optimize"
        if args.supervise:
            cmd_str += " --opt-supervisor-steps 2"
    if args.model:
        cmd_str += f" --model {args.model}"
    if args.engine:
        cmd_str += f"  [engine: {args.engine}]"
    print(cmd_str)
    print("="*70)
    
    # Ask user if they want to run the command (unless --yes flag is set)
    if args.yes:
        print("\nAuto-confirming (--yes flag set)")
        print("Running command...")
        print("="*70)
        # Run the command
        result = subprocess.run(cmd, check=False)
        sys.exit(result.returncode)
    else:
        try:
            response = input("\nDo you want to run this command now? [Y/n]: ").strip().lower()
            if response == '' or response == 'y' or response == 'yes':
                print("\nRunning command...")
                print("="*70)
                # Run the command
                result = subprocess.run(cmd, check=False)
                sys.exit(result.returncode)
            else:
                print("Command not executed. You can run it manually later.")
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Command not executed.")
            sys.exit(1)

if __name__ == "__main__":
    main()
