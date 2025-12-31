#!/usr/bin/env python3
"""
Script to kill all GPU processes that were launched from this WSL machine.
Uses nvidia-smi to find GPU processes and kills only those accessible from WSL.
"""

import subprocess
import os
import signal
import sys
import time
from typing import List, Optional


def get_gpu_processes() -> List[int]:
    """Get list of PIDs using the GPU."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            pids = []
            for line in result.stdout.strip().split('\n'):
                line = line.strip()
                if line:
                    try:
                        pids.append(int(line))
                    except ValueError:
                        pass
            return pids
    except subprocess.TimeoutExpired:
        print("Error: nvidia-smi command timed out")
    except FileNotFoundError:
        print("Error: nvidia-smi not found. Is NVIDIA driver installed?")
    except Exception as e:
        print(f"Error getting GPU processes: {e}")
    return []


def is_process_accessible(pid: int) -> bool:
    """Check if a process is accessible from this WSL instance."""
    try:
        # Try to send signal 0 (doesn't kill, just checks if process exists and is accessible)
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        # Process doesn't exist
        return False
    except PermissionError:
        # Process exists but we don't have permission (likely Windows process)
        return False
    except Exception:
        return False


def kill_gpu_processes(exclude_pids: Optional[List[int]] = None) -> tuple[int, int, int]:
    """
    Kill all GPU processes accessible from this WSL machine.
    
    Returns:
        (total_found, killed_count, failed_count)
    """
    exclude_pids = exclude_pids or []
    
    print("Finding GPU processes...")
    gpu_pids = get_gpu_processes()
    
    if not gpu_pids:
        print("No GPU processes found.")
        return 0, 0, 0
    
    print(f"Found GPU processes: {gpu_pids}")
    print()
    
    total = len(gpu_pids)
    killed = 0
    failed = 0
    
    for pid in gpu_pids:
        if pid in exclude_pids:
            print(f"Skipping PID {pid} (excluded)")
            continue
        
        if not is_process_accessible(pid):
            print(f"  - PID {pid} not accessible from WSL (likely a Windows process)")
            continue
        
        print(f"Killing PID {pid}...", end=" ")
        try:
            os.kill(pid, signal.SIGKILL)
            killed += 1
            print("✓ Successfully killed")
        except ProcessLookupError:
            print("- Process already terminated")
        except PermissionError:
            failed += 1
            print("✗ Permission denied")
        except Exception as e:
            failed += 1
            print(f"✗ Failed: {e}")
    
    return total, killed, failed


def main():
    """Main entry point."""
    print("=" * 60)
    print("GPU Process Killer for WSL")
    print("=" * 60)
    print()
    
    total, killed, failed = kill_gpu_processes()
    
    print()
    print("Summary:")
    print(f"  Total GPU processes found: {total}")
    print(f"  Successfully killed: {killed}")
    print(f"  Failed to kill: {failed}")
    
    if killed > 0:
        print()
        print("Waiting 2 seconds and checking for remaining processes...")
        time.sleep(2)
        remaining = get_gpu_processes()
        if remaining:
            print(f"Warning: {len(remaining)} GPU process(es) still running:")
            print(f"  {remaining}")
            print("  (These may be Windows processes not accessible from WSL)")
        else:
            print("All WSL GPU processes have been terminated.")
    
    print()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

