#!/usr/bin/env python3
"""
Cleanup script to fix stuck/orphaned jobs in the database
Run this if jobs are stuck in "running" state after being killed manually
"""

import sqlite3
import os
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent / 'jobs.db'

def check_process_exists(pid):
    """Check if a process with given PID exists"""
    if pid is None:
        return False
    try:
        # Sending signal 0 doesn't kill the process, just checks if it exists
        os.kill(pid, 0)
        return True
    except (OSError, TypeError):
        return False

def cleanup_orphaned_jobs():
    """Find and fix jobs marked as running but whose processes are dead"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    # Find all running jobs
    c.execute("SELECT id, pid FROM jobs WHERE status = 'running'")
    running_jobs = c.fetchall()
    
    fixed_count = 0
    for job in running_jobs:
        job_id = job['id']
        pid = job['pid']
        
        if not check_process_exists(pid):
            # Process is dead, mark job as failed
            print(f"Found orphaned job: {job_id} (PID: {pid})")
            c.execute('''
                UPDATE jobs 
                SET status = 'failed', 
                    completed_at = ?, 
                    error = 'Process was terminated externally'
                WHERE id = ?
            ''', (datetime.now(), job_id))
            fixed_count += 1
            print(f"  ✓ Marked as failed")
    
    conn.commit()
    conn.close()
    
    if fixed_count > 0:
        print(f"\n✅ Fixed {fixed_count} orphaned job(s)")
    else:
        print("✓ No orphaned jobs found")

if __name__ == '__main__':
    print("Paracodex Job Cleanup")
    print("=" * 50)
    cleanup_orphaned_jobs()
