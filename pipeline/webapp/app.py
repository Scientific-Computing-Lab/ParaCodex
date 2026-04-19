#!/usr/bin/env python3
"""
Flask web application for Paracodex Pipeline
Provides a modern web interface to configure and run code translation pipelines
"""

import os
import sys
import json
import sqlite3
import subprocess
import uuid
import threading
import time
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from flask_cors import CORS

# Add parent directory to path to import pipeline modules
PIPELINE_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PIPELINE_ROOT))

# Now we can import local modules
from utils import parbench_utils

app = Flask(__name__, static_folder='static', template_folder='static')
CORS(app)

# Configuration
DB_PATH = Path(__file__).parent / 'jobs.db'
JOBS = {}  # In-memory job tracking
LOGS = {}  # In-memory log storage
LOGS_DIR = Path(__file__).parent.parent / 'logs'
LOGS_DIR.mkdir(parents=True, exist_ok=True)

def init_db():
    """Initialize SQLite database for job tracking"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            source_directory TEXT NOT NULL,
            from_api TEXT NOT NULL,
            to_api TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            workdir TEXT,
            output_dir TEXT,
            pid INTEGER,
            exit_code INTEGER,
            error TEXT
        )
    ''')
    conn.commit()
    # Check if model column exists, if not add it
    try:
        c = conn.cursor()
        c.execute('SELECT model FROM jobs LIMIT 1')
    except sqlite3.OperationalError:
        c.execute('ALTER TABLE jobs ADD COLUMN model TEXT')
        conn.commit()
    # Check if parbench_spec column exists, if not add it
    try:
        c.execute('SELECT parbench_spec FROM jobs LIMIT 1')
    except sqlite3.OperationalError:
        c.execute('ALTER TABLE jobs ADD COLUMN parbench_spec TEXT')
        conn.commit()
    # Check if engine column exists, if not add it
    try:
        c.execute('SELECT engine FROM jobs LIMIT 1')
    except sqlite3.OperationalError:
        c.execute('ALTER TABLE jobs ADD COLUMN engine TEXT')
        conn.commit()
    # kernel_name column
    try:
        c.execute('SELECT kernel_name FROM jobs LIMIT 1')
    except sqlite3.OperationalError:
        c.execute('ALTER TABLE jobs ADD COLUMN kernel_name TEXT')
        conn.commit()
    # traces column (JSON array of trace filenames created during this job)
    try:
        c.execute('SELECT traces FROM jobs LIMIT 1')
    except sqlite3.OperationalError:
        c.execute('ALTER TABLE jobs ADD COLUMN traces TEXT')
        conn.commit()
    # supervise column
    try:
        c.execute('SELECT supervise FROM jobs LIMIT 1')
    except sqlite3.OperationalError:
        c.execute('ALTER TABLE jobs ADD COLUMN supervise INTEGER DEFAULT 0')
        conn.commit()
    # baseline column
    try:
        c.execute('SELECT baseline FROM jobs LIMIT 1')
    except sqlite3.OperationalError:
        c.execute('ALTER TABLE jobs ADD COLUMN baseline INTEGER DEFAULT 0')
        conn.commit()
    conn.close()
    recover_stale_jobs()

def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def recover_stale_jobs():
    """On startup, detect 'running' jobs whose process died and recover their state."""
    import glob as _glob
    base_dir = Path(os.environ.get('CODEX_BASE_DIR', '/root/codex_baseline'))
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT id, pid, from_api, to_api, workdir, output_dir FROM jobs WHERE status = 'running'")
    stale = [dict(r) for r in c.fetchall()]
    conn.close()

    for job in stale:
        job_id = job['id']
        pid = job['pid']

        # Check if process is still alive
        process_alive = False
        if pid:
            try:
                os.kill(pid, 0)
                process_alive = True
            except (ProcessLookupError, PermissionError, OSError):
                pass

        if process_alive:
            continue  # Still running — leave it

        # Process is dead. Try to recover workdir/output_dir from filesystem.
        workdir = job['workdir']
        output_dir = job['output_dir']

        if not workdir:
            from_api = job['from_api']
            to_api = job['to_api']
            pattern = str(base_dir / f'custom_{from_api}_to_{to_api}_workdir_*')
            matches = sorted(_glob.glob(pattern), key=os.path.getmtime, reverse=True)
            if matches:
                workdir = matches[0]

        if not output_dir and workdir:
            output_dir = workdir.replace('_workdir_', '_output_')
            if not Path(output_dir).exists():
                output_dir = None

        # Determine status from output directory contents
        status = 'failed'
        if output_dir and Path(output_dir).exists():
            try:
                if any(Path(output_dir).iterdir()):
                    status = 'completed'
            except Exception:
                pass

        conn = get_db_connection()
        c = conn.cursor()
        c.execute('''
            UPDATE jobs SET status = ?, completed_at = ?, workdir = ?, output_dir = ?
            WHERE id = ? AND status = 'running'
        ''', (status, datetime.now().isoformat(), workdir, output_dir, job_id))
        conn.commit()
        conn.close()
        print(f"[recovery] Job {job_id[:8]}: stale process recovered → {status}, workdir={workdir}")

@app.route('/')
def index():
    """Serve the main web interface"""
    return send_from_directory('static', 'index.html')

@app.route('/api/browse-directory')
def browse_directory():
    """API endpoint to browse directories on the server"""
    import os
    default_path = str(Path.home())
    path = request.args.get('path', default_path)
    
    try:
        target_path = Path(path).resolve()
        
        # Security: ensure path is within allowed directories
        # Allow home directory, common Unix roots, and Windows Users dirs
        home = Path.home()
        allowed_roots = [
            home,
            Path('/home'),
            Path('/root'),
            Path('/Users'),  # macOS
            Path('C:/Users'),  # Windows
        ]
        is_allowed = any(
            str(target_path).startswith(str(root))
            for root in allowed_roots
        )
        # Also allow if target_path IS one of the roots (e.g. /home itself)
        if not is_allowed:
            is_allowed = any(
                root == target_path or str(target_path) == str(root)
                for root in allowed_roots
            )
        
        if not is_allowed:
            return jsonify({'error': 'Access denied to this directory'}), 403
        
        if not target_path.exists() or not target_path.is_dir():
            return jsonify({'error': 'Directory not found'}), 404
        
        # Get directory contents
        items = []
        try:
            for item in sorted(target_path.iterdir()):
                # Skip hidden files and certain directories
                if item.name.startswith('.') or item.name in ['__pycache__', 'node_modules']:
                    continue
                    
                items.append({
                    'name': item.name,
                    'path': str(item),
                    'is_dir': item.is_dir(),
                    'size': item.stat().st_size if item.is_file() else None
                })
        except PermissionError:
            return jsonify({'error': 'Permission denied'}), 403
        
        # Get parent directory
        parent = str(target_path.parent) if target_path.parent != target_path else None
        
        return jsonify({
            'current_path': str(target_path),
            'parent': parent,
            'items': items
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/start-pipeline', methods=['POST'])
def start_pipeline():
    """Start a new pipeline job"""
    data = request.json
    
    # Check if a ParBench spec was provided
    parbench_spec = data.get('parbench_spec')
    
    if parbench_spec:
        # Load from ParBench Spec
        try:
            with open(parbench_spec, "r") as f:
                spec_data = json.load(f)
            from_api = spec_data.get("identity", {}).get("parallel_api", "serial")
            # Target API must be provided by user or use a default (from UI) length
            to_api = data.get('to_api')
            # Extract source directory path from spec
            spec_source_path = spec_data.get("provenance", {}).get("source_path")
            
            if spec_source_path:
                # Resolve relative to the spec file's parent directory (or the parbench root)
                # Typically specs are in `parbench/specs/X.json` and src in `parbench/src/X`
                # So if source_path is "src/kernel", it's relative to the `parbench` root
                spec_path = Path(parbench_spec).resolve()
                
                # Check if the spec_source_path is an absolute path
                if Path(spec_source_path).is_absolute():
                     source_dir = spec_source_path
                else:
                    # Use parbench_utils for robust resolution
                    resolved = parbench_utils.resolve_source_dir(spec_data, parbench_spec)
                    
                    if resolved:
                        source_dir = resolved
                    else:
                        # Fallback for UI display even if not found yet
                        source_dir = str(spec_path.parent.parent / spec_source_path)
            else:
                 source_dir = f"ParBench: {spec_data.get('identity', {}).get('source_suite', 'unknown')}-{spec_data.get('identity', {}).get('kernel_name', 'unknown')}"
        except Exception as e:
            return jsonify({'error': f'Failed to parse ParBench spec: {str(e)}'}), 400
    else:
        # Standard input
        source_dir = data.get('source_directory')
        from_api = data.get('from_api')
        to_api = data.get('to_api')
        
    model = data.get('model')
    engine = data.get('engine') or None
    if engine not in ('opencode', 'codex', None):
        engine = None
    supervise = bool(data.get('supervise', False))
    baseline = bool(data.get('baseline', False))

    # Extract kernel_name from parbench spec or directory name pattern {kernel_name}-{api}
    kernel_name = None
    if parbench_spec:
        try:
            with open(parbench_spec) as _f:
                _spec = json.load(_f)
            kernel_name = (_spec.get('identity', {}).get('kernel_name')
                           or _spec.get('identity', {}).get('name'))
        except Exception:
            pass
    if not kernel_name and source_dir:
        _dn = Path(source_dir).name
        for _api in ('serial', 'omp', 'cuda', 'ocl', 'acc', 'hip', 'sycl'):
            if _dn.endswith(f'-{_api}'):
                kernel_name = _dn[:-len(f'-{_api}')]
                break
        if not kernel_name:
            kernel_name = _dn

    if not parbench_spec and not all([source_dir, from_api, to_api]):
        return jsonify({'error': 'Missing required parameters'}), 400
    
    # Validate source directory exists if not using ParBench
    if not parbench_spec:
        source_path = Path(source_dir)
        if not source_path.exists() or not source_path.is_dir():
            return jsonify({'error': f'Source directory not found: {source_dir}'}), 400
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Create job record in database
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        INSERT INTO jobs (id, source_directory, from_api, to_api, model, parbench_spec, engine, kernel_name, supervise, baseline, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (job_id, source_dir, from_api, to_api, model, parbench_spec, engine, kernel_name, int(supervise), int(baseline), 'running'))
    conn.commit()
    conn.close()

    # Initialize log storage
    LOGS[job_id] = []

    # Start pipeline in background thread
    thread = threading.Thread(
        target=run_pipeline_job,
        args=(job_id, source_dir, from_api, to_api, model, parbench_spec, engine, kernel_name, supervise, baseline)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'job_id': job_id,
        'status': 'running',
        'message': 'Pipeline job started successfully'
    })

def run_pipeline_job(job_id, source_dir, from_api, to_api, model=None, parbench_spec=None, engine=None, kernel_name=None, supervise=False, baseline=False):
    """Run the pipeline job in a background thread"""
    try:
        _traces_dir = PIPELINE_ROOT / 'traces'

        # Log start
        log_message(job_id, f"Starting pipeline job {job_id}")
        log_message(job_id, f"Source: {source_dir}")
        log_message(job_id, f"Translation: {from_api} -> {to_api}")
        active_engine = engine or 'codex'
        log_message(job_id, f"Engine: {active_engine}")
        if baseline:
            log_message(job_id, "Mode: baseline (single-session translation + optimization)")
        if supervise:
            log_message(job_id, "Supervisor: enabled (correctness verification after baseline)")
        if model:
            log_message(job_id, f"Model: {model}")
        log_message(job_id, "")
        
        # Run setup_pipeline_workdir.py
        setup_script = PIPELINE_ROOT / 'setup_pipeline_workdir.py'
        
        log_message(job_id, "Running setup_pipeline_workdir.py...")
        
        # Prepare command args
        cmd_args = [
            sys.executable,
            str(setup_script),
            '-f', from_api,
            '-t', to_api,
            '--yes'  # Auto-confirm without prompting
        ]
        
        if parbench_spec:
            cmd_args.extend(['-s', source_dir])
            cmd_args.extend(['--parbench_spec', parbench_spec])
        else:
            cmd_args.extend(['-s', source_dir])
        
        if model:
            cmd_args.extend(['--model', model])

        if engine:
            cmd_args.extend(['--engine', engine])

        if baseline:
            cmd_args.append('--baseline')

        if supervise:
            cmd_args.append('--supervise')

        # Record timestamp just before the subprocess starts so we can
        # identify only the traces it creates (avoids cross-job contamination
        # when multiple jobs run concurrently).
        _job_start_time = time.time()

        # Create process with pipes to capture output
        # Use --yes flag to skip interactive prompt
        process = subprocess.Popen(
            cmd_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(PIPELINE_ROOT),
            preexec_fn=os.setpgrp  # Create new process group
        )
        
        # Store PID
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('UPDATE jobs SET pid = ? WHERE id = ?', (process.pid, job_id))
        conn.commit()
        conn.close()
        
        # Variables to capture workdir and output_dir
        workdir = None
        output_dir = None
        
        # Read output line by line
        for line in process.stdout:
            line = line.rstrip()
            log_message(job_id, line)
            
            # Capture workdir and output_dir from logs and persist immediately
            if line.startswith('Workdir: '):
                workdir = line.replace('Workdir: ', '').strip()
                try:
                    _c = get_db_connection()
                    _c.execute('UPDATE jobs SET workdir = ? WHERE id = ?', (workdir, job_id))
                    _c.commit()
                    _c.close()
                except Exception:
                    pass
            elif line.startswith('Output directory: '):
                output_dir = line.replace('Output directory: ', '').strip()
                try:
                    _c = get_db_connection()
                    _c.execute('UPDATE jobs SET output_dir = ? WHERE id = ?', (output_dir, job_id))
                    _c.commit()
                    _c.close()
                except Exception:
                    pass
        
        # Wait for completion
        exit_code = process.wait()

        # Find traces created during this job by checking mtime against the
        # timestamp recorded just before the subprocess started.
        _job_end_time = time.time()
        _job_traces = []
        if _traces_dir.exists():
            for _f in _traces_dir.glob('*.jsonl'):
                if 'prompt' not in _f.name:
                    try:
                        _mtime = _f.stat().st_mtime
                        if _job_start_time <= _mtime <= _job_end_time:
                            _job_traces.append(_f.name)
                    except OSError:
                        pass
        _job_traces = sorted(_job_traces)

        # Check if job was killed (don't overwrite 'killed' with 'failed')
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('SELECT status FROM jobs WHERE id = ?', (job_id,))
        row = c.fetchone()
        current_status = row['status'] if row else None

        _traces_json = json.dumps(_job_traces) if _job_traces else None

        # Only update status if it's not already killed
        if current_status != 'killed':
            status = 'completed' if exit_code == 0 else 'failed'
            error = None if exit_code == 0 else f"Process exited with code {exit_code}"

            c.execute('''
                UPDATE jobs
                SET status = ?, completed_at = ?, exit_code = ?, error = ?, workdir = ?, output_dir = ?, traces = ?
                WHERE id = ?
            ''', (status, datetime.now(), exit_code, error, workdir, output_dir, _traces_json, job_id))
            conn.commit()

        else:
            # Just update metadata for killed job
            c.execute('''
                UPDATE jobs
                SET exit_code = ?, workdir = ?, output_dir = ?, traces = ?
                WHERE id = ?
            ''', (exit_code, workdir, output_dir, _traces_json, job_id))
            conn.commit()

        conn.close()
        
        log_message(job_id, "")
        if exit_code == 0:
            log_message(job_id, "✓ Pipeline completed successfully!")
            log_message(job_id, f"Workdir: {workdir}")
            log_message(job_id, f"Output: {output_dir}")
        else:
            log_message(job_id, f"✗ Pipeline failed with exit code {exit_code}")
    
    except Exception as e:
        # Update job as failed
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('''
            UPDATE jobs 
            SET status = 'failed', completed_at = ?, error = ?
            WHERE id = ?
        ''', (datetime.now(), str(e), job_id))
        conn.commit()
        conn.close()
        
        log_message(job_id, f"ERROR: {str(e)}")

def log_message(job_id, message):
    """Add a log message for a job"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    log_entry = {
        'timestamp': timestamp,
        'message': message
    }
    if job_id not in LOGS:
        LOGS[job_id] = []
    LOGS[job_id].append(log_entry)
    # Also persist to disk
    try:
        with open(LOGS_DIR / f'{job_id}.log', 'a', encoding='utf-8') as _f:
            _f.write(f'[{timestamp}] {message}\n')
    except Exception:
        pass

@app.route('/api/job-status/<job_id>')
def get_job_status(job_id):
    """Get the current status of a job"""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM jobs WHERE id = ?', (job_id,))
    row = c.fetchone()
    conn.close()
    
    if not row:
        return jsonify({'error': 'Job not found'}), 404
    
    job = dict(row)
    return jsonify(job)


@app.route('/api/parbench-verify/<job_id>', methods=['POST'])
def parbench_verify_job(job_id):
    """Run ParBench verification on a completed job's translated output."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM jobs WHERE id = ?', (job_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return jsonify({'error': 'Job not found'}), 404

    job = dict(row)
    parbench_spec = job.get('parbench_spec')
    output_dir = job.get('output_dir')
    to_api = job.get('to_api')

    if not parbench_spec:
        return jsonify({'error': 'This job was not started with a ParBench spec — nothing to verify'}), 400
    if not output_dir or not Path(output_dir).exists():
        return jsonify({'error': f'Output directory not found: {output_dir}'}), 400

    # Find translated kernel directories inside output_dir
    output_path = Path(output_dir)
    # The actual output might be structured as output_dir/<kernel_name>
    # We ignore 'data', 'logs', etc.
    kernel_dirs = [d for d in output_path.iterdir() if d.is_dir() and d.name not in ['data', 'logs', '.gemini']]
    if not kernel_dirs:
        return jsonify({'error': 'No kernel output directories found in output_dir'}), 400

    verify_script = PIPELINE_ROOT / 'parbench_verify.py'
    parbench_root = str(Path(parbench_spec).resolve().parent.parent)
    body = request.get_json(silent=True) or {}
    config = body.get('config', 'correctness')

    results = []
    for kernel_dir in kernel_dirs:
        # The true translated dir is typically in data/src/<kernel_name> or <kernel_name>/step2
        # Let's find the deepest directory with actual source files
        true_translated_dir = kernel_dir
        
        # Check standard pipeline output path: data/src/kernel_name
        data_src_path = output_path / 'data' / 'src' / kernel_dir.name
        if data_src_path.exists() and data_src_path.is_dir():
            true_translated_dir = data_src_path
        else:
            # Check for step2 output
            step2_path = kernel_dir / 'step2'
            if step2_path.exists() and step2_path.is_dir():
                true_translated_dir = step2_path
                
        cmd = [
            sys.executable, str(verify_script),
            '--parbench-spec', parbench_spec,
            '--translated-dir', str(true_translated_dir),
            '--to-api', to_api,
            '--config', config,
            '--parbench-root', parbench_root,
            '--json-out',
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            verify_result = {'kernel': kernel_dir.name, 'status': 'unknown', 'stdout': proc.stdout[-3000:]}
            json_start = (proc.stdout or '').rfind('{')
            if json_start >= 0:
                try:
                    verify_result.update(json.loads(proc.stdout[json_start:]))
                except json.JSONDecodeError:
                    pass
            verify_result['status'] = 'pass' if proc.returncode == 0 else 'fail'
            if proc.stderr:
                verify_result['stderr'] = proc.stderr[-1000:]
        except subprocess.TimeoutExpired:
            verify_result = {'kernel': kernel_dir.name, 'status': 'timeout'}
        except Exception as e:
            verify_result = {'kernel': kernel_dir.name, 'status': 'error', 'error': str(e)}
        results.append(verify_result)

    passed = sum(1 for r in results if r.get('status') == 'pass')
    return jsonify({
        'job_id': job_id,
        'parbench_spec': parbench_spec,
        'config': config,
        'to_api': to_api,
        'results': results,
        'summary': {
            'total': len(results),
            'passed': passed,
            'failed': len(results) - passed,
        }
    })

@app.route('/api/logs/<job_id>')
def stream_logs(job_id):
    """Stream logs for a job using Server-Sent Events"""
    def generate():
        last_index = 0
        max_wait = 300  # 5 minutes max
        start_time = time.time()
        
        while True:
            # Check if job exists
            conn = get_db_connection()
            c = conn.cursor()
            c.execute('SELECT status FROM jobs WHERE id = ?', (job_id,))
            row = c.fetchone()
            conn.close()
            
            if not row:
                yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                break
            
            status = row['status']
            
            # Send new log entries
            if job_id in LOGS:
                logs = LOGS[job_id]
                if last_index < len(logs):
                    new_logs = logs[last_index:]
                    for log in new_logs:
                        yield f"data: {json.dumps(log)}\n\n"
                    last_index = len(logs)
            
            # If job is complete, send final message and stop
            if status in ['completed', 'failed']:
                yield f"data: {json.dumps({'status': status, 'done': True})}\n\n"
                break
            
            # Check timeout
            if time.time() - start_time > max_wait:
                yield f"data: {json.dumps({'error': 'Timeout', 'done': True})}\n\n"
                break
            
            time.sleep(0.5)
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/jobs')
def list_jobs():
    """List all jobs"""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM jobs ORDER BY created_at DESC LIMIT 50')
    rows = c.fetchall()
    conn.close()
    
    jobs = [dict(row) for row in rows]
    return jsonify(jobs)

@app.route('/api/jobs/<job_id>', methods=['DELETE'])
def delete_job(job_id):
    """Delete a job record (cannot delete running jobs)"""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT status FROM jobs WHERE id = ?', (job_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return jsonify({'error': 'Job not found'}), 404
    if row['status'] == 'running':
        conn.close()
        return jsonify({'error': 'Cannot delete a running job — kill it first'}), 400
    c.execute('DELETE FROM jobs WHERE id = ?', (job_id,))
    conn.commit()
    conn.close()
    LOGS.pop(job_id, None)
    return jsonify({'success': True})


@app.route('/api/recover-job/<job_id>', methods=['POST'])
def recover_job(job_id):
    """Manually set workdir/output_dir/status for a stuck job."""
    data = request.get_json(silent=True) or {}
    workdir = data.get('workdir')
    output_dir = data.get('output_dir')
    status = data.get('status', 'completed')
    traces_list = data.get('traces')  # optional list of trace filenames

    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT id FROM jobs WHERE id = ?', (job_id,))
    if not c.fetchone():
        conn.close()
        return jsonify({'error': 'Job not found'}), 404

    fields, vals = [], []
    if workdir is not None:
        fields.append('workdir = ?'); vals.append(workdir)
    if output_dir is not None:
        fields.append('output_dir = ?'); vals.append(output_dir)
    if status:
        fields.append('status = ?'); vals.append(status)
        fields.append('completed_at = ?'); vals.append(datetime.now().isoformat())
    if traces_list is not None:
        fields.append('traces = ?'); vals.append(json.dumps(traces_list))

    if fields:
        c.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?", vals + [job_id])
        conn.commit()
    conn.close()
    return jsonify({'success': True})


@app.route('/api/active-jobs')
def get_active_jobs():
    """Return list of all jobs (running + recent completed/failed)"""
    conn = get_db_connection()
    c = conn.cursor()
    # Get jobs from last 24 hours, ordered by most recent first
    c.execute('''
        SELECT id, source_directory, from_api, to_api, model, engine, status,
               created_at, completed_at, exit_code
        FROM jobs
        WHERE created_at > datetime('now', '-1 day')
        ORDER BY created_at DESC
        LIMIT 20
    ''')
    rows = c.fetchall()
    conn.close()
    
    jobs = [dict(row) for row in rows]
    
    # Add 'is_active' flag for jobs that have running processes
    active_jobs = list(JOBS.keys())
    for job in jobs:
        job['is_active'] = job['id'] in active_jobs
    
    return jsonify({'jobs': jobs})

@app.route('/api/config')
def get_config():
    """Get configuration information"""
    return jsonify({
        'default_source_dir': str(Path.home()),
        'apis': ['serial', 'omp', 'cuda', 'ocl', 'acc', 'hip', 'sycl'],
        'pipeline_root': str(PIPELINE_ROOT)
    })

# Pipeline stages in order
PIPELINE_STAGES = [
    {'id': 'analysis', 'name': 'Analysis', 'description': 'Analyzing source code'},
    {'id': 'translation', 'name': 'Translation', 'description': 'Initial translation'},
    {'id': 'optimization', 'name': 'Optimization', 'description': 'Optimizing code'},
    {'id': 'supervision', 'name': 'Verification', 'description': 'Running correctness checks'},
]

def detect_current_stage(logs):
    """Detect current stage from log messages"""
    if not logs:
        return 'analysis'
    
    # Join all log messages
    all_text = ' '.join([l.get('message', '') for l in logs if isinstance(l, dict)])
    
    # Check in reverse order of stages
    if 'Supervisor' in all_text or 'correctness' in all_text.lower():
        return 'supervision'
        
    # Optimization detection - be specific to avoid matching "Optimize enabled" config log
    if 'Running step' in all_text or 'Optimization Phase' in all_text:
        return 'optimization'
        
    if 'Translation Phase' in all_text or 'Translating' in all_text:
        return 'translation'
        
    # Default to analysis if we have logs but haven't hit the above
    return 'analysis'

def _extract_trace_cwd(filepath):
    """Read the session_meta record from a trace file and return its cwd, or None."""
    import ast as _ast
    try:
        with open(filepath, encoding='utf-8', errors='replace') as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line:
                    continue
                try:
                    _d = json.loads(_line)
                except json.JSONDecodeError:
                    continue
                if _d.get('type') == 'session_meta':
                    _payload = _d.get('payload', {})
                    if isinstance(_payload, str):
                        try:
                            _payload = _ast.literal_eval(_payload)
                        except Exception:
                            _payload = {}
                    if isinstance(_payload, dict):
                        return _payload.get('cwd') or None
    except Exception:
        pass
    return None


def _extract_trace_label(filepath):
    """Read the first user message from a trace file to determine its stage label."""
    import re as _re, ast as _ast
    try:
        with open(filepath, encoding='utf-8', errors='replace') as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line:
                    continue
                try:
                    _d = json.loads(_line)
                except json.JSONDecodeError:
                    continue
                _payload = _d.get('payload', {})
                if isinstance(_payload, str):
                    try:
                        _payload = _ast.literal_eval(_payload)
                    except Exception:
                        _payload = {}
                _dtype = _d.get('type', '')
                _ptype = _payload.get('type', '') if isinstance(_payload, dict) else ''
                if _dtype == 'event_msg' and _ptype == 'user_message':
                    _msg = str(_payload.get('message', ''))
                    # Match "Task: {apis} {stage} for kernel ..."
                    _m = _re.match(r'Task:\s+\S+\s+->\s+\S+\s+(.+?)(?:\s+for\s+|\s*\n|$)', _msg, _re.IGNORECASE)
                    if _m:
                        _stage = _m.group(1).strip()
                        if 'analysis' in _stage.lower():
                            return 'Analysis'
                        _m2 = _re.search(r'optimization step\s*(\d+)', _stage, _re.IGNORECASE)
                        if _m2:
                            return f'Opt Step {_m2.group(1)}'
                        if 'translation' in _stage.lower():
                            return 'Translation'
                        if 'supervisor' in _stage.lower() or 'supervise' in _stage.lower():
                            return 'Supervisor'
                        return _stage[:30]
                    # Fallback heuristics
                    _ml = _msg.lower()
                    if 'previous step transcript summary' in _ml:
                        return 'Opt Step 2'
                    if 'analysis' in _ml[:100]:
                        return 'Analysis'
                    return _msg.split('\n')[0].strip()[:30] or None
    except Exception:
        pass
    return None


def find_artifacts(job_id, workdir=None, output_dir=None):
    """Find markdown artifacts in workdir and output directories"""
    artifacts = []
    
    # Get job info for directories
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT workdir, output_dir FROM jobs WHERE id = ?', (job_id,))
    row = c.fetchone()
    conn.close()
    
    if not row:
        return artifacts
    
    workdir = row['workdir'] or workdir
    output_dir = row['output_dir'] or output_dir
    
    # Search for .md files in both directories
    search_dirs = []
    if workdir and Path(workdir).exists():
        search_dirs.append(Path(workdir))
    if output_dir and Path(output_dir).exists():
        search_dirs.append(Path(output_dir))
    
    seen_files = set()
    for search_dir in search_dirs:
        for md_file in search_dir.rglob('*.md'):
            # Skip certain files
            if md_file.name in ['README.md', 'AGENTS.md', 'Agents.md']:
                continue
            
            # Deduplicate by filename (keep first occurrence)
            if md_file.name in seen_files:
                continue
            seen_files.add(md_file.name)
            
            # Determine artifact type based on filename
            artifact_type = 'document'
            if 'analysis' in md_file.name.lower():
                artifact_type = 'analysis'
            elif 'plan' in md_file.name.lower():
                artifact_type = 'plan'
            elif 'report' in md_file.name.lower() or 'gate' in md_file.name.lower():
                artifact_type = 'report'
            
            artifacts.append({
                'name': md_file.name,
                'path': str(md_file),
                'type': artifact_type,
                'size': md_file.stat().st_size,
                'modified': md_file.stat().st_mtime
            })
    
    # Sort by modification time (newest first)
    artifacts.sort(key=lambda x: x['modified'], reverse=True)
    return artifacts

@app.route('/api/job-progress/<job_id>')
def get_job_progress(job_id):
    """Get detailed job progress including current stage and artifacts"""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM jobs WHERE id = ?', (job_id,))
    row = c.fetchone()
    conn.close()
    
    if not row:
        return jsonify({'error': 'Job not found'}), 404
    
    job = dict(row)
    
    # Get current stage from logs
    logs = LOGS.get(job_id, [])
    current_stage = detect_current_stage(logs)
    
    # Find stage index
    stage_index = 0
    for i, stage in enumerate(PIPELINE_STAGES):
        if stage['id'] == current_stage:
            stage_index = i
            break
    
    # Get artifacts
    artifacts = find_artifacts(job_id)
    
    # Build response
    return jsonify({
        'job': job,
        'stages': PIPELINE_STAGES,
        'current_stage': current_stage,
        'stage_index': stage_index,
        'artifacts': artifacts,
        'log_count': len(logs)
    })

@app.route('/api/artifact/<job_id>/<path:artifact_path>')
def get_artifact(job_id, artifact_path):
    """Get artifact content by path"""
    print(f"[get_artifact] Job ID: {job_id}, Raw path: {repr(artifact_path)}")
    
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT workdir, output_dir FROM jobs WHERE id = ?', (job_id,))
    row = c.fetchone()
    conn.close()
    
    if not row:
        print(f"[get_artifact] Job {job_id} not found in database")
        return jsonify({'error': 'Job not found'}), 404
    
    print(f"[get_artifact] workdir={row['workdir']}, output_dir={row['output_dir']}")
    
    # Build list of paths to try
    possible_paths = []
    
    # helper to normalize path
    def normalize_path(p_str):
        # If it looks like an absolute path but missing leading slash (common issue with URL decoding)
        if not p_str.startswith('/') and (p_str.startswith('root/') or p_str.startswith('home/') or p_str.startswith('Users/')):
            return '/' + p_str
        return p_str

    normalized_artifact_path = normalize_path(artifact_path)
    artifact_abs = Path(normalized_artifact_path)
    
    print(f"[get_artifact] Normalized path: {normalized_artifact_path} (is_absolute={artifact_abs.is_absolute()})")

    # If artifact_path is absolute (or we made it absolute), try it first
    if artifact_abs.is_absolute():
        possible_paths.append(artifact_abs)
        print(f"[get_artifact] Added absolute path: {artifact_abs}")
        
        # If the absolute path contains workdir, also try substituting with output_dir
        if row['workdir'] and row['output_dir'] and str(row['workdir']) in str(artifact_abs):
            substituted = str(artifact_abs).replace(str(row['workdir']), str(row['output_dir']))
            possible_paths.append(Path(substituted))
            print(f"[get_artifact] Added substituted path (workdir->output_dir): {substituted}")
        
        # Also try just the basename in workdir and output_dir
        basename = artifact_abs.name
        if row['workdir']:
            for md_file in Path(row['workdir']).rglob(basename):
                possible_paths.append(md_file)
                print(f"[get_artifact] Found in workdir: {md_file}")
        if row['output_dir']:
            for md_file in Path(row['output_dir']).rglob(basename):
                possible_paths.append(md_file)
                print(f"[get_artifact] Found in output_dir: {md_file}")
    else:
        # Relative path - join with workdir and output_dir
        if row['workdir']:
            possible_paths.append(Path(row['workdir']) / artifact_path)
        if row['output_dir']:
            possible_paths.append(Path(row['output_dir']) / artifact_path)
    
    print(f"[get_artifact] Trying {len(possible_paths)} possible paths")
    
    # Try each path
    for file_path in possible_paths:
        print(f"[get_artifact] Checking: {file_path} (exists={file_path.exists()})")
        if file_path.exists() and file_path.is_file():
            try:
                content = file_path.read_text()
                print(f"[get_artifact] SUCCESS! Read {file_path} ({len(content)} bytes)")
                return jsonify({
                    'name': file_path.name,
                    'path': str(file_path),
                    'content': content
                })
            except Exception as e:
                print(f"[get_artifact] Failed to read {file_path}: {e}")
                return jsonify({'error': f'Could not read file: {str(e)}'}), 500
    
    print(f"[get_artifact] ERROR: Artifact not found in any of {len(possible_paths)} paths")
    return jsonify({'error': f'Artifact not found. Searched {len(possible_paths)} locations.'}), 404

@app.route('/api/kill-job/<job_id>', methods=['POST'])
def kill_job(job_id):
    """Kill a running job"""
    try:
        # Get job from database
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('SELECT pid, status FROM jobs WHERE id = ?', (job_id,))
        row = c.fetchone()
        
        if not row:
            conn.close()
            return jsonify({'error': 'Job not found'}), 404
        
        pid = row['pid']
        status = row['status']
        
        if status != 'running':
            conn.close()
            return jsonify({'error': f'Job is not running (status: {status})'}), 400
        
        # Kill the process if PID exists
        if pid:
            import signal
            try:
                # Check if process exists first
                try:
                    os.kill(pid, 0)
                except OSError:
                    log_message(job_id, f"Process {pid} not found (already dead)")
                else:
                    # Kill process group to ensure all child processes are killed
                    try:
                        os.killpg(os.getpgid(pid), signal.SIGTERM)
                        log_message(job_id, f"Sent SIGTERM to process group (PID: {pid})")
                    except ProcessLookupError:
                        log_message(job_id, f"Process group for {pid} not found")
                        # Try killing just the process
                        try:
                            os.kill(pid, signal.SIGTERM)
                            log_message(job_id, f"Sent SIGTERM to process {pid}")
                        except OSError:
                            pass
                    except Exception as e:
                        log_message(job_id, f"Error killing process group: {str(e)}")
                        # Try forceful kill
                        try:
                            os.kill(pid, signal.SIGKILL)
                            log_message(job_id, f"Sent SIGKILL to process {pid}")
                        except OSError:
                            pass
            except Exception as e:
                log_message(job_id, f"Unexpected error during kill: {str(e)}")
        
        # Update job status - ALWAYS do this
        c.execute('''
            UPDATE jobs 
            SET status = 'killed', completed_at = ?, error = 'Job killed by user'
            WHERE id = ?
        ''', (datetime.now(), job_id))
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Job killed successfully',
            'job_id': job_id
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze-performance/<job_id>', methods=['POST'])
def analyze_performance(job_id):
    """Run performance analysis on original and translated code"""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT workdir, output_dir, from_api, to_api, parbench_spec FROM jobs WHERE id = ?', (job_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return jsonify({'error': 'Job not found'}), 404

    workdir = row['workdir']
    output_dir = row['output_dir']

    if not workdir or not output_dir:
        return jsonify({'error': 'Job mismatch: Missing workdir or output_dir'}), 400

    # Load performance run args from parbench spec so both APIs run the same workload
    run_args = None
    if row['parbench_spec']:
        try:
            with open(row['parbench_spec']) as _f:
                _spec = json.load(_f)
            _perf_args = (_spec.get('run', {})
                          .get('input_configurations', {})
                          .get('performance', {})
                          .get('arguments'))
            if _perf_args:
                run_args = _perf_args
                print(f"Using parbench performance args: {run_args}")
        except Exception:
            pass

    results = {
        'original': {'api': row['from_api']},
        'translated': {'api': row['to_api']},
        'speedup': None,
        'run_args': run_args,
    }

    sys.path.insert(0, str(Path(__file__).parent))
    from performance import run_nsys_profile

    print(f"Analyzing original code (golden reference) in {workdir}")
    orig_res = run_nsys_profile(Path(workdir), is_original=True, run_args=run_args)
    results['original'].update(orig_res)

    print(f"Analyzing translated code (optimized) from {output_dir}")
    trans_res = run_nsys_profile(Path(workdir), is_original=False,
                                 output_dir=Path(output_dir), run_args=run_args)
    results['translated'].update(trans_res)
    
    # Calculate speedup if both succeeded and have gpu_time
    # Speedup = Original / Translated (>1.0 means translation is faster)
    if orig_res.get('success') and trans_res.get('success'):
        t_orig = orig_res.get('gpu_time_ms')
        t_trans = trans_res.get('gpu_time_ms')
        
        if t_orig is not None and t_trans is not None and t_trans > 0:
            speedup = t_orig / t_trans
            results['speedup'] = speedup
            
    return jsonify(results)

@app.route('/api/dashboard/stats')
def get_dashboard_stats():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT status, COUNT(*) as count FROM jobs GROUP BY status')
    status_counts = {row['status']: row['count'] for row in c.fetchall()}
    c.execute('''
        SELECT from_api || ' -> ' || to_api AS pair,
               COUNT(*) AS total,
               SUM(CASE WHEN status="completed" THEN 1 ELSE 0 END) AS succeeded
        FROM jobs GROUP BY from_api, to_api ORDER BY total DESC LIMIT 12
    ''')
    pairs = [dict(row) for row in c.fetchall()]
    c.execute('''
        SELECT id, kernel_name, from_api, to_api, engine, model, status, created_at, completed_at,
               CAST(ROUND((julianday(completed_at)-julianday(created_at))*86400) AS INTEGER) AS duration_seconds
        FROM jobs ORDER BY created_at DESC LIMIT 10
    ''')
    recent = [dict(row) for row in c.fetchall()]
    conn.close()
    total = sum(status_counts.values())
    completed = status_counts.get('completed', 0)
    return jsonify({
        'status_counts': status_counts,
        'total': total,
        'success_rate': round(completed / total * 100, 1) if total > 0 else 0,
        'pairs': pairs,
        'recent': recent,
    })


@app.route('/api/history')
def get_history():
    search  = request.args.get('search', '').strip()
    status  = request.args.get('status', '').strip()
    limit   = min(int(request.args.get('limit', 100)), 500)
    offset  = int(request.args.get('offset', 0))
    conn = get_db_connection()
    c = conn.cursor()
    like = f'%{search}%'
    c.execute('''
        SELECT id, source_directory, from_api, to_api, engine, model, parbench_spec,
               kernel_name, supervise, baseline, status, created_at, completed_at, workdir, output_dir, traces,
               CAST(ROUND((julianday(completed_at)-julianday(created_at))*86400) AS INTEGER) AS duration_seconds
        FROM jobs
        WHERE (? = '' OR source_directory LIKE ? OR from_api LIKE ? OR to_api LIKE ? OR COALESCE(model,'') LIKE ?
               OR COALESCE(kernel_name,'') LIKE ?)
          AND (? = '' OR status = ?)
        ORDER BY created_at DESC LIMIT ? OFFSET ?
    ''', (search, like, like, like, like, like, status, status, limit, offset))
    jobs = [dict(row) for row in c.fetchall()]
    c.execute('''
        SELECT COUNT(*) FROM jobs
        WHERE (? = '' OR source_directory LIKE ? OR from_api LIKE ? OR to_api LIKE ? OR COALESCE(model,'') LIKE ?
               OR COALESCE(kernel_name,'') LIKE ?)
          AND (? = '' OR status = ?)
    ''', (search, like, like, like, like, like, status, status))
    total = c.fetchone()[0]
    conn.close()
    return jsonify({'jobs': jobs, 'total': total, 'offset': offset, 'limit': limit})


@app.route('/api/logs-download/<job_id>')
def download_job_log(job_id):
    # Try persistent file first
    log_file = LOGS_DIR / f'{job_id}.log'
    if log_file.exists():
        text = log_file.read_text(encoding='utf-8', errors='replace')
        return Response(text, mimetype='text/plain',
                        headers={'Content-Disposition': f'attachment; filename="paracodex-{job_id[:8]}.log"'})
    # Fall back to in-memory
    logs = LOGS.get(job_id, [])
    if not logs:
        return jsonify({'error': 'Log not available (server may have restarted or job not found)'}), 404
    text = '\n'.join(f"[{e['timestamp']}] {e['message']}" for e in logs)
    return Response(text, mimetype='text/plain',
                    headers={'Content-Disposition': f'attachment; filename="paracodex-{job_id[:8]}.log"'})


@app.route('/api/job-log/<job_id>')
def get_job_log(job_id):
    """Get log content for a job (for inline viewing)"""
    log_file = LOGS_DIR / f'{job_id}.log'
    if log_file.exists():
        text = log_file.read_text(encoding='utf-8', errors='replace')
        return jsonify({'log': text, 'source': 'file', 'lines': text.count('\n')})
    logs = LOGS.get(job_id, [])
    if logs:
        text = '\n'.join(f"[{e['timestamp']}] {e['message']}" for e in logs)
        return jsonify({'log': text, 'source': 'memory', 'lines': len(logs)})
    return jsonify({'log': '', 'source': 'none', 'lines': 0})


@app.route('/api/job-detail/<job_id>')
def get_job_detail(job_id):
    """Full job detail: metadata + artifacts + traces"""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM jobs WHERE id = ?', (job_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return jsonify({'error': 'Job not found'}), 404
    job = dict(row)
    # Artifacts
    artifacts = find_artifacts(job_id)
    # Traces — only include traces whose cwd matches this job's workdir
    traces = []
    job_workdir = job.get('workdir') or ''
    try:
        trace_files = json.loads(job.get('traces') or '[]')
        traces_dir = PIPELINE_ROOT / 'traces'
        for fname in trace_files:
            fp = traces_dir / fname
            if not fp.exists():
                continue
            # Filter: trace's cwd must match the job's workdir (when workdir is known)
            if job_workdir:
                trace_cwd = _extract_trace_cwd(fp) or ''
                if trace_cwd and not trace_cwd.startswith(job_workdir):
                    continue
            parts = fname.replace('.jsonl', '').split('-', 2)
            tid = parts[2] if len(parts) >= 3 else fname.replace('.jsonl', '')
            label = _extract_trace_label(fp)
            traces.append({
                'id': tid,
                'filename': fname,
                'size': fp.stat().st_size,
                'label': label,
            })
    except Exception:
        pass
    # Log availability
    log_file = LOGS_DIR / f'{job_id}.log'
    has_log = log_file.exists() or bool(LOGS.get(job_id))
    return jsonify({
        'job': job,
        'artifacts': artifacts,
        'traces': traces,
        'has_log': has_log,
    })


@app.route('/api/traces/by-job')
def list_traces_by_job():
    """Return jobs that have associated traces, with their trace file list."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        SELECT id, kernel_name, from_api, to_api, engine, status, created_at, traces, workdir
        FROM jobs
        WHERE traces IS NOT NULL AND traces != '[]'
        ORDER BY created_at DESC LIMIT 200
    ''')
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    traces_dir = PIPELINE_ROOT / 'traces'
    result = []
    for row in rows:
        try:
            trace_files = json.loads(row['traces'] or '[]')
        except Exception:
            trace_files = []
        job_workdir = row.get('workdir') or ''
        enriched = []
        for fname in trace_files:
            fp = traces_dir / fname
            if not fp.exists():
                continue
            # Filter: trace's cwd must match this job's workdir
            if job_workdir:
                trace_cwd = _extract_trace_cwd(fp) or ''
                if trace_cwd and not trace_cwd.startswith(job_workdir):
                    continue
            parts = fname.replace('.jsonl', '').split('-', 2)
            tid = parts[2] if len(parts) >= 3 else fname.replace('.jsonl', '')
            enriched.append({
                'id': tid,
                'filename': fname,
                'size': fp.stat().st_size,
                'label': _extract_trace_label(fp),
            })
        if enriched:
            result.append({
                'job_id': row['id'],
                'kernel_name': row['kernel_name'],
                'from_api': row['from_api'],
                'to_api': row['to_api'],
                'engine': row['engine'],
                'status': row['status'],
                'created_at': row['created_at'],
                'traces': enriched,
            })
    return jsonify({'jobs': result})


@app.route('/api/traces')
def list_traces():
    traces_dir = PIPELINE_ROOT / 'traces'
    if not traces_dir.exists():
        return jsonify({'traces': []})
    files = []
    for f in sorted(traces_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if f.suffix == '.jsonl' and 'prompt' not in f.name:
            prompt_file = traces_dir / f.name.replace('.jsonl', '-prompt.txt')
            # Extract trace ID from filename: codex-trace-<uuid>.jsonl
            parts = f.stem.split('-', 2)
            trace_id = parts[2] if len(parts) >= 3 else f.stem
            files.append({
                'id': trace_id,
                'filename': f.name,
                'size': f.stat().st_size,
                'modified': f.stat().st_mtime,
                'has_prompt': prompt_file.exists(),
            })
    return jsonify({'traces': files[:100]})


@app.route('/api/traces/<path:trace_id>')
def get_trace_content(trace_id):
    import ast as _ast
    traces_dir = PIPELINE_ROOT / 'traces'
    # Find matching file
    filename = f'codex-trace-{trace_id}.jsonl'
    file_path = traces_dir / filename
    if not file_path.exists():
        # Try direct filename
        safe = Path(trace_id).name
        file_path = traces_dir / safe
        if not file_path.exists():
            return jsonify({'error': 'Trace not found'}), 404

    events = []
    meta = {}
    try:
        with open(file_path, encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                payload = d.get('payload', {})
                if isinstance(payload, str):
                    try:
                        payload = _ast.literal_eval(payload)
                    except Exception:
                        payload = {}
                dtype = d.get('type', '')
                ptype = payload.get('type', '') if isinstance(payload, dict) else ''

                if dtype == 'session_meta':
                    if isinstance(payload, dict):
                        meta = {'id': payload.get('id', ''), 'cwd': payload.get('cwd', '')}
                    elif isinstance(payload, str):
                        try:
                            pm = _ast.literal_eval(payload) if isinstance(payload, str) else payload
                            meta = {'id': pm.get('id', ''), 'cwd': pm.get('cwd', '')}
                        except Exception:
                            pass
                    continue

                if dtype == 'event_msg':
                    if ptype == 'user_message':
                        events.append({'type': 'user', 'text': str(payload.get('message', ''))[:2000]})
                    elif ptype == 'agent_message':
                        events.append({'type': 'assistant', 'text': str(payload.get('message', ''))[:3000]})
                    elif ptype == 'agent_reasoning':
                        events.append({'type': 'reasoning', 'text': str(payload.get('text', ''))[:500]})
                    # skip token_count

                elif dtype == 'response_item':
                    if ptype == 'function_call':
                        events.append({
                            'type': 'tool_call',
                            'name': payload.get('name', 'tool'),
                            'args': str(payload.get('arguments', ''))[:800],
                        })
                    elif ptype == 'function_call_output':
                        events.append({
                            'type': 'tool_result',
                            'output': str(payload.get('output', ''))[:800],
                        })
                    elif ptype == 'custom_tool_call':
                        events.append({
                            'type': 'tool_call',
                            'name': payload.get('name', 'tool'),
                            'args': str(payload.get('input', ''))[:800],
                        })
                    elif ptype == 'custom_tool_call_output':
                        events.append({
                            'type': 'tool_result',
                            'output': str(payload.get('output', ''))[:800],
                        })
                    # skip message items (developer/system context)

        # Cap at 300 events
        truncated = len(events) > 300
        events = events[:300]
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'id': trace_id, 'meta': meta, 'events': events, 'truncated': truncated})


@app.route('/api/skills')
def list_skills():
    import re as _re
    skills_dir = PIPELINE_ROOT / '.opencode' / 'skills'
    skills = []
    if not skills_dir.exists():
        return jsonify({'skills': []})
    for skill_dir in sorted(skills_dir.iterdir()):
        skill_md = skill_dir / 'SKILL.md'
        if not skill_md.exists():
            continue
        try:
            content = skill_md.read_text(encoding='utf-8', errors='replace')
        except Exception:
            continue
        name = skill_dir.name
        description = ''
        compatibility = ''
        body = content
        fm = _re.match(r'^---\n(.*?)\n---\n?(.*)', content, _re.DOTALL)
        if fm:
            fm_text = fm.group(1)
            body = fm.group(2).strip()
            m = _re.search(r'^name:\s*"?(.+?)"?\s*$', fm_text, _re.MULTILINE)
            if m: name = m.group(1).strip()
            m = _re.search(r'^description:\s*"?(.+?)"?\s*$', fm_text, _re.MULTILINE)
            if m: description = m.group(1).strip()
            m = _re.search(r'^compatibility:\s*"?(.+?)"?\s*$', fm_text, _re.MULTILINE)
            if m: compatibility = m.group(1).strip()
        skills.append({
            'name': name,
            'description': description,
            'compatibility': compatibility,
            'body': body,
            'has_examples': (skill_dir / 'references' / 'examples.md').exists(),
            'has_output': (skill_dir / 'references' / 'output.md').exists(),
            'has_scripts': (skill_dir / 'scripts').exists(),
        })
    return jsonify({'skills': skills})


if __name__ == '__main__':
    # Initialize database
    init_db()
    
    # Start server
    print("=" * 70)
    print("Paracodex Pipeline Web Application")
    print("=" * 70)
    print(f"Server starting at http://localhost:5000")
    print(f"Pipeline root: {PIPELINE_ROOT}")
    print("=" * 70)
    print()
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
