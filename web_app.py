"""
Flask Web Application for Optimal Sample Selection System
==========================================================

Features:
- S1: Main computation interface (parameter input, execute, store, print)
- S2: Database browser (view, display, delete saved results)
- Mobile-responsive design
- Real-time progress updates via Server-Sent Events (SSE)
- Background computation with threading
"""

import os
import sys
import json
import random
import threading
import time
from flask import Flask, render_template, request, jsonify, redirect, url_for, Response, send_file
from queue import Queue
import contextlib

# Import the core algorithm
from algorithm import DEFAULT_TIME_LIMIT, solve

app = Flask(__name__)
app.config['SECRET_KEY'] = 'optimal-sample-selection-2024'

# File-based storage
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RESULTS_INDEX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")

os.makedirs(RESULTS_DIR, exist_ok=True)

# Global queue for SSE progress updates
progress_queues = {}

# ── Storage helpers ──────────────────────────────────────────────────────────

def load_results():
    """Load all saved results from results/ folder"""
    records = []
    if os.path.exists(RESULTS_DIR):
        for fname in os.listdir(RESULTS_DIR):
            if fname.endswith('.json'):
                fpath = os.path.join(RESULTS_DIR, fname)
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        records.append(json.load(f))
                except json.JSONDecodeError:
                    pass
    # Also migrate legacy results.json entries
    if os.path.exists(RESULTS_INDEX):
        try:
            with open(RESULTS_INDEX, 'r', encoding='utf-8') as f:
                legacy = json.load(f)
            existing_ids = {r['id'] for r in records}
            for rec in legacy:
                if rec['id'] not in existing_ids:
                    # Migrate to folder
                    fpath = os.path.join(RESULTS_DIR, f"{rec['id']}.json")
                    with open(fpath, 'w', encoding='utf-8') as f:
                        json.dump(rec, f, indent=2, ensure_ascii=False)
                    records.append(rec)
            # Clear legacy after migration
            os.remove(RESULTS_INDEX)
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    # Sort by id descending
    records.sort(key=lambda r: r.get('id', ''), reverse=True)
    return records

def save_result(record):
    """Save a single result to results/ folder"""
    fpath = os.path.join(RESULTS_DIR, f"{record['id']}.json")
    with open(fpath, 'w', encoding='utf-8') as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

def delete_result_file(record_id):
    """Delete a result file from results/ folder"""
    fpath = os.path.join(RESULTS_DIR, f"{record_id}.json")
    if os.path.exists(fpath):
        os.remove(fpath)

def next_run_id(m, n, k, j, s):
    """Calculate next run ID for given parameters"""
    count = sum(
        1 for r in load_results()
        if (r["params"]["m"] == m and r["params"]["n"] == n
            and r["params"]["k"] == k and r["params"]["j"] == j
            and r["params"]["s"] == s)
    )
    return count + 1

# ── Progress writer for SSE ──────────────────────────────────────────────────

class ProgressWriter:
    """Redirect stdout to SSE queue"""
    def __init__(self, queue):
        self.queue = queue

    def write(self, text):
        if text.strip():
            self.queue.put(text)

    def flush(self):
        pass

# ── Routes: S1 Main Interface ────────────────────────────────────────────────

@app.route('/')
def index():
    """Main computation interface"""
    return render_template('index.html')

@app.route('/execute', methods=['POST'])
def execute():
    """Execute algorithm in background thread"""
    data = request.get_json()

    # Parse and validate parameters
    try:
        if not data:
            return jsonify({'error': 'Invalid request: empty body'}), 400
        m = int(data.get('m') or 45)
        n = int(data.get('n') or 9)
        k = int(data.get('k') or 6)
        j = int(data.get('j') or 5)
        s = int(data.get('s') or 5)
        T = int(data.get('T') or 1)
        mode = data.get('mode', 'random')
        manual_input = data.get('manual_input', '')

        # Validation
        if not (45 <= m <= 54):
            return jsonify({'error': 'm must be between 45 and 54'}), 400
        if not (7 <= n <= 25):
            return jsonify({'error': 'n must be between 7 and 25'}), 400
        if not (4 <= k <= 7):
            return jsonify({'error': 'k must be between 4 and 7'}), 400
        if not (3 <= s <= 7):
            return jsonify({'error': 's must be between 3 and 7'}), 400
        if not (s <= j <= k):
            return jsonify({'error': 'j must satisfy s ≤ j ≤ k'}), 400
        if T < 1:
            return jsonify({'error': 'T must be ≥ 1'}), 400
        if n > m:
            return jsonify({'error': 'n cannot exceed m'}), 400

        # Generate samples
        if mode == 'random':
            samples = sorted(random.sample(range(1, m + 1), n))
        else:
            try:
                samples = [int(x.strip()) for x in manual_input.split(',') if x.strip()]
                if len(samples) != n:
                    return jsonify({'error': f'Expected {n} samples, got {len(samples)}'}), 400
                if len(set(samples)) != n:
                    return jsonify({'error': 'Sample IDs must be distinct'}), 400
                if any(x < 1 or x > m for x in samples):
                    return jsonify({'error': f'Sample IDs must be in range 1-{m}'}), 400
                samples = sorted(samples)
            except ValueError:
                return jsonify({'error': 'Invalid sample IDs format'}), 400

    except (KeyError, ValueError) as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400

    # Create unique session ID
    session_id = f"{int(time.time() * 1000)}"
    progress_queues[session_id] = Queue()

    # Start background computation
    def worker():
        try:
            with contextlib.redirect_stdout(ProgressWriter(progress_queues[session_id])):
                groups, info = solve(
                    samples, k, j, s,
                    T=T, time_limit=DEFAULT_TIME_LIMIT, verbose=True
                )

            # Send completion message
            result = {
                'groups': [list(g) for g in groups],
                'info': info,
                'params': {'m': m, 'n': n, 'k': k, 'j': j, 's': s, 'T': T},
                'samples': samples
            }
            progress_queues[session_id].put(f"__DONE__{json.dumps(result)}")
        except Exception as e:
            progress_queues[session_id].put(f"__ERROR__{str(e)}")

    threading.Thread(target=worker, daemon=True).start()

    return jsonify({'session_id': session_id, 'samples': samples})

@app.route('/progress/<session_id>')
def progress(session_id):
    """Polling endpoint for progress updates — returns JSON"""
    if session_id not in progress_queues:
        return jsonify({'status': 'error', 'message': 'Session not found'}), 404

    queue = progress_queues[session_id]
    logs = []

    # Drain all log messages from the queue (non-blocking peek)
    while not queue.empty():
        try:
            msg = queue.get_nowait()
            if msg.startswith('__DONE__'):
                result = json.loads(msg[8:])
                # Clean up
                del progress_queues[session_id]
                return jsonify({'status': 'done', 'result': result})
            elif msg.startswith('__ERROR__'):
                error_msg = msg[9:]
                del progress_queues[session_id]
                return jsonify({'status': 'error', 'message': error_msg})
            else:
                logs.append(msg)
        except:
            break

    return jsonify({'status': 'running', 'logs': logs})

@app.route('/store', methods=['POST'])
def store():
    """Store current result to database"""
    data = request.get_json()

    try:
        groups = data['groups']
        info = data['info']
        params = data['params']
        samples = data['samples']

        # Generate record ID
        run_id = next_run_id(params['m'], params['n'], params['k'],
                             params['j'], params['s'])
        rec_id = (f"{params['m']}-{params['n']}-{params['k']}-"
                  f"{params['j']}-{params['s']}-{run_id}-{info['solution_size']}")

        record = {
            'id': rec_id,
            'params': params,
            'samples': samples,
            'info': info,
            'groups': groups
        }

        save_result(record)

        return jsonify({'success': True, 'id': rec_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/export', methods=['POST'])
def export():
    """Export result as text file"""
    data = request.get_json()

    try:
        groups = data['groups']
        info = data['info']
        params = data['params']
        samples = data['samples']

        # Generate text content
        method = info.get('method', 'heuristic')
        optimal = info.get('optimal', False)

        lines = [
            "Optimal Sample Selection Result",
            "=" * 60,
            f"Parameters: m={params['m']}  n={params['n']}  k={params['k']}  "
            f"j={params['j']}  s={params['s']}  T={params['T']}",
            f"Samples:    {samples}",
            f"Groups: {info['solution_size']}  lb={info['lower_bound']}  "
            f"gap={info['gap']}  valid={info['valid']}  method={method}  "
            f"optimal={optimal}  time={info['time']}s",
            "-" * 60,
            ""
        ]

        for g in groups:
            lines.append(str(g))

        content = '\n'.join(lines)

        # Create temporary file
        filename = f"result_{params['m']}-{params['n']}-{params['k']}-{params['j']}-{params['s']}.txt"
        filepath = os.path.join('/tmp', filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        return send_file(filepath, as_attachment=True, download_name=filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Routes: S2 Database Interface ────────────────────────────────────────────

@app.route('/db')
def database():
    """Database browser interface"""
    return render_template('database.html')

@app.route('/api/records')
def get_records():
    """Get all saved records"""
    records = load_results()
    return jsonify(records)

@app.route('/api/record/<record_id>')
def get_record(record_id):
    """Get specific record by ID"""
    records = load_results()
    for rec in records:
        if rec['id'] == record_id:
            return jsonify(rec)
    return jsonify({'error': 'Record not found'}), 404

@app.route('/api/delete/<record_id>', methods=['DELETE'])
def delete_record(record_id):
    """Delete a record"""
    delete_result_file(record_id)
    return jsonify({'success': True})

@app.route('/api/export/<record_id>')
def export_record(record_id):
    """Export a saved record as text file"""
    records = load_results()
    rec = None
    for r in records:
        if r['id'] == record_id:
            rec = r
            break

    if not rec:
        return jsonify({'error': 'Record not found'}), 404

    # Generate text content
    params = rec['params']
    info = rec['info']
    samples = rec.get('samples', [])
    groups = rec['groups']

    method = info.get('method', 'heuristic')
    optimal = info.get('optimal', False)

    lines = [
        "Optimal Sample Selection Result",
        "=" * 60,
        f"Record ID:  {rec['id']}",
        f"Parameters: m={params['m']}  n={params['n']}  k={params['k']}  "
        f"j={params['j']}  s={params['s']}  T={params['T']}",
        f"Samples:    {samples}",
        f"Groups: {info['solution_size']}  lb={info['lower_bound']}  "
        f"gap={info['gap']}  valid={info['valid']}  method={method}  "
        f"optimal={optimal}  time={info['time']}s",
        "-" * 60,
        ""
    ]

    for g in groups:
        lines.append(str(g))

    content = '\n'.join(lines)

    # Create temporary file
    filename = f"{record_id}.txt"
    filepath = os.path.join('/tmp', filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    return send_file(filepath, as_attachment=True, download_name=filename)

# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import socket

    # Get local IP for mobile access
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    print("=" * 60)
    print("  Optimal Sample Selection System - Web Application")
    print("=" * 60)
    print(f"  Local:  http://localhost:3000")
    print(f"  Mobile: http://{local_ip}:3000")
    print("=" * 60)
    print()

    app.run(host='0.0.0.0', port=3000, debug=False, threaded=True)
