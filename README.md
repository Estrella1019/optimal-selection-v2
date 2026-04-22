# An Optimal Samples Selection System

## How to Run

```bash
# Install dependencies (first time only)
pip install flask numpy numba

# Start web app
./start.sh          # or: python web_app.py
# Access at: http://localhost:3000

# Run tests (optional)
python test_algorithm.py --tier 1   # Quick correctness test
python test_algorithm.py --tier 2   # Medium scale quality test
python test_algorithm.py --tier 3   # Worst-case stress test (~18 min)
```

On first run, numba takes ~10–30 seconds to JIT-compile; subsequent runs use the cached build and start instantly.
The current version uses a **hybrid solver**: exact branch first (proven optimal for small instances), then enhanced heuristic with up to **30 seconds** for web use.

---

## File Overview

| File | Description |
|------|-------------|
| `web_app.py` | Flask web application (S1 compute + S2 database) |
| `algorithm.py` | Core solver (numba-accelerated exact + heuristic) |
| `test_algorithm.py` | Three-tier automated test suite |
| `results/` | Individual JSON files per saved result (auto-created) |
| `start.sh` | Launcher script (start / install / test / help) |

---

## Parameter Guide

| Param | Range | Meaning |
|-------|-------|---------|
| m | 45–54 | Total sample pool size |
| n | 7–25 | Number of samples to use |
| k | 4–7 | Size of each k-group |
| j | s–k | Size of j-subsets to be covered |
| s | 3–7 | Minimum intersection size to count as "covered" |
| T | ≥ 1 | Each j-subset must be covered by ≥ T k-groups |

**Constraint:** s ≤ j ≤ k ≤ n ≤ m

---

## Interface Guide

### S1 — Main Interface

1. **Fill in parameters** — m, n, k, j, s, T within valid ranges
2. **Choose sample source:**
   - *Random*: randomly pick n samples from m
   - *Manual*: enter n sample IDs manually (comma-separated, range 1–m)
3. **Click Execute** — algorithm runs in background, progress streams live
4. **On completion:**
   - *Save Result*: store to `results/` folder, named `m-n-k-j-s-runid-groups.json`
   - *Export to File*: download as `.txt`
   - *Clear*: reset all fields
5. **Click "View Saved Results"** → S2 interface

### S2 — Database Interface

- Lists all saved records sorted by most recent
- **View Details**: pop-up showing full k-groups
- **Delete**: remove selected record
- **Export to File**: re-download as `.txt`
- **Back to Home**: return to S1

---

## Output Guide

Each result shows:

```
Found N groups  exact/heuristic  [optimal]
  Lower bound: X | Gap: Y | Valid: True | Time: Zs
```

- **valid=True**: 100% guarantee — every j-subset meets coverage requirement
- **lb**: theoretical lower bound (true optimum ≥ lb)
- **gap**: distance to lower bound (gap=0 → at lower bound, very likely optimal)
- **method**: `exact` (proven optimal) / `hybrid` / `heuristic`
- **optimal=True**: proven optimal by exact solver or reached lower bound

---

## Result File Format

Each file in `results/` is a standalone JSON, e.g. `results/45-9-6-5-5-1-14.json`:

```json
{
  "id": "45-9-6-5-5-1-14",
  "params": {"m": 45, "n": 9, "k": 6, "j": 5, "s": 5, "T": 1},
  "samples": [3, 7, 12, 18, 22, 33, 41, 44, 45],
  "info": {
    "solution_size": 14,
    "lower_bound": 9,
    "gap": 5,
    "restarts": 120,
    "time": 3.2,
    "valid": true,
    "method": "hybrid",
    "optimal": false
  },
  "groups": [[3,7,12,18,33,41], [5,9,14,22,35,44], ...]
}
```

---

## Performance Guide

| Parameters | Est. Time | Notes |
|------------|-----------|-------|
| n≤10, k≤6 | < 5s | Most enter exact or very-fast heuristic |
| n=15, k=6, j=5, s=5 | 5–20s | Medium scale |
| n=20, k=7, j=6, s=5 | 30–180s | Slower |
| n=25, k=7, j=7, s=5 | ≤ 30s | Heuristic time cap (web mode) |

Heuristic budget is capped at **30 seconds** for web use. If the exact branch completes within budget, it returns a proven optimal result.
