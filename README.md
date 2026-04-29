# An Optimal Samples Selection System

**CS360 Artificial Intelligence — Group Project**

> A high-performance covering-design solver with a mobile-friendly web interface for selecting optimal sample groupings.

---

## Overview

This project implements a **hybrid optimization solver** for the **Covering Design Problem** (also known as the Set Cover problem in combinatorial design). Given a set of *n* samples, the goal is to find the minimum number of *k-sized groups* such that every *j-sized subset* is covered by at least *s* groups.

The system features:
- **Exact + Heuristic hybrid solver** with Numba JIT acceleration (30× faster on large instances)
- **Quality search with full-bitset compression** — LNS destroy-and-repair for tighter solutions on medium instances
- **Adaptive runtime budgets** — 120s–600s based on problem scale
- **Real-time progress streaming** via polling-based SSE
- **Mobile-responsive web UI** supporting both random and manual sample input
- **Persistent result database** with export and management features
- **Automated test suite** with three tiers (correctness, quality, stress)

---

## Problem Definition

> Given *n* samples, find the smallest collection of *k-element groups* such that every *j-element subset* appears in at least *s* groups.

| Symbol | Meaning |
|--------|---------|
| `m` | Total pool size (45–54) |
| `n` | Number of selected samples (n ≤ m) |
| `k` | Size of each group to form (k ≤ n) |
| `j` | Size of subset to be covered (s ≤ j ≤ k) |
| `s` | Minimum coverage threshold per j-subset |
| `T` | Number of groups each j-subset must appear in (multi-cover) |

### Example

For *n=9* samples, forming *k=6* groups, every *j=5* subset must appear in at least *s=5* groups:

```
Samples: [1, 3, 7, 12, 18, 22, 33, 41, 44]
Found 14 groups (optimal)

Group 1:  [1, 3, 7, 12, 18, 33]
Group 2:  [1, 3, 22, 33, 41, 44]
...
```

---

## Algorithm Architecture

### Four-Layer Acceleration Design

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1 — Problem Reduction                                 │
│  Preprocessing: filter dominated subsets, bitmask encoding  │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 2 — Lower Bound (counting bound + Schönheim bound)   │
│  Provides quality guarantee: gap = solution - lower_bound    │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 3 — Hybrid Solver                                     │
│                                                        ║    │
│    ┌─────────────┐        ┌─────────────────────┐    ║    │
│    │ Exact Branch │──OK──▶│  Branch & Bound     │    ║    │
│    │ (small n)   │        │  + LNS + Swap       │    ║    │
│    └─────────────┘        └─────────────────────┘    ║    │
│           │                      │                   ║    │
│           ▼                      ▼                   ║    │
│    ┌─────────────┐        ┌─────────────────────┐  ║    │
│    │ Proven       │        │ Warm-start heuristic│  ║    │
│    │ Optimal      │        │ (best-known upper) │  ║    │
│    └─────────────┘        └─────────────────────┘  ║    │
│                                                    ║    │
│  ═══════════════════════════════════════════════════╝    │
│              Global optimal or best-known bound            │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 4 — Quality Search (v2, T=1 medium instances)        │
│                                                             │
│  ┌──────────────────┐  ┌────────────────┐  ┌────────────┐  │
│  │ Global greedy    │→ │ Fixed-B descent│→ │ LNS destroy│  │
│  │ (bitset encode)  │  │ compression    │  │ & repair   │  │
│  └──────────────────┘  └────────────────┘  └────────────┘  │
│                                                             │
│  + Local search swap polish (remaining time budget)         │
└─────────────────────────────────────────────────────────────┘
```

### Key Optimizations

1. **Bitmask Encoding** — Each j-subset stored as uint32 bitmask. Intersection test is a single `&` operation.
2. **Counting Lower Bound** — For each sample, count j-subsets containing it, divide by C(k-1, j-1), take max.
3. **Greedy Initialization** — Pick groups maximizing new coverage first.
4. **Minimization via LNS** — Iteratively drop groups, re-cover uncovered subsets with perturbation.
5. **Local Search (Swap)** — Try replacing each group with a sample outside it.
6. **Warm-start Mechanism** — Pre-solve smaller variants to seed the heuristic.
7. **Numba JIT + Parallel** — 40+ parallel loops, ~30× speedup on large instances.
8. **Full-Bitset Quality Search** *(v2)* — Global greedy construction + fixed-B descent compression using Python big-integer bitsets for medium instances (e.g. n=20, k=6, j=5, s=4: ~185 → ~178 groups).
9. **LNS Destroy-and-Repair** *(v2)* — Randomly removes overlapping groups and repairs with bounded DFS; adaptive destroy size when search stalls.
10. **Adaptive Time Budgets** *(v2)* — Automatically allocates 120s / 300s / 600s based on `n_j × n_cand`; partitions time between construction phase (65%) and quality search phase.
11. **Scale Protection** *(v2)* — Skips full-bitset search when `n_j × n_cand > 50M` to avoid memory blowup on large instances.
12. **K-Group Deduplication** *(v2)* — Prevents duplicate groups from being selected, including multi-cover (T>1) scenarios.

---

## System Features

### S1 — Computation Interface
- 4 quick preset buttons for common parameter sets
- Random or manual sample selection
- Real-time progress streaming with log output
- Save / Export / Clear operations
- Mobile-responsive layout

### S2 — Database Browser
- View all saved computation results
- Modal detail view with full k-groups
- Delete unwanted records
- Re-export results as text files

### Technical Highlights
- **Background threading**: algorithm runs in daemon thread, never blocks UI
- **Progress polling**: client polls `/progress/<session_id>` every 200–300ms
- **File-based storage**: each result stored as `{id}.json` in `results/` directory
- **Migration support**: auto-migrates legacy `results.json` entries

---

## Quick Start

```bash
# Install dependencies
pip install flask numpy numba

# Start the web application
python web_app.py

# Or use the launcher script
./start.sh          # Shows menu: start / install / test / help
```

Access the app at **http://localhost:3000** (or `http://<your-ip>:3000` for mobile access).

> **First run note:** Numba takes 10–30 seconds to JIT-compile on first run. Subsequent runs use cached builds and start instantly.

---

## Automated Testing

```bash
# Tier 1 — Correctness (tiny instances, < 5s each)
python test_algorithm.py --tier 1

# Tier 2 — Medium quality (< 60s each)
python test_algorithm.py --tier 2

# Tier 3 — Worst-case stress (~18 minutes total)
python test_algorithm.py --tier 3

# Run all tiers (default)
python test_algorithm.py
```

### Known Optimal Results (Tier 1 verification)

| Instance | Expected | Method |
|----------|----------|--------|
| n=7, k=6, j=5, s=5 | 6 | exact |
| n=7, k=5, j=4, s=4 | 9 | exact |
| n=6, k=4, j=3, s=3 | 5+ | exact |

---

## File Structure

```
optimal-selection-v2/
├── algorithm.py                # Core solver (hybrid exact + heuristic + quality search)
├── web_app.py                  # Flask web app (S1 + S2)
├── app.py                      # Desktop GUI (tkinter)
├── algorithm_documentation.md  # Full algorithm documentation
├── IMPROVEMENTS.md             # v2 improvement summary
├── test_algorithm.py           # Three-tier automated test suite
├── start.sh                    # Launcher script
├── static/
│   ├── css/style.css           # Styling
│   └── favicon.svg             # Favicon
├── templates/
│   ├── index.html              # S1 — Computation UI
│   └── database.html           # S2 — Database browser
└── results/                    # Saved results (auto-created, one JSON per record)
```

---

## Parameter Guide

| Param | Range | Description |
|-------|-------|-------------|
| `m` | 45–54 | Total number of samples in the pool |
| `n` | 7–25 | Number of samples to select for grouping |
| `k` | 4–7 | Size of each group (k ≤ n) |
| `j` | s–k | Size of subsets that must be covered |
| `s` | 3–7 | Minimum number of groups covering each j-subset |
| `T` | ≥ 1 | Multi-cover count (each j-subset must appear in ≥ T groups) |

**Constraint:** `s ≤ j ≤ k ≤ n ≤ m`

---

## Output Interpretation

```
Found N groups   exact   [optimal]
  Lower bound: X | Gap: Y | Valid: True | Time: Zs
```

| Field | Meaning |
|-------|---------|
| `valid=True` | Every j-subset is covered — 100% guarantee |
| `lower_bound` | Theoretical minimum (true optimum ≥ lb) |
| `gap` | Distance from solution to lower bound (gap=0 → likely optimal) |
| `method` | `exact` (proven optimal) / `hybrid` / `heuristic` |
| `optimal=True` | Proven optimal by exact solver or gap=0 |

---

## Team Contributions

| Module | Student | Work Description |
|--------|---------|-----------------|
| **Algorithm Core** | [PANJIAYING] | Problem modeling, exact branch-and-bound solver, Schönheim/counting lower bounds |
| **Web Application** | [JIN XINYI] | Flask UI, background threading, progress streaming, database browser |
| **Heuristic Engine** | [LIU XINGZHE] | Greedy construction, LNS perturbation, local search (swap), warm-start mechanism |
| **Performance Acceleration** | [CHENG YUXIANG] | Bitmask encoding, Numba JIT parallelization, memory optimization |
| **Documentation & Testing** | [All Members] | Algorithm documentation, test suite (tier 1/2/3), presentation outline |

> **Note:** Please update the team member names and adjust roles based on actual contributions.

---

## License

This project was developed for CS360 Artificial Intelligence Group Project.

---

## References

- Schönheim, J. (1964). coverings of a complete graph.
- Covering Design — Wikipedia. https://en.wikipedia.org/wiki/Covering_design
- Numba JIT Compiler. https://numba.pydata.org/