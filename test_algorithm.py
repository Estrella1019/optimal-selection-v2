"""
test_algorithm.py — Test suite for the covering-design solver.

Tests are grouped by scale:
  Tier 1 — tiny   : verify correctness against known values (< 1 s each)
  Tier 2 — medium : quality check on moderate instances   (< 60 s each)
  Tier 3 — stress : worst-case timing test                (≤ 1200 s)

Run:
    python test_algorithm.py            # all tiers
    python test_algorithm.py --tier 1   # correctness only
    python test_algorithm.py --tier 3   # worst-case only
"""

import sys
import time
from math import comb
from algorithm import solve, verify, preprocess, counting_lower_bound


# ── helpers ─────────────────────────────────────────────────────────────────

def banner(msg: str):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print('='*60)


def check(label: str, n_samples, k, j, s, T=1,
          expected_size=None, expected_method=None,
          time_limit=60, seed=42):
    """
    Run solve() and print a one-line verdict.

    expected_size : if given, assert solution_size == expected_size.
    """
    banner(label)
    sol, info = solve(n_samples, k, j, s, T=T,
                      time_limit=time_limit, seed=seed, verbose=True)

    ok_valid  = info['valid']
    ok_size   = (expected_size is None) or (info['solution_size'] == expected_size)
    ok_method = (expected_method is None) or (info.get('method') == expected_method)

    status = "PASS" if (ok_valid and ok_size and ok_method) else "FAIL"
    if not ok_valid:
        status += " [INVALID COVERAGE]"
    if not ok_size:
        status += f" [expected {expected_size}, got {info['solution_size']}]"
    if not ok_method:
        status += f" [expected method={expected_method}, got {info.get('method')}]"

    print(f"\n  [{status}]  size={info['solution_size']}  "
          f"lb={info['lower_bound']}  gap={info['gap']}  "
          f"time={info['time']}s  restarts={info['restarts']}  "
          f"method={info.get('method')}  optimal={info.get('optimal')}")

    print("\n  Solution groups:")
    for g in sol:
        print(f"    {list(g)}")

    return status.startswith("PASS"), info


# ── Tier 1: correctness ──────────────────────────────────────────────────────

def tier1():
    banner("TIER 1 — Correctness / tiny instances")
    results = []

    # n=7, k=6, j=5, s=5
    # Proof that 6 is optimal:  any two 6-groups (from 7 samples) share exactly
    # one 5-subset; 5 groups cover at most 5*6 - C(5,2)*1 = 20 < C(7,5)=21.
    p, _ = check("n=7, k=6, j=5, s=5  (known OPT=6)",
                 list(range(1, 8)), k=6, j=5, s=5,
                 expected_size=6, expected_method="exact")
    results.append(("n=7,k=6,j=5,s=5", p))

    # n=7, k=5, j=4, s=4
    # Counting lb = ceil(35/5) = 7, but Steiner system S(4,5,7) doesn't exist
    # (3 ∤ 10), so OPT ≥ 9. Schönheim bound = 9. Algorithm should find 9.
    p, _ = check("n=7, k=5, j=4, s=4  (Schönheim lb=9, expect 9)",
                 list(range(1, 8)), k=5, j=4, s=4,
                 expected_size=9, expected_method="exact")
    results.append(("n=7,k=5,j=4,s=4", p))

    # n=6, k=4, j=3, s=3
    # Each 4-group covers C(4,3)=4 three-subsets; C(6,3)=20 total.
    # lb = ceil(20/4) = 5.
    p, _ = check("n=6, k=4, j=3, s=3  (lb=5)",
                 list(range(1, 7)), k=4, j=3, s=3,
                 expected_method="exact")
    results.append(("n=6,k=4,j=3,s=3", p))

    # n=8, k=6, j=5, s=5
    p, _ = check("n=8, k=6, j=5, s=5",
                 list(range(1, 9)), k=6, j=5, s=5,
                 expected_method="exact")
    results.append(("n=8,k=6,j=5,s=5", p))

    # T=2 (multi-cover): n=7, k=6, j=5, s=5, T=2
    p, _ = check("n=7, k=6, j=5, s=5, T=2",
                 list(range(1, 8)), k=6, j=5, s=5, T=2)
    results.append(("n=7,k=6,j=5,s=5,T=2", p))

    return results


# ── Tier 2: medium quality ───────────────────────────────────────────────────

def tier2():
    banner("TIER 2 — Medium instances / quality")
    results = []

    cases = [
        ("n=10, k=5, j=4, s=4", list(range(1, 11)), 5, 4, 4, 1, 60),
        ("n=12, k=6, j=5, s=4", list(range(1, 13)), 6, 5, 4, 1, 60),
        ("n=15, k=6, j=5, s=5", list(range(1, 16)), 6, 5, 5, 1, 60),
        ("n=15, k=7, j=6, s=5", list(range(1, 16)), 7, 6, 5, 1, 60),
        ("n=20, k=6, j=5, s=4", list(range(1, 21)), 6, 5, 4, 1, 120),
        ("n=20, k=7, j=6, s=5", list(range(1, 21)), 7, 6, 5, 1, 120),
    ]

    for label, samp, k, j, s, T, tl in cases:
        p, info = check(label, samp, k, j, s, T=T, time_limit=tl)
        results.append((label, p, info))

    return results


# ── Tier 3: stress / worst-case timing ──────────────────────────────────────

def tier3():
    banner("TIER 3 — Worst-case stress test (time_limit=1200s)")
    results = []

    cases = [
        # Large j-subset count, hard coverage
        ("n=25, k=7, j=7, s=5  [hardest coverage]",
         list(range(1, 26)), 7, 7, 5, 1, 1200),

        # Large candidate set per step, but small OPT
        ("n=25, k=7, j=6, s=3  [many candidates, small OPT]",
         list(range(1, 26)), 7, 6, 3, 1, 1200),

        # Max n, challenging balance
        ("n=25, k=7, j=6, s=5",
         list(range(1, 26)), 7, 6, 5, 1, 1200),
    ]

    for label, samp, k, j, s, T, tl in cases:
        p, info = check(label, samp, k, j, s, T=T, time_limit=tl)
        results.append((label, p, info))

    return results


# ── entry point ─────────────────────────────────────────────────────────────

def main():
    tier_arg = None
    if "--tier" in sys.argv:
        idx = sys.argv.index("--tier")
        tier_arg = int(sys.argv[idx + 1])

    t_total = time.perf_counter()
    all_results = []

    if tier_arg in (None, 1):
        all_results += [(name, ok) for name, ok in tier1()]

    if tier_arg in (None, 2):
        all_results += [(r[0], r[1]) for r in tier2()]

    if tier_arg in (None, 3):
        all_results += [(r[0], r[1]) for r in tier3()]

    # Summary
    banner("SUMMARY")
    passed = sum(1 for _, ok in all_results if ok)
    total  = len(all_results)
    for name, ok in all_results:
        mark = "✓" if ok else "✗"
        print(f"  {mark}  {name}")
    print(f"\n  {passed}/{total} passed  |  "
          f"total time: {time.perf_counter()-t_total:.1f}s")


if __name__ == "__main__":
    main()
