from collections import OrderedDict

import numpy as np
from itertools import combinations
from math import comb
import time


DEFAULT_TIME_LIMIT = 30.0
EXACT_MAX_CANDIDATES = 126
EXACT_MAX_SECONDS = 45.0
HEURISTIC_CACHE_SIZE = 32
HEURISTIC_CACHEABLE_CANDIDATES = 1500
NUMBA_CACHE = False

# ── numba JIT setup ──────────────────────────────────────────────────────────

try:
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:                         # graceful fallback
    _HAS_NUMBA = False
    def njit(**kw):
        return lambda f: f
    def prange(n):
        return range(n)


@njit(cache=NUMBA_CACHE)
def _popcount(x: np.uint32) -> np.int32:

    c = np.int32(0)
    while x:
        x = x & (x - np.uint32(1))
        c += np.int32(1)
    return c


@njit(parallel=True, cache=NUMBA_CACHE)
def _gains_parallel(cand_masks: np.ndarray,
                    uncov_j_masks: np.ndarray,
                    s: np.int32) -> np.ndarray:
    n_c = len(cand_masks)
    n_u = len(uncov_j_masks)
    gains = np.zeros(n_c, dtype=np.int32)
    for ci in prange(n_c):
        g = cand_masks[ci]
        gain = np.int32(0)
        for ui in range(n_u):
            if _popcount(g & uncov_j_masks[ui]) >= s:
                gain += np.int32(1)
        gains[ci] = gain
    return gains


@njit(parallel=True, cache=NUMBA_CACHE)
def _weighted_gains_parallel(cand_masks: np.ndarray,
                             uncov_j_masks: np.ndarray,
                             weights: np.ndarray,
                             s: np.int32) -> np.ndarray:
    n_c = len(cand_masks)
    n_u = len(uncov_j_masks)
    gains = np.zeros(n_c, dtype=np.int32)
    for ci in prange(n_c):
        g = cand_masks[ci]
        gain = np.int32(0)
        for ui in range(n_u):
            if _popcount(g & uncov_j_masks[ui]) >= s:
                gain += weights[ui]
        gains[ci] = gain
    return gains



@njit(cache=NUMBA_CACHE)
def _update_cover(best_mask: np.uint32,
                  j_masks: np.ndarray,
                  cover_count: np.ndarray,
                  s: np.int32) -> None:
    for i in range(len(j_masks)):
        if _popcount(best_mask & j_masks[i]) >= s:
            cover_count[i] += np.int32(1)


@njit(cache=NUMBA_CACHE)
def _covered_indices(g_mask: np.uint32,
                     j_masks: np.ndarray,
                     s: np.int32) -> np.ndarray:
    n = len(j_masks)
    buf = np.empty(n, dtype=np.int32)
    cnt = 0
    for i in range(n):
        if _popcount(g_mask & j_masks[i]) >= s:
            buf[cnt] = i
            cnt += 1
    return buf[:cnt]


@njit(parallel=True, cache=NUMBA_CACHE)
def _filter_cover_all(cand_masks: np.ndarray,
                      j_masks: np.ndarray,
                      under_indices: np.ndarray,
                      s: np.int32) -> np.ndarray:
    n_c = len(cand_masks)
    result = np.zeros(n_c, dtype=np.bool_)
    for ci in prange(n_c):
        gm = cand_masks[ci]
        ok = True
        for ui in range(len(under_indices)):
            if _popcount(gm & j_masks[under_indices[ui]]) < s:
                ok = False
        result[ci] = ok
    return result


# ── warm-up: trigger compilation on import ───────────────────────────────────

def _warmup():
    dummy_c = np.array([np.uint32(0b111)], dtype=np.uint32)
    dummy_j = np.array([np.uint32(0b110), np.uint32(0b101)], dtype=np.uint32)
    dummy_w = np.array([1, 2], dtype=np.int32)
    dummy_cc = np.zeros(2, dtype=np.int32)
    dummy_under = np.array([0, 1], dtype=np.int32)
    _ = _gains_parallel(dummy_c, dummy_j, np.int32(2))
    _ = _weighted_gains_parallel(dummy_c, dummy_j, dummy_w, np.int32(2))
    _update_cover(np.uint32(0b111), dummy_j, dummy_cc, np.int32(2))
    _ = _covered_indices(np.uint32(0b111), dummy_j, np.int32(2))
    _ = _filter_cover_all(dummy_c, dummy_j, dummy_under, np.int32(2))


if _HAS_NUMBA:
    _warmup()


# ── helpers ──────────────────────────────────────────────────────────────────

def _mask(sub: tuple) -> np.uint32:
    m = np.uint32(0)
    for x in sub:
        m |= np.uint32(1) << np.uint32(x)
    return m


# ── preprocessing ────────────────────────────────────────────────────────────

def preprocess(n: int, j: int):
    j_subsets = list(combinations(range(n), j))
    j_masks = np.array([_mask(sub) for sub in j_subsets], dtype=np.uint32)
    return j_subsets, j_masks


# ── candidate generation (the pruning step) ──────────────────────────────────

def candidates_for(j_sub: tuple, n: int, k: int, s: int) -> list:
    j_set = set(j_sub)
    others = [x for x in range(n) if x not in j_set]
    result = []
    jl = len(j_sub)
    for t in range(s, min(jl, k) + 1):
        for from_j in combinations(j_sub, t):
            for from_o in combinations(others, k - t):
                result.append(tuple(sorted(from_j + from_o)))
    return result


# ── lower bound ──────────────────────────────────────────────────────────────

def counting_lower_bound(n: int, k: int, j: int, s: int, T: int) -> int:
    N_j = comb(n, j)
    max_cov = sum(
        comb(k, t) * comb(n - k, j - t)
        for t in range(s, min(j, k) + 1)
        if 0 <= j - t <= n - k
    )
    if max_cov == 0:
        return 0
    return -(-N_j * T // max_cov)   # 等价于 ceil(N_j*T / max_cov)


def _get_candidate_pool(seed_idx: int, j_subsets: list,
                        n: int, k: int, s: int,
                        cand_cache: OrderedDict | None):
    if cand_cache is not None and seed_idx in cand_cache:
        cands, cand_masks = cand_cache.pop(seed_idx)
        cand_cache[seed_idx] = (cands, cand_masks)
        return cands, cand_masks

    cands = candidates_for(j_subsets[seed_idx], n, k, s)
    cand_masks = np.array([_mask(g) for g in cands], dtype=np.uint32)

    if (cand_cache is not None
            and len(cands) <= HEURISTIC_CACHEABLE_CANDIDATES):
        cand_cache[seed_idx] = (cands, cand_masks)
        while len(cand_cache) > HEURISTIC_CACHE_SIZE:
            cand_cache.popitem(last=False)

    return cands, cand_masks


def _greedy_exact_cover(cover_bits: list[int], full_mask: int) -> list[int]:
    covered = 0
    selected = []
    available = set(range(len(cover_bits)))

    while covered != full_mask:
        best_idx = None
        best_gain = 0
        uncovered = full_mask & ~covered
        for idx in available:
            gain = (cover_bits[idx] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        if best_idx is None or best_gain == 0:
            break
        selected.append(best_idx)
        covered |= cover_bits[best_idx]
        available.remove(best_idx)

    return selected


def exact_cover_solve(n: int, k: int, j: int, s: int,
                      j_masks: np.ndarray,
                      time_budget: float,
                      verbose: bool = True):
    if comb(n, k) > EXACT_MAX_CANDIDATES or time_budget <= 0:
        return None

    t0 = time.perf_counter()
    deadline = t0 + min(EXACT_MAX_SECONDS, max(2.0, time_budget * 0.25), time_budget)
    s32 = np.int32(s)

    dedup = {}
    for g in combinations(range(n), k):
        gm = _mask(g)
        cover_bits = 0
        for idx in _covered_indices(gm, j_masks, s32):
            cover_bits |= 1 << int(idx)
        if cover_bits and cover_bits not in dedup:
            dedup[cover_bits] = (gm, g)

    if not dedup:
        return None

    ordered = sorted(dedup.items(),
                     key=lambda item: item[0].bit_count(),
                     reverse=True)
    cover_bits = [item[0] for item in ordered]
    cand_masks = [item[1][0] for item in ordered]
    cand_tuples = [item[1][1] for item in ordered]

    n_j = len(j_masks)
    full_mask = (1 << n_j) - 1
    greedy_idx = _greedy_exact_cover(cover_bits, full_mask)
    greedy_bits = 0
    for idx in greedy_idx:
        greedy_bits |= cover_bits[idx]
    if greedy_bits != full_mask:
        return None

    best_idx = list(greedy_idx)
    best_size = len(best_idx)
    timed_out = False

    if verbose:
        print(f"  Exact search: {len(cover_bits)} canonical candidates "
              f"| budget {deadline - t0:.1f}s")

    coverers_by_j = [[] for _ in range(n_j)]
    for idx, bits in enumerate(cover_bits):
        work = bits
        while work:
            low_bit = work & -work
            coverers_by_j[low_bit.bit_length() - 1].append(idx)
            work ^= low_bit

    suffix_union = [0] * (len(cover_bits) + 1)
    suffix_best = [0] * (len(cover_bits) + 1)
    for idx in range(len(cover_bits) - 1, -1, -1):
        suffix_union[idx] = suffix_union[idx + 1] | cover_bits[idx]
        suffix_best[idx] = max(suffix_best[idx + 1], cover_bits[idx].bit_count())

    def pick_target(uncovered: int, start: int):
        chosen = -1
        best_count = None
        work = uncovered
        while work:
            low_bit = work & -work
            subset_idx = low_bit.bit_length() - 1
            cnt = 0
            for cand_idx in coverers_by_j[subset_idx]:
                if cand_idx >= start:
                    cnt += 1
                    if best_count is not None and cnt >= best_count:
                        break
            if cnt == 0:
                return subset_idx, 0
            if best_count is None or cnt < best_count:
                chosen = subset_idx
                best_count = cnt
                if cnt == 1:
                    return chosen, best_count
            work ^= low_bit
        return chosen, best_count

    def backtrack(start: int, covered: int, selected: list[int]) -> None:
        nonlocal best_idx, best_size, timed_out

        if time.perf_counter() >= deadline:
            timed_out = True
            return
        if covered == full_mask:
            if len(selected) < best_size:
                best_idx = list(selected)
                best_size = len(selected)
            return
        if start >= len(cover_bits) or len(selected) >= best_size:
            return
        if (covered | suffix_union[start]) != full_mask:
            return

        uncovered = full_mask & ~covered
        max_single = suffix_best[start]
        if max_single == 0:
            return
        lb = -(-uncovered.bit_count() // max_single)
        if len(selected) + lb >= best_size:
            return

        target_idx, cnt = pick_target(uncovered, start)
        if cnt == 0:
            return

        relevant = [
            cand_idx for cand_idx in coverers_by_j[target_idx]
            if cand_idx >= start and (cover_bits[cand_idx] & uncovered)
        ]
        relevant.sort(
            key=lambda cand_idx: (cover_bits[cand_idx] & uncovered).bit_count(),
            reverse=True,
        )

        for cand_idx in relevant:
            selected.append(cand_idx)
            backtrack(cand_idx + 1, covered | cover_bits[cand_idx], selected)
            selected.pop()
            if timed_out:
                return

    backtrack(0, 0, [])

    elapsed = time.perf_counter() - t0
    result = {
        "elapsed": elapsed,
        "timed_out": timed_out,
        "masks": [cand_masks[idx] for idx in best_idx],
        "tuples": [cand_tuples[idx] for idx in best_idx],
    }

    if verbose:
        if timed_out:
            print(f"  Exact search paused at {elapsed:.1f}s; "
                  "continuing with heuristic improvement.")
        else:
            print(f"  Exact search solved optimally in {elapsed:.2f}s.")

    return result


# ── single greedy run ─────────────────────────────────────────────────────────

def greedy_once(n: int, k: int, j: int, s: int, T: int,
                j_subsets: list, N_j: int,
                j_masks: np.ndarray, rng,
                cand_cache: OrderedDict | None = None,
                t_deadline: float = float('inf'),
                init_masks=None, init_tuples=None) -> list:
    cover_count = np.zeros(N_j, dtype=np.int32)
    selected_masks = []
    selected_tuples = []
    s32 = np.int32(s)

    # LNS warm-start: pre-load surviving groups
    if init_masks is not None:
        for gm in init_masks:
            _update_cover(gm, j_masks, cover_count, s32)
        selected_masks = list(init_masks)
        selected_tuples = list(init_tuples)

    while True:
        uncov_indices = np.where(cover_count < T)[0]
        if len(uncov_indices) == 0:
            break

        # Deadline check (time budget enforcement)
        if time.perf_counter() >= t_deadline:
            # Fast completion: cover remaining by greedy with any valid candidate
            for idx in uncov_indices:
                if cover_count[idx] >= T:
                    continue
                cands, cand_masks = _get_candidate_pool(
                    int(idx), j_subsets, n, k, s, cand_cache)
                if not cands:
                    continue
                # Pick first candidate that covers this j-subset
                for pos, g in enumerate(cands):
                    gm = cand_masks[pos]
                    if _popcount(gm & j_masks[int(idx)]) >= s32:
                        _update_cover(gm, j_masks, cover_count, s32)
                        selected_masks.append(gm)
                        selected_tuples.append(g)
                        break
            break

        # Pick most-urgent uncovered j-subset (min count), random tie-break
        min_cnt = int(cover_count[uncov_indices].min())
        urgent = uncov_indices[cover_count[uncov_indices] == min_cnt]
        pick = int(rng.choice(urgent))

        # Generate focused candidate set
        cands, cand_masks = _get_candidate_pool(
            pick, j_subsets, n, k, s, cand_cache)

        uncov_j_masks = j_masks[uncov_indices]
        deficits = (T - cover_count[uncov_indices]).astype(np.int32)

        # Score candidates by the total deficit they repair across all pending subsets.
        gains = _weighted_gains_parallel(cand_masks, uncov_j_masks, deficits, s32)

        best_gain = int(gains.max())
        if best_gain <= 0:
            break

        # Random tie-break among candidates with maximum gain
        top_idx = np.where(gains == best_gain)[0]
        chosen = int(top_idx[rng.integers(len(top_idx))])
        best_g = cands[chosen]
        best_mask = cand_masks[chosen]

        _update_cover(best_mask, j_masks, cover_count, s32)
        selected_masks.append(best_mask)
        selected_tuples.append(best_g)

    return selected_masks, selected_tuples


# ── post-processing: remove redundant groups ─────────────────────────────────

def minimize_solution(selected_masks: list, selected_tuples: list,
                      j_masks: np.ndarray, N_j: int,
                      s: int, T: int,
                      t_deadline: float = float('inf')):
    s32 = np.int32(s)
    cover_count = np.zeros(N_j, dtype=np.int32)
    cov_cache = {}
    for gm in selected_masks:
        cov = _covered_indices(gm, j_masks, s32)
        cov_cache[int(gm)] = cov
        cover_count[cov] += 1

    masks = list(selected_masks)
    tuples = list(selected_tuples)
    changed = True
    while changed and time.perf_counter() < t_deadline:
        changed = False
        for i in range(len(masks) - 1, -1, -1):
            if time.perf_counter() >= t_deadline:
                break
            gm = int(masks[i])
            cov = cov_cache[gm]
            # Safe to remove iff cover_count[cov] > T for all positions in cov
            if len(cov) == 0 or bool(np.all(cover_count[cov] > T)):
                cover_count[cov] -= 1
                masks.pop(i)
                tuples.pop(i)
                changed = True

    return masks, tuples


# ── local search: swap-based polish ──────────────────────────────────────────

def local_search_swap(masks: list, tuples: list,
                      j_subsets: list, j_masks: np.ndarray,
                      N_j: int, n: int, k: int, j: int, s: int, T: int,
                      rng, cand_cache: OrderedDict | None = None,
                      t_deadline: float = float('inf')):
    s32 = np.int32(s)
    masks = list(masks)
    tuples = list(tuples)

    # Upper bound on j-subsets one k-group can cover (filter: skip if |U| > this)
    max_single_cov = sum(
        comb(k, t) * comb(n - k, j - t)
        for t in range(s, min(j, k) + 1)
        if 0 <= j - t <= n - k
    )

    improved = True
    while improved and time.perf_counter() < t_deadline:
        improved = False

        # Rebuild coverage structures at the start of each pass
        cover_count = np.zeros(N_j, dtype=np.int32)
        cov_cache: dict = {}
        for gm in masks:
            key = int(gm)
            if key not in cov_cache:
                cov_cache[key] = _covered_indices(gm, j_masks, s32)
            cover_count[cov_cache[key]] += 1

        order = list(range(len(masks)))
        rng.shuffle(order)

        for i in order:
            if time.perf_counter() >= t_deadline:
                break
            if i >= len(masks):
                break

            gm_i = masks[i]
            cov_i = cov_cache[int(gm_i)]

            # Coverage after hypothetically removing group i
            reduced = cover_count.copy()
            reduced[cov_i] -= 1
            under = np.where(reduced < T)[0]

            # Group i is redundant — remove directly (shouldn't happen post-minimize)
            if len(under) == 0:
                masks.pop(i)
                tuples.pop(i)
                improved = True
                break

            # No single candidate can cover all of U → skip
            if len(under) > max_single_cov:
                continue

            # Seed candidates from a random element of U (try up to 5 seeds)
            n_seeds = min(len(under), 5)
            seed_positions = rng.choice(len(under), n_seeds, replace=False)

            found = False
            for sp in seed_positions:
                if found:
                    break
                seed_idx = int(under[sp])
                cands, cand_masks_arr = _get_candidate_pool(
                    seed_idx, j_subsets, n, k, s, cand_cache)
                if not cands:
                    continue
                under_arr = under.astype(np.int32)

                # Keep only candidates covering ALL under-covered j-subsets
                valid_flags = _filter_cover_all(cand_masks_arr, j_masks,
                                                under_arr, s32)
                valid_indices = np.where(valid_flags)[0]
                if len(valid_indices) == 0:
                    continue

                for ci in valid_indices:
                    cm = cand_masks_arr[int(ci)]
                    key_new = int(cm)
                    if key_new not in cov_cache:
                        cov_cache[key_new] = _covered_indices(cm, j_masks, s32)
                    cov_new = cov_cache[key_new]

                    # Coverage after: remove g_i, add g_new
                    test_count = reduced.copy()
                    test_count[cov_new] += 1

                    # Check if any other group is now redundant
                    for j_idx in range(len(masks)):
                        if j_idx == i:
                            continue
                        cov_j = cov_cache[int(masks[j_idx])]
                        if len(cov_j) == 0 or bool(np.all(test_count[cov_j] > T)):
                            # Accept: replace g_i with g_new, remove g_j
                            masks[i] = cm
                            tuples[i] = cands[int(ci)]
                            masks.pop(j_idx)
                            tuples.pop(j_idx)
                            improved = True
                            found = True
                            break

                    if found:
                        break

            if improved:
                break

    return masks, tuples


# ── verification ─────────────────────────────────────────────────────────────

def verify(selected_masks: list, j_masks: np.ndarray,
           N_j: int, s: int, T: int) -> bool:
    s32 = np.int32(s)
    cover_count = np.zeros(N_j, dtype=np.int32)
    for gm in selected_masks:
        cover_count[_covered_indices(gm, j_masks, s32)] += 1
    return bool(np.all(cover_count >= T))


# ── public solver ─────────────────────────────────────────────────────────────

def solve(n_samples: list, k: int, j: int, s: int,
          T: int = 1, time_limit: float = DEFAULT_TIME_LIMIT,
          seed: int = 42, verbose: bool = True):
    n = len(n_samples)
    samples = list(n_samples)
    assert s <= j <= k <= n, f"Need s≤j≤k≤n, got s={s},j={j},k={k},n={n}"

    if verbose:
        nc = sum(comb(j, t) * comb(n - j, k - t)
                 for t in range(s, min(j, k) + 1))
        print(f"Problem: n={n}, k={k}, j={j}, s={s}, T={T}")
        print(f"  C(n,j) = {comb(n,j):,} j-subsets | ~{nc:,} candidates/step")

    t_pre = time.perf_counter()
    j_subsets, j_masks = preprocess(n, j)
    N_j = len(j_subsets)
    lb = counting_lower_bound(n, k, j, s, T)
    if verbose:
        print(f"  Preprocessing: {time.perf_counter()-t_pre:.2f}s | "
              f"Lower bound >= {lb}")

    overall_deadline = time.perf_counter() + time_limit
    rng = np.random.default_rng(seed)
    cand_cache = OrderedDict()
    best_masks, best_tuples = None, None
    method = "heuristic"
    exact_phase_used = False
    t0 = time.perf_counter()

    if T == 1:
        exact_result = exact_cover_solve(
            n, k, j, s, j_masks,
            time_budget=max(0.0, overall_deadline - time.perf_counter()),
            verbose=verbose,
        )
        if exact_result is not None:
            exact_phase_used = True
            best_masks = exact_result["masks"]
            best_tuples = exact_result["tuples"]
            if best_masks:
                best_masks, best_tuples = minimize_solution(
                    best_masks, best_tuples, j_masks, N_j, s, T,
                    t_deadline=overall_deadline)
            if not exact_result["timed_out"]:
                elapsed = time.perf_counter() - t0
                result = [tuple(sorted(samples[i] for i in g)) for g in best_tuples]
                info = {
                    'solution_size': len(result),
                    'lower_bound': lb,
                    'gap': len(result) - lb,
                    'restarts': 0,
                    'time': round(elapsed, 2),
                    'valid': True,
                    'method': 'exact',
                    'optimal': True,
                    'exact_phase_used': True,
                }
                if verbose:
                    print(f"\nResult: {len(result)} groups | lb={lb} | "
                          f"gap={info['gap']} | valid=True | "
                          f"method=exact | optimal=True | time={elapsed:.1f}s")
                return result, info
            method = "hybrid"
            if verbose and best_masks is not None:
                print(f"  Hybrid warm start: {len(best_masks)} groups from "
                      "the exact phase incumbent.")

    restarts = 0
    no_improve = 0
    heuristic_budget = max(0.0, overall_deadline - time.perf_counter())
    max_no_improve = max(300, min(2500, int(heuristic_budget) + N_j // 40))
    lns_trigger = 4 if heuristic_budget >= 300 else 5

    while time.perf_counter() < overall_deadline:
        restarts += 1
        remaining = overall_deadline - time.perf_counter()
        if remaining <= 0:
            break
        deadline = time.perf_counter() + remaining * 0.92

        # LNS perturbation or full random restart
        use_lns = (best_masks is not None
                   and no_improve >= lns_trigger
                   and rng.random() < (0.85 if no_improve < lns_trigger + 8 else 0.95))
        remove_frac = min(0.65, 0.25 + 0.03 * max(0, no_improve - lns_trigger))

        if use_lns:
            n_keep = max(1, int(len(best_masks) * (1 - remove_frac)))
            keep_idx = sorted(rng.choice(len(best_masks), n_keep, replace=False).tolist())
            init_m = [best_masks[i] for i in keep_idx]
            init_t = [best_tuples[i] for i in keep_idx]
            m, t = greedy_once(n, k, j, s, T, j_subsets, N_j,
                               j_masks, rng, cand_cache=cand_cache,
                               t_deadline=deadline,
                               init_masks=init_m, init_tuples=init_t)
        else:
            m, t = greedy_once(n, k, j, s, T, j_subsets, N_j,
                               j_masks, rng, cand_cache=cand_cache,
                               t_deadline=deadline)

        m, t = minimize_solution(
            m, t, j_masks, N_j, s, T, t_deadline=overall_deadline)

        if best_masks is None or len(m) < len(best_masks):
            best_masks, best_tuples = m, t
            no_improve = 0
            if verbose:
                elapsed = time.perf_counter() - t0
                mode = "LNS" if use_lns else "rnd"
                print(f"  restart {restarts:3d} [{mode}]: {len(m)} groups  "
                      f"(lb={lb}, gap={len(m)-lb})  @ {elapsed:.1f}s")
        else:
            no_improve += 1

        if len(best_masks) <= lb:
            if verbose:
                print("  Reached lower bound — stopping.")
            break
        if no_improve >= max_no_improve:
            if verbose:
                print(f"  Converged ({no_improve} restarts without improvement).")
            break

    # ── local search polish (uses remaining time budget) ──────────────────────
    elapsed = time.perf_counter() - t0
    remaining = max(0.0, overall_deadline - time.perf_counter())
    if best_masks is None:
        raise RuntimeError("Solver failed to produce a feasible solution.")
    if remaining > 10.0 and len(best_masks) > lb:
        if verbose:
            print(f"\n  Local search polish  (budget {remaining:.0f}s) ...")
        ls_deadline = time.perf_counter() + remaining - 1.0
        before_ls = len(best_masks)
        best_masks, best_tuples = local_search_swap(
            best_masks, best_tuples, j_subsets, j_masks,
            N_j, n, k, j, s, T, rng,
            cand_cache=cand_cache, t_deadline=ls_deadline)
        if verbose:
            elapsed = time.perf_counter() - t0
            print(f"  After local search: {len(best_masks)} groups  "
                  f"(reduced by {before_ls - len(best_masks)})  @ {elapsed:.1f}s")

    elapsed = time.perf_counter() - t0

    # Restore original sample IDs
    result = [tuple(sorted(samples[i] for i in g)) for g in best_tuples]

    ok = verify(best_masks, j_masks, N_j, s, T)
    optimal = bool(len(result) == lb)
    info = {
        'solution_size': len(result),
        'lower_bound':   lb,
        'gap':           len(result) - lb,
        'restarts':      restarts,
        'time':          round(elapsed, 2),
        'valid':         ok,
        'method':        method,
        'optimal':       optimal,
        'exact_phase_used': exact_phase_used,
    }

    if verbose:
        print(f"\nResult: {len(result)} groups | lb={lb} | "
              f"gap={info['gap']} | valid={ok} | "
              f"method={method} | optimal={optimal} | "
              f"restarts={restarts} | time={elapsed:.1f}s")

    return result, info
