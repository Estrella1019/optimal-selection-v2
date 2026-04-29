# Algorithm Improvements Summary

This version improves the solver with a quality-first strategy while keeping runtime under control.

- Added adaptive runtime budgets: medium cases use about 120 seconds by default, while larger cases can use up to about 600 seconds.
- Added a full-bitset quality search for suitable medium-sized cases, combining global greedy construction with fixed-B descent compression.
- Added scale protection so very large cases do not build an oversized full coverage model.
- Prevented duplicate k-groups from being selected, including multi-cover cases.
- Improved fixed-B search sampling to avoid bias toward lower-index subsets.
- Added final metadata for the actual time budget and whether quality search was enabled.
- Every returned solution is still fully verified: `Valid: true` means every j-subset is covered at least T times under the rule `|group intersection subset| >= s`.

For example, `n=20, k=6, j=5, s=4, T=1` now runs the extra quality search and typically returns around 178 groups in about 120 seconds on the tested machine.
