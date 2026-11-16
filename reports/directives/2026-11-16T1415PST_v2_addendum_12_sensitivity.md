# [2026-11-16 14:15 PST] Addendum 12: per-tile concavity sensitivity and equal-parameter comparisons

This addendum follows the closed v2 refactor (Addenda 1–11; release `v2.0.0`). Work happens on a new branch `v2.1-sensitivity` cut from `main` at the `v2.0.0` tag. Addendum 4 §1 (timestamped titles, one report per checkpoint), Addendum 2 §4–§5 (Slurm, storage, push) and Addendum 7 §0.4 (submit guard) still apply. Do not move the `v2.0.0` tag or change `main`.

## 0. Orientation (you are a new agent instance)

1. **Read:**
   - `reports/v2/20261116T1330PST_release.md` and `20261113T1330PST_wrapup.md` (or the latest wrap-up report);
   - `reports/v2/20261110T1540PST_checkpoint_4b_5.md` §5 (the site-m decision and S(m));
   - `docs/measurement_methods.md` §0;
   - this file.
2. **Confirm** that `main` is at the `v2.0.0` merge commit, that the tests pass on `serc`, and that no job is running.
3. **Where this addendum and the repo disagree,** the repo wins. Report it and stop.

**Why this addendum.** A review of the v2 brief raised two questions a referee will ask:
- **Concavity.** Do the headline results depend on using one concavity per site (AP m = 0.47, GM m = 0.31, fitted by the model)?
- **Parameter count.** Does the partitioned model beat AD only because it has a third free parameter (A_crit)?

This addendum answers both, mostly from runs that already exist.

## 1. Per-tile concavity sensitivity

### 1.1 Per-tile m

For each tile, take m_t, the minimum of that tile's own exponent profile MSE*(m), from the metre m-profiles of the 4b/5 checkpoint (grid 0.05–0.90). Report m_t with its 1% and 5% intervals in m.
- **Flag every m_t on a grid edge.** AP4 was at 0.21 on the old grid.
- **Beside m_t, report the measured site-pooled θ.** Use the pooled and equal-weight χ estimators at A_c = 10³ and 10⁴ m², with their 95% bands, from the Addendum 4 §6.4 ladder rerun. Say whether m_t and the site m fall inside them.

### 1.2 Rerun at m_t, all ten tiles, metres, v2 optimizer

1. **K₀(m_t):** k_s by χ–z at θ = m_t over the network above 10³ m², pooled through-origin estimator, as in Phase 6. D₀ does not depend on m and is unchanged; AP keeps its bracket.
2. **Linear partitioned joint profile at m_t:** thin to 2000 values and refine, as in Phase 6.
3. **AD profile at m_t:** over r, with Brent refinement.

### 1.3 Report, side by side at site m and at m_t

- K/K₀, D/D₀ (both ends of the bracket on AP) and A_crit;
- the MSE of AD, the MSE of the partitioned model, and the ratio AD/partitioned;
- the AD K/K₀ and AD D/D₀.

### 1.4 Rule, written now

The site-m choice is **immaterial** at a site if both of these hold:
- the partitioned MSE is below the AD MSE on 5 of 5 tiles at m_t;
- the site's median partitioned D/D₀,A changes by less than 25% and its median AD K/K₀ by less than 25%.

Otherwise, report which conclusion changes, and on which tiles.

## 2. Equal-parameter comparisons

All at the site m. Use the exact reductions:
- the partitioned model at fixed A_crit: Z = a + (U/K)b + (U/D)c;
- the AD model at fixed r: Z = h(r) + (U/D)g(r).

Every fixed-parameter variant is a constrained case of the existing least-squares problem or profile. No new solver is needed.

### 2.1 Variants (free parameters in brackets)

| Count | Partitioned | AD |
| --- | --- | --- |
| 1 | P1: stage 2, K₀ and D₀ fixed [A_crit] (exists) | A1: K fixed at K₀ [D]; A2: D fixed at D₀ [K] |
| 2 | P2: D fixed at D₀ [K, A_crit]; P3: K fixed at K₀ [D, A_crit] | AD: [K, D] (exists) |
| 3 | P4: joint [K, D, A_crit] (exists) | — |

On AP, run every variant that fixes D at both D₀,A and D₀,B.

### 2.2 Held-out comparison

Split each tile's basins into two halves with equal total cell count:
- **Basins:** outlets on the tile edge, at the 10³ m² network.
- **Split:** greedy, largest basin first, each to the lighter half.
- **Report** the cell share of each half; this replaces the lopsided split of Addendum 8.

Fit P4 and AD on one half, score the MSE on the other, both ways, and average. Do the same for P2 against AD, which have equal parameter counts.

### 2.3 Rule, written now (per site)

- **Equal count, in-sample.** The partitioned model wins at a count if its MSE is lower than every AD variant of the same count on at least 4 of 5 tiles. Report counts 1 and 2 separately. On AP, report both D₀ ends.
- **Held-out.** P4 beats AD out of sample if its held-out MSE is lower on at least 4 of 5 tiles.
- **Report every case,** including those where AD wins. No combined score.

## 3. Cost and order

1. Print the core-hour estimate first: about 10 tiles × (AD profile ≈ 1.6 h + partitioned ≈ 3–12 h) for §1, plus the §2 variants, which are cheap by reduction. Submit without asking only if:
   - the total is at most 300 core-hours;
   - every submission is at most 300;
   - every array is throttled.

   Otherwise stop and report.
2. Do §2 first, since it is mostly existing reductions, then §1.

## 4. Report and stop

One report, `reports/v2/<YYYYMMDDTHHMM>PST_sensitivity.md`, titled with its timestamp. Fixed outline:

1. State check.
2. Equal-parameter comparisons, in-sample and held-out, with the rule applied.
3. Per-tile m: m_t against the measured θ and the site m.
4. Results at m_t against the site m, with the rule applied.
5. What the evidence supports, and where it is weak.
6. Decisions needed.
7. Appendices: all generated tables.

**Stop for review.** Push the branch after the Addendum 2 §5 check. Do not merge or tag.
