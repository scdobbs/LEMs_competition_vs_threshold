# Directive: Correct the Allegheny Plateau vertical-unit error

## Context

Repo: `~/LEMs_competition_vs_threshold-JGR-branch`. Environment: `$GROUP_HOME/scdobbs-envs/lem-widths`.

**The problem.** The five Allegheny Plateau (AP) tiles have horizontal coordinates in metres (0.9752 m pixels, reprojected from the 3.2 ft PAMAP cell) but elevations in **US survey feet**. Evidence:

- filled-grid elevations run from 1043 to 1521, i.e. 318–464 m once converted;
- the median pixel slope is 0.88;
- no 0.3048 factor appears anywhere in the repo.

The Gabilan Mesa (GM) grids are in metres (135–425 m).

**Let c = 3.2808** (feet per metre, US survey). On the AP tiles, every quantity built from z is off by a power of c.

| quantity | stored value relative to true | correction |
|---|---|---|
| elevation, relief, depth, MSE^½ | × c | × 0.3048 |
| slope, k_s, hilltop curvature, any ∇z or ∇²z | × c | ÷ c |
| MSE | × c² | ÷ c² (× 0.0929) |
| K, D (measured as K₀ = U/k_s, D₀ = U/C_ht, and optimized) | ÷ c | × c |
| A_crit, drainage area, D8 directions, θ | unchanged | none |

**Why most results survive.** U = 5.5×10⁻⁵ m/yr was used as metres. Both the advection–diffusion (AD) model and the linear partitioned model are linear in z (n = 1, linear diffusion, Dirichlet boundaries taken from the same DEM). Scaling z by c is therefore exactly equivalent to dividing K and D by c. In consequence:

- the optimal A_crit is unchanged;
- every fitted/measured ratio is unchanged;
- within-tile model rankings are unchanged.

This holds for the objective function. It is **not guaranteed for the optimizer path**, for two reasons:

- the scipy Nelder–Mead initial simplex scales with x₀ in log space;
- the parameter bounds were not scaled.

**Not invariant, and these must be recomputed:**

- the nonlinear partitioned model, because the term (|∇z|/S_c)² with S_c = 1.27 depends on the absolute slope;
- any comparison of AP values with external values, such as Perron et al. (2012);
- anything with an absolute elevation, depth or curvature threshold: the width analysis, Test 1, Test 2 and the D8 epsilon fill.

**Constraints.**

- Do not modify or overwrite any existing grid, notebook, output or figure. New files only.
- Do not push to the remote without asking.
- Do not edit the manuscript or the response letter. Produce a corrections table instead (section 6).

---

## 0. Setup

Create a new branch `fix/ap-vertical-units` from the current working branch (`feature/head-extraction`). Put new code in `ap_units/` and outputs in `outputs/ap_units/`.

## 1. Inventory (do first; write `outputs/ap_units/inventory.md`)

List every product in the repo and on group storage that reads an AP elevation grid or a quantity derived from one, either directly or through saved outputs. For each, record:

- what it is;
- where it is used: manuscript figure or table, SI, response letter, or test report;
- which correction class from the table above applies;
- whether it must be recomputed or can be corrected algebraically.

At minimum, cover:

- `notebooks/model_runs/allegheny_tile*_complete.ipynb`, all stages:
  - AD with initial (measured) parameters;
  - AD optimized;
  - partitioned with A_crit only;
  - partitioned nonlinear;
  - partitioned joint.
- The source of the per-tile measured k_s, θ and hilltop curvature hard-coded in those notebooks. The tiles were prepared outside the repo, so find the measurement scripts, or report that they cannot be found.
- The Fig. 2 forward runs that use the Perron et al. (2012) published parameters.
- Fig. 3 (ΔK, ΔD).
- Table 1.
- `notebooks/figures/fig_slope_area.ipynb`, `supp_fig5.ipynb` and `d8_vs_dinf_comparison.ipynb`.
- The nonlinear-transport figure, wherever it is built. `supp_fig6.ipynb` has no literal AP path, so check whether it reads AP outputs indirectly.
- `width_analysis/` (AP widths, depths, bank-pick window and stage).
- `acrit_slope/` (Test 1).
- `head_extraction/` (Test 2).
- Tile selection: the criterion that matched per-tile k_s, θ and ∇²z to regional values.

Also confirm that `notebooks/figures/resolution_study_dinf.ipynb` and the synthetic sweep use no AP data.

## 2. Corrected grids and site sanity checks

1. Write corrected AP filled grids, `data/AP/tile{n}/*_filled_m`, with z × 0.3048. Keep the georeferencing identical.
2. Regenerate D8 flow directions and drainage area from the corrected filled grids, using the same code and settings as the originals. Compare them with the originals cell by cell and report the number of cells that differ:
   - **zero differences, or differences confined to filled flats:** use the original direction and area grids;
   - **otherwise:** report where the differences are and use the regenerated grids.
3. Run the same sanity checks on all ten tiles and tabulate them:
   - elevation range against a published or regional elevation for the area;
   - median pixel slope;
   - median S_up_10 at Test 1 heads.

   This confirms GM is in metres and corrected AP is plausible. Flag anything outside a soil-mantled-landscape range.
4. Recompute the per-tile measured parameters on the corrected grids:
   - **k_s and θ**, by χ–elevation and slope–area regression;
   - **hilltop curvature**, with `robust_curv_mag`, 30 m block mean, and the same hilltop-pixel selection as before.

   Use the original measurement code if section 1 finds it. If it cannot be found:
   - correct algebraically (k_s ÷ c, curvature ÷ c, θ unchanged);
   - re-measure hilltop curvature with `robust_curv_mag` as an independent check;
   - report both.

## 3. Invariance gate (pass/fail; STOP if it fails)

Run the full notebook pipeline on corrected grids for **AP1 and AP5**:

- **AP1:** largest stage-2 MSE ratio.
- **AP5:** lowest A_crit; the tile the paper discusses most.

Make copies of the two notebooks (`allegheny_tile{1,5}_complete_m.ipynb`). Change only the parameter cell:

- corrected grid paths;
- corrected measured k_s and hilltop curvature (from step 2.4);
- K_bounds and D_bounds multiplied by c at both ends.

Leave every other cell unchanged. Record the git diff of each copy against its original in the report.

**Pass** if all of the following hold, for every stage:

| check | criterion |
|---|---|
| A_crit (A_crit-only, nonlinear, joint stages) | within 5% of original |
| K_fit/K₀ and D_fit/D₀ (AD-optimized and joint stages) | within 5% of original |
| MSE_new / MSE_old (AD, A_crit-only, joint) | within 5% of c⁻² = 0.0929 |

The nonlinear stage is **not** expected to pass. Report its new A_crit and MSE as results, not as a gate failure.

Report the wall time per tile for the full pipeline.

**If the gate passes:**

- **If one tile's pipeline takes under 6 hours on one node:** rerun AP2–AP4 directly, in the same way.
- **Otherwise:** correct the AP2–AP4 linear stages algebraically, using the table in Context, and state that in the report. Run the nonlinear stage on all five tiles regardless, since it cannot be corrected algebraically.

**If the gate fails:** stop. Report which quantities moved and by how much. Include a plot of the joint-stage optimizer trajectory, old against new (log A_crit, log D, log K), so Steve can see whether the optimizer ended at a different point along a ridge.

## 4. Recompute the non-invariant products (after the gate passes)

1. **Fig. 2 forward runs.** Rerun the AD and partitioned forward models on corrected AP grids with the Perron et al. (2012) published AP parameters, exactly as configured for Fig. 2. Report the new MSEs and regenerate the AP panels into `outputs/ap_units/figures/`.
2. **Nonlinear transport on AP.** Rerun the nonlinear stage on all five tiles (S_c = 1.27, same initial A_crit and bounds). Report:
   - A_crit, MSE, and the fraction of hillslope pixels with |∇z| > 0.5 S_c, before and after correction;
   - whether nonlinear still beats linear on each AP tile.

   Regenerate any figure built from these results.
3. **Slope–area and SI figures.** Regenerate `fig_slope_area`, `supp_fig5` and `d8_vs_dinf_comparison` from corrected AP grids into `outputs/ap_units/figures/`. Do not overwrite the originals.
4. **Tile selection.** Recompute the selection criterion (per-tile k_s, θ and ∇²z against the regional values) for all candidate AP tiles, if the candidate list exists. Report whether the same five tiles would be chosen. If the candidate list cannot be found, say so.
5. **Width analysis, AP points only.** Rerun on corrected grids with the same configuration, which is in metres: bank picks 0.3–4 m above the thalweg, bed width at a 0.5 m stage. Report the AP table and exceedance fractions old against new. GM is unchanged.
6. **Test 1 (`acrit_slope`).** Rerun all ten tiles, since its conclusions are cross-site. Report:
   - the tile summary and every fit, old against new;
   - whether the site separation in head slope and the AS² climate ordering survive.
7. **Test 2 (`head_extraction`), AP only.** Rerun the extract phase on corrected AP grids at κt = 0.05, 0.1 and 0.2 m⁻¹, with OWF and DT = 2. Report:
   - head counts, classes and median head area;
   - whether AP heads are still crest pixels;
   - the Wiener-fit diagnostics again (β, noise floor), because the spectrum's shape is unchanged but the noise-to-signal threshold behaves differently.

   Stop at the review gate as before.

## 5. Tests

Add to `tests/test_ap_units.py`:

- **Scale invariance of the linear models.** On a synthetic tile, solve the AD model and the linear partitioned model with (z_bc, K, D) and with (c·z_bc, K/c, D/c). The second solution must equal c times the first to floating-point tolerance.
- **Nonlinear model is not invariant.** The same check on the nonlinear model must fail. This documents why its stage is rerun.
- **Grid conversion.** Corrected grid = original × 0.3048, with matching georeferencing.
- **Algebraic correction.** The helper that applies the Context table reproduces the gate-tile rerun values within 5%, if the gate passed.

## 6. Corrections table for the manuscript and letter (`outputs/ap_units/corrections_table.md`)

List every AP number that appears in the manuscript, the SI, the Reviewer 1 response draft, or the game-plan memo, with the old value, the new value, and the source (rerun or algebraic). Steve will apply them by hand.

Read the project files to find these numbers. At minimum, include:

- Table 1, AP rows: measured and optimized K, D, A_crit and MSE for each model.
- AP k_s and hilltop curvature, wherever they are quoted.
- The Reviewer 1 response sentence: "AD-optimized D values range from 4.3 × 10⁻⁵ to 6.7 × 10⁻⁴ m²/yr — factors of 15 to 230 smaller than Perron et al.'s published D". Give corrected values:
  - against Perron's 1.0×10⁻² m²/yr;
  - against per-tile measured D.
- Any statement that AP measured K or D differs from Perron et al.'s regional values, with the corrected ratio. Reviewer 1 states that AP median K is more than an order of magnitude smaller than Perron's.
- The AD-model "1–2 orders of magnitude" D-suppression wording, as it applies to AP.
- AP width-analysis numbers in the game plan and draft letter wording: n resolved, median trench and bed width, depth, exceedance fractions.
- Test 1 and Test 2 statements about AP that change.

Flag every sentence whose **claim**, not just its number, changes. For example:

- whether measured AP D now agrees with Perron's regional value;
- whether AP and GM still differ in head slope.

## 7. Report (`reports/2026-11-XX_ap_vertical_units_fix.md`)

Include:

1. Inventory summary.
2. Grid checks: the D8 and area comparison, and the sanity table.
3. Re-measured per-tile parameters, old and new.
4. Gate result, with every number in the pass/fail table and wall times.
5. Results of each section 4 item, old against new, each with a one-line statement of whether any conclusion changes.
6. The corrections table, or a link to it.
7. A short paragraph Steve can adapt for the response letter, stating what was found and what changed. Write it plainly, without minimizing:
   - a vertical-unit error in the AP elevations was found and corrected;
   - A_crit and every fitted-to-measured ratio are unchanged by construction;
   - the absolute AP K and D values change by a factor of 3.28;
   - the nonlinear-transport results on AP were recomputed.

**Commits.** Commit in logical steps on `fix/ap-vertical-units`.
