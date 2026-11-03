# Channel-width extraction: Phase 3 full-run report

Date: 2026-11-02
Branch: `feature/channel-width` (outputs committed at c36f767; code at 0cd1102, 12fda6b, d1eef04)
Directive: `channel_width_directive.md` Phase 3 plus Steve's follow-up decisions of 2026-11-02
Outputs: `outputs/widths/` (`points.csv`, `excluded.csv`, `transects.csv`, `points.gpkg`,
`validation_points.gpkg` and `.csv`, `fits.json`, `regime_fraction_vs_dx.csv`,
`regime_boundary.csv`, `figures/fig1..fig7`, `qc/` (random 20 per site), `qc_all/`
(every point, untracked), `acrit_profile/AP5.csv` and `.png`)
Environment: `$GROUP_HOME/scdobbs-envs/lem-widths` (Python 3.11.18, numpy 2.4.6,
scipy 1.17.1, rasterio 1.4.4, geopandas 1.2.0); Slurm jobs 50839645 (full run,
147 s) and 50839646 (A_crit profile, 21 min). Tests: 17 of 17 pass.

## 1. Bottom line

The full run completed as specified, but the headline fit is not a channel-width
scaling. The bank-break width shows no dependence on drainage area at either
site (b = -0.02 on AP, -0.10 on GM) because, at drainage areas below about
10^5 m², the first convex breaks flanking the thalweg are the walls of the
hollow or valley, not channel banks. The channelized flag (both breaks present,
inner relief >= 0.3 m) does not screen these out, since hollow walls have more
than 0.3 m of relief. The regime-III fraction and the regime boundary derived
from that fit should not be used. The fixed-stage widths tell a different and
more robust story: at 0.25 m above the thalweg the median width is 2.0 to 2.8 m
at every drainage area on both sites, i.e. at the 2-pixel resolution floor.
Section 7 gives the evidence and a proposed fix; nothing further should be
built on the W = cA^b fit until the bank-break pick is constrained.

## 2. What was implemented (Steve's decisions)

- The min(W/h) bankfull criterion is gone. Widths are `W_inner_bankbreak`,
  `W_valley_floor` and `W_stage_0.25 / 0.5 / 1.0`, never conflated.
- `W_primary` = bank-break width where both banks exist, valley-floor width
  where `shape_class = flat_bed`, otherwise unresolved and censored at 2δ;
  `W_primary_source` records which.
- `channelized` = both bank breaks present and `inner_relief_m` >= 0.3 m.
- Fits (censored MLE, 1000-sample bootstrap, OLS-on-uncensored labelled
  biased) use channelized points only, with (`all`) and without (`no_steep`)
  points with local slope > 0.5 (`steep_rill_flag`).
- Model-free statistic = fraction of points with a resolved inner channel or a
  flat bed wider than 2δ (`resolved_width`), per A bin and site (figure 2).
- Channelized fraction vs A with a logistic fit and bootstrap CI on A50
  (figure 6); bank-break vs 0.25 m stage width, log-log, 1:1 line (figure 7).
- `validation_points.gpkg`: channelized points with A >= 10^5 m² (priority 1:
  22 AP, 17 GM) plus 10 random others per site (priority 2), with stage and
  bank-break widths as attributes.
- `acrit_profile.py`: MSE vs A_crit for AP tile 5 with K, D at their optimized
  values, using the repo's partitioned model unchanged.
- README explains why a fixed-stage width on a V-shaped section measures side
  slope (W = 2h / S_side), not channel width.

## 3. Sample

95 points: 47 AP (one bin short by one), 48 GM, 6 per log-A bin from 10^3 m²
to the site maximum (1.7 x 10^7 m² AP, 8.9 x 10^6 m² GM).

| Exclusion reason | AP | GM |
|---|---|---|
| too_close_to_sample (< 50 m) | 319 | 8 |
| tile_edge (< 30 m) | 147 | 14 |
| confluence (jump in A within 15 m) | 71 | 17 |
| flat_fill_proxy (no fall along trace) | 42 | 3 |
| no_incision (confinement test) | 12 | 12 |
| knickpoint (> 3 x reach median) | 4 | 2 |
| bin_exhausted | 1 | 0 |

Slope method: 83 vertical-interval, 12 horizontal fallback (low-gradient
trunks). Steep (S > 0.5) points: 13 AP, 2 GM.

| Site | channelized | resolved width > 2δ | flat_bed | unresolved primary |
|---|---|---|---|---|
| AP | 36/47 | 40/47 (0.85) | 19/47 (0.40) | 3/47 |
| GM | 31/48 | 37/48 (0.77) | 22/48 (0.46) | 3/48 |

Per-bin tallies (n = 6 per bin unless stated; medians in metres):

| Site | bin (log10 A) | channelized | resolved | flat_bed | median W_bankbreak | median W_stage_0.25 | median S |
|---|---|---|---|---|---|---|---|
| AP | 3.0-3.5 | 4 | 5 | 3 | 11.8 | 4.1 | 1.05 |
| AP | 3.5-4.1 | 3 | 4 | 4 | 13.8 | 4.3 | 0.63 |
| AP | 4.1-4.6 | 4 | 4 | 1 | 9.3 | 3.3 | 0.52 |
| AP | 4.6-5.1 | 6 | 6 | 1 | 7.9 | 1.4 | 0.25 |
| AP | 5.1-5.6 | 5 | 5 | 1 | 9.8 | 2.0 | 0.13 |
| AP | 5.6-6.2 | 5 | 6 | 2 | 5.8 | 2.1 | 0.06 |
| AP | 6.2-6.7 | 4 | 5 | 2 | 8.0 | 3.0 | 0.05 |
| AP | 6.7-7.2 (n=5) | 5 | 5 | 5 | 13.3 | 5.4 | 0.01 |
| GM | 3.0-3.5 | 2 | 4 | 5 | 19.5 | 9.0 | 0.31 |
| GM | 3.5-4.0 | 5 | 5 | 5 | 17.0 | 5.3 | 0.16 |
| GM | 4.0-4.5 | 2 | 3 | 5 | 23.9 | 6.3 | 0.12 |
| GM | 4.5-5.0 | 5 | 5 | 1 | 7.8 | 1.9 | 0.08 |
| GM | 5.0-5.5 | 4 | 5 | 1 | 7.6 | 2.7 | 0.07 |
| GM | 5.5-6.0 | 2 | 4 | 2 | 4.9 | 5.4 | 0.05 |
| GM | 6.0-6.5 | 6 | 6 | 1 | 7.0 | 2.0 | 0.04 |
| GM | 6.5-6.9 | 5 | 5 | 2 | 9.0 | 2.6 | 0.03 |

## 4. Fits (`fits.json`, figure 1)

Censored MLE of log10 W_inner_bankbreak = log10 c + b log10 A on channelized
points. No channelized point is censored (all bank-break widths exceed 2δ), so
the Tobit and OLS estimates coincide.

| Site | variant | n | c (m) [95 % CI] | b [95 % CI] | sigma (log10) |
|---|---|---|---|---|---|
| AP | all | 36 | 12.2 [6.5, 23.7] | -0.021 [-0.079, 0.031] | 0.21 |
| AP | no_steep | 27 | 8.9 [2.4, 38.1] | 0.002 [-0.106, 0.096] | 0.22 |
| GM | all | 31 | 29.0 [8.9, 73.9] | -0.099 [-0.171, -0.013] | 0.21 |
| GM | no_steep | 29 | 32.6 [7.8, 94.4] | -0.108 [-0.192, -0.006] | 0.21 |

Both sites give b indistinguishable from zero or slightly negative. Removing
the steep rills does not change this. The OLS fit (uncensored only) is
identical because nothing is censored.

For comparison, OLS of the stage widths on the same points: W_stage_0.5,
AP c = 4.6, b = -0.01; GM c = 12.5, b = -0.10. W_stage_0.25: AP c = 2.0,
b = 0.02; GM c = 8.0, b = -0.09. No width measure in this dataset scales
with drainage area between 10^3 and 10^7 m².

**Channelized fraction and A50 (figure 6).** The channelized fraction is 0.33
to 1.0 in every bin at both sites with no trend, so the logistic fit is
undetermined: AP A50 = 1.2 x 10^2 m² with a bootstrap interval from 10^-28
to 8.7 x 10^3 m²; GM A50 = 4.4 x 10^3 m² with an interval from 4 x 10^-5 to
6.2 x 10^4 m². These numbers carry no information; the flag does not separate
hollows from channels (Section 7).

## 5. Model-free result and shape class (figures 2, 5)

Fraction of points with a resolved inner channel or a flat bed wider than 2δ:
40/47 on AP, 37/48 on GM, and between 3/6 and 6/6 in every bin (table in
Section 3). This statistic is honest about what it counts: a feature wider than
2 m around the thalweg exists almost everywhere, but below about 10^5 m² that
feature is the hollow floor or hollow walls, not a channel.

Flat-bed fraction: AP 19/47, GM 22/48. On GM the three lowest bins are 5/6
flat_bed and the five bins above 3 x 10^4 m² are 1/6 to 2/6: the small-A
Gabilan hollows have flat or parabolic floors several metres wide, the larger
channels are V-shaped inner channels. AP is mixed except the trunk bin
(5/5 flat_bed, valley floors 2 to 15 m).

## 6. Regime-III fraction and regime boundary (figures 3, 4; `regime_*.csv`)

With the fitted W(A) of Section 4 (about 9 to 12 m independent of A), 100 % of
channel length (A >= tile A_crit) on every tile has W/δ > 1 at δ = 1 m, and
the fraction drops to zero between δ = 9 and 14 m (first sweep spacing with
fraction below 0.5 is 10.6 m at both sites). A* = (δ/c)^(1/b) is undefined
for b <= 0, so `regime_boundary.csv` has no finite A* (the column is NaN by
construction, with c and b recorded). These outputs are a faithful consequence
of the fit, and the fit measures hollow width, so they are not a regime-III
result. Do not quote them.

What survives independently of the fit: at δ = 1 m the measured feature around
the thalweg is wider than 2 pixels in 77 to 85 % of points, and the stage
widths at 0.25 m sit at the resolution floor at all A.

## 7. Why the bank-break width fails, with evidence

1. Figure 7: on channelized points the bank-break width is 3.5 times the 0.25 m
   stage width (median ratio 3.5 on both sites); 25/36 AP and 20/31 GM points
   exceed a 3:1 ratio. A bank 0.3 to 1 m high bounding a 2 m wide channel
   cannot be 3.5 times wider at its top than 0.25 m up unless the "bank" is the
   valley side.
2. QC frames: GM2_013 (A = 1.05 x 10^4 m²): picks at -16 and +20 m on a
   parabolic hollow 36 m wide and 3.8 m deep; AP5_011 (5.3 x 10^3 m²): right pick
   on a ridge crest 13 m away; AP1_014 (1.7 x 10^4 m²): picks at the valley-wall
   breaks 20 m apart over a 0.5 m inner notch; GM4_008 (5.3 x 10^3 m²): left pick
   20 m up the hillslope with 0.39 m relief; AP3_045 (1.1 x 10^7 m²): picks on a
   terrace edge and the opposite valley wall, 22 m apart, over a channel the
   0.25 m stage puts at 7 m.
3. Medians: channelized points below 10^5 m² have bank-break widths of 11.9 m
   (AP) and 9.9 m (GM); above 10^5 m², 8.1 and 7.0 m. The stage-0.25 widths are
   2.8 and 2.0 m respectively. Widths do not shrink headward because the
   method measures a different landform headward.
4. The pick rule is "first local convexity maximum with convexity >= 0.05 m^-1
   and mean slope from the thalweg >= 0.10". On a smooth parabolic hollow there
   is no inner break, so the first qualifying maximum is the hollow lip; on a
   large channel with a bench, the inner bank is often picked on one side and
   the valley wall on the other (AP3_045).

Proposed fix, for Steve to decide: constrain the bank-break search to a
channel-scale window, e.g. accept a pick only if its height above the thalweg
is between 0.3 and `bank_max_relief_m` (about 1.5 m) and its distance from the
thalweg is less than `bank_max_offset_m` (about 3 to 5 m, or a multiple of the
0.5 m stage half-width), and declare "no inner channel" otherwise. Points
without an inner channel would then be unresolved or valley-floor by the
existing rules, and the channelized fraction would start meaning something.
The alternative is to abandon the bank-break width on these DEMs and treat the
0.5 m stage width as the channel proxy, which is what the pilot's clean cases
and figure 7 suggest is the only channel-scale measure here.

## 8. A_crit misfit profile, AP tile 5 (`acrit_profile/AP5.png`)

MSE(A_crit) at the jointly optimized K = 3.44 x 10^-6, D = 3.63 x 10^-3,
U = 5.5 x 10^-5, m = 0.40, 31 log-spaced A_crit from 10^2 to 10^5 m² plus the
optimum.

| A_crit (m²) | MSE (m²) | channel pixels |
|---|---|---|
| 100 | 1287 | 111726 |
| 316 | 582 | 55578 |
| 1000 | 485 | 22904 |
| 1585 | 329 | 15039 |
| 1887 (optimum) | 290 | 13046 |
| 1995 | 301 | 12661 |
| 2512 | 445 | 10174 |
| 3162 | 664 | 8566 |
| 7943 | 3105 | 3995 |
| 10000 | 3758 | 3488 |
| 31623 | 8640 | 2470 |
| 100000 | 18794 | 1835 |

The minimum is sharp and sits exactly at the notebook optimum (289.6 m²). Moving
A_crit to the AP median of 7.6 x 10^3 m² raises the misfit about ten-fold; to
10^4 m², thirteen-fold. The 8873 pixels that the low optimum adds to the
channel network (1887 <= A < 7646 m²) have a median pixel-scale slope of 0.71
and 71 % of them are steeper than 0.5; the pixels above 7646 m² have a median
of 0.26. In the width sample, 4 of the 5 AP tile 5 points below 10^4 m² have
local slopes of 0.63 to 1.47 and lie on the parallel rills seen in the pilot.
So yes: on tile 5 the optimizer needs the rilled slopes in the channel domain
to reduce elevation misfit, and the profile shows there is no comparable
minimum at channel-like areas. Two caveats. The pixel-scale slope of the 1 m
AP DEMs is dominated by gridding noise (all five AP tiles return a median
pixel slope of 0.88 to 0.89 and 81 to 86 % of pixels above 0.5, which is not a
landscape property), so the 0.71 vs 0.26 contrast is indicative, not a
measurement of hillslope gradient; the path-based local slopes of the sample
points (0.6 to 1.5 on the rills) are the better number. And the profile shows
that the rills reduce the misfit, not that they are channels.

## 9. Main uncertainties

- The bank-break width measures hollow and valley geometry below about
  10^5 m² and sometimes above; every fit, the regime fraction and the regime
  boundary inherit this. Section 7 is the fix to decide on.
- The inner channel, where the pilot showed one, is 2 to 8 m wide and its 0.25 m
  stage width sits at the 2δ floor: on these DEMs the inner channel is at or
  below the resolution limit at all drainage areas. That is itself the
  regime-III statement these tiles can support, and it is weaker than "W > δ
  everywhere".
- Filled DEMs only: the flat-fill proxy removed 45 candidates; filled reaches
  that passed the proxy would read as flat beds.
- D8 lines off the thalweg (GM4_002 in the pilot) and parallel rills on steep
  slopes (13 AP points) are in the sample by design and flagged.
- 6 points per bin: the bin-level fractions have binomial uncertainty of about
  +/- 0.2.

## 10. Reproducing

```bash
cd ~/LEMs_competition_vs_threshold-JGR-branch
PY=/home/groups/hilley/scdobbs-envs/lem-widths/bin/python
$PY -m pytest tests/test_width_analysis.py -q
$PY -m width_analysis.run --site all --mode full --out outputs/widths      # ~150 s, 24 GB requested
$PY -m width_analysis.acrit_profile --site AP --tile 5                     # ~21 min, 48 GB requested
```

Job scripts: `$SCRATCH/width_jobs/full.sbatch`, `$SCRATCH/width_jobs/acrit_profile_AP5.sbatch`.
