# [2026-11-24 18:30 PST] Addendum 23: Tennessee Valley on one clipped domain, the road check, a 1 m head finder, and a fair measure of channelization

Directive: `reports/directives/2026-11-24T0915PST_v2_addendum_23_ttv_domain_roads_heads_channelization.md` (outline fixed by its section 8).
Branch `v2.3-tiles` (from c694e02, the Addendum 22 report). **Package code changed** where decision 2.1 needed it, and nowhere else:
`lemcvt.models.common.extract_grid_data(..., outlets=, noflux_outside=)` carried by the AD model, the partitioned model and
`PartitionedProfiler` (`docs/numerics_changes.md` N14; `tests/test_domain.py`, nine tests; defaults bit-identical; the Addendum 18
regression gate in Appendix E). Every number is tied to the run directory
`$GROUP_HOME/lemcvt_results/20261124T173105Z_a23_ttv_domain_roads_heads_channelization/` or to a Slurm job (Appendix D). Scripts
`reports/v2/scripts/a23_{common,domain,roads,heads1m,fit,extra,measures,post,figs}.py`; figures `reports/v2/figures/a23/`; the
machine-generated tables are in Appendix F and the state check with every choice made in applying the directive is Appendix E.
**This is the addendum's stop.**

"Split model" means the threshold-partitioned model: stream-power incision where a channel rule holds, linear diffusion elsewhere.

## 1. Summary

**Every model is fitted on the 1 m lidar grid.** The 12 m block mean enters only the head finding and the field thresholds.

1. **One domain replaces the three tiles:** the drainage basins above the four cells where flow leaves the M&D study area with at least
   10⁴ m² (3.145 km², 0.978 of the study area; 0.36 of the domain lies outside it). Only those four cells are held at their observed
   elevation; every other domain edge is a divide. On it K₀ = 1.34 × 10⁻⁵ (TTV2's value) and the relief D = 0.0200 m²/yr (between
   the three tile values).
2. **The AD gorge line on TTV2 is not a road:** 13 of its 54 wall heads lie within 15 m of a TIGER road (0.24 against the reading's
   0.5) and 0.008 of the observed trunk path lies within 5 m of one; the road runs 20–60 m from the stream along the valley side.
   TIGER carries no trail-class feature in the study area.
3. **No 1 m head finder is accepted.** The three candidates find 0.84–0.97 of the mapped heads only by making 2,000–13,000 heads on the
   test ground (share real 0.002–0.05); the 12 m method finds 0.38 with 0.16 real. Channel heads are not a usable test with these
   methods and are reported in Appendix H only.
4. **On the domain every split model beats the AD model inside the study area:** 408–557 m² against 612 (0.67–0.91 ×). The misfits are
   3–4 × the tile misfits because the tiles pinned 3,996 edge cells to the lidar and the domain pins 4.
5. **The split models fit with the measured K** (0.89–0.90 × K₀) **and a D within the relief range** (0.83–0.99 × the relief D); the AD
   model needs K 8.6 × K₀ and D 0.045 × the relief D (0.36 × Heimsath's). Fitting the area × slope² threshold returns 823 m², within
   7 % of M&D's 876 m² (on the tiles it was 559–4,607 m²).
6. **Channelization: no model matches the lidar under reading C-1.** The AD model has 1.6–2.7 × the lidar's spectral power at 27–75 m
   wavelength and 1.35 × its valley share (finer dissection, as expected). The split models have the lidar's valley share (0.99–1.12 ×)
   but only 0.12–0.44 × its power over 20–200 m (0.35–0.66 × its amplitude): they put valleys where the lidar has them, with too little
   relief at those scales and none of the hollows and roughness between them.
7. **The planner's expectation holds for the AD model** (more power, larger valley share) **and for the valley share of the M&D-rule split
   models, not for their spectra**, which lie farther from the lidar than the AD model's in the 20–60 m band.
8. Through the 12 m method the real landscape pairs 18 of the 64 mapped heads; the split models 13–16; the AD model 3, with 80 extra
   heads, 12 of them within 15 m of a road.
9. The package change the domain needed (outlet anchors and no-flux divides) leaves every default result bit-identical, reproduces the
   Addendum 22 TTV2 fits exactly through the domain route and passes the Addendum 18 regression gate. 34 core-hours in all (Part B 20).

## 2. Which grid each quantity uses, and the model key

Fitting is on the 1 m grid throughout: the models are solved cell by cell on the 1 m lattice of the domain and the misfit is the mean
squared elevation error over 1 m cells. The 12 m block mean is used only where the table says so.

| quantity | grid | window or smoothing | routing |
| --- | --- | --- | --- |
| the elevation fit (misfit, K, D, fitted threshold) | 1 m, the domain's 3,145,269 cells; misfit over the 2,012,633 cells inside the M&D polygon | none | the full-site 1 m D8 of Addendum 15 (the domain is a union of its basins; the four outlets are fixed) |
| K₀ (χ–z) | 1 m | none; channel network at A ≥ 10³ m², basins by outlet, m = 0.33 | the same 1 m D8 |
| the crest curvature | 1 m | least-squares quadratic over an 11 m window (5-cell half-width); hilltop set on the 10³ m² network, slope ≤ 0.4, 5.5 m from the network, 10 m from a TIGER road, no segment filter | the same 1 m D8 |
| the relief D | 1 m | crest pixels inside the polygon traced to the mapped-head network; D = U L_H² / (2R) | the same 1 m D8 |
| the field thresholds (876 and 6,120 m²) | 12 m block mean of the full-site 1 m grid (Addendum 21, lattice origin 529495 / 4198505) | 12 m blocks; slope to the largest-area donor | D8 of the filled 12 m surface |
| head finding (the 12 m method, every surface) | 12 m block mean of each 1 m surface on the domain box lattice | 12 m blocks | D8 of the filled 12 m surface |
| head finding (the 1 m candidates, lidar only) | 1 m | multiple-flow-direction area (Freeman 1991), or Gaussian smoothing σ = 3 or 6 m; slope over 10 m downstream | the lidar's 1 m D8 for the multiple-flow candidate; D8 of the smoothed, filled 1 m surface for the other two |
| power spectrum | 1 m, the largest square inside the domain (1,008 m) | plane detrend, 2-D Hann window; 14 logarithmic wavelength bins, 4–500 m | none |
| valley share | 1 m | Gaussian σ = 10 m (5 m as sensitivity); contour curvature | none |

| Plain name | What it is | Fitted parameters |
| --- | --- | --- |
| **Real landscape (lidar)** | The observed surface, run through the same measures as the models | none |
| **AD model** | Co-located advection–diffusion; incision and diffusion act in every cell | K, D |
| **Split model, M&D area × slope² rule** | Stream power where area × slope² ≥ 876 m² (measured at M&D's heads on the 12 m lattice), diffusion elsewhere | K, D |
| **Split model, M&D area rule** | Stream power where drainage area ≥ 6,120 m², diffusion elsewhere | K, D |
| **Split model, nothing fitted** | M&D area × slope² rule, K = the domain's measured K₀, D = the domain's relief D | none |
| **Split model, area × slope² rule fitted** | As the first split model, but the threshold is chosen by the fit | K, D, threshold |

Fixed values: U = 65 m/Myr, m = 0.33, physical D = 0.0025 m²/yr (Heimsath et al. 1999); K₀ and the relief D re-measured on the domain
(section 3); the K and D box of Addendum 18 (the TTV2 config bounds × 65/77, as Addenda 21–22). The two "area rule fitted" and
"nothing fitted, physical D" models of Addendum 22 are not repeated.

## 3. The domain and the re-measured inputs (figure 1)

**The domain (decision 2.1).** On the full-site 1 m D8 routing, 2,116 cells on the M&D polygon boundary pass flow out of the polygon;
nine of them carry a drainage area of at least 10⁴ m². Five of the nine are "nested": their flow leaves the polygon and re-enters it
(the polygon boundary runs along or across the trunk valley on the east), so they lie inside another outlet's basin and are ordinary
domain cells. The remaining four are the outlets, held at their observed elevation: drainage areas 1,226,719, 1,134,623, 758,368 and
25,559 m². The domain, the union of the cells draining to those four, holds 3,145,269 cells (3.145 km²). It covers 0.978 of the
polygon's 2,057,654 valid cells; the hillslope fringe (polygon cells draining out at less than 10⁴ m²) is 45,021 cells, 0.022 of the
polygon; and 0.360 of the domain lies outside the polygon, almost all of it one basin east of the study area whose stream reaches the
polygon boundary at the two eastern outlets (figure 1). Every domain cell's donors are domain cells (checked), so the full-grid areas are
the domain areas; no domain cell touches the window edge. The box cut around the domain is 1,867 × 2,918 cells (origin x 539160,
y 4191578, EPSG:6339). All 64 of Steve's picks lie inside the domain and the polygon (Addendum 22 counted 61 on tiles TTV1–TTV3; the
other three lie in the polygon between the tiles); per tile footprint 22 / 34 / 5, and 3 on no tile.

**Re-measured inputs**, by the same methods as Addenda 17–21, beside the per-tile values:

| quantity | domain | TTV1 | TTV2 | TTV3 |
| --- | --- | --- | --- | --- |
| k_s (χ–z, m = 0.33, A ≥ 10³ m²) | 4.859 (4 basins, 64,931 channel pixels; basin bootstrap 68 % 4.70–5.01, n_eff 2.6) | 6.85 | 4.87 | 2.92 |
| K₀ = U / k_s (m^0.34/yr) | 1.338 × 10⁻⁵ | 9.49 × 10⁻⁶ | 1.33 × 10⁻⁵ | 2.23 × 10⁻⁵ |
| crest curvature at 11 m (m⁻¹) | 0.0141 (2,039 crest pixels, 341 segments; 68 % 0.0134–0.0149; inside the polygon 0.0139) | 0.0159 | 0.0149 | 0.0147 |
| D_form = U / crest curvature (m²/yr) | 0.0046 | 0.0041 | 0.0044 | 0.0044 |
| relief D on the mapped-head network (m²/yr) | 0.0200 (1,569 crest pixels inside the polygon, 285 segments; 68 % 0.0189–0.0224; L_H median 173 m, R median 49 m) | 0.0175 | 0.0159 | 0.0230 |

The domain K₀ equals TTV2's (the tile that holds most of the polygon) within 0.3 %; the relief D on the domain, 0.0200 m²/yr, lies
between the three tile values (its 68 % interval, 0.0189–0.0224, overlaps TTV1's and TTV3's); the relief D traced to the 876 m² and
6,120 m² networks instead is 0.0193 and 0.0178. The physical-D check: a linear model at Heimsath's D = 0.0025 m²/yr needs a crest
curvature of U/D = 0.026 m⁻¹ to lower the crests at U; the measured crest curvature is 0.54 × that, so Heimsath's D again cannot
reproduce the crests at this U (Addendum 21's F-phys result, restated on the domain). With four fixed cells the k_s bootstrap has
n_eff 2.6 (its unit is the basin), so the k_s interval is indicative only.

**The domain check (decision 2.1).** Tile TTV2 cut as before, with its 3,996 edge cells given as the outlets through the new domain
route (no-flux faces on), reproduces the Addendum 22 TTV2 polygon fits exactly: M&D area rule 68.180795 m² (K, D identical), M&D
area × slope² rule 104.740981, nothing fitted 125.655549, AD at its Addendum 22 K and D 206.728375; relative differences 0 (Appendix
B). The unit tests add the two identities the directive did not ask for: a NaN-padded tile with its former edge ring as outlets
reproduces the unpadded solution, and the half of a mirror-symmetric strip with no-flux at the centre line reproduces the full strip's
solution on that half (an exact discrete identity), which the old ghost value of zero does not.


## 4. The road check (figure 2)

**The road layer.** TIGER/Line 2024 ROADS for Marin County holds 29 features on the domain box, every one of class S1400 (local
road): Tennessee Valley Rd, Smith Rd, Headlands Ave, five named residential streets, and 21 unnamed segments (the longest, 5,191 and
3,863 line cells, are the fire roads and tracks of the Headlands). It holds **no trail-class feature** (S1500 vehicular trail, S1710
walkway, S1820 bike path, S1830 bridle path) anywhere on the box, so "roads and trails (TIGER)" here means roads only, and the
Tennessee Valley Trail along the valley floor is represented only where TIGER carries Tennessee Valley Rd. On tile TTV2 itself the
layer has three features: Tennessee Valley Rd (546 line cells), one unnamed road (3,422) and Smith Rd (225).

**Is the AD gorge line a road? No.** On the Addendum 22 TTV2 surfaces (figure 2):

| quantity | 5 m | 15 m | 25 m |
| --- | --- | --- | --- |
| share of the 54 too-steep-wall extra AD heads within the distance of a TIGER road | 0.07 (4) | **0.24 (13)** | 0.46 (25) |
| share of all 66 extra AD heads | 0.06 (4) | 0.21 (14) | 0.39 (26) |
| share of the observed trunk flow path (area ≥ 10⁵ m², 1,458 cells in the polygon) | **0.008** | 0.048 | 0.21 |

The wall heads are 27 m from a road at the median, the trunk cells 65 m. Neither half of the reading is met (0.24 against 0.5 for the
heads; 0.008 against 0.5 for the trunk), so the reading is **not road-influenced**. Figure 2 shows why: the road (Tennessee Valley Rd
and its unnamed continuation) runs along the eastern valley side, 20–60 m from the stream, while the AD gorge follows the stream itself
and its wall heads sit on both walls of the slot. The AD slot is a property of the fit (K 4.8 × K₀ and D 0.08 × Heimsath's, Addendum
22 section 6), not of the routing: the observed routing follows the natural thalweg, and a road bed 20–60 m away cannot force it into a
line. The caveat is TIGER's positional accuracy in rural ground (tens of metres) and its missing trails; the 25 m column is the
sensitivity to that.

**Extra heads within 15 m of a road on every surface of this addendum** (section 8, Appendix H): the real landscape 1 of its 68 extra heads;
the AD model 12 of 80; the M&D area × slope² rule 6 of 56; the M&D area rule 10 of 72; nothing fitted 5 of 45; area × slope² fitted 5 of 53.


## 5. The head finder on the real landscape: candidates and reading H-A (figure 3)

**Candidates (decision 2.4)**, each run on the 1 m lidar of the domain: (1) multiple-flow-direction area (Freeman 1991, exponent
1.1) on the filled 1 m surface with the lidar D8 for the 10 m downstream slope and the upstream maximum; (2) D8 on the surface smoothed
with a Gaussian of σ = 3 m, and (3) σ = 6 m, each filled and routed again; (4) the Addendum 22 12 m method on the box lattice as the
reference. The statistic is area × slope² (slope over 10 m downstream); a head is a cell where its upstream maximum first reaches the
threshold. At a pick the statistic is the largest value within 3 m (1 and 5 m as sensitivities; the 12 m reference uses the pick's 3 × 3
block as Addendum 21). Thresholds are the median at the calibration half's picks (TTV2 ground: 34 picks; TTV1 + TTV3 ground: 27), the
test is one-to-one matching on the other half's ground, 25 m primary.

| candidate | threshold set on TTV2 | tested on TTV1 + TTV3: found / real (heads) | threshold set on TTV1 + TTV3 | tested on TTV2: found / real (heads) | pooled found / real | H-A |
| --- | --- | --- | --- | --- | --- | --- |
| multiple-flow-direction area, 1 m | 45 m² | 0.96 / 0.002 (12,591) | 55 m² | 0.97 / 0.004 (7,793) | 0.97 / 0.003 | fails |
| D8 on the 1 m surface smoothed 3 m | 18 m² | 1.00 / 0.008 (3,576) | 54 m² | 0.76 / 0.055 (474) | 0.87 / 0.013 | fails |
| D8 on the 1 m surface smoothed 6 m | 25 m² | 0.85 / 0.011 (2,018) | 23 m² | 0.82 / 0.018 (1,562) | 0.84 / 0.014 | fails |
| the 12 m method (reference) | 560 m² | 0.48 / 0.14 (94) | 718 m² | 0.29 / 0.20 (49) | 0.38 / 0.16 | fails |

**No candidate is accepted.** The 1 m candidates find almost every mapped head because they make thousands of heads: the median
area × slope² at the picks is 25–55 m² on the 1 m grids (median largest area within 3 m: 126–416 m²; the 12 m lattice gives 627 m² and
4,536 m² for the same picks), and a threshold that low starts a channel in every hollow and on every thread of the planar slopes, 2,000–
13,000 heads on the test ground against 27–34 picks. This is the Addendum 20 finding (1 m areas at heads are 10–60 × too small
because hollow flow splits into parallel threads) seen from the other side: multiple-flow routing and 3–6 m smoothing raise the area
at the picks only to a few hundred m². The median matching distance of the found heads is 3 m for the multiple-flow candidate and
9–13 m for the smoothed ones, so position is not the problem; discrimination is.

The sensitivity the directive did not ask for makes the same point: the threshold that maximises the F1 score of the two shares on the
calibration half (grid of 37 values, 0.05–20 × the median) is 370–760 m² for the 1 m candidates, and at it the test half gives found /
real 0.26 / 0.01 and 0.32 / 0.01 (multiple-flow), 0.37 / 0.12 and 0.32 / 0.26 (smoothed 3 m), 0.37 / 0.18 and 0.32 / 0.20 (smoothed
6 m); the 12 m reference at its F1-optimal 661 and 1,001 m² gives 0.44 / 0.19 and 0.26 / 0.25. No threshold on any candidate brings
both shares to the bar on either half; the best any 1 m candidate reaches is 0.37 found with 0.26 real, no better than the 12 m method.
Figure 3 shows the shares at 10–50 m: the found share of the 1 m candidates rises to 1 at 30–50 m because their heads are everywhere,
and the real share never leaves the floor.

**Consequence (decision 2.4, "if none is accepted").** The split models' channel networks keep the 12 m method (M&D's 876 m² and
6,120 m² on the full-site 12 m lattice, through the 12 m head cells' seeds onto the domain's 1 m routing: 20,995 and 23,354 channel
cells, 117 and 111 1 m heads); the head comparison on the model surfaces is reported in Appendix H only, and channel heads are
not a usable test with these methods.


## 6. Misfit, K and D on the domain (figures 4, 5)

**Setting.** Every model is solved on the domain's 3,145,269 1 m cells with the four outlets held at their observed elevation and
every other domain edge a no-flux divide; the misfit is the mean squared elevation error over the 2,012,633 domain cells inside the
M&D polygon; the whole-domain misfit of the same parameters is reported beside it. K₀ = 1.338 × 10⁻⁵ and the relief D = 0.0200 m²/yr
are the domain values of section 3; the K and D box is the Addendum 18 box (TTV2 config × 65/77).

| model | misfit in polygon (m²) | ÷ AD | whole-domain misfit at these parameters | K ÷ K₀ | D (m²/yr) | D ÷ relief D | D ÷ Heimsath's | threshold | channel cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AD model | 611.6 | 1 | 724.8 | 8.58 | 0.00091 | 0.045 | 0.36 | – | – |
| Split model, M&D area × slope² rule | 416.0 | 0.68 | 963.0 | 0.89 | 0.0198 | 0.99 | 7.9 | 876 m² (fixed) | 20,995 |
| Split model, M&D area rule | 436.5 | 0.71 | 611.1 | 0.90 | 0.0167 | 0.83 | 6.7 | 6,120 m² (fixed) | 23,354 |
| Split model, nothing fitted | 557.1 | 0.91 | 913.1 | 1 (fixed) | 0.0200 (fixed) | 1 | 8.0 | 876 m² (fixed) | 20,995 |
| Split model, area × slope² rule fitted | 408.4 | 0.67 | 990.0 | 0.90 | 0.0188 | 0.94 | 7.5 | 823 m² (M&D 876) | 21,559 |

**The AD model.** The profile over r = K/D (200 log-uniform points from 1.4 × 10⁻¹⁰ to 5 × 10⁷ with the closed-form U/D at each,
bounded Brent refinement; 217 factorisations at 115 s) has one minimum: K = 1.15 × 10⁻⁴ (8.58 × K₀), D = 0.00091 m²/yr, interior to the
box, polygon misfit 611.6 m². The 1 % level of the profile spans K 7.9–9.1 × K₀ and D 0.00073–0.0011 m²/yr, the 5 % level 7.3–10.7 × K₀
and D 0.00057–0.0017: the optimum is sharp. The whole-domain optimum is at K 10.5 × K₀, D 0.0017 (misfit 669.6; 635.5 inside the
polygon). As on the tiles (Addendum 22: K 2.2–10 ×, D 1.4–200 × below Heimsath's) the AD model needs a K far above the channels' and a D far
below the hillslopes': one surface cannot carry channels that incise at the measured K and hillslopes that diffuse at the relief D when
both act in every cell. Its D is 0.36 × Heimsath's and 0.045 × the relief D.

**The split models.** With M&D's area × slope² rule fixed, the fit returns K = 0.89 × K₀ and D = 0.0198 m²/yr, 0.99 × the relief D
measured on the same domain (and 7.9 × Heimsath's); with the area rule, 0.90 × K₀ and 0.83 × the relief D. Both K readings hold
(0.5–2) and both D readings hold (0.67–1.5). The nothing-fitted model (K₀, relief D, M&D's rule, no free parameter) misfits 557 m²,
1.34 × the two-parameter fit on the same network. Fitting the threshold as well moves it from M&D's 876 m² to 823 m² (the 1 % level of the profile
761–847 m² in two pieces with two isolated points near 4,000 m²; the 5 % level 450–1,083 m² in two pieces plus four higher islands up to 4,151 m²) and lowers the misfit by
2 % (408 against 416 m²); K and D stay at 0.90 × K₀ and 0.94 × the relief D, interior to the box. The fitted threshold is M&D's value
to within 7 %: on the domain, unlike on the tiles (Addendum 22: 559–4,607 m²), the free threshold agrees with the field one.

**Why the misfits are three to four times the tile misfits.** The Addendum 22 tile fits (68–152 m² for the same two models) fixed all
3,996 edge cells of each 1 km tile at their observed elevation, so every hillslope and every divide was pinned to the lidar at 1 km
spacing along the perimeter and the channels were anchored wherever they crossed the edge; the domain fixes 4 cells and lets every
divide float. The tile parameters applied on the domain show the difference (Appendix F, "tile parameters on the domain"):
TTV2's area × slope² parameters (K 1.32 × 10⁻⁵, D 0.0128) give a polygon misfit of 468 m² on the domain, 4.5 × their 105 m² on the tile,
and TTV1's and TTV3's (K 0.81 and 2.14 × 10⁻⁵) give 3,164 and 3,256 m², because on a 2 km trunk anchored only at the outlets a K that
is 0.6 or 1.6 × the domain's K₀ puts the channel profile tens of metres off. The tile misfits were therefore partly the edge
constraint's; the domain misfits are the models'. The whole-domain misfits (611–990 m²) exceed the polygon misfits again because the
eastern basin outside the polygon (0.36 of the domain) is not in the objective; its hillslopes are fitted only through the shared K
and D. The misfit over those 1,132,636 outside cells, implied by the two reported values ((n_domain × whole-domain − n_polygon ×
polygon) ÷ n_outside), is 926 m² for the AD model, 921 for the M&D area rule, 1,546 for the nothing-fitted model, and 1,935 and 2,023 for
the two area × slope² models: the eastern basin is fitted twice as badly by the area × slope² network as by the area network or by the
AD model. The whole-domain fits say the same (Appendix F): the M&D area × slope² fit over the whole domain runs D to the box bound
(0.0422 m²/yr, misfit 611.9) while the area-rule whole-domain fit stays interior (K 0.97 × K₀, D 0.0143, misfit 578.8). M&D did not map
the eastern basin, and the 876 m² rule measured at their heads does not transfer to it at the same K and D.

**Readings** (section 7 of the directive, on the domain): the split models beat the AD model, holds for all four (0.67–0.91 ×); K within
0.5–2 × measured holds for every fitted split model (0.89–0.90 ×) and fails for the AD model (8.58 ×); D within 0.67–1.5 × the relief D holds
for every fitted split model (0.83–0.99 ×) and fails for the AD model (0.045 ×).


## 7. Channelization: power spectrum and valley share; reading C-1 (figures 6, 7)

**Setting (decision 2.5).** Both measures are applied identically to the lidar and to the five model surfaces on the 1 m domain. The power
spectrum is taken on the largest axis-aligned square inside the domain, 1,008 m on a side (0.98 of it inside the polygon), after a
least-squares plane detrend and a 2-D Hann window, and binned in 14 logarithmic wavelength bins from 4 to 500 m; the quantity reported is
model power ÷ lidar power in each bin. The valley share is the share of domain cells whose contour curvature, after Gaussian smoothing
with σ = 10 m (5 m as the sensitivity), exceeds one threshold fixed from the lidar, its mean + 2 sd over the domain (0.086 m⁻¹ at 10 m,
0.156 at 5 m); the lidar's own share is 0.018 at 10 m and 0.010 at 5 m.

**The lidar's spectrum.** Normalised by a power law fitted over all 14 bins (exponent 4.58; 4.50 over ≥ 100 m), the lidar's normalised spectrum
peaks at 106 m (2.1 × the background), the valley-spacing wavelength in the sense of Perron et al. (2008), and falls back below its
background above the peak at 211 m (the roll-off). The
detrended variance of the square is 714 m² for the lidar, 339 for the AD model and 453–533 for the split models.

**Power ratio, model ÷ lidar (figure 6; every bin in Appendix F).**

| model | 19 m | 27 m | 38 m | 53 m | 75 m | 106 m | 150 m | 211 m | within 0.5–2 in every bin 20–200 m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AD model | 0.70 | 1.63 | 2.72 | 2.67 | 1.89 | 0.99 | 1.28 | 2.10 | no (above, 27–75 m) |
| Split model, M&D area × slope² rule | 0.23 | 0.25 | 0.26 | 0.23 | 0.18 | 0.12 | 0.20 | 0.53 | no (below, every bin) |
| Split model, M&D area rule | 0.38 | 0.40 | 0.44 | 0.44 | 0.31 | 0.19 | 0.33 | 0.71 | no (below, every bin) |
| Split model, nothing fitted | 0.21 | 0.24 | 0.25 | 0.23 | 0.17 | 0.12 | 0.19 | 0.50 | no (below, every bin) |
| Split model, area × slope² rule fitted | 0.24 | 0.27 | 0.28 | 0.25 | 0.19 | 0.13 | 0.22 | 0.59 | no (below, every bin) |

The AD model exceeds the lidar's power by 1.6–2.7 × from 27 to 75 m, matches it at the lidar's peak wavelength (0.99 at 106 m) and
exceeds it again in the 211 m bin (2.1); below 20 m it is as smooth as every other model (0.07–0.32), and above 250 m it has less than
half the lidar's power (0.38–0.48). Every split model lies below the lidar at every wavelength: 0.12–0.26 (M&D area × slope² rule),
0.19–0.44 (M&D area rule), 0.12–0.25 (nothing fitted) and 0.13–0.28 (area × slope² fitted) over 20–200 m, with the deepest deficit at
the lidar's own peak (0.12–0.19 at 106 m), and 0.5–0.8 above 200 m. Power is amplitude squared, so the split models carry 0.35–0.66 of
the lidar's amplitude at 20–200 m and the AD model 1.3–1.6 × at 27–75 m.

**Valley share (figure 7).** Model ÷ lidar at σ = 10 m (5 m): AD 1.35 (1.50); M&D area × slope² rule 0.99 (1.17); M&D area rule 1.12
(1.33); nothing fitted 1.06 (1.26); area × slope² fitted 1.02 (1.21). At 10 m every model is within 0.67–1.5 of the lidar; at 5 m the AD
model sits on the band's upper edge and the split models at 1.17–1.33.

**Reading C-1: no model matches the lidar's channelization.** Every model fails the spectral half (the AD model on the high side in the
27–75 m bins, the split models on the low side in every 20–200 m bin) and every model passes the valley-share half at σ = 10 m.
**The planner's expectation** (section 2.5): the AD model shows more power than the lidar at short wavelengths (yes: median ratio 2.67 over
20–60 m) and a larger valley share (yes: 1.35): held. The split models with M&D's rules are closer to the lidar than the AD model: on the
valley share yes (0.99 and 1.12 against 1.35); on the 20–60 m spectrum no for the area × slope² rule (0.25 is farther from 1 on a log scale
than 2.67) and yes for the area rule (0.44). Under the compound reading as written, **not held**.

**What the two measures say together.** The valley share counts cells of strong convergence, and the split models reproduce it because
their channel cells are the lidar's: the 876 m² or 6,120 m² network forced from the observed 12 m statistic. The spectrum measures relief
amplitude at each scale, and there the split models have too little. Between the forced channels the steady-state linear-diffusion
hillslope at D = 0.02 m²/yr is a smooth surface of constant Laplacian (−U/D) with none of the lidar's unchannelled hollows, benches or
metre-scale roughness, and the forced channels themselves carry less relief than the lidar's (misfit rms 20 m inside the polygon). The AD
model makes relief at 27–75 m that the lidar does not have, its channels reaching into every hollow at K 8.6 × K₀ with a D too small to
fill them; that is the finer dissection the planner expected, and figure 8 shows it. But its long-wavelength relief is low (less than half
the lidar's variance), so "more channelized" is true of the AD model at 27–75 m and false above 250 m. The valley share is the measure
that separates "channels where the lidar has them" (the split models) from "channels everywhere" (the AD model); the spectrum says that
neither class reproduces the amplitude of the real hillslope-scale topography, the split models by a larger factor.

## 8. Heads on the model surfaces

No 1 m finder was accepted (section 5), so the head comparison on the model surfaces uses the 12 m method and is reported in Appendix
H only. In one line: through the 12 m method on a block mean of each 1 m surface, the real landscape pairs 18 of the 64 mapped heads
one-to-one within 25 m with 68 extra; the split models pair 13–16 with 45–72 extra (0.72–0.89 × the lidar's share found, 0.87–1.13 × its
share real); the AD model pairs 3 with 80 extra (0.17 × the lidar on both shares), 12 of its extra heads within 15 m of a road against 1 of
the lidar's and 5–10 of the split models'. Channel heads are not a usable test with these methods.

## 9. What the evidence supports, and where it is weak

**Supported.**
1. **On one seam-free domain anchored only at its four outlets, the split models with M&D's rules fit the M&D area better than the AD
   model** (0.68–0.71 ×), with the measured K (0.89–0.90 ×) and a D in the relief range (0.83–0.99 ×); the AD model needs K 8.6 × K₀ and
   D 0.045 × the relief D (0.36 × Heimsath's). This is the Addendum 21–22 result with the tile-edge constraint removed and one grid, and
   on the domain the fitted area × slope² threshold (823 m²) agrees with M&D's 876 m² to 7 %, which it did not on the tiles.
2. **The AD model's line of extra channels on TTV2 is not a road artefact** (0.24 and 0.008 against the reading's 0.5; the road is 20–60 m
   from the stream).
3. **The AD model dissects the landscape more finely than the lidar** (1.6–2.7 × the power at 27–75 m; 1.35 × the valley share), **and the
   split models with M&D's rules put their valleys where the lidar has them** (valley share 0.99–1.12 ×). Both measures need no head
   finder and no channel definition.
4. **No head finder on these grids resolves the mapped heads.** The 1 m candidates have no discrimination (share real below 0.06 at the
   threshold the picks set and at most 0.26 at the F1-optimal one); the 12 m method has a low ceiling (0.38 found). Hollow flow on a 1 m
   grid splits into parallel threads (Addendum 20), and 3–6 m smoothing or multiple-flow routing does not rejoin it.

**Weak.**
1. **The split models carry only 0.12–0.44 of the lidar's spectral power over 20–200 m** (0.35–0.66 of its amplitude): they reproduce
   valley position, not valley-scale relief. Steady state, a single U, linear diffusion and a channel network forced from the lidar's
   12 m statistic all limit this, and the misfit (rms 20 m inside the polygon) says the same. The spectral half of C-1 is failed by every
   model, the split models by a larger factor than the AD model.
2. **The domain misfits (408–612 m²) are 3–4 × the tile misfits (68–252 m²)** because the tiles pinned 3,996 edge cells to the lidar. Neither
   is "the" misfit of the model; the domain values are the ones to carry, since on the domain the edges no longer do the models' work,
   but they are not comparable with the tile values.
3. **0.36 of the domain (the eastern basin) is outside the objective and fits badly at the polygon parameters under the area × slope²
   network** (implied misfit 1,935–2,023 m², against 921–926 for the area rule and the AD model); the whole-domain area × slope² fit runs
   D to the box bound. The rule measured in the M&D area does not transfer to the next basin at the same K and D.
4. **The k_s on the domain rests on four basins** (n_eff 2.6, the largest basin half the channel pixels), so its interval is indicative; the
   relief D's 68 % interval is 0.0189–0.0224 m²/yr and the D verdicts (0.83–0.99 ×) are inside it.
5. **TIGER's positional accuracy in rural ground and its missing trails:** the 25 m column (0.46 of the wall heads) is the sensitivity; the
   reading does not change, but the valley-floor trail that TIGER lacks is not in the check.
6. **The valley share depends on the smoothing scale:** at σ = 5 m the AD model sits on the band's edge (1.50) and the split models at
   1.17–1.33. The spectrum ratios are from one 1,008 m square, one realisation, with no uncertainty computed.
7. **Non-steady state and the single U** remain as in Addenda 21–22: the relief D is a calibrated quantity, and the fine dissection the AD
   model produces and the hollows the split models lack may both be transient forms.

## 10. Decisions needed

1. **Channel heads.** The 12 m method and the three 1 m candidates all fail on the real lidar (sections 5 and 8). I recommend that the
   paper carries no channel-head test and keeps the head maps as supporting evidence only, as the directive's "if none is accepted"
   branch says. The remaining route is M&D's own source-area measurement at each pick, which is field-scale work outside this project.
2. **Which channelization measure the paper carries.** The valley share is the test that separates the models in the expected direction
   (split models 0.99–1.12 ×, AD 1.35 ×) and needs no head finder; the power spectrum is a diagnostic that shows the AD model's finer
   dissection (27–75 m) and the split models' amplitude deficit at every scale. I recommend carrying both, with the amplitude
   (square-root) reading stated, and dropping the spectral half of C-1 as a pass/fail rule, since no model passes it and the two classes
   fail it in opposite directions. The alternative is to keep C-1 as written and report that nothing matches.
3. **Domain or tiles for the paper's TTV result.** I recommend the domain fits (one grid, no seams, four anchors; the split models at
   0.67–0.71 × AD, K 0.89–0.90 ×, D 0.83–0.99 ×) as the primary result, with the tile fits of Addendum 22 as the sensitivity, and a
   sentence that the two misfit scales are not comparable (item 2 of section 9).
4. **The eastern basin.** Outside the polygon the area × slope² network fits twice as badly as the area network or the AD model. Options:
   (a) keep the objective inside the M&D polygon and report the implied outside misfit beside it, as here; (b) score the whole domain,
   which moves the area × slope² D to the box bound. I recommend (a), with the eastern basin named as ground where M&D's rule was
   never measured.
5. **The package change (N14).** `outlets=` and `noflux_outside=` are on `v2.3-tiles` only, with tests, a clean suite and the regression
   gate. I recommend merging them with the next release; the defaults are bit-identical.
6. **The intermittent Addendum 7 test** (`test_slope_area_model_grids_are_not_converted_twice`: the ftUS and metre runs' AD slope–area
   exponents differ by 8 × 10⁻⁶ relative on one node and agree on another; tolerance 10⁻⁶ on an exponent of −0.014). I recommend an
   absolute tolerance on the exponent difference (10⁻⁴); left for Steve.
7. Nothing is merged or tagged.

## 11. Appendices

### A. Rules and verdicts (directive section 7)

| reading | statement | result |
| --- | --- | --- |
| H-A, head finder accepted | on the real lidar, both test halves at 25 m: share of mapped heads found ≥ 0.6 and share of found heads that are real ≥ 0.5 | fails for all four candidates: the 1 m candidates 0.76–1.00 found with 0.002–0.055 real; the 12 m reference 0.29–0.48 found with 0.14–0.20 real; none accepted |
| Road | the AD gorge line is road-influenced if ≥ 0.5 of its too-steep-wall extra heads lie within 15 m of a road or trail, or ≥ 0.5 of the observed trunk path lies within 5 m of a road | 0.24 (13 of 54) and 0.008: not road-influenced |
| C-1, spectrum | model ÷ lidar power within 0.5–2 in every wavelength bin from 20 to 200 m | fails for every model: AD 0.99–2.72 (above in the 27–75 m bins), split models 0.12–0.44 (below in every bin) |
| C-1, valley share | model ÷ lidar within 0.67–1.5 (σ = 10 m) | holds for every model: AD 1.35, split models 0.99–1.12 |
| C-1, both | matches the lidar's channelization | fails for every model |
| Planner's expectation (section 2.5) | AD more power at short wavelengths and a larger valley share; the M&D-rule split models closer to the lidar | AD halves held (2.67 median over 20–60 m; 1.35); the split models closer on the valley share only; not held as written |
| Misfit | the split models beat the AD model (polygon misfit) | holds for all four: 0.67, 0.68, 0.71, 0.91 × |
| K physical | K ÷ K₀ within 0.5–2 | holds for the three fitted split models (0.89–0.90); fails for the AD model (8.58) |
| D explained by relief | D ÷ relief D within 0.67–1.5 | holds for the three fitted split models (0.83–0.99); fails for the AD model (0.045) |

### B. The domain check (decision 2.1)

Tile TTV2 cut as in Addendum 22 (its 3,996 valid edge cells given as the outlets, no-flux faces on), through `PartitionedProfiler` and the AD model with the new options, against the Addendum 22 polygon fits (`tiles/TTV2.json`); tolerance relative 10⁻⁸.


| check | Addendum 22 value | domain route | relative difference | pass |
|---|---|---|---|---|
| PF_A | 68.180795 | 68.180795 | 0.0e+00 | yes |
| PF_C | 104.740981 | 104.740981 | 0.0e+00 | yes |
| F2_relief | 125.655549 | 125.655549 | 0.0e+00 | yes |
| AD_at_a22_K_D | 206.728375 | 206.728375 | 0.0e+00 | yes |

### C. Run ids, inputs and checksums

- Run directory `$GROUP_HOME/lemcvt_results/20261124T173105Z_a23_ttv_domain_roads_heads_channelization/`: `domain/` (domain.json, domain.npz: the box arrays, polygon, domain, outlets, roads, networks, picks, measured inputs, the domain check), `roads/` (road_check.json, .npz), `heads/` (heads1m.json, .npz, accepted.json), `fits/` (fixed.json, fixed_surfaces.npz, AD.json, AD_profile.parquet, AD_surfaces.npz, SJ_C.json, SJ_C_profile.parquet, SJ_C_surfaces.npz, AD_pilot.json, SJ_C_pilot.json, tile_params_on_domain.json), `measures/` (measures.json, spectra.npz, surfaces12.npz), `post/` (metrics.json, rules.json, tables.md, captions.json), `gate/` (a18_gate_after_domain_options.json); every job wrote a `job.json`.
- Full-site TTV 1 m grids (filled, flow direction, area; EPSG:6339): `$GROUP_HOME/lemcvt_results/20261118T160326Z_a15_stage_b/grid_TTV/TTV_1m_{filled,fd,area_cells}.tif`; the window read is rows 5416–10191, columns 8209–13067 of those grids.
- M&D georeference: `$GROUP_HOME/lemcvt_results/20261118T160326Z_a15_stage_b/md_georef/md1989_fig3.gpkg`, layers `study_area` and `channels`.
- Roads: `$GROUP_HOME/lemcvt_data/source/TIGER2024_roads/tl_2024_06041_roads.zip` (TIGER/Line 2024 ROADS, Marin County), read through `/vsizip/`.
- Steve's heads: `data/ttv/ttv_heads_manual_steve_20261119.csv`, sha256 ac87de3d4ca1f479e6eb2ceda2fedb033740b3d8f37186fd267329d5e4e702b6 (checked in code before use; 64 picks, all inside the domain and the polygon).
- Addendum 21 run (the full-site 12 m statistics `grid12_arrays.npz`, the pick seeds, the per-tile K₀, crest curvature and relief D): `$GROUP_HOME/lemcvt_results/20261123T184712Z_a21_ttv_channel_head_test/`; Addendum 22 run (the TTV2 polygon fits for the domain check, the TTV2 AD surface and extra heads for the road check, the TTV2 config bounds): `$GROUP_HOME/lemcvt_results/20261124T000138Z_a22_ttv_cleanup_head_metrics/`; Addendum 18 run (the regression gate's reference values): `$GROUP_HOME/lemcvt_results/20261119T143945Z_a18_expanded_fits/`.
- Git: every job ran at c694e02 with this addendum's package change and scripts in the working tree (`git=c694e02` in every job log); the scripts, tests, package change and figures are committed with this report.

### D. Slurm accounting

| job name | tasks | core-h used | core-h of limits | states | job ids |
| --- | --- | --- | --- | --- | --- |
| a23_tests (at c694e02 with the change, 09:27 PST) | 1 | 0.14 | 2.0 | COMPLETED (pytest exit 1: the Addendum 7 test) | 53563635 |
| a23_gate (Addendum 18 regression gate) | 4 | 13.41 | 48.0 | COMPLETED 4 | 53564415 (raw task ids 53564453, 53564454, 53564909, 53564415) |
| a23_gatecheck | 1 | 0.01 | 0.5 | COMPLETED (GATE PASS) | 53564450 |
| a23_domain | 2 | 0.03 | 5.0 | FAILED 1 (OGR `Layer_GetExtent` on the zipped TIGER shapefile, fixed by opening the datasource through `/vsizip/`), COMPLETED 1 | 53565023, 53565153 |
| a23_heads1m | 1 | 0.08 | 4.0 | COMPLETED | 53565288 |
| a23_fit_fixed | 1 | 0.03 | 6.0 | COMPLETED | 53565315 |
| a23_pilot (AD 6 evaluations, area × slope² 12) | 2 | 0.50 | 12.0 | COMPLETED 2 | 53565318 (raw task ids 53566269, 53565318) |
| a23_fit_SJ_C (area × slope² profile; pilot estimate 3.41 h printed before submission, limit 6.5 h) | 1 | 5.74 | 13.0 | COMPLETED (2.86 h) | 53566854 |
| a23_roads (road check, figures 1–3) | 1 | 0.01 | 0.5 | COMPLETED | 53566967 |
| a23_fit_AD (AD profile; pilot estimate 6.79 h printed before submission, limit 10 h) | 1 | 13.92 | 20.0 | COMPLETED (6.96 h) | 53567316 |
| a23_extra (tile parameters scored on the domain) | 1 | 0.04 | 2.0 | COMPLETED | 53567562 |
| a23_post_debug (pipeline check without the AD fit) | 1 | 0.03 | 4.0 | FAILED (figure script formatted a missing AD ratio) | 53581917 |
| a23_figs | 2 | 0.03 | 1.3 | COMPLETED 2 (debug figures without AD; the figure 6 legend fix) | 53583041, 53604926 |
| a23_post (measures, post-processing, figures; afterok the two profiles) | 1 | 0.04 | 4.0 | COMPLETED | 53567319 |
| a23_tests_after (full suite after the change, 16:51 PST) | 1 | 0.13 | 2.0 | COMPLETED (283 passed) | 53601118 |

Core-hours used: **34.1** (sacct CPU time; `serc` allocated 2 CPUs to most 1-CPU requests, so wall-hours are about half). Part B alone
20.3 (directive estimate ≤ 50). Core-hours of limits 124; largest single submission the gate array, 48; no submission over 300. Peak
MaxRSS 10.6 GB (a23_fit_AD; 48 GB requested). 17 submissions (21 Slurm jobs counting array tasks). The report was assembled on the login node
from the post files (`a23_assemble_report.py`, no computation).

### E. State check, decisions as applied, deviations

**At the start (2026-11-24 09:15 PST, sh03-ln06).** `v2.3-tiles` = `origin` = c694e02 (the Addendum 22 report); no job running;
`/etc/agents/AGENTS.md` and its Slurm and storage topic files read. The session was cut when the login node was killed at about
16:40 PST with the AD profile still running; a second session resumed at 16:40 PST on sh03-ln07 (AGENTS.md and the Slurm topic file read
again), ran the full suite after the change, waited for the profiles, fixed figure 6, and wrote this report from the post files. Tests on `serc` at c694e02 (53563635): ruff and mypy clean, pytest
exit 1: **one pre-existing failure**, `tests/test_addendum7.py::test_slope_area_model_grids_are_not_converted_twice` (the AD_opt
slope–area exponent of the ftUS and metre runs of the Addendum 7 synthetic tile differ by 8 × 10⁻⁶ relative, tolerance 10⁻⁶). It fails
on the committed tree without this addendum's edits, on `serc` (sh03-09n56) and on the login node, and the environment has not changed
since 2026-11-09; the same suite passed on 2026-11-23 (Addendum 22, 53477999) and **passed again at 16:51 PST on sh04-07n18 (53601118,
283 passed)**, so the failure is node-dependent rounding in a tolerance on a near-zero exponent (−0.014), not a change of result; no code
of this addendum touches it; it is reported here and left for Steve (section 10, item 6). The directive and the repo did not
disagree anywhere that required a stop. The Addendum 22 run directory, the full-site grids, the M&D georeference, the TIGER roads and
Steve's heads CSV (checksum checked in code) are where the directive says.

**Package change (directive 1, "a package change is allowed only where 2.1 needs it").** The v1/v2 assembly fixes the valid
grid-edge cells at their observed elevation and, for a hillslope cell beside a NaN or off-grid cell, subtracts the neighbour weight from
the Laplacian diagonal with nothing on the right-hand side: a ghost value of zero, not a divide. A basin domain embedded in NaN has no
valid grid-edge cell, so without a change the chi integration had no anchor and every divide would have been a wall at z = 0.
`extract_grid_data(..., outlets=mask, noflux_outside=True)` (common.py), carried by `boundary_cells`, `_boundary_anchor`,
`assemble_hillslope_system`, `advection_diffusion_model`, `partitioned_threshold_model` and `PartitionedProfiler`: the fixed cells are
the outlets (the largest-area cell must be one; the perimeter-outlet rule is not applied) and an invalid face adds nothing to the
diagonal. Defaults unchanged bit for bit; `tests/test_domain.py` (nine tests: flags and fixed set; the edge ring as outlets bit-identical
to the default for both models and both no-flux settings; the NaN-padded tile; the mirror-symmetric strip; the validation errors);
`docs/numerics_changes.md` N14. Full suite on `serc` after the change (53601118, 16:51 PST): ruff and mypy clean, pytest 283 passed including the nine domain tests.
The Addendum 18 regression gate (GM1 at m = 0.31 and
AP2 PAMAP, partitioned and AD, against v2.0.0 at 10⁻⁹): passes, every parameter and misfit at relative 0 to 3.8 × 10⁻¹⁵ (53564415, four tasks,
13.4 core-h; 53564450; `gate/a18_gate_after_domain_options.json`).

**Decisions as applied.** Every choice the directive left open (the same list heads `a23_common.py`):

1. **The window and the box.** The polygon's bounding box plus 1,500 m on every side is read from the full-site grids (4,775 × 4,858
   cells); the domain is checked not to touch the window edge; the arrays are cut to the domain's bounding box plus one NaN cell
   (1,867 × 2,918).
2. **Outlets.** Polygon cells with A ≥ 10⁴ m² whose D8 receiver is outside the polygon (or absent). Basins by breadth-first labelling
   upstream along the D8. An outlet whose receiver is a domain cell (its path leaves and re-enters the polygon) is nested: a domain
   cell, not fixed. FD is set to 0 at the four fixed outlets; everywhere else the full-grid D8 is kept, which is exact because every
   domain cell's donors are domain cells (checked: zero leaks, zero receiver-less cells besides the outlets).
3. **The polygon** is rasterised with cell centres (ALL_TOUCHED=FALSE) as in Addenda 19–22; its raster area (2,057,654 m²) matches the
   vector area (2,057,638 m²).
4. **Roads.** TIGER/Line 2024 ROADS, Marin County, every class the file holds, reprojected to EPSG:6339, rasterised with ALL_TOUCHED;
   distances by Euclidean distance transform; "within d m" is distance ≤ d.
5. **The 12 m networks** for the split models' channels are built from the Addendum 21 full-site 12 m statistics (M, Mdon at α = 2 and
   0): a 12 m head cell seeds the largest-area 1 m cell of its block inside the domain and the channel set is every domain cell at or
   downstream of a seed. The mapped-head network uses the Addendum 21 pick seeds the same way (all 64 inside the domain).
6. **K₀, crest curvature, relief D** as section 2's table: `ks_at_theta` with basins labelled by their fixed outlet (200 basin
   bootstraps); the Addendum 19 hilltop set on the 10³ m² network with the 10 m road buffer and without the segment filter (the Addendum
   20 D₀,N set; numerical-zero curvatures dropped), median curvature at 11 m with the 2,000-draw segment bootstrap (seed 20261124);
   relief D from the crest pixels inside the polygon traced to the mapped-head network, segment bootstrap of U L_H² / (2R).
7. **The 1 m head finder.** T = A S² with S the steepest-descent slope over 10 m downstream (`valley_slope_reach`); M the upstream
   maximum along the surface's D8; a head where M ≥ C and no donor has M ≥ C (`criterion_sets`). Multiple-flow area: Freeman (1991),
   exponent 1.1, descending-elevation sweep, a filled flat passes everything to its D8 receiver. Smoothing: NaN-aware Gaussian, then
   fill and D8 with `border="outlets"` (the domain's data-edge cells are outlets; they carry negligible area). The 12 m reference is
   `coarse_products` on the box lattice. Statistic at a pick: the largest T within 3 m (1 and 5 m reported). Halves: picks inside the
   TTV2 footprint (34) and inside TTV1 or TTV3 (27); three picks on no tile take part in neither half. Heads are counted on the test
   half's footprints ∩ domain ∩ polygon. The F1-optimal threshold is a reported sensitivity, not the reading.
8. **The fits.** The domain profiler with the four outlets fixed and no-flux divides; the polygon misfit is the masked Gram sum
   (`_terms_masked`), the whole-domain misfit beside it; bounds the Addendum 18 box × 65/77 (TTV2 config); AD: 200 log-uniform r,
   closed-form U/D on the polygon cells, bounded Brent, both objectives from the same LU; area × slope² fitted: the 12 m statistic of the
   17,770 12 m cells touching the domain through their seeds, the C range found by bisection so that the 1 m channel set is as large as
   A ≥ 10³ and 10⁵ m² (176–8,634 m²: at 176 m² the 1 m channel set is 64,935 cells, the size of the A ≥ 10³ m² network, and at 8,634 m² it is 6,272 cells, the size of A ≥ 10⁵ m²; 13,224 distinct statistic values, 3,079 in range, thinned to 400; 17 refinement points evaluated; 418 evaluations in all), 400 thinned values, refinement ≤ 400 around both minima, M&D's 876 m² added to the grid.
9. **Power spectrum.** Largest axis-aligned square of domain cells (1,008 m); plane detrend by least squares; 2-D Hann window
   (outer product); periodogram normalised by the window's sum of squares; 14 logarithmic bins from 4 to 500 m, mean power per bin;
   the lidar's characteristic wavelength and roll-off (Perron et al. 2008; a description, not a reading): a background power law fitted
   to every lidar bin by log–log least squares, the normalised spectrum = power ÷ background, the characteristic wavelength = the peak of
   the normalised spectrum at ≥ 20 m, the roll-off = the shortest wavelength above the peak at which the normalised spectrum falls back
   below 1; the exponent over the bins ≥ 100 m is reported beside the background exponent.
10. **Valley share.** Contour curvature (Perron et al. 2012 eq. 1) of the Gaussian-smoothed surface after nearest-neighbour filling of
    NaN; threshold = the lidar's mean + 2 sd over domain cells, applied to every surface; shares over domain cells (and over polygon
    cells, reported).
11. **Figures.** Every figure was rendered and looked at; one overlap was found and fixed: figure 6's legend covered the AD model's points at 27–38 m
    and was moved below the axes (53604926); the eight figures were looked at again after that. Hillshades are 1 m, lit from
    the north-west; the only 12 m image is the AD surface in figure 2, which the directive asks to redraw from Addendum 22 and whose
    heads were found on that 12 m block mean (the caption says so).
12. **Cost.** 34.1 core-hours used (sacct CPU time over the 20 a23 jobs; `serc` allocated 2 CPUs to most 1-CPU requests, so wall-hours
    are about half): the Addendum 18 gate 13.4; the AD profile 13.9 (6.96 h wall, 217 factorisations at 115 s; pilot estimate 6.79 h); the
    area × slope² profile 5.7 (2.86 h, 418 evaluations at 15–25 s; pilot estimate 3.41 h); the pilots 0.5; the two test suites 0.3; everything
    else 0.3. Part B alone (fits, measures, post-processing, figures) 20.3 against the directive's ≤ 50; core-hours of limits 124; the largest
    single submission the gate array (48 core-h of limits); peak MaxRSS 10.6 GB (the AD profile, 48 GB requested). Both pilot estimates
    were printed before the full profiles were submitted (Appendix D).

**Deviations from the directive, stated.** (a) The road layer carries no trails (section 4), so "roads and trails" is roads. (b) The
head-finder calibration adds the F1-optimal threshold as a labelled sensitivity. (c) The road-check buffers 5 and 25 m are added as
sensitivities beside the directive's 15 and 5 m. (d) The tile-parameter sensitivity of section 6 (the Addendum 22 parameters scored on
the domain) is added to explain the misfit change. (e) Nothing else was changed, dropped or added.

### F. Machine-generated tables (`post/tables.md`, verbatim; headings demoted)

#### Model key

| Plain name | What it is | Fitted parameters | internal code |
|---|---|---|---|
| **Real landscape (lidar)** | The observed surface, run through the same head-finding method as the models | none | obs |
| **AD model** | Co-located advection–diffusion; incision and diffusion act in every cell | K, D | AD |
| **Split model, M&D area × slope² rule** | Stream power where area × slope² ≥ 876 m² (measured at M&D's heads), diffusion elsewhere | K, D | PF_C |
| **Split model, M&D area rule** | Stream power where drainage area ≥ 6,120 m² (measured at M&D's heads), diffusion elsewhere | K, D | PF_A |
| **Split model, nothing fitted** | M&D area × slope² rule, K = measured K₀ (domain), D = relief D (domain) | none | F2_relief |
| **Split model, area × slope² rule fitted** | As above, but the threshold is chosen by the fit | K, D, threshold | SJ_C |

#### The domain

| quantity | value |
|---|---|
| domain area | 3.145 km² (3,145,269 1 m cells) |
| outlets (polygon exits with area ≥ 10⁴ m²) | 9 (4 fixed, 5 nested; polygon exit cells of any size 2,116) |
| largest outlet area | 1,226,719 m² |
| M&D polygon (valid cells) | 2,057,654 m² (vector area 2,057,638 m²) |
| share of the polygon the domain covers | 0.978 |
| hillslope fringe (polygon cells draining out below 10⁴ m²) | 45,021 cells, 0.022 of the polygon |
| share of the domain outside the polygon | 0.360 |
| box | 1867 × 2918 cells, origin x 539160, y 4191578 (EPSG:6339) |
| picks in the domain / in the polygon / both | 64 / 64 / 64 of 64 (per tile in domain {'TTV1': 22, 'TTV2': 34, 'TTV3': 5}) |
| M&D traced channel length inside the polygon | 31,079 m |
| TIGER features on the box | 29 ({'S1400': 29}); trail classes present: none |

#### Re-measured inputs on the domain, beside the per-tile values

| quantity | domain | TTV1 | TTV2 | TTV3 |
|---|---|---|---|---|
| k_s (χ–z at m = 0.33, A ≥ 10³ m²) | 4.859 (n basins 4, 68 % 4.702–5.010) | 6.850 | 4.871 | 2.921 |
| K₀ = U / k_s (m^0.34/yr) | 1.338e-05 | 9.489e-06 | 1.334e-05 | 2.226e-05 |
| crest curvature at 11 m (m⁻¹) | 0.0141 (n 2,039; 68 % 0.0134–0.0149) | 0.0159 | 0.0149 | 0.0147 |
| D_form = U / crest curvature (m²/yr) | 0.0046 | 0.0041 | 0.0044 | 0.0044 |
| physical-D check: measured crest curvature ÷ the curvature a linear model needs at D = 0.0025 (U/D = 0.026 m⁻¹) | 0.54 | – | – | – |
| relief D on the mapped-head network (m²/yr) | 0.0200 (n 1,569 crest pixels in the polygon; 68 % 0.0189–0.0224; L_H median 173 m) | 0.0175 | 0.0159 | 0.0230 |

#### The road check (Addendum 22 TTV2 surfaces)

| quantity | 5 m | 15 m | 25 m |
|---|---|---|---|
| share of the 54 too-steep-wall extra AD heads within the distance of a TIGER road | 0.07 (4) | 0.24 (13) | 0.46 (25) |
| share of all 66 extra AD heads | 0.06 (4) | 0.21 (14) | 0.39 (26) |
| share of the observed trunk path (A ≥ 10⁵ m², 1,458 cells in the polygon) | 0.008 | 0.05 | 0.21 |

Median distance of the wall heads to a road 27 m; of the trunk cells 65 m. TIGER features on TTV2: 3 ({'S1400': 3}); trail classes: none. Reading: road-influenced = no.


#### The head-finder candidates on the lidar (reading H-A: at 25 m, share found ≥ 0.6 and share real ≥ 0.5 on both test halves)

| candidate | calibrated on TTV2 (34 picks): threshold | tested on TTV1 + TTV3: found / real (heads) | calibrated on TTV1 + TTV3: threshold | tested on TTV2: found / real (heads) | pooled found / real | F1-optimal threshold on the calibration half (TTV2 / TTV1+3) and its test-half found / real | H-A |
|---|---|---|---|---|---|---|---|
| multiple-flow-direction area on the 1 m surface | 45 m² | 0.96 / 0.002 (12591) | 55 m² | 0.97 / 0.004 (7793) | 0.97 / 0.003 | 758 → 0.26 / 0.01; 405 → 0.32 / 0.01 | no |
| D8 on the 1 m surface smoothed 3 m | 18 m² | 1.00 / 0.008 (3576) | 54 m² | 0.76 / 0.05 (474) | 0.87 / 0.01 | 369 → 0.37 / 0.12; 652 → 0.32 / 0.26 | no |
| D8 on the 1 m surface smoothed 6 m | 25 m² | 0.85 / 0.01 (2018) | 23 m² | 0.82 / 0.02 (1562) | 0.84 / 0.01 | 421 → 0.37 / 0.18; 464 → 0.32 / 0.20 | no |
| the 12 m method of Addendum 22 (reference) | 560 m² | 0.48 / 0.14 (94) | 718 m² | 0.29 / 0.20 (49) | 0.38 / 0.16 | 661 → 0.44 / 0.19; 1,001 → 0.26 / 0.25 | no |

#### Head-finder candidates at every tolerance (pooled over both test halves; share found / share real)

| candidate | 10 m | 15 m | 20 m | 25 m | 30 m | 40 m | 50 m |
|---|---|---|---|---|---|---|---|
| multiple-flow-direction area on the 1 m surface | 0.70 / 0.002 | 0.90 / 0.003 | 0.97 / 0.003 | 0.97 / 0.003 | 1.00 / 0.003 | 1.00 / 0.003 | 1.00 / 0.003 |
| D8 on the 1 m surface smoothed 3 m | 0.36 / 0.005 | 0.57 / 0.009 | 0.79 / 0.01 | 0.87 / 0.01 | 0.90 / 0.01 | 0.95 / 0.01 | 0.98 / 0.01 |
| D8 on the 1 m surface smoothed 6 m | 0.43 / 0.007 | 0.54 / 0.009 | 0.72 / 0.01 | 0.84 / 0.01 | 0.89 / 0.02 | 0.97 / 0.02 | 1.00 / 0.02 |
| the 12 m method of Addendum 22 (reference) | 0.08 / 0.03 | 0.15 / 0.06 | 0.31 / 0.13 | 0.38 / 0.16 | 0.51 / 0.22 | 0.61 / 0.26 | 0.74 / 0.31 |

The statistic at the picks (largest area × slope² within 3 m; median over the 61 picks in the domain and polygon), and the median largest drainage area within 3 m:

| candidate | median area × slope² (m²) | 25th–75th | median area (m²) | heads in the domain ∩ polygon at that threshold |
|---|---|---|---|---|
| multiple-flow-direction area on the 1 m surface | 55 | 21–158 | 416 | 18833 |
| D8 on the 1 m surface smoothed 3 m | 41 | 14–150 | 212 | 1951 |
| D8 on the 1 m surface smoothed 6 m | 25 | 14–88 | 126 | 3612 |
| the 12 m method of Addendum 22 (reference) | 627 | 417–1,062 | 4,536 | 137 |

#### Misfit, K and D on the domain (misfit inside the M&D polygon; m² mean squared elevation error)

| model | misfit in polygon | ÷ AD | whole-domain misfit at these parameters | K ÷ K₀ | K physical (0.5–2) | D (m²/yr) | D ÷ relief D | explained by relief (0.67–1.5) | D ÷ Heimsath's | fitted threshold | channel cells |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AD model | 611.58 | 1.00 | 724.83 | 8.58 | no | 9.08e-04 | 0.05 | no | 0.4 | – | – |
| Split model, M&D area × slope² rule | 415.97 | 0.68 | 962.95 | 0.89 | yes | 0.0198 | 0.99 | yes | 7.9 | 876 m² area × slope² (fixed) | 20,995 |
| Split model, M&D area rule | 436.51 | 0.71 | 611.12 | 0.90 | yes | 0.0167 | 0.83 | yes | 6.7 | 6,120 m² area (fixed) | 23,354 |
| Split model, nothing fitted | 557.06 | 0.91 | 913.11 | 1.00 | – | 0.0200 | 1.00 | – | 8.0 | 876 m² area × slope² (fixed) | 20,995 |
| Split model, area × slope² rule fitted | 408.45 | 0.67 | 989.96 | 0.90 | yes | 0.0188 | 0.94 | yes | 7.5 | 823 m² area × slope² (M&D 876) | 21,559 |

#### Channelization (reading C-1)

Square of side 1008 m inside the domain (0.98 of it in the polygon). Lidar spectrum: background power-law exponent 4.58 (over ≥ 100.0 m: 4.50); characteristic wavelength (peak of the normalised spectrum) 106 m (2.11 × the background); roll-off above the peak 211 m.

| model | 5 m | 7 m | 9 m | 13 m | 19 m | 27 m | 38 m | 53 m | 75 m | 106 m | 150 m | 211 m | 298 m | 421 m | within 0.5–2 in every bin 20–200 m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AD model | 0.07 | 0.13 | 0.19 | 0.32 | 0.70 | 1.63 | 2.72 | 2.67 | 1.89 | 0.99 | 1.28 | 2.10 | 0.48 | 0.38 | no |
| Split model, M&D area × slope² rule | 0.04 | 0.07 | 0.12 | 0.18 | 0.23 | 0.25 | 0.26 | 0.23 | 0.18 | 0.12 | 0.20 | 0.53 | 0.62 | 0.60 | no |
| Split model, M&D area rule | 0.06 | 0.12 | 0.19 | 0.28 | 0.38 | 0.40 | 0.44 | 0.44 | 0.31 | 0.19 | 0.33 | 0.71 | 0.83 | 0.70 | no |
| Split model, nothing fitted | 0.04 | 0.07 | 0.11 | 0.17 | 0.21 | 0.24 | 0.25 | 0.23 | 0.17 | 0.12 | 0.19 | 0.50 | 0.57 | 0.53 | no |
| Split model, area × slope² rule fitted | 0.04 | 0.08 | 0.12 | 0.19 | 0.24 | 0.27 | 0.28 | 0.25 | 0.19 | 0.13 | 0.22 | 0.59 | 0.64 | 0.63 | no |

| model | valley share (σ = 10 m) | ÷ lidar | within 0.67–1.5 | valley share (σ = 5 m) | ÷ lidar | matches the lidar's channelization (C-1) |
|---|---|---|---|---|---|---|
| Real landscape (lidar) | 0.018 | 1.00 | – | 0.010 | 1.00 | – |
| AD model | 0.024 | 1.35 | yes | 0.016 | 1.50 | no |
| Split model, M&D area × slope² rule | 0.018 | 0.99 | yes | 0.012 | 1.17 | no |
| Split model, M&D area rule | 0.020 | 1.12 | yes | 0.014 | 1.33 | no |
| Split model, nothing fitted | 0.019 | 1.06 | yes | 0.013 | 1.26 | no |
| Split model, area × slope² rule fitted | 0.018 | 1.02 | yes | 0.013 | 1.21 | no |

Planner's expectation (AD more power than the lidar at 20–60 m: yes, median ratio 2.67; AD larger valley share: yes; M&D-rule split models closer than AD on both: no): held = no.


#### Heads on every surface through the 12 m method (heads inside the domain and the polygon; 25 m; the Addendum 22 metrics)

| model | heads | matched | missed | extra | share found | ÷ lidar | share real | ÷ lidar | extra heads within 15 m of a road |
|---|---|---|---|---|---|---|---|---|---|
| Real landscape (lidar) | 86 | 18 | 46 | 68 | 0.28 | 1.00 | 0.21 | 1.00 | 1 |
| AD model | 83 | 3 | 61 | 80 | 0.05 | 0.17 | 0.04 | 0.17 | 12 |
| Split model, M&D area × slope² rule | 69 | 13 | 51 | 56 | 0.20 | 0.72 | 0.19 | 0.90 | 6 |
| Split model, M&D area rule | 88 | 16 | 48 | 72 | 0.25 | 0.89 | 0.18 | 0.87 | 10 |
| Split model, nothing fitted | 59 | 14 | 50 | 45 | 0.22 | 0.78 | 0.24 | 1.13 | 5 |
| Split model, area × slope² rule fitted | 66 | 13 | 51 | 53 | 0.20 | 0.72 | 0.20 | 0.94 | 5 |

Mapped heads inside the domain and the polygon: 64. Area rule (6,120 m²) at 25 m, share found / share real: Real landscape (lidar) 0.28 / 0.22; AD model 0.27 / 0.20; Split model, M&D area × slope² rule 0.20 / 0.26; Split model, M&D area rule 0.16 / 0.20; Split model, nothing fitted 0.17 / 0.22; Split model, area × slope² rule fitted 0.19 / 0.24.


#### Appendix: the domain check (TTV2 cut as before, its edge ring as the outlets, through the domain route)

| check | Addendum 22 value | domain route | relative difference | pass |
|---|---|---|---|---|
| PF_A | 68.180795 | 68.180795 | 0.0e+00 | yes |
| PF_C | 104.740981 | 104.740981 | 0.0e+00 | yes |
| F2_relief | 125.655549 | 125.655549 | 0.0e+00 | yes |
| AD_at_a22_K_D | 206.728375 | 206.728375 | 0.0e+00 | yes |

### G. Figures (`reports/v2/figures/a23/`; captions as printed on each figure, from `post/captions.json`)

- **Figure 1. The domain.** `figures/a23/fig1_domain.png` (2.20 MB)
  - How to read: Hillshade of the full-site 1 m lidar around the M&D study area (dashed). The blue outline and tint are the domain: every cell that drains to one of the red outlets, which are the cells where flow leaves the study area with drainage area of at least 10,000 m². Orange triangles are the mapped channel heads. Scale bar 500 m; north up.
  - What it shows: Domain 3.15 km² (3,145,269 cells), 4 outlets held at their observed elevation (5 nested exits inside the domain); the domain covers 0.98 of the study area, 0.36 of the domain lies outside it, and the hillslope fringe that drains out below 10,000 m² is 0.02 of the study area. 64 of 64 picks lie in the domain and the study area.
- **Figure 2. The road check.** `figures/a23/fig2_road_check.png` (1.43 MB)
  - How to read: Left: the 1 m lidar of tile TTV2 with the TIGER roads (grey), the observed trunk flow path (blue), the M&D study area (dashed) and the extra heads of the AD model from Addendum 22 (filled: on too-steep gorge walls; open: elsewhere). Right: the same symbols on the AD model's own surface. Heads were found on a 12 m block mean of the AD surface and are drawn at their cell centres.
  - What it shows: 13 of 54 wall heads lie within 15 m of a TIGER road (share 0.24); 0.01 of the trunk path lies within 5 m of a road. TIGER holds 3 features on the tile and no trail-class feature. Reading 'road-influenced': no.
- **Figure 3. The head-finder test on the real lidar.** `figures/a23/fig3_head_finder_test.png` (0.25 MB)
  - How to read: Top row: the share of the test half's mapped heads that a candidate pairs one-to-one within the tolerance on the x axis. Bottom row: the share of the candidate's heads on that ground that are real (paired). Left column: threshold set on TTV2 ground (34 picks), tested on TTV1 + TTV3 ground (27); right column: the other way. A candidate is accepted if both shares clear the dotted bars at 25 m in both columns.
  - What it shows: Pooled over both test halves at 25 m: multiple-flow-direction area on the 1 m surface: 0.97 found, 0.00 real; D8 on the 1 m surface smoothed 3 m: 0.87 found, 0.01 real; D8 on the 1 m surface smoothed 6 m: 0.84 found, 0.01 real; the 12 m method of Addendum 22 (reference): 0.38 found, 0.16 real. Accepted: none.
- **Figure 4. Misfit per model on the domain.** `figures/a23/fig4_misfit.png` (0.09 MB)
  - How to read: One bar per model: the mean squared difference between the model surface and the lidar over the domain cells inside the M&D study area. Lower is better.
  - What it shows: AD model 611.6 m² (1.00 × AD); Split model, M&D area × slope² rule 416.0 m² (0.68 × AD); Split model, M&D area rule 436.5 m² (0.71 × AD); Split model, area × slope² rule fitted 408.4 m² (0.67 × AD); Split model, nothing fitted 557.1 m² (0.91 × AD).
- **Figure 5. K and D per model against the measured K₀, the relief D and Heimsath's D.** `figures/a23/fig5_K_D.png` (0.13 MB)
  - How to read: Left: each fitted model's K divided by the K measured from the channel profiles (K₀ = U/k_s on the domain); the grey band is a factor of 2 either way. Right: each model's D against the relief D measured on the domain (black line, grey band 0.67–1.5 ×) and Heimsath et al.'s 0.0025 m²/yr (red dashed).
  - What it shows: K₀ 1.338e-05, relief D 0.0200 m²/yr. AD model: K 8.58 × K₀, D 0.0009 m²/yr (0.05 × relief D, 0.4 × Heimsath's); Split model, M&D area × slope² rule: K 0.89 × K₀, D 0.0198 m²/yr (0.99 × relief D, 7.9 × Heimsath's); Split model, M&D area rule: K 0.90 × K₀, D 0.0167 m²/yr (0.83 × relief D, 6.7 × Heimsath's); Split model, area × slope² rule fitted: K 0.90 × K₀, D 0.0188 m²/yr (0.94 × relief D, 7.5 × Heimsath's).
- **Figure 6. Power spectrum ratio, model ÷ lidar, against wavelength.** `figures/a23/fig6_power_spectrum_ratio.png` (0.25 MB)
  - How to read: For each model, the binned 2-D power spectrum of its surface divided by the lidar's, on the largest square inside the domain, against wavelength. Above 1 the model is rougher than the real landscape at that scale; below 1 smoother. The grey band is a factor of 2 either way; the blue band is the reading's range of wavelengths.
  - What it shows: Square side 1008 m; the lidar's characteristic wavelength 106 m. AD model: 0.99–2.72 over 20–200 m; Split model, M&D area × slope² rule: 0.12–0.26 over 20–200 m; Split model, M&D area rule: 0.19–0.44 over 20–200 m; Split model, nothing fitted: 0.12–0.25 over 20–200 m; Split model, area × slope² rule fitted: 0.13–0.28 over 20–200 m.
- **Figure 7. Valley share per model ÷ lidar.** `figures/a23/fig7_valley_share.png` (0.15 MB)
  - How to read: A cell is valley where the contour curvature of the smoothed surface exceeds the lidar's mean plus two standard deviations (the same threshold for every surface). Bars: each model's valley share of the domain divided by the lidar's, at 10 m smoothing (solid) and 5 m (hatched). The grey band is the reading's range.
  - What it shows: Lidar valley share 0.018 at 10 m, 0.010 at 5 m. AD model 1.35 (5 m: 1.50); Split model, M&D area × slope² rule 0.99 (5 m: 1.17); Split model, M&D area rule 1.12 (5 m: 1.33); Split model, nothing fitted 1.06 (5 m: 1.26); Split model, area × slope² rule fitted 1.02 (5 m: 1.21).
- **Figure 8. Maps of the lidar and each model on 1 m hillshades with the roads.** `figures/a23/fig8_maps.png` (2.02 MB)
  - How to read: Six panels: the lidar and the five models, each shown as the hillshade of its own 1 m surface. Grey lines are TIGER roads and trails, the dashed line the M&D study area, the blue line the domain. Orange triangles are the mapped heads; circles are the heads the 12 m method finds on a block mean of each surface (filled where matched one-to-one within 25 m, open where extra).
  - What it shows: Real landscape (lidar): 18 matched of 64, 68 extra; AD model: 3 matched of 64, 80 extra; Split model, M&D area × slope² rule: 13 matched of 64, 56 extra; Split model, M&D area rule: 16 matched of 64, 72 extra; Split model, nothing fitted: 14 matched of 64, 45 extra; Split model, area × slope² rule fitted: 13 matched of 64, 53 extra.

Every figure was rendered and looked at before it went in; the one text overlap found (figure 6's legend over the AD model's points) was fixed and the figure looked at again. Hillshades are the 1 m surfaces lit from the north-west; the only 12 m image is the AD surface of Addendum 22 in figure 2, whose heads were found on that 12 m block mean (its caption says so).

### H. Heads on the model surfaces through the 12 m method (the directive's appendix branch)

No 1 m finder was accepted, so this is the Addendum 22 comparison repeated on the domain: the 12 m block mean of each 1 m surface on the domain box lattice, D8 of the filled 12 m surface, M&D's area × slope² rule at 876 m² (the area rule at 6,120 m² beside it), heads inside the domain and the polygon, Hungarian one-to-one matching to the 64 mapped heads at 25 m. The mapped heads here are 64, not Addendum 22's 61, because the three picks on no tile are inside the domain. Channel heads are not a usable test with these methods (section 5); the table is supporting evidence only.


| model | heads | matched | missed | extra | share found | ÷ lidar | share real | ÷ lidar | extra heads within 15 m of a road |
|---|---|---|---|---|---|---|---|---|---|
| Real landscape (lidar) | 86 | 18 | 46 | 68 | 0.28 | 1.00 | 0.21 | 1.00 | 1 |
| AD model | 83 | 3 | 61 | 80 | 0.05 | 0.17 | 0.04 | 0.17 | 12 |
| Split model, M&D area × slope² rule | 69 | 13 | 51 | 56 | 0.20 | 0.72 | 0.19 | 0.90 | 6 |
| Split model, M&D area rule | 88 | 16 | 48 | 72 | 0.25 | 0.89 | 0.18 | 0.87 | 10 |
| Split model, nothing fitted | 59 | 14 | 50 | 45 | 0.22 | 0.78 | 0.24 | 1.13 | 5 |
| Split model, area × slope² rule fitted | 66 | 13 | 51 | 53 | 0.20 | 0.72 | 0.20 | 0.94 | 5 |

Mapped heads inside the domain and the polygon: 64. Area rule (6,120 m²) at 25 m, share found / share real: Real landscape (lidar) 0.28 / 0.22; AD model 0.27 / 0.20; Split model, M&D area × slope² rule 0.20 / 0.26; Split model, M&D area rule 0.16 / 0.20; Split model, nothing fitted 0.17 / 0.22; Split model, area × slope² rule fitted 0.19 / 0.24.

