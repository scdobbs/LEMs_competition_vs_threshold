# Measured parameters: methods

`lemcvt measure` produces, for every tile, the concavity θ, the channel steepness
k_s, the hilltop curvature C_ht and the derived initial diffusivity D₀ = U/C_ht,
from the tile's own 1 m grid with z in metres, using the same D8 drainage-area grid
the models use and no model output (addendum 1 to the v2 directive). This document
states the definitions and the reasons for each choice in enough detail for a
Methods section. Where the choice has a free parameter, the sensitivity to it is
part of the output (`params.json` and `curves.parquet` of a `measure` run).

## 1. Consistency with the models

The partitioned model integrates channel elevations as

    z = z_bc + (U/K) Σ A^(-m) ΔL,

with m = θ, n = 1 and the drainage area A in m². With χ defined by

    χ = ∫ (A₀/A)^θ dx,   A₀ = 1 m²,

the steady-state channel profile is z − z_outlet = k_s χ and k_s = U/K exactly.
`lemcvt measure` therefore (i) uses the tile's D8 area grid, (ii) sets A₀ = 1 m²,
(iii) integrates χ along D8 links with the model's own rule, χ_up = χ_down +
(A₀/A_up)^θ ΔL (the partitioned model integrates z_up = z_down + (U/K) A_up^(−m) ΔL),
with link lengths dx (east-west), dy (north-south) and √(dx²+dy²) (diagonal), and
(iv) takes k_s as the slope of z − z_outlet against χ through the origin. K₀ = U/k_s
then enters the tile config without any further conversion. On a landscape generated
by the partitioned model itself the fit recovers θ exactly and k_s = U/K to 1e-6
(`tests/test_measure.py`); the trapezoid rule of Perron & Royden (2013) is also
evaluated at θ* and reported (`ks_trapezoid_rule_at_theta_star`); it biases k_s by
about 1 % because A_down > A_up. On the AP tiles dx = 0.9752 m and
dy = 0.9751 m; the 7e-5 relative anisotropy is carried, not ignored.

## 2. Channel pixels and the area threshold

Channel pixels are those with A ≥ A_c. The primary A_c is chosen from the data:
the slope–area rollover (Montgomery & Foufoula-Georgiou 1993; Tarolli & Dalla
Fontana 2009). The D8 steepest-descent slope of the filled grid is binned by
log₁₀ A in 0.1-decade bins (≥ 50 pixels per bin), the bin medians are smoothed
with a 3-bin running median to suppress the TIN-facet noise of the AP grids, and
A_c* is the area of the maximum binned slope: below it slope increases with area
(hillslopes), above it slope decreases (channels). The curve, the chosen bin and a
note if the maximum sits at the edge of the range are written out.

Because any threshold is a judgement, θ and k_s are also computed at a fixed ladder
A_c ∈ {1e3, 2e3, 5e3, 1e4, 2e4, 5e4} m² and reported side by side (`chi_fits`), so
the dependence of the result on the threshold is visible rather than hidden.

Basins are the channel networks inside the tile: an outlet is a channel pixel whose
D8 receiver leaves the grid. The area grids were computed on the full DEM and clipped
to the tile, so every pixel carries its true drainage area, and χ is integrated
upstream from each in-tile outlet with those areas. A catchment cut by the tile edge
therefore yields a valid, shorter χ–z profile: its χ and z are correct over the
reach the tile contains, and what the edge removes is channel length, not accuracy.
Truncation is a sample-size issue for the fit, not a bias (addendum 3, §4.2). Networks
that enter the tile already large are treated as their own basins with χ = 0 and z
relative to the pixel where they leave. Only basins with ≥ 10 channel pixels enter
the fits; their number and the number of edge outlets are reported.

## 3. Concavity θ

θ is scanned on a grid 0.10–0.90 in steps of 0.01. For each θ, χ is computed for
every channel pixel, and in every basin z − z_outlet is regressed on χ through the
origin. The primary objective is the total squared residual over all basins
(Perron & Royden 2013, "best χ–z fit"); θ* is its minimum. The whole misfit curve
is stored (`ssr_total` against `thetas`) and plotted so the sharpness of the
minimum can be judged.

As an independent check that does not assume a linear χ–z relation, the disorder
statistic of Hergarten et al. (2016) as used by Mudd et al. (2018) is computed on
the same χ arrays: within a basin, sort the pixels by χ and take
(Σ|Δz| − (z_max − z_min))/(z_max − z_min); collinear tributaries give small values.
Its minimiser θ*_disorder is reported next to θ*.

The legacy v1 method (`legacy/measurement_notebooks/updated_best_fit_tile_finder.ipynb`,
cell 11) also minimised the total through-origin SSR, but with basins defined by
every tile-edge pixel (TopoAnalysis `outlets_at_coastlines`), channel pixels above
Taylor's thresholds (5.04e4 m² on AP, 1e4 m² on GM), a coarse–fine θ grid, link
lengths dx and dx√2, and, on AP, z in feet. It is re-run on the metre grids for
comparison (`comparisons.legacy_v1_method_rerun`).

## 4. Steepness k_s

At θ*, k_s is the pooled through-origin slope of z − z_outlet against χ over all
basins (one value per tile, the one the models need), together with the per-basin
k_s distribution (median, 16–84 %). The binned slope–area regression of v1
(`analysis.py`: 24 log bins, bin medians, count-weighted least squares, 10 m
vertical interval for the channel slope) is run at A_c* as a cross-check and
reported as `slope_area` (its θ and k_s are not used for K₀).

### 3.1 Estimators and their effective sample size (addendum 3, §4.1)

The pooled through-origin objective weights every basin by its χ spread, so a few
large networks can dominate it. `lemcvt audit measure` therefore reports, beside the
pooled θ*, (i) n_eff = 1/Σwᵢ², with wᵢ basin i's share of the total Σ(χ − χ̄)² (and,
for reference, of the through-origin leverage Σχ²), evaluated at θ*; where n_eff < 3
the basin bootstrap is not a usable uncertainty; (ii) an equal-weight estimator,
θ* = argmin Σᵢ SSRᵢ(θ)/SSRᵢ,null with SSRᵢ,null the basin's sum of squares about its
own mean elevation, so every basin counts equally; (iii) the median over basins of
the per-basin best θ. The three are reported side by side in
`reports/v2/02e_audits.md`; none of them is yet an input to a tile config.

## 5. Uncertainty of θ and k_s

Both are bootstrapped over basins (200 draws by default): basins are resampled with
replacement and θ* and the pooled k_s recomputed from the per-basin SSR(θ) curves
and cross products. 16–84 % and 2.5–97.5 % percentile intervals are reported. The
unit of resampling is the basin because residuals within a channel network are
strongly correlated along the profile.

## 6. Hilltop pixels

A pixel is a hilltop pixel when all of the following hold:

1. it receives no D8 flow (a divide cell, TopoAnalysis `divides`);
2. it is adjacent (8-neighbourhood) to a pixel draining to a different basin of
   the channel network at A_c*, i.e. it lies on the drainage divide between two
   catchments, after Hurst et al. (2012), who take hilltops as the divides between
   basins of a channel network;
3. it is at least two pixels from the tile edge;
4. its local slope |∇z| ≤ 0.4 (Hurst et al. 2012 exclude steeper hilltops, where
   the linear-diffusion assumption behind D = U/C_ht fails). The number removed is
   reported.

Every pixel's basin is the channel pixel it reaches first along D8 receivers,
relabelled by that channel pixel's network outlet.

## 7. Curvature and the fitting scale

C_ht is the Laplacian ∇²z = ∂²z/∂x² + ∂²z/∂y² (not the mean, Gaussian or profile
curvature). This is the quantity in the steady-state hilltop balance U = −D ∇²z, so
D₀ = U/|∇²z| is consistent with the models' diffusion operator, whose discrete form
is the 5-point Laplacian in `lemcvt.models.common`. The Laplacian is taken from the
least-squares quadratic surface z = a x² + b y² + c xy + d x + e y + f over a (2w+1)²
window (Evans 1980; Hurst et al. 2012): ∇²z = 2(a + b), i.e. the Laplacian of the
locally smoothed surface; for w = 1 on a regular grid it reduces to the standard
9-point finite-difference Laplacian. For a fixed window this is a linear filter, so C
is computed exactly as a convolution with a kernel built from the pseudo-inverse of
the design matrix (`curvature_kernel`), using dx and dy. Hilltops are convex up
(C < 0); C_ht is the median of −C over hilltop pixels, a positive magnitude, and
D₀ = U/C_ht.

Near-channel exclusion (curvature option (i), addendum 3, §4.3). Because a window
longer than the crest averages the straight flanks into the fit and biases the
curvature low (the synthetic-ridge test of `02c`, §7), hilltop pixels closer than
L*/2 to a channel pixel of the network at A_c are dropped before the median is taken.
`lemcvt audit measure` reports C_ht and D₀ with and without the exclusion, on the
networks at A_c = 10³ and 10⁴ m² and on the rollover network that `lemcvt measure`
uses to define the divides, so that the dependence on the network threshold is visible.

Lidar hilltop curvature depends on the window because of metre-scale roughness and,
on AP, TIN facets. C_ht is computed for windows L = (2w+1)·dx with
w ∈ {1, 2, 3, 4, 5, 7, 10, 12, 15, 20, 25, 30}, i.e. L ≈ 3–61 m. The chosen scale
L* is the smallest window at which the median has stopped changing: the relative
change between successive windows is below 5 % for that step and the next. The same
rule is applied at both sites; the whole curve (median and 16–84 % spread of −C,
and the fraction of hilltop pixels that are convex) is stored. v1 instead fixed the
scale by block-averaging to 30 m (AP) and 10 m (GM) before taking a 3×3 Laplacian;
that is reproduced approximately for comparison (the `topo_analysis` module v1 used
is not available; the approximation is labelled as such).

## 8. Uncertainty of C_ht and D₀

Hilltop pixels are grouped into 8-connected segments; segments are resampled with
replacement (200 draws) and the weighted median recomputed. The 16–84 % and
2.5–97.5 % intervals of C_ht are reported and propagated to D₀ = U/C_ht.

## 9. Comparisons (reported, not acted on)

For each tile the run records: the v1 hard-coded θ, k_s, C_ht (AP values divided by
3.2808 to bring the feet-based k_s and curvature to metres; θ is unit-free), the
regional values from the site config (flagged there as still to be verified against
Perron et al. 2012), the pooled k_s at the v1 θ on the new channel set, the ratios
new/v1, and the re-run of the legacy method. These comparisons are for the record;
Phase 6 uses the new values.

## References

- Evans, I. S. (1980). An integrated system of terrain analysis and slope mapping. Z. Geomorph. Suppl. 36, 274–295.
- Hergarten, S., Robl, J., Stüwe, K. (2016). Tectonic geomorphology at small catchment sizes – extensions of the stream-power approach and the χ method. Earth Surf. Dynam. 4, 1–9.
- Hurst, M. D., Mudd, S. M., Walcott, R., Attal, M., Yoo, K. (2012). Using hilltop curvature to derive the spatial distribution of erosion rates. J. Geophys. Res. 117, F02017.
- Montgomery, D. R., Foufoula-Georgiou, E. (1993). Channel network source representation using digital elevation models. Water Resour. Res. 29, 3925–3934.
- Mudd, S. M., Attal, M., Milodowski, D. T., Grieve, S. W. D., Valters, D. A. (2014). A statistical framework to quantify spatial variation in channel gradients using the integral method of channel profile analysis. J. Geophys. Res. Earth Surf. 119, 138–152.
- Mudd, S. M., Clubb, F. J., Gailleton, B., Hurst, M. D. (2018). How concave are river channels? Earth Surf. Dynam. 6, 505–523.
- Perron, J. T., Royden, L. (2013). An integral approach to bedrock river profile analysis. Earth Surf. Process. Landforms 38, 570–576.
- Tarolli, P., Dalla Fontana, G. (2009). Hillslope-to-valley transition morphology: new opportunities from high resolution DTMs. Geomorphology 113, 47–56.

Citations above are from memory of the literature and have not been checked against
the sources in this session; verify volume and page numbers before use.
