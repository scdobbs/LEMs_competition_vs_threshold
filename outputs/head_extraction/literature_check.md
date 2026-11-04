# Literature check for Test 2 (head extraction)

Date: 2026-11-04. Sources are the PDFs in `papers/`; page numbers are the
journal pages printed on each PDF. Each item says whether the directive's
reading was confirmed, corrected, or could not be checked.

## 1. Clubb et al. (2014), WRR 50, 4283–4304, doi 10.1002/2013WR015167

**Field-mapped head areas at the Ohio sites.** No per-site median or range
is tabulated. What the paper gives (p. 4296, section 4.2, and Figure 7):

- the *average* contributing area of the mapped heads, used as the "A
  threshold": 413 m² at Mid Bailey Run (53 heads), 1035 m² at Indian Creek
  (36 heads); for comparison 12,262 m² at Feather River (15 heads) and
  20,816 m² in the Virginia Piedmont (63 heads);
- the corresponding average AS² thresholds: 278 m² (Mid Bailey Run), 640 m²
  (Indian Creek), 1805 m² (Feather River), 1721 m² (Virginia);
- across all four sites, head areas "ranging from 1000 to 1,000,000 m²"
  (p. 4300, section 5.3). This range is stated for the pooled data set and
  is not consistent with the Ohio averages of 413 and 1035 m² unless the
  lower bound is loose; the paper does not resolve this. Figure 7 (p. 4297)
  plots the individual heads but values cannot be read from it precisely.

**R² of the area–slope relation at the mapped heads** (p. 4296): power-law
regressions of area on gradient gave R² = 0.0009 (Feather River), 0.0391
(Mid Bailey Run), 0.1342 (Indian Creek), 0.1196 (Virginia). Quote: "we find
no statistically significant relationship between the slope and drainage
area of the mapped channel heads, implying that a slope-area threshold is
not effective in identifying channel head locations." **Directive's reading
confirmed** (0.0009–0.13).

Also relevant: the mapped heads sit at lower drainage area than the
slope–area scaling break at Cascade Ridge, Mid Bailey Run and Indian Creek
(p. 4296); the A and AS² thresholds mislocate heads by 26 m (Mid Bailey Run)
and 26 m (Indian Creek) for A, 29 and 18 m for AS² (p. 4296). Pelletier's
method located the mapped heads with mean errors of −11, −7, +5 and −24 m
(negative = predicted upstream of mapped) at Feather River, Mid Bailey Run,
Indian Creek and Piedmont (abstract; p. 4295); DrEICH −34, +3, +12, −58 m
(abstract) or, in the discussion, "22 m in Indian Creek, 10 m in Mid Bailey
Run, 34 m in the Feather River, 66 m in the Piedmont" (p. 4301; the two
statements differ, the abstract values are the ones in Figure 5).

Site context (p. 4286): Wayne National Forest is on the "southern,
unglaciated Allegheny Plateau", MAP 1025 mm, Carboniferous–Permian
sandstone, shale, siltstone, limestone and coal, "mature dissected plateau,
with moderate to steep slopes, narrow ridges". That is the same
physiographic province and lithology as our AP tiles.

**Why it matters.** With the closest field analogue to AP showing no
area–slope relation at mapped heads (R² ≤ 0.13), a null in 6b at AP is the
expected outcome. Note also that Clubb et al. attribute part of the scatter
to the spread of head areas (three orders of magnitude) and to differing
initiation processes, not to measurement error alone.

## 2. Pelletier (2013), WRR 49, 75–89, doi 10.1029/2012WR012452

**Wiener-filter procedure** (section 2, p. 78–80; Figure 3): compute the
radially averaged power spectrum S(ν) of the DEM up to the Nyquist
frequency; model the signal spectrum |H(ν)|² by a power-law fit to the
low-frequency part; model the noise spectrum |N(ν)|² by a constant fit to
the high-frequency part "just below the Nyquist frequency"; filter in
frequency space with the transfer function Φ(ν) = |H|²/(|H|² + |N|²)
(equation 4, p. 79). "A key advantage of the OWF is that it involves no free
parameters" (p. 80). For the analytic test case the OWF is equivalent to
diffusive smoothing with DT ≈ 0.5 m² (p. 80).

**Contour curvature** (equation 1, p. 76, after Mitasova and Hofierka
1993): κ = (h_xx h_y² − 2 h_xy h_x h_y + h_yy h_x²) / ((h_x² + h_y²)
√(1 + h_x² + h_y²)); positive values are convergent (valleys).

**Head definition** (paragraph 17, p. 80): after filtering, "the method
computes the contour curvature ... and then searches the landscape, from
the highest elevation to the lowest elevation in rank order, identifying
locations where the contour curvature, κ, is greater than a user-prescribed
threshold value, κt. Once a valley head is found, the method 'flags' all
pixels located downstream (along any downslope flow pathway, not just the
direction of steepest descent), and the algorithm removes those downstream
points from consideration as valley heads regardless of their value of κ."
The filtered DEM is hydrologically conditioned (pits and flats raised by
1 cm, Pelletier 2008) before the flagging step. **This differs from the
directive's wording** ("the most upslope pixel of each connected valley
segment"): Pelletier's heads are the highest unflagged pixels exceeding κt
along every descending path, which is not the same as the top of each
connected component of the κ > κt mask. Both are computed in this test;
Pelletier's definition is primary.

**Default threshold:** κt = 0.1 m⁻¹ (p. 84, paragraph 30; p. 85; conclusions
p. 88). Clubb et al. used the same value. **Confirmed.**

**Area or discharge thresholds.** Head location (steps 1–3) uses none. Steps
4–6 route a unit discharge from each head by multiple flow direction and
remove *discontinuous reaches* where discharge per pixel divided by the
number of upstream heads falls below Rt = 0.1 (p. 84–85); that step affects
which valley segments are kept downstream, not where heads are placed. The
directive's prohibition on area thresholds in head location is therefore
compatible with Pelletier's steps 1–3 as published.

## 3. Kargère et al. (2025), ESurf 13, 403–415

**How heads were extracted at Gabilan** (p. 411, section 5): "Our data
originate from a 2015 DEM, as utilized by Grieve et al. (2016). These data
include channel heads using the Pelletier algorithm (Pelletier, 2013), which
estimates channel head locations from contour curvature. Floodplains were
manually mapped and excluded. We did not need to run a sink-filling
algorithm, since Gabilan features very little topographic roughness."
Routing: "Throughout this paper, we use the D∞ algorithm" (p. 407, section
2.2). The Pelletier heads are shown as yellow stars in their Figure 8
(p. 412). **Directive's reading confirmed**, with the qualification that the
heads were inherited from Grieve et al. (2016), so the filter and threshold
settings are those of that study, not restated here.

**r and r²** (p. 411, Figure 7 caption p. 412): "The characteristic length
corresponding to the inflection point is 62 m"; channel heads are "nodes
with A = (62 m)²", i.e. r² = 3844 m², "with r = 62 m in Eq. (12)"; they also
write r = 1.3 √(D/K) ≈ 62 m. **Confirmed** (62 m, 3.8 × 10³ m²). The value
comes from the inflection point of the binned slope–area curve of one
section of the DEM (Figure 7), not from the Pelletier heads; the agreement
of the two is their Figure 8.

**Same DEM as our GM tiles?** Could not confirm. Theirs is "a 2015 DEM, as
utilized by Grieve et al. (2016)". The manuscript says our Gabilan Mesa
lidar was collected and processed by NCALM but gives no acquisition year,
and the repo holds only the clipped 1 m tiles. If the Grieve et al. (2016)
DEM is the NCALM Gabilan Mesa acquisition, the source is the same, but
their analysed section and our five tiles need not overlap.

## 4. Montgomery & Dietrich (1989), WRR 25, 1907–1918, and (1992), Science 255, 826–830

**Regression** (1989, p. 1912, equation 10): "A = Λ tan θ^−1.65, R² = 0.75,
where Λ is 1978 m²", by simple least squares of the log-transformed data
with source area as the dependent variable, for 63 Tennessee Valley heads
(Table 1, p. 1910). Source-basin length: L = λ tan θ^−0.83, R² = 0.47,
λ = 67 m (equation 9, p. 1912). **Test 1 report confirmed.**

**Slope above channel heads** (1989). Abstract, p. 1907: "The local valley
slope at the channel head is inversely related to both source area and
source-basin length as well as to the contributing area per unit contour
length at the channel head. In contrast, valley slope does not vary with
drainage area upslope of channel heads." Body, p. 1912: "drainage areas for
points upslope of the channel head are well below those for channel heads
on similar slopes. Furthermore, for several locations upslope of two channel
heads, drainage area is independent of slope and both channels begin within
the variance for channel heads" (Figure 8, p. 1914). The 1988 Nature letter
(p. 233, Figure 1c caption) says the same: "downslope from the drainage
divide in two hollows the slope within the source area is relatively
constant". **Test 1 report confirmed**, with the caveat that the statement
rests on two hollows in Tennessee Valley.

**Envelope** (1992, p. 828, equations 3 and 4): "Upper and lower bounds to
the range in channel head locations are defined by the relations
AS² = 4000 m² (3), AS² = 500 m² (4)", for the Marin County (Tennessee
Valley) data of Figure 4B; converted to area per unit contour length with
b = 20 m they become (A/b)S² = 200 m and 25 m (equations 5 and 6). The
bounds are drawn to the range of the data, not fitted. **Confirmed.**

**Climate ordering** (1988, p. 234): the southern Sierra data (260 mm)
plot above Marin County (760 mm), which plot above Coos Bay (1500 mm):
"for a given local slope, the source area size required to initiate a
channel should increase with increasing aridity".
