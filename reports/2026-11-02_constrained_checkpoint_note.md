# Constrained re-run: checkpoint (Section 4 of the constrained directive)

Date: 2026-11-02. Job 50848713, outputs in `outputs/widths_constrained/checkpoint/`
(`points.csv`, `excluded.csv`, `transects.csv`, `points.gpkg`, `qc/*.png`, one frame per point).
Default window: relief 0.3-1.5 m, offset <= 6 m; five transects at -10, -5, 0, +5, +10 m.

Sample: 2 per bin, 5 bins from 1e5 m² to the site maximum. GM: 10 points. AP: 8 points.
AP bin 3 (2.2e6-6.1e6 m²) has no channel pixels on any AP tile (the AP tiles jump from
1.8e6 to 9e6 m²), and AP bin 4 reached only 5 of 10 in the full draw (all 546 candidates
tested; the trunk on tiles 3 and 4 is short, with edge, confluence and spacing rules
excluding the rest). The rules were not relaxed.

| id | A (m²) | S | class (counts over 5 transects) | L / R side class | W_bank (n, spread) | W_stage_bank | ratio | relief L / R (m) | nearest break L / R (offset m, relief m) | W_stage_0.5 | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AP2_000 | 1.6e5 | 0.21 | ambiguous (amb 5) | bank / no_break | - | - | - | 0.94 / - | (1.8, 0.97) / (7.0, 6.61) | 2.6 | |
| AP5_001 | 1.1e5 | 0.19 | unresolved_narrow (un 4, amb 1) | no_break / no_break | - | - | - | - / 1.25 | (4.0, 9.76) / (1.8, 1.25) | 2.1 | |
| AP5_002 | 7.6e5 | 0.07 | unresolved_narrow (un 3, amb 2) | no_break / no_break | - | - | - | 1.24 / - | (4.5, 2.94) / (2.0, 1.87) | 2.8 | |
| AP1_003 | 7.2e5 | 0.09 | unresolved_narrow (un 4, amb 1) | no_break / no_break | - | - | - | - / - | (2.2, 4.03) / (4.0, 3.74) | 2.0 | |
| AP5_004 | 1.2e6 | 0.05 | ambiguous (amb 3, ic 2) | no_break / bank | 5.12 (2, 0.25) | 5.53 | 0.93 | 0.52 / 0.66 | (2.5, 0.47) / (2.5, 0.85) | 6.4 | |
| AP1_005 | 9.7e5 | 0.04 | ambiguous (amb 3, un 2) | no_break / no_break | - | - | - | 0.81 / - | (2.2, 1.22) / (13.2, 22.6) | 3.6 | |
| AP4_006 | 1.7e7 | 0.01 | ambiguous (amb 5) | no_break / no_break | - | - | - | 0.86 / - | (3.0, 0.62) / (3.5, 2.86) | 6.0 | |
| AP3_007 | 1.1e7 | 0.01 | ambiguous (amb 4, ic 1) | no_break / bank | 7.75 (1, 0) | 10.86 | 0.71 | 1.16 / 0.57 | (2.8, 2.35) / (3.0, 0.45) | 5.9 | asymmetric_banks |
| GM3_000 | 1.6e5 | 0.07 | ambiguous (amb 3, ic 1, un 1) | no_break / no_break | 4.75 (1, 0) | 4.67 | 1.02 | 0.57 / 0.69 | (5.2, 1.86) / (17.0, 2.25) | 5.4 | |
| GM2_001 | 2.1e5 | 0.06 | ambiguous (amb 3, un 2) | bank / no_break | - | - | - | 0.41 / - | (2.0, 0.43) / (21.2, 8.73) | 6.5 | |
| GM5_002 | 3.3e5 | 0.05 | ambiguous (amb 5) | no_break / no_break | - | - | - | - / - | none / none | 15.6 | |
| GM5_003 | 3.8e5 | 0.05 | ambiguous (amb 5) | no_break / no_break | - | - | - | 0.48 / - | (3.0, 0.35) / none | 27.9 | |
| GM1_004 | 1.1e6 | 0.04 | inner_channel (ic 5) | bank / bank | 5.75 (5, 1.25) | 6.30 | 0.91 | 0.89 / 1.13 | (3.0, 0.89) / (3.5, 1.05) | 3.7 | |
| GM1_005 | 1.1e6 | 0.06 | unresolved_narrow (un 4, amb 1) | no_break / no_break | - | - | - | 1.05 / - | (4.0, 2.05) / (3.0, 1.93) | 3.1 | |
| GM4_006 | 2.4e6 | 0.03 | ambiguous (amb 4, ic 1) | no_break / bank | 7.50 (1, 0) | 7.56 | 0.99 | 1.41 / 1.11 | (4.2, 1.75) / (4.2, 1.15) | 5.0 | |
| GM5_007 | 2.7e6 | 0.03 | unresolved_narrow (un 3, amb 2) | no_break / no_break | - | - | - | 1.42 / - | (3.0, 1.50) / (3.5, 1.82) | 2.4 | |
| GM2_008 | 8.8e6 | 0.02 | ambiguous (amb 5) | bank / no_break | - | - | - | 0.70 / 0.50 | (3.5, 0.58) / (2.8, 1.91) | 5.0 | |
| GM2_009 | 8.8e6 | 0.02 | ambiguous (amb 2, ic 1, un 2) | no_break / bank | 7.50 (1, 0) | 7.49 | 1.00 | 1.21 / 1.46 | (6.2, 1.75) / (5.2, 1.53) | 4.9 | class_tie |

"nearest break" is the first convex break outward from the thalweg on the centre
transect, whatever its relief; relief L / R are medians over the transects whose side
class is `bank`.

Class counts: inner_channel 1, unresolved_narrow 5, ambiguous 12, wide_or_terrace 0.

Reading (for Steve's review of the frames):

1. The relief ceiling of 1.5 m is what fails most points. On 10 of the 18 the
   nearest break on at least one side sits 2 to 6 m from the thalweg with 1.5 to
   4 m of relief (AP1_003: 4.0 / 3.7 m; AP5_002: 2.9 / 1.9; GM1_005: 2.1 / 1.9;
   GM5_007: 1.5 / 1.8; GM2_008 right: 1.9; GM4_006 left: 1.75; GM2_009: 1.75 /
   1.5). These are incised channels 3 to 8 m wide and 2 to 4 m deep, not sub-2 m
   features; the window calls them `no_break`, and the class rule then labels a
   V-shaped one `unresolved_narrow` (W <= 2δ), which is the wrong direction.
2. `ambiguous` is mostly bank + wall: one accepted bank and, on the other side, a
   steep valley wall whose first break is above the relief ceiling (AP2_000,
   GM2_001, GM2_008). The channel is real and 3 to 5 m wide at the accepted
   bank's height (W_stage_0.5 is 2.6 to 6.5 m). A "one bank plus wall" rule
   (width at the accepted bank's relief, taken from W_stage_bank) would resolve
   these without widening the window.
3. The offset limit of 6 m is not binding: no side was `none_within_window`.
4. GM5_002 and GM5_003 are flat valley floors (16 and 28 m at 0.5 m) with no
   break at all or a 0.35 m notch: genuinely unresolved on this DEM.
5. GM1_004 is the one clean `inner_channel`: 5.75 m, five transects agree, ratio
   0.91. AP5_004, AP3_007, GM3_000, GM4_006 and GM2_009 each have one or two
   transects with both banks (W_bank 4.8 to 7.8 m, ratio 0.7 to 1.0) and the rest
   failing on one side by the relief ceiling.
