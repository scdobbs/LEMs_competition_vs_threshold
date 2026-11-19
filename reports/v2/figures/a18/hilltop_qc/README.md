# Addendum 18, section 3.2: hilltop QC figures

Made by `reports/v2/scripts/a18_hilltop_qc.py` from the addendum 17 Part B `measure` runs (the option (iii) hilltop sets at L_f = 11 m, recomputed from each tile's rollover A_c with `lemcvt.measure.phase6_inputs.option_iii_mask`; no new measurement). PNG at 200 dpi (downscaled where a file would exceed 5 MB); hillshade azimuth 315°, altitude 45°, no vertical exaggeration, from the tile's own 1 m filled grid.

## Colour legend (the same on every figure; Okabe-Ito colours at 60 % opacity over the hillshade)

- (i) `#0072B2`: channel (rollover network, A ≥ A_c)
- (ii) `#E69F00`: divide, excluded: within L_f/2 of a channel cell
- (iii) `#D55E00`: counted hilltop (option (iii) at L_f = 11 m): gives C_ht and D₀
- (iv) `#009E73`: divide, dropped: not between two basins, within 2 cells of the edge, or |∇z| > 0.4
- (v) uncoloured: not part of the measurement
- thin `////` hatch: cells of `nnfill_mask` or `gapfill_mask` (NCALM and PAMAP tiles have no mask)
- survey-pair overlays: `#CC79A7` old survey only, `#56B4E9` new survey only, black both

Zoom boxes: the 200 × 200 m box holding the most class (iii) pixels of the tile (on pair figures, the box of the old survey). Panel (c) of the per-tile figures uses one colour scale per site (the 2nd-98th percentiles of the site's class (iii) curvature values).

## Figures

### Per-tile QC (item 2)

- `AP10_hilltop_qc.png`
- `AP11_hilltop_qc.png`
- `AP13_hilltop_qc.png`
- `AP14_hilltop_qc.png`
- `AP18_hilltop_qc.png`
- `AP19_hilltop_qc.png`
- `AP1_3dep_hilltop_qc.png`
- `AP1_hilltop_qc.png`
- `AP20_hilltop_qc.png`
- `AP23_hilltop_qc.png`
- `AP24_hilltop_qc.png`
- `AP25_hilltop_qc.png`
- `AP27_hilltop_qc.png`
- `AP28_hilltop_qc.png`
- `AP29_hilltop_qc.png`
- `AP2_3dep_hilltop_qc.png`
- `AP2_hilltop_qc.png`
- `AP3_3dep_hilltop_qc.png`
- `AP3_hilltop_qc.png`
- `AP4_3dep_hilltop_qc.png`
- `AP4_hilltop_qc.png`
- `AP5_3dep_hilltop_qc.png`
- `AP5_hilltop_qc.png`
- `AP6_hilltop_qc.png`
- `AP7_hilltop_qc.png`
- `AP8_hilltop_qc.png`
- `AP9_hilltop_qc.png`
- `GM11_hilltop_qc.png`
- `GM12_hilltop_qc.png`
- `GM13_hilltop_qc.png`
- `GM14_hilltop_qc.png`
- `GM15_hilltop_qc.png`
- `GM16_hilltop_qc.png`
- `GM17_hilltop_qc.png`
- `GM18_hilltop_qc.png`
- `GM19_hilltop_qc.png`
- `GM1_fema_hilltop_qc.png`
- `GM1_hilltop_qc.png`
- `GM20_hilltop_qc.png`
- `GM21_hilltop_qc.png`
- `GM22_hilltop_qc.png`
- `GM24_hilltop_qc.png`
- `GM25_hilltop_qc.png`
- `GM26_hilltop_qc.png`
- `GM27_hilltop_qc.png`
- `GM28_hilltop_qc.png`
- `GM29_hilltop_qc.png`
- `GM2_fema_hilltop_qc.png`
- `GM2_hilltop_qc.png`
- `GM30_hilltop_qc.png`
- `GM3_fema_hilltop_qc.png`
- `GM3_hilltop_qc.png`
- `GM4_fema_hilltop_qc.png`
- `GM4_hilltop_qc.png`
- `GM5_fema_hilltop_qc.png`
- `GM5_hilltop_qc.png`
- `GM6_fema_hilltop_qc.png`
- `GM6_hilltop_qc.png`
- `GM7_fema_hilltop_qc.png`
- `GM7_hilltop_qc.png`
- `GM8_hilltop_qc.png`
- `GM9_hilltop_qc.png`
- `TTV11_hilltop_qc.png`
- `TTV15_hilltop_qc.png`
- `TTV1_hilltop_qc.png`
- `TTV2_hilltop_qc.png`
- `TTV3_hilltop_qc.png`
- `TTV4_hilltop_qc.png`
- `TTV5_hilltop_qc.png`
- `TTV6_hilltop_qc.png`
- `TTV7_hilltop_qc.png`
- `TTV8_hilltop_qc.png`
- `TTV9_hilltop_qc.png`

### Survey pairs (item 3)

- `AP1_AP1_3dep_hilltop_pair.png`
- `AP2_AP2_3dep_hilltop_pair.png`
- `AP3_AP3_3dep_hilltop_pair.png`
- `AP4_AP4_3dep_hilltop_pair.png`
- `AP5_AP5_3dep_hilltop_pair.png`
- `GM1_GM1_fema_hilltop_pair.png`
- `GM2_GM2_fema_hilltop_pair.png`
- `GM3_GM3_fema_hilltop_pair.png`
- `GM4_GM4_fema_hilltop_pair.png`
- `GM5_GM5_fema_hilltop_pair.png`
- `GM6_GM6_fema_hilltop_pair.png`
- `GM7_GM7_fema_hilltop_pair.png`

### Galleries (item 4)

- `AP_hilltop_gallery_p1.png`
- `AP_hilltop_gallery_p2.png`
- `GM_hilltop_gallery_p1.png`
- `GM_hilltop_gallery_p2.png`
- `TTV_hilltop_gallery_p1.png`
