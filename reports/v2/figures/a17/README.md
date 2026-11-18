# Addendum 17 Part A: figures for visual review

Made by `reports/v2/scripts/a17_figures.py` (job 52870152) from the run directory
`$GROUP_HOME/lemcvt_results/20261118T225314Z_a17_register_figures/` (copies of record in its `figures/`, sizes and panel statistics in
`figures.json`). PNG at 200 dpi (each under 5 MB, committed) and PDF (same name, not committed: `.gitignore`). Report:
`reports/v2/20261118T1600PST_register_figures.md`, section 4.

Conventions, the same everywhere: hillshade azimuth 315°, altitude 45°, no vertical exaggeration, Horn gradient on the tile's or grid's own
1 m filled surface; overview hillshades block-averaged to 5 m (GM, TTV) or 10 m (AP) on a world-aligned lattice; tile panels 1 × 1 km at the
same scale, channel network A ≥ 10³ m² from the tile's registered area grid (`area_model` for GM1-GM5, `area` otherwise), line width
∝ log₁₀ A; panel title = tile · kind · survey · z min-max · largest A inside · nearest-neighbour-filled fraction (n/a where the grid was
delivered as a raster: NCALM 2003, PAMAP). Tile kinds: original (red), drawn (blue), M&D by design (green), control (orange, dashed).

| figure | file | what it shows |
| --- | --- | --- |
| GM overview | `GM_overview.png` / `.pdf` | FEMA 2018 hillshade (the 167 km² extension grid + the 44 km² control grid, 5 m), lidar footprints, the NCALM 2003 footprint (purple dashed), 33 + 49 eligible windows (grey), GM1-GM7 (red/blue), GM8-GM30 (blue), GM1_fema-GM7_fema (orange dashed, on the GM1-GM7 footprints); EPSG:6339 |
| AP overview | `AP_overview.png` / `.pdf` | 3DEP 2019-20 hillshade (10 m), lidar footprint, 54 eligible windows, AP1_3dep-AP5_3dep (red), AP6-AP29 (blue); EPSG:6346 |
| TTV overview | `TTV_overview.png` / `.pdf` | Marin 2019 hillshade (5 m), lidar footprint, 34 eligible windows, the clipped M&D (1989) study area and its 77 digitised heads (green), TTV1-TTV3 (green), TTV4-TTV15 (blue); EPSG:6339 |
| GM gallery, page 1 | `GM_gallery_p1.png` / `.pdf` | GM1-GM7 (NCALM 2003) and GM8-GM15 (FEMA 2018), 15 panels |
| GM gallery, page 2 | `GM_gallery_p2.png` / `.pdf` | GM16-GM30 (FEMA 2018), 15 panels |
| GM gallery, control | `GM_gallery_control.png` / `.pdf` | GM1_fema-GM7_fema (FEMA 2018 at the NCALM footprints), 7 panels |
| AP gallery, page 1 | `AP_gallery_p1.png` / `.pdf` | AP1_3dep-AP5_3dep and AP6-AP15, 15 panels |
| AP gallery, page 2 | `AP_gallery_p2.png` / `.pdf` | AP16-AP29, 14 panels |
| TTV gallery | `TTV_gallery.png` / `.pdf` | TTV1-TTV15, 15 panels |
| GM survey-control pairs | `GM_control_pairs.png` / `.pdf` | GM1-GM7: NCALM 2003 beside FEMA 2018, same rendering, 14 panels in 4 columns (N, F, N, F) |
| AP survey pairs | `AP_pamap_3dep_pairs.png` / `.pdf` | AP1-AP5: the PAMAP 2006-08 v1/v2 tile (EPSG:26917, 0.975 m cells, metre version) beside the 3DEP 2019-20 tile, 10 panels |

Every panel of the galleries is one registered tile; the exclusion review (addendum 17 Part A, section 5) is made on these.
