#!/usr/bin/env python3
"""Addendum 23: assemble the report from the hand-written draft (reports/v2/_draft_a23.md) and the run's post files (tables, captions).
usage: a23_assemble_report.py <out.md> <stamp like '2026-11-24 18:30 PST'>"""
import json
import os
import sys

RUN = os.path.join(os.environ.get("GROUP_HOME", "/home/groups/hilley"), "lemcvt_results", "20261124T173105Z_a23_ttv_domain_roads_heads_channelization")
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(HERE)
out, stamp = sys.argv[1], sys.argv[2]
s = open(os.path.join(V2, "_draft_a23.md")).read().replace("[[DATE]]", "[" + stamp + "]")
tables = open(os.path.join(RUN, "post", "tables.md")).read()
caps = json.load(open(os.path.join(RUN, "post", "captions.json")))

def section(md: str, title_start: str) -> str:
    """One '### ...' section of tables.md, headings demoted one level."""
    parts = md.split("\n### ")
    for p in parts:
        if p.startswith(title_start):
            return "#### " + p.strip() + "\n"
    raise KeyError(title_start)

A = """### A. Rules and verdicts (directive section 7)

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

"""
B = "### B. The domain check (decision 2.1)\n\nTile TTV2 cut as in Addendum 22 (its 3,996 valid edge cells given as the outlets, no-flux faces on), through `PartitionedProfiler` and the AD model with the new options, against the Addendum 22 polygon fits (`tiles/TTV2.json`); tolerance relative 10⁻⁸.\n\n" + section(tables, "Appendix: the domain check").replace("#### Appendix: the domain check (TTV2 cut as before, its edge ring as the outlets, through the domain route)\n", "") + "\n"
C = """### C. Run ids, inputs and checksums

- Run directory `$GROUP_HOME/lemcvt_results/20261124T173105Z_a23_ttv_domain_roads_heads_channelization/`: `domain/` (domain.json, domain.npz: the box arrays, polygon, domain, outlets, roads, networks, picks, measured inputs, the domain check), `roads/` (road_check.json, .npz), `heads/` (heads1m.json, .npz, accepted.json), `fits/` (fixed.json, fixed_surfaces.npz, AD.json, AD_profile.parquet, AD_surfaces.npz, SJ_C.json, SJ_C_profile.parquet, SJ_C_surfaces.npz, AD_pilot.json, SJ_C_pilot.json, tile_params_on_domain.json), `measures/` (measures.json, spectra.npz, surfaces12.npz), `post/` (metrics.json, rules.json, tables.md, captions.json), `gate/` (a18_gate_after_domain_options.json); every job wrote a `job.json`.
- Full-site TTV 1 m grids (filled, flow direction, area; EPSG:6339): `$GROUP_HOME/lemcvt_results/20261118T160326Z_a15_stage_b/grid_TTV/TTV_1m_{filled,fd,area_cells}.tif`; the window read is rows 5416–10191, columns 8209–13067 of those grids.
- M&D georeference: `$GROUP_HOME/lemcvt_results/20261118T160326Z_a15_stage_b/md_georef/md1989_fig3.gpkg`, layers `study_area` and `channels`.
- Roads: `$GROUP_HOME/lemcvt_data/source/TIGER2024_roads/tl_2024_06041_roads.zip` (TIGER/Line 2024 ROADS, Marin County), read through `/vsizip/`.
- Steve's heads: `data/ttv/ttv_heads_manual_steve_20261119.csv`, sha256 ac87de3d4ca1f479e6eb2ceda2fedb033740b3d8f37186fd267329d5e4e702b6 (checked in code before use; 64 picks, all inside the domain and the polygon).
- Addendum 21 run (the full-site 12 m statistics `grid12_arrays.npz`, the pick seeds, the per-tile K₀, crest curvature and relief D): `$GROUP_HOME/lemcvt_results/20261123T184712Z_a21_ttv_channel_head_test/`; Addendum 22 run (the TTV2 polygon fits for the domain check, the TTV2 AD surface and extra heads for the road check, the TTV2 config bounds): `$GROUP_HOME/lemcvt_results/20261124T000138Z_a22_ttv_cleanup_head_metrics/`; Addendum 18 run (the regression gate's reference values): `$GROUP_HOME/lemcvt_results/20261119T143945Z_a18_expanded_fits/`.
- Git: every job ran at c694e02 with this addendum's package change and scripts in the working tree (`git=c694e02` in every job log); the scripts, tests, package change and figures are committed with this report.

"""
D = """### D. Slurm accounting

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

"""
F = "### F. Machine-generated tables (`post/tables.md`, verbatim; headings demoted)\n\n" + tables.replace("\n### ", "\n#### ").replace("### Model key", "#### Model key", 1) + "\n"
G = "### G. Figures (`reports/v2/figures/a23/`; captions as printed on each figure, from `post/captions.json`)\n\n"
titles = {"fig1_domain.png": "The domain", "fig2_road_check.png": "The road check", "fig3_head_finder_test.png": "The head-finder test on the real lidar",
          "fig4_misfit.png": "Misfit per model on the domain", "fig5_K_D.png": "K and D per model against the measured K₀, the relief D and Heimsath's D",
          "fig6_power_spectrum_ratio.png": "Power spectrum ratio, model ÷ lidar, against wavelength", "fig7_valley_share.png": "Valley share per model ÷ lidar",
          "fig8_maps.png": "Maps of the lidar and each model on 1 m hillshades with the roads"}
for i, name in enumerate(["fig1_domain.png", "fig2_road_check.png", "fig3_head_finder_test.png", "fig4_misfit.png", "fig5_K_D.png", "fig6_power_spectrum_ratio.png", "fig7_valley_share.png", "fig8_maps.png"], 1):
    size = os.path.getsize(os.path.join(V2, "figures", "a23", name)) / 1e6
    G += f"- **Figure {i}. {titles[name]}.** `figures/a23/{name}` ({size:.2f} MB)\n  - How to read: {caps[name]['how']}\n  - What it shows: {caps[name]['what']}\n"
G += "\nEvery figure was rendered and looked at before it went in; the one text overlap found (figure 6's legend over the AD model's points) was fixed and the figure looked at again. Hillshades are the 1 m surfaces lit from the north-west; the only 12 m image is the AD surface of Addendum 22 in figure 2, whose heads were found on that 12 m block mean (its caption says so).\n\n"
H = ("### H. Heads on the model surfaces through the 12 m method (the directive's appendix branch)\n\nNo 1 m finder was accepted, so this is the Addendum 22 comparison repeated on the domain: the 12 m block mean of each 1 m surface on "
     "the domain box lattice, D8 of the filled 12 m surface, M&D's area × slope² rule at 876 m² (the area rule at 6,120 m² beside it), heads inside the domain and the "
     "polygon, Hungarian one-to-one matching to the 64 mapped heads at 25 m. The mapped heads here are 64, not Addendum 22's 61, because the three picks on no tile are "
     "inside the domain. Channel heads are not a usable test with these methods (section 5); the table is supporting evidence only.\n\n"
     + section(tables, "Heads on every surface through the 12 m method").split("\n", 1)[1] + "\n")
apps = A + B + C + D + "[[E]]" + F + G + H
# the state check (E) is already in the draft, before the placeholder; put A–D before it and F–H after it
head, tail = s.split("### E. State check, decisions as applied, deviations", 1)
tail_body, _ = tail.split("[[APPENDICES_REST]]", 1)
E = "### E. State check, decisions as applied, deviations" + tail_body
s = head + A + B + C + D + E + F + G + H
assert "[[" not in s, [l for l in s.splitlines() if "[[" in l][:5]
open(out, "w").write(s)
print(out, len(s.splitlines()), "lines")
