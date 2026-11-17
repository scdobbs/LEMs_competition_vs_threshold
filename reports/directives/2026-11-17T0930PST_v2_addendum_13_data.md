# [2026-11-17 09:30 PST] Addendum 13: data inventory, USGS 3DEP point clouds for AP, and Tennessee Valley staging

This addendum follows Addendum 12 and its report `reports/v2/20261117T0900PST_sensitivity.md`. Addendum 4 §1 (timestamped titles, one report per checkpoint), Addendum 2 §4–§5 (Slurm, storage, push) and Addendum 7 §0.4 (submit guard) still apply.

**This is a data checkpoint. No model runs, no gridding into model tiles, and no existing grid is replaced or edited.**

## 0. Orientation (you are a new agent instance)

1. **Read:**
   - `reports/v2/20261117T0900PST_sensitivity.md`;
   - `reports/v2/20261116T1330PST_release.md`;
   - `docs/measurement_methods.md` §0;
   - this file.
2. **Confirm** the state of `main`, `v2.1-sensitivity` and the `v2.0.0` tag, that the tests pass on `serc`, and that no job is running.
3. **Where this addendum and the repo disagree,** the repo wins. Report it and stop.

**Why this addendum.**
- Every weak AP result so far traces back to the PAMAP DEM: TIN facets, no stable crest-curvature window (so AP D₀ is a bracket), and the AP1 seam.
- PAMAP was collected to QL3 at a nominal point spacing of 1.4 m. Regridding PAMAP points would mostly reproduce the facets.
- Greene County, PA is covered by the 2019–2020 USGS 3DEP QL2 Western PA lidar (work unit PA_WesternPA_1), at a nominal pulse spacing of 0.71 m. This addendum finds and downloads only the point-cloud files under AP1–AP5, and checks them.
- It also inventories new data Steve placed on Sherlock, including the Tennessee Valley (TTV) lidar for the planned third site.

## 1. Decisions

Steve will strike any he does not accept before this reaches you.

1. **Addendum 12 decisions** (its §6), as recommended there:
   1. The greedy basin split stands as the held-out test; report it with its shares. No rerun.
   2. AP wording: "at equal parameter counts the partitioned model beats AD on AP only when D is free (and then D ≈ 4–8 × D₀,A)".
   3. The AP AD K/K₀ statement is reported as conditional on the site m (1.3–15 across per-tile m).
   4. Cost rule from now on: the **expected** total core-hours must be under the stated cap, **and** the sum of the submissions' time limits must be under twice that cap; every submission ≤ 300 core-hours of limits.
   5. Count-2 results name P2 and P3 separately.
   6. **Merge** `v2.1-sensitivity` into `main` with a merge commit once CI is green. Do not tag.
2. **Branch:** cut `v2.2-data` from `main` after the merge. Code and reports only go in git. **No point clouds, rasters or PDFs are committed.** Check `.gitignore` covers every new data path before the first commit.
3. **New data are additions.** The 3DEP AP data get their own data version, `AP-3DEP`, beside the PAMAP-based `AP` data. Nothing under `data/AP/` or the existing results is changed.
4. **Download scope:** only the files under AP1–AP5 plus a **500 m buffer**, chosen by the rule in §3. Not the county, and not the bounding box of all five tiles (about 400 files).
5. **Budget:**
   - expected ≤ 20 core-hours, time limits ≤ 40 core-hours in all; every job on `serc`; arrays throttled;
   - raw download ≤ 40 GB. If the §3 list implies more, stop and report before downloading.
6. **Scratch is not storage of record.** Sherlock purges `$SCRATCH` files that have not been modified for 90 days. Copy everything Steve placed in `$SCRATCH/lem_topography/` to group storage (§2.3) before using it.

## 2. Inventory of `$SCRATCH/lem_topography/` and the new papers (read-only)

Steve placed these in `/scratch/users/scdobbs/lem_topography/` (confirm the path):
- **TTV:** the NCALM 2006 "Lone Tree Creek and Tennessee Valley, CA" files (bare-earth grids, unfiltered grids, and/or LAS);
- **Original PA grids:** the PAMAP grids the AP tiles came from;
- **Gabilan Mesa grids;**
- **Perron's coarse grids:** a 10 m GM grid and a 30 m AP grid, which Perron et al. (2012) used in part to set some model parameters.

He also placed Perron et al. (2012) and its supplementary information in `papers/` (never committed).

### 2.1 For every file, record

- path, size, format, sha256;
- CRS: horizontal datum and projection, **horizontal units**, **vertical datum and vertical units**, and how each was established (header, `.prj`, metadata, or inference). Mark every inferred unit as inferred;
- resolution, extent, nodata value, value range;
- for point clouds: point count, density (points/m²), the classes present and their counts, return numbers, the GPS-time range (collection dates).

### 2.2 Match each item to what the repo already holds

1. **Original PA grids against `data/AP/`.** Are the AP model tiles cut from these grids? Test it:
   - Locate each AP tile inside them.
   - Compare cell by cell after the feet-to-metres factor: report max |Δz|, the count of differing cells and where they are.
   - Say whether these grids include the area outside the tiles that a drainage-area computation would need (Addendum 2 §6.5).
2. **GM grids against `data/GM/`.** The same test.
3. **TTV.**
   - Which LAS file or grid covers the Montgomery & Dietrich (1989) study area: 2.1 km² on the east flank of Coyote Ridge, elevations 61–314 m, on the USGS Point Bonita 7.5′ quadrangle (Project PDF; Steve can supply it).
   - The file names `LTTV06_533293_4191792` and `LTTV06_538471_4188025` look like UTM 10N coordinates. Confirm or refute from the headers; do not assume.
   - **Vertical datum:** state whether the heights are ellipsoidal or orthometric (NAVD88). Older NCALM deliveries can be ellipsoidal. That makes a difference of about 30 m in absolute elevation, but almost none in gradients.
   - Ground classification: whether ground points are classified; ground density over the whole tile and in hollows.
4. **Perron's 10 m GM and 30 m AP grids.**
   - Identify their source (e.g. NED or SRTM) from metadata where possible.
   - From Perron et al. (2012) and its SI, list every parameter or quantity the paper says came from these grids, with page, section or figure references, quoted in under 15 words or paraphrased. Do not interpret beyond that.

### 2.3 Storage of record

- Copy everything to `$GROUP_HOME/lemcvt_data/source/<item>/`, unchanged, with a `MANIFEST.sha256`, before any use.
- Verify the checksums after the copy.
- Report the total size and the group-storage quota before and after.

## 3. Select the USGS 3DEP point-cloud files for AP1–AP5

### 3.1 Source

- **Dataset:** 2019–2020 USGS 3DEP QL2 Lidar, Western PA, work unit PA_WesternPA_1 (covers Greene County).
- **Distribution:** PASDA, `https://pasda.psu.edu/download/usgs/LiDAR2020/LAS/PA_South/`, one zipped LAS per file. Ancillary files (tile index, project report) are under `.../LiDAR2020/Ancillary_files/` (`PA_South/` and `Western_PA_QL2_South_LiDAR_ProjectReport.pdf`).
- **Expected CRS (from the metadata, to be verified in §5):** NAD83(2011) State Plane Pennsylvania South (FIPS 3702), **US survey feet**; vertical NAVD88 (Geoid12b), **US survey feet**. Both axes in feet, unlike PAMAP's mixed units.

### 3.2 Naming rule (to be verified, not trusted)

- The PAMAP block scheme: blocks are 10,000 × 10,000 ft in State Plane South.
- A block's name is the first four digits of the State Plane northing (6 digits) and easting (7 digits) of its **north-west corner**, then `PAS`. So `26001220PAS` has its NW corner at N 260,000 ft, E 1,220,000 ft.
- The files split each block into 5,000 ft quadrants: `<block>PAS_<NE|NW|SE|SW>_LAS.zip`.
- Some blocks lack quadrants (e.g. `15001350` has only NE, NW and SE).

### 3.3 Steps

1. **Footprints.** Read AP1–AP5's bounds and CRS from the model grids' georeferencing (the actual files under `data/AP/`). Report the CRS exactly, including the datum realization (NAD83, NAD83(2011) or WGS84). At 1 m cells a datum mismatch of about 1 m matters.
   - Steve's overall box for the five tiles is UTM 17N, about E 550,980–584,961 and N 4,405,724–4,429,556 (Greene County, 39.80–40.02 N, 80.40–80.00 W). Every tile must fall inside it; if one does not, stop.
2. **Candidate list.** Add `reports/v2/scripts/pasda_tiles_for_ap.py` (Appendix A) and run:
   ```
   python reports/v2/scripts/pasda_tiles_for_ap.py --buffer 500 --from-rasters <the five AP model grids>
   ```
   If a grid has no EPSG code, pass `--tile` with its bounds and the correct `--src-epsg`, and say which code you used and why.
3. **Check against the listing.** Download the directory listing once, save it, and parse it.
   - Report every candidate that is absent from the listing.
   - For each absent quadrant, record whether the tile area it would cover lies inside the PA border.
4. **Check against the tile index**, if `Ancillary_files/PA_South/` has one. Every candidate's polygon must exist and intersect its tile's buffered footprint. Report any listed file the index puts elsewhere.
5. **Size gate.** Sum the listed sizes. If the total exceeds 40 GB, stop and report before downloading.

## 4. Download

1. Use a Sherlock data-transfer node or a `serc` batch job, not the login node; check Sherlock's documentation for the current recommended route.
2. Use `wget -c` (resumable) from the saved file list, throttled to a few parallel transfers.
3. Raw zips go to `$SCRATCH/lem_topography/pa3dep_raw/`. Unzip to `$GROUP_HOME/lemcvt_data/source/AP-3DEP/las/`, with a `MANIFEST.sha256` built at download time, and verify it.
4. Also download the ancillary tile index and the project report into the same source directory. The report is not committed.
5. Report: files requested, received, failed (with HTTP status), bytes, wall time.

## 5. Header, unit and coverage checks (rules written now)

For every downloaded file, run `pdal info --metadata` (and `--stats` on the class-2 points) and report:
- the CRS as written in the header (WKT), horizontal and vertical units, point count, classes and their counts, the GPS-time range, and the header bounds.

**Rule 5.1, naming.** Each file's header bounds must lie within its computed 5,000 ft quadrant, to within 50 ft on each side. If any file fails, stop: the naming rule in §3.2 is wrong.

**Rule 5.2, units.** Header units must be US survey feet horizontally and vertically, matching §3.1. If the header disagrees with the metadata, or any unit is missing, stop and report. **Do not convert on an assumption.** Conversion everywhere uses 1 ftUS = 1200/3937 m. The v1 error was a feet/metres mismatch, and the v1 international-foot factor 0.3048 differs from 1200/3937 by 2 ppm, which is negligible here but must be stated.

**Rule 5.3, agreement with the existing AP grids** (the test that would have caught v1).
- For each AP tile, take the class-2 points inside it, convert to metres in the tile's CRS, and compare them with the existing PAMAP-based metre grid by bilinear lookup.
- **Pass** when, per tile, both hold:
  - the median Δz is within ±1.0 m;
  - the slope of a robust (Theil–Sen) fit of z_3DEP against z_PAMAP is within 0.98–1.02.
- A units error shows up as a slope near 3.28 or 0.305. A datum or geoid error shows up as an offset with a slope near 1.
- Report the median, the 5th–95th percentile of Δz and the slope for each tile. Map where |Δz| > 2 m: expect roads, quarries, mining and stream banks over 12+ years. Do not edit or filter anything to make a tile pass. If a tile fails, stop.

**Rule 5.4, coverage.** Inside each AP tile's 500 m-buffered footprint, report:
- the fraction of 1 m cells with at least one class-2 point, and with zero;
- the class-2 density (points/m²): overall; on the option (iii) hilltop pixels from Phase 6; and in hollows (the hollow mask of the hollow-excluded refit; cite its run, or state what you used instead);
- the same three densities from the PAMAP LAS for the same footprints, only if the PAMAP LAS are already on Sherlock. **Do not download PAMAP for this.**

No pass/fail on density. It is the evidence for whether a 1 m grid from 3DEP is honest, and especially whether hilltop curvature can be measured on crests.

**Rule 5.5, collection dates.** Report the GPS-time range per tile and whether it was leaf-off. The metadata say fall 2019 to spring 2020.

## 6. Cost

Print the estimate first: header and stats jobs, the §5.3 comparison and the §5.4 density, about 10 short jobs, each well under 1 core-hour. Submit without asking only within the §1.5 caps.

## 7. Report and stop

One report, `reports/v2/<YYYYMMDDTHHMM>PST_data.md`, titled with its timestamp. Fixed outline:

1. State check, including the Addendum 12 merge.
2. Inventory of `lem_topography` (§2.1), as one table per item.
3. Matches against the repo: original PA grids, GM grids, TTV coverage of the M&D area, Perron's coarse grids and what they were used for (§2.2).
4. 3DEP file selection: footprints and CRS, candidate list, listing and index checks, size (§3).
5. Download log (§4).
6. Header, unit, agreement and coverage checks, with each rule applied (§5).
7. What the data can and cannot support. In particular: does 3DEP class-2 density on AP crests support a 1 m grid and a crest-curvature measurement, and is it clearly better than PAMAP?
8. Decisions needed: the gridding method for AP-3DEP and TTV (not chosen here), and anything that failed.
9. Appendices: all generated tables, the file list with checksums.

**Stop for review.** Push `v2.2-data` after the Addendum 2 §5 check. Do not grid into model tiles, do not run any model, do not merge or tag.

## Appendix A. `reports/v2/scripts/pasda_tiles_for_ap.py`

Written by the planner. Tested only on the block-name rule (known names reproduce). §5.1 is the real test of the rule.

```python
#!/usr/bin/env python3
"""List the PASDA LiDAR2020 PA_South LAS files that cover given tile footprints.

Usage:
  python pasda_tiles_for_ap.py --buffer 200 \
      --tile AP1 xmin ymin xmax ymax [--tile AP2 ...] [--src-epsg 6346]
  python pasda_tiles_for_ap.py --buffer 200 --from-rasters data/AP/tile*/*_filled_m.tif

Footprints are in the source CRS (default EPSG:6346, NAD83(2011) UTM 17N, metres).
With --from-rasters, bounds and CRS are read from each raster (needs rasterio).

Naming rule (PAMAP block scheme, reused by the PASDA LiDAR2020 delivery):
  block = 10,000 x 10,000 ft in NAD83 PA State Plane South (US survey ft);
  name  = first 4 digits of the NW-corner northing (6 digits) + first 4 digits of the
          NW-corner easting (7 digits) + 'PAS'; e.g. 18001230PAS = NW corner N 180,000, E 1,230,000.
  Each block is split into 5,000 ft quadrants: _NW, _NE, _SW, _SE.
  File: <block>PAS_<Q>_LAS.zip
Check the file list against the directory listing: border blocks may lack quadrants.
"""
import argparse, math, sys
from pyproj import Transformer

SP_SOUTH_FTUS = 6565  # NAD83(2011) / Pennsylvania South (ftUS)
BLOCK, HALF = 10_000.0, 5_000.0


def sp_bounds(xmin, ymin, xmax, ymax, src_epsg, buffer_m):
    t = Transformer.from_crs(src_epsg, SP_SOUTH_FTUS, always_xy=True)
    xmin, ymin, xmax, ymax = xmin - buffer_m, ymin - buffer_m, xmax + buffer_m, ymax + buffer_m
    # densify the edges: the projection change bends straight lines slightly
    pts = []
    for i in range(21):
        f = i / 20
        pts += [(xmin + f * (xmax - xmin), ymin), (xmin + f * (xmax - xmin), ymax),
                (xmin, ymin + f * (ymax - ymin)), (xmax, ymin + f * (ymax - ymin))]
    xs, ys = zip(*(t.transform(x, y) for x, y in pts))
    return min(xs), min(ys), max(xs), max(ys)


def quadrant_files(E0, N0, E1, N1):
    """All quadrant files whose 5,000 ft square intersects [E0,E1] x [N0,N1] (ftUS)."""
    out = set()
    for qe in range(math.floor(E0 / HALF), math.floor(E1 / HALF) + 1):
        for qn in range(math.floor(N0 / HALF), math.floor(N1 / HALF) + 1):
            e_sw, n_sw = qe * HALF, qn * HALF            # SW corner of the quadrant
            be = math.floor(e_sw / BLOCK) * BLOCK       # block W edge
            bn = math.floor(n_sw / BLOCK) * BLOCK + BLOCK  # block N edge (NW-corner northing)
            ew = "E" if e_sw - be >= HALF else "W"
            ns = "N" if n_sw >= bn - HALF else "S"
            name = f"{int(bn):06d}"[:4] + f"{int(be):07d}"[:4]
            out.add(f"{name}PAS_{ns}{ew}_LAS.zip")
    return sorted(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tile", nargs=5, action="append", metavar=("NAME", "XMIN", "YMIN", "XMAX", "YMAX"))
    p.add_argument("--from-rasters", nargs="+")
    p.add_argument("--src-epsg", type=int, default=6346)
    p.add_argument("--buffer", type=float, default=200.0, help="metres, in the source CRS")
    a = p.parse_args()
    tiles = []
    for name, *b in a.tile or []:
        tiles.append((name, *map(float, b), a.src_epsg))
    for path in a.from_rasters or []:
        import rasterio
        with rasterio.open(path) as r:
            epsg = r.crs.to_epsg()
            if epsg is None:
                sys.exit(f"{path}: CRS has no EPSG code; pass --tile instead")
            tiles.append((path, *r.bounds, epsg))
    if not tiles:
        p.error("give --tile or --from-rasters")
    allfiles = set()
    for name, x0, y0, x1, y1, epsg in tiles:
        E0, N0, E1, N1 = sp_bounds(x0, y0, x1, y1, epsg, a.buffer)
        files = quadrant_files(E0, N0, E1, N1)
        allfiles.update(files)
        print(f"# {name}: SP South ftUS E {E0:.0f}-{E1:.0f}, N {N0:.0f}-{N1:.0f}: {len(files)} files")
        for f in files:
            print(f"#   {f}")
    print(f"# total unique files: {len(allfiles)}")
    for f in sorted(allfiles):
        print(f"https://pasda.psu.edu/download/usgs/LiDAR2020/LAS/PA_South/{f}")


if __name__ == "__main__":
    main()
```
