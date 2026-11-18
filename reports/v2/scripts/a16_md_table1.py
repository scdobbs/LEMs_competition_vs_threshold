#!/usr/bin/env python3
"""Addendum 16, section 2.4 (decision 7): transcription of Montgomery & Dietrich (1989) Table 1.

Source: the WRR scan in papers/ (never committed), PDF page 4 = journal page 1910, right-hand column: "TABLE 1. Source Area
Data From Tennessee Valley, California" with four columns: local valley slope above the channel head (m/m, Brunton
compass), source area (m2, average of several measurements rounded to the nearest 100 m2), source-basin length (m, rounded
to the nearest 10 m) and width of colluvial fill at the channel head (m). The rows were read from the page rendered at
200 dpi (pdftoppm, poppler 0.47 on Sherlock) and are checked here against the scan's OCR text layer (pdftotext -layout):
every row's (area, length, width) triple must occur as a line of the OCR text and its slope must occur in the text, and the
row count must be 63 (68 channelled hollows minus the 5 trail-affected heads the paper omitted). The rows carry no
coordinates and are not linked to the digitised heads.
Output: md1989_table1.csv (row, local_valley_slope, source_area_m2, source_basin_length_m, colluvial_fill_width_m,
source) and md1989_table1_check.json.
usage: a16_md_table1.py --out DIR
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a15_md_georef import PDF, POPPLER  # noqa: E402
from a16_common import MD_TABLE1_PAGE, dump  # noqa: E402

# (slope m/m, source area m2, source-basin length m, colluvial fill width m), in the printed order
ROWS = [
    (0.45, 7400, 120, 60), (0.43, 12000, 180, 60), (0.49, 7900, 140, 50), (0.42, 9100, 180, 60), (0.62, 2700, 50, 36),
    (0.78, 3800, 100, 40), (0.44, 9300, 130, 5), (0.53, 3700, 80, 45), (0.55, 6700, 110, 30), (0.42, 10000, 190, 5),
    (0.42, 7900, 160, 5), (0.47, 9900, 140, 16), (0.58, 5400, 130, 8), (0.53, 2100, 100, 30), (0.51, 6100, 110, 50),
    (0.53, 4500, 120, 32), (0.60, 4900, 80, 20), (0.44, 8500, 120, 60), (0.47, 9800, 160, 28), (0.44, 5500, 100, 28),
    (0.62, 4300, 100, 20), (0.34, 8200, 160, 32), (0.57, 8000, 110, 16), (0.65, 2600, 50, 50), (0.24, 40100, 400, 28),
    (0.58, 8900, 140, 12), (0.49, 14500, 150, 48), (0.58, 8400, 90, 48), (0.38, 15100, 150, 43), (0.36, 9300, 120, 28),
    (0.36, 8300, 120, 32), (0.58, 6300, 160, 40), (0.44, 4200, 120, 24), (0.55, 5400, 140, 24), (0.47, 11400, 120, 48),
    (0.67, 5200, 120, 30), (0.42, 7200, 100, 44), (0.40, 11900, 150, 32), (0.81, 4900, 90, 28), (0.51, 3800, 80, 5),
    (0.34, 11400, 170, 48), (0.25, 14900, 170, 56), (0.27, 14000, 190, 40), (0.36, 7500, 170, 40), (0.29, 13300, 170, 52),
    (0.25, 24000, 190, 28), (0.45, 7400, 120, 52), (0.45, 12700, 260, 8), (0.47, 4700, 110, 12), (0.36, 10700, 180, 32),
    (0.32, 13100, 170, 28), (0.45, 4000, 90, 40), (0.30, 11000, 170, 32), (0.33, 6700, 120, 24), (0.46, 3800, 80, 24),
    (0.84, 1800, 60, 30), (0.76, 1200, 80, 20), (0.70, 6800, 140, 48), (0.91, 1200, 60, 8), (0.62, 6600, 160, 8),
    (0.67, 5600, 110, 8), (0.45, 10800, 170, 5), (0.78, 2000, 80, 5),
]
SOURCE = "Montgomery & Dietrich (1989) WRR 25(8) 1907-1918, Table 1, p. 1910 (PDF page 4); footnotes: slope by Brunton compass; area = average of several measurements rounded to nearest 100 m2; length rounded to nearest 10 m"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "md1989_table1.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "local_valley_slope_m_per_m", "source_area_m2", "source_basin_length_m", "colluvial_fill_width_m", "source"])
        for i, (s, ar, ln, wd) in enumerate(ROWS, 1):
            w.writerow([i, f"{s:.2f}", ar, ln, wd, SOURCE])
    # OCR check
    env = dict(os.environ, LD_LIBRARY_PATH=POPPLER + "/lib:" + os.environ.get("LD_LIBRARY_PATH", ""))
    pg = MD_TABLE1_PAGE["pdf_page"]
    txt = subprocess.run([POPPLER + "/bin/pdftotext", "-layout", "-f", str(pg), "-l", str(pg), PDF, "-"], check=True, capture_output=True, text=True, env=env).stdout
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in txt.splitlines()]
    triples_missing, slopes_missing = [], []
    for i, (s, ar, ln, wd) in enumerate(ROWS, 1):
        tri = f"{ar:,} {ln} {wd}"
        if not any(l.endswith(tri) or (" " + tri + " ") in (" " + l + " ") for l in lines):
            triples_missing.append(i)
        if not any(re.search(rf"(^|\s){s:.2f}(\s|$)", l) for l in lines):
            slopes_missing.append(i)
    areas = [r[1] for r in ROWS]
    slopes = [r[0] for r in ROWS]
    full = subprocess.run([POPPLER + "/bin/pdftotext", "-layout", PDF, "-"], check=True, capture_output=True, text=True, env=env).stdout
    rec = {"rows": len(ROWS), "rows_expected": MD_TABLE1_PAGE["rows_expected"], "row_count_pass": len(ROWS) == MD_TABLE1_PAGE["rows_expected"],
           "source_area_min_m2": min(areas), "source_area_max_m2": max(areas), "source_area_median_m2": sorted(areas)[len(areas) // 2],
           "slope_min": min(slopes), "slope_max": max(slopes), "length_min_m": min(r[2] for r in ROWS), "length_max_m": max(r[2] for r in ROWS),
           "width_min_m": min(r[3] for r in ROWS), "width_max_m": max(r[3] for r in ROWS),
           "ocr_triples_not_found_rows": triples_missing, "ocr_slopes_not_found_rows": slopes_missing,
           "paper_text_mentions": {k: bool(re.search(k, full)) for k in (r"1,?200", r"40,?100", r"68 hollows", r"Twenty-two", r"Five hollows")},
           "values_unreadable": [], "page": MD_TABLE1_PAGE, "csv": path}
    dump(rec, os.path.join(a.out, "md1989_table1_check.json"))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
