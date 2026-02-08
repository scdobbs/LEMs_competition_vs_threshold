#!/usr/bin/env python3
"""
Transform supp_fig6.ipynb:
  - Cell 0 (ba4094e0): rewrite imports
  - Cell 1 (d3c21950): delete (plot_maps_area_slope now imported from src/plotting)
  - Cell 2 (10fae1ea): remove inline function defs + redundant imports;
                        replace plt.rcParams.update({...}) with set_nature_style()
  - Cell 3 (2670f8aa): remove duplicate matplotlib.ticker import line
  - Cell 4 (f15515a3): leave as-is
"""

import json
import sys

NB_PATH = "/home/hilley/Documents/GitHub/Manuscript_submission/notebooks/figures/supp_fig6.ipynb"

# ── Load ──────────────────────────────────────────────────────────────
with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# Build id -> index map (use short prefix match)
def find_cell(prefix):
    for i, c in enumerate(cells):
        cid = c.get("id", "")
        if cid.startswith(prefix):
            return i
    raise KeyError(f"No cell with id prefix {prefix!r}")

idx0 = find_cell("ba4094e0")
idx1 = find_cell("d3c21950")
idx2 = find_cell("10fae1ea")
idx3 = find_cell("2670f8aa")
idx4 = find_cell("f15515a3")

# ── Cell 0: rewrite imports ──────────────────────────────────────────
NEW_CELL0 = """\
import sys
from pathlib import Path
sys.path.insert(0, str(Path('../../src')))

from models import (
    advection_diffusion_model,
    partitioned_threshold_model,
    partitioned_threshold_model_nonlinear,
)
from plotting import set_nature_style, plot_maps_area_slope
from utils import find_project_root, resample_from_1m
from analysis import (
    ks_from_loglog_fixed_theta,
    slope_area_arrays_from_elevation,
    ks_obs_from_observed,
    ks_mod_from_modeled,
)

import numpy as np
import copy
import TopoAnalysis.dem as d
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext
"""
cells[idx0]["source"] = NEW_CELL0.split("\n")
# Rejoin as a list of lines, each ending with \n (except possibly last)
lines = NEW_CELL0.rstrip("\n").split("\n")
cells[idx0]["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]

# ── Cell 1: mark for deletion ────────────────────────────────────────
# We'll delete after all edits to avoid shifting indices
delete_idx = idx1

# ── Cell 2: surgical removal of inline functions + redundant imports ─
src2 = "".join(cells[idx2]["source"])

# --- Remove inline function definitions ---
# Each function to remove: identify by "def funcname(" up to (but not including)
# the next top-level block.
# Strategy: split into lines, walk through, skip function bodies.

lines2 = src2.split("\n")

FUNCTIONS_TO_REMOVE = {
    "_infer_dx_from_georef",
    "_nanaware_zoom",
    "resample_from_1m",
    "ks_from_loglog_fixed_theta",
    "slope_area_arrays_from_elevation",
    "ks_obs_from_observed",
    "ks_mod_from_modeled",
}

# Also remove these import lines
IMPORTS_TO_REMOVE = [
    "from scipy.ndimage import zoom",
    "from scipy.stats import linregress",
    "import copy",
]

# Also remove the plt.rcParams.update({...}) block and replace with set_nature_style()
# We'll handle that specially.

new_lines2 = []
i = 0
in_function_to_remove = False
in_rcparams_block = False
rcparams_brace_depth = 0
rcparams_replaced = False

while i < len(lines2):
    line = lines2[i]
    stripped = line.strip()

    # --- Skip import lines to remove ---
    skip_import = False
    for imp in IMPORTS_TO_REMOVE:
        if stripped == imp:
            skip_import = True
            break
    if skip_import:
        i += 1
        continue

    # --- Handle plt.rcParams.update block ---
    if "plt.rcParams.update({" in stripped and not rcparams_replaced:
        in_rcparams_block = True
        # Count braces on this line
        rcparams_brace_depth = stripped.count("{") - stripped.count("}")
        if rcparams_brace_depth <= 0:
            # Single-line rcParams block
            in_rcparams_block = False
            new_lines2.append("set_nature_style()")
            rcparams_replaced = True
        i += 1
        continue

    if in_rcparams_block:
        rcparams_brace_depth += stripped.count("{") - stripped.count("}")
        if rcparams_brace_depth <= 0:
            in_rcparams_block = False
            new_lines2.append("set_nature_style()")
            rcparams_replaced = True
        i += 1
        continue

    # --- Handle function removal ---
    if stripped.startswith("def "):
        # Extract function name
        fname = stripped[4:].split("(")[0].strip()
        if fname in FUNCTIONS_TO_REMOVE:
            in_function_to_remove = True
            i += 1
            continue

    if in_function_to_remove:
        # We're inside a function to remove. Skip lines until we hit
        # a non-indented, non-blank line (next top-level statement).
        if stripped == "":
            # Could be blank line within function or between functions
            # Peek ahead: if next non-blank line is indented, still in function
            i += 1
            continue
        if line[0] not in (" ", "\t") and not stripped.startswith("def "):
            # Back to top level, not a def -> stop skipping
            in_function_to_remove = False
            # Fall through to add this line
        elif stripped.startswith("def "):
            fname = stripped[4:].split("(")[0].strip()
            if fname in FUNCTIONS_TO_REMOVE:
                # Another function to remove
                i += 1
                continue
            else:
                in_function_to_remove = False
                # Fall through to add this line
        else:
            # Still indented -> part of function body, skip
            i += 1
            continue

    new_lines2.append(line)
    i += 1

# Now clean up excessive blank lines (collapse 3+ consecutive blanks to 2)
cleaned = []
blank_count = 0
for line in new_lines2:
    if line.strip() == "":
        blank_count += 1
        if blank_count <= 2:
            cleaned.append(line)
    else:
        blank_count = 0
        cleaned.append(line)

# Also remove the section header comments for removed functions:
# "# Helpers: file roots + nan-aware resampling from 1 m base"
# "# KS estimation utilities (fixed theta)"
# These are standalone comment blocks. Let's check and remove them.
final_lines2 = []
i = 0
while i < len(cleaned):
    stripped = cleaned[i].strip()
    # Remove the helper section headers that only introduced removed functions
    if stripped == "# Helpers: file roots + nan-aware resampling from 1 m base":
        # Skip this line and any surrounding decoration lines (===)
        # Check if previous line is a decoration
        # Check if next line is a decoration
        i += 1
        continue
    if stripped == "# KS estimation utilities (fixed theta)":
        i += 1
        continue
    # Skip decoration lines that are orphaned (=== lines around removed sections)
    # But only if they're adjacent to removed content. We'll handle this by checking
    # context — if a === line is between two blank lines, it might be orphaned.
    final_lines2.append(cleaned[i])
    i += 1

# Remove orphaned section separator lines (lines of just "# " + "="*N)
# that now sit between blank lines with no following content section
result_lines2 = []
for i, line in enumerate(final_lines2):
    stripped = line.strip()
    # Check for orphaned "# ===" separator lines
    if stripped.startswith("# ") and stripped[2:].replace("=", "").strip() == "":
        # It's a separator line like "# ============"
        # Check if there's a following comment line that's a section header
        next_nonblank = None
        for j in range(i + 1, len(final_lines2)):
            if final_lines2[j].strip():
                next_nonblank = final_lines2[j].strip()
                break
        # If the next non-blank line is also a separator or starts with #, keep it
        # If there's a real section header after, keep the separator
        if next_nonblank and next_nonblank.startswith("# "):
            result_lines2.append(line)
        else:
            # Orphaned separator, skip
            pass
    else:
        result_lines2.append(line)

# Rejoin and set source
src2_new = "\n".join(result_lines2)
# Convert to notebook source format (list of lines with \n)
lines_final = src2_new.split("\n")
if lines_final and lines_final[-1] == "":
    lines_final = lines_final[:-1]
cells[idx2]["source"] = [l + "\n" for l in lines_final[:-1]] + [lines_final[-1]]

# ── Cell 3: remove the duplicate matplotlib.ticker import ────────────
src3 = "".join(cells[idx3]["source"])
src3_new = src3.replace("from matplotlib.ticker import LogLocator, LogFormatterMathtext\n\n", "")
src3_new = src3_new.replace("from matplotlib.ticker import LogLocator, LogFormatterMathtext\n", "")
lines3 = src3_new.split("\n")
if lines3 and lines3[-1] == "":
    lines3 = lines3[:-1]
cells[idx3]["source"] = [l + "\n" for l in lines3[:-1]] + [lines3[-1]]

# ── Delete Cell 1 ────────────────────────────────────────────────────
del cells[delete_idx]

# ── Post-cleanup: collapse stacked separator lines in Cell 2 ─────────
# After removing helper/KS sections, three consecutive "# ===" lines
# may remain before the "Core per-(tile,dx)" section.  Collapse to one.
cell2 = [c for c in cells if c.get("id", "").startswith("10fae1ea")][0]
src2_post = "".join(cell2["source"])
triple_sep = (
    "# ============================================================\n"
    "# ============================================================\n"
    "# ============================================================\n"
)
single_sep = "# ============================================================\n"
while triple_sep in src2_post:
    src2_post = src2_post.replace(triple_sep, single_sep)
# Also collapse double separators that may result
double_sep = single_sep + single_sep
while double_sep in src2_post:
    src2_post = src2_post.replace(double_sep, single_sep)
lines_c2 = src2_post.split("\n")
if lines_c2 and lines_c2[-1] == "":
    lines_c2 = lines_c2[:-1]
cell2["source"] = [l + "\n" for l in lines_c2[:-1]] + [lines_c2[-1]]

# ── Save ──────────────────────────────────────────────────────────────
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

print("Done. supp_fig6.ipynb has been updated.")
print(f"  Remaining cells: {len(cells)}")
for i, c in enumerate(cells):
    cid = c.get("id", "???")
    src_preview = "".join(c["source"])[:80].replace("\n", " | ")
    print(f"  [{i}] {cid[:8]}  {src_preview!r}")
