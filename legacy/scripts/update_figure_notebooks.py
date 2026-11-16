#!/usr/bin/env python3
"""
Update the 3 figure notebooks to use refactored src/ modules.

Changes:
1. fig_slope_area.ipynb: Replace inline function definitions with imports from
   src/analysis.py and src/utils.py. Replace matplotlib style block with
   set_nature_style(). Add markdown with LaTeX.
2. supp_fig5.ipynb: Replace inline planform/solver code with imports from
   src/synthetic.py. Add markdown with LaTeX for Peclet analysis.
3. supp_fig6.ipynb: Delete the 1072-line Cell 0 that duplicates src/models.py.
   Replace with imports. Add sys.path setup. Add markdown.

Usage:
    python scripts/update_figure_notebooks.py          # dry-run
    python scripts/update_figure_notebooks.py --write  # apply
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FIG_DIR = Path(__file__).resolve().parent.parent / "notebooks" / "figures"


def _cell_source(cell):
    src = cell.get("source", [])
    return "".join(src) if isinstance(src, list) else src


def _set_cell_source(cell, text):
    lines = []
    for line in text.split("\n"):
        lines.append(line + "\n")
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1][:-1]
    cell["source"] = lines


def clear_outputs(nb):
    cleared = 0
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            if cell.get("outputs") or cell.get("execution_count"):
                cell["outputs"] = []
                cell["execution_count"] = None
                cleared += 1
    return cleared


# ---------------------------------------------------------------------------
# fig_slope_area.ipynb
# ---------------------------------------------------------------------------

def update_fig_slope_area(nb):
    """Replace inline function defs with imports; add style helper and LaTeX."""
    changes = []

    # Cell 0: the large function-definition cell containing slope_area_regression_binned
    cell0 = nb["cells"][0]
    src0 = _cell_source(cell0)

    if "def slope_area_regression_binned" in src0:
        # Replace with imports from src modules
        new_src = (
            "import sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, str(Path('../../src')))\n"
            "\n"
            "from analysis import slope_area_regression_binned, compute_iqr_errors, weighted_r2\n"
            "from utils import find_project_root\n"
            "from plotting import set_nature_style\n"
        )
        _set_cell_source(cell0, new_src)
        changes.append("Cell 0: replaced inline function defs with src/ imports")

    # Cell 1: the main code cell with matplotlib style, helpers, plotting
    cell1 = nb["cells"][1]
    src1 = _cell_source(cell1)

    if "def find_project_root" in src1:
        # Remove the inline find_project_root definition
        # Remove the inline compute_iqr_errors definition
        # Remove the inline weighted_r2 definition
        # Replace plt.rcParams block with set_nature_style()
        # Keep: LOC_CONFIGS, color constants, unpack_dbg, prepare_location, plotting code

        # Strategy: replace the style block and remove redundant function defs
        new_lines = []
        skip_until_next_section = False
        in_function_def = False
        function_name = ""
        indent_level = 0

        for line in src1.split("\n"):
            stripped = line.strip()

            # Skip the plt.rcParams.update block (already handled by set_nature_style())
            if "plt.rcParams.update(" in line:
                skip_until_next_section = True
                new_lines.append("set_nature_style()")
                new_lines.append("")
                continue
            if skip_until_next_section:
                if stripped == "})" or stripped == "})":
                    skip_until_next_section = False
                continue

            # Skip the inline find_project_root definition
            if line.startswith("def find_project_root"):
                in_function_def = True
                function_name = "find_project_root"
                continue
            # Skip the inline compute_iqr_errors definition
            if line.startswith("def compute_iqr_errors"):
                in_function_def = True
                function_name = "compute_iqr_errors"
                continue
            # Skip the inline weighted_r2 definition
            if line.startswith("def weighted_r2"):
                in_function_def = True
                function_name = "weighted_r2"
                continue

            if in_function_def:
                # A function def ends when we encounter a non-indented, non-empty line
                if stripped == "" or line.startswith("    ") or line.startswith("\t"):
                    continue
                else:
                    in_function_def = False
                    # Fall through to add this line

            new_lines.append(line)

        _set_cell_source(cell1, "\n".join(new_lines))
        changes.append("Cell 1: removed inline function defs, replaced style block with set_nature_style()")

    return changes


# ---------------------------------------------------------------------------
# supp_fig5.ipynb
# ---------------------------------------------------------------------------

def update_supp_fig5(nb):
    """Add sys.path and imports for src/synthetic.py to first code cell."""
    changes = []

    # Find the first code cell
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = _cell_source(cell)

        # Check if already patched
        if "from synthetic import" in src:
            break

        # Check if this is the imports/setup cell
        if "import numpy as np" in src:
            # Add sys.path and synthetic imports at the top
            import_block = (
                "import sys\n"
                "from pathlib import Path\n"
                "sys.path.insert(0, str(Path('../../src')))\n"
                "\n"
                "from synthetic import (\n"
                "    compute_area_planform,\n"
                "    downstream_link,\n"
                "    solve_advection_diffusion_planform,\n"
                "    inlet_area_from_Pe,\n"
                "    D_from_Pe,\n"
                "    Pe_from_D_Ain,\n"
                "    A_top_for_target_inlet_area,\n"
                "    channel_slope_centerline,\n"
                ")\n"
                "from plotting import set_nature_style\n"
                "\n"
            )
            _set_cell_source(cell, import_block + src)
            changes.append("First code cell: added sys.path + synthetic/plotting imports")
            break

    return changes


# ---------------------------------------------------------------------------
# supp_fig6.ipynb
# ---------------------------------------------------------------------------

def update_supp_fig6(nb):
    """Delete the 1072-line Cell 0 that duplicates src/models.py, replace with imports."""
    changes = []

    cell0 = nb["cells"][0]
    src0 = _cell_source(cell0)

    if "def advection_diffusion_model" in src0 and cell0["cell_type"] == "code":
        # Replace the entire cell with imports
        new_src = (
            "import sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, str(Path('../../src')))\n"
            "\n"
            "from models import (\n"
            "    advection_diffusion_model,\n"
            "    partitioned_threshold_model,\n"
            "    partitioned_threshold_model_nonlinear,\n"
            ")\n"
            "from plotting import set_nature_style\n"
            "from utils import find_project_root\n"
            "\n"
            "import numpy as np\n"
            "import copy\n"
            "import TopoAnalysis.dem as d\n"
            "from scipy.stats import linregress\n"
            "import matplotlib.pyplot as plt\n"
            "from scipy.ndimage import zoom\n"
            "from matplotlib.ticker import LogLocator, LogFormatterMathtext\n"
        )
        _set_cell_source(cell0, new_src)
        changes.append("Cell 0: replaced 1072-line models duplicate with imports from src/")

    return changes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Update figure notebooks")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    mode = "WRITE" if args.write else "DRY-RUN"
    print(f"[{mode}] Updating figure notebooks in {FIG_DIR}\n")

    handlers = {
        "fig_slope_area.ipynb": update_fig_slope_area,
        "supp_fig5.ipynb": update_supp_fig5,
        "supp_fig6.ipynb": update_supp_fig6,
    }

    for name, handler in handlers.items():
        nb_path = FIG_DIR / name
        if not nb_path.exists():
            print(f"  SKIP {name} (not found)")
            continue

        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        n_cleared = clear_outputs(nb)
        changes = handler(nb)

        print(f"  {name}:")
        print(f"    outputs cleared: {n_cleared} cells")
        for c in changes:
            print(f"    {c}")
        if not changes:
            print(f"    (no structural changes needed)")

        if args.write:
            with open(nb_path, "w", encoding="utf-8") as f:
                json.dump(nb, f, indent=1, ensure_ascii=False)
                f.write("\n")

    print(f"\nDone. {'Changes written.' if args.write else 'Re-run with --write to apply.'}")


if __name__ == "__main__":
    main()
