#!/usr/bin/env python3
"""
Update the 10 model-run Jupyter notebooks in notebooks/model_runs/.

This script programmatically applies three categories of changes to each
notebook while preserving all tile-specific parameters and code:

1. **Clear outputs** -- removes all cell outputs to keep diffs small.
2. **Augment imports** -- adds ``from optimization import ...`` and
   ``from utils import save_model_outputs, compute_slope_magnitude``
   to the first code cell (the import / config cell).
3. **Enhance markdown** -- inserts LaTeX governing equations into the
   existing section-header markdown cells so the notebooks are
   self-documenting.

Cell identification is *content-based* (pattern matching on the cell
source text), not index-based, so the script is robust to the extra
markdown cell present in ``allegheny_tile5.ipynb``.

Usage
-----
    python scripts/update_model_notebooks.py          # dry-run (default)
    python scripts/update_model_notebooks.py --write  # apply changes

The script prints a summary of every change it makes (or would make).
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks" / "model_runs"

NOTEBOOKS = sorted(NB_DIR.glob("*.ipynb"))

# New imports to splice into the import cell.  We add them right after the
# existing ``from utils import crop`` line so the grouping stays tidy.
NEW_IMPORTS = (
    "from optimization import (\n"
    "    optimize_advection_diffusion,\n"
    "    optimize_threshold_acrit,\n"
    "    optimize_threshold_acrit_nonlinear,\n"
    "    optimize_threshold_full,\n"
    ")\n"
    "from utils import save_model_outputs, compute_slope_magnitude"
)

# ---------------------------------------------------------------------------
# LaTeX equation blocks to inject into markdown cells
# ---------------------------------------------------------------------------

# Each entry is (pattern_to_find_in_cell, text_to_append_if_absent).
# The pattern is matched against the raw markdown source; we only add the
# LaTeX block when it is not already present (idempotent).

LATEX_AD = (
    "\n\n**Governing equation (advection--diffusion):**\n\n"
    "$$\n"
    "0 = U - K A^m \\frac{\\partial z}{\\partial x} + D \\nabla^2 z\n"
    "$$"
)

LATEX_THRESHOLD = (
    "\n\n**Governing equations (threshold landscape model):**\n\n"
    "Channels where $A > A_{\\mathrm{crit}}$:\n\n"
    "$$\n"
    "0 = U - K A^m \\frac{\\partial z}{\\partial x}\n"
    "$$\n\n"
    "Hillslopes where $A \\le A_{\\mathrm{crit}}$, with linear sediment flux "
    "$q_s = -D\\nabla z$:\n\n"
    "$$\n"
    "0 = U + D\\nabla^2 z\n"
    "$$"
)

LATEX_NONLINEAR = (
    "\n\n**Nonlinear hillslope sediment flux:**\n\n"
    "$$\n"
    "q_s = \\frac{-D \\nabla z}{1 - (|\\nabla z|/S_c)^2}\n"
    "$$"
)

# Mapping: (substring that uniquely identifies the markdown cell) ->
#           LaTeX block to append.
MARKDOWN_ENHANCEMENTS: list[tuple[str, str]] = [
    # Cell 1 -- AD section intro
    ("## Advection-Diffusion model exploration", LATEX_AD),
    # Cell 7 -- Threshold model section header (use specific wording to
    # avoid matching cell 26 which also mentions "threshold landscape model"
    # but in a different context).
    ("We next consider the threshold landscape model", LATEX_THRESHOLD),
    # Cell 16 -- Nonlinear section
    ("non-linear diffusion on the best-fitting", LATEX_NONLINEAR),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cell_source(cell: dict) -> str:
    """Return a cell's source as a single string."""
    src = cell.get("source", [])
    if isinstance(src, list):
        return "".join(src)
    return src


def _set_cell_source(cell: dict, text: str) -> None:
    """Set a cell's source from a single string, converting to line-list."""
    # nbformat 4 stores source as a list of strings, each ending with \n
    # except possibly the last.
    lines: list[str] = []
    for line in text.split("\n"):
        lines.append(line + "\n")
    # Strip trailing newline on the very last element.
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1][:-1]
    cell["source"] = lines


def _new_cell_id() -> str:
    """Generate a random cell ID (nbformat >= 4.5)."""
    return str(uuid.uuid4())


def clear_outputs(nb: dict) -> int:
    """Clear outputs and execution counts from all code cells.  Returns count."""
    cleared = 0
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            if cell.get("outputs") or cell.get("execution_count"):
                cell["outputs"] = []
                cell["execution_count"] = None
                cleared += 1
    return cleared


def augment_imports(nb: dict) -> bool:
    """Add optimization / utils imports to the first code cell.

    Returns True if the cell was modified.
    """
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = _cell_source(cell)
        # This is the import cell if it contains the sys.path manipulation.
        if "sys.path.insert" not in src:
            continue

        # Already patched?
        if "from optimization import" in src:
            return False

        # Find the ``from utils import crop`` line and insert after it.
        marker = "from utils import crop"
        if marker in src:
            src = src.replace(
                marker,
                marker + "\n" + NEW_IMPORTS,
            )
        else:
            # Fallback: append at the end of the import block (before the
            # first blank line that separates imports from config).
            src = src.rstrip() + "\n" + NEW_IMPORTS + "\n"

        _set_cell_source(cell, src)
        return True

    return False


def enhance_markdown(nb: dict) -> int:
    """Append LaTeX equation blocks to the relevant markdown cells.

    Returns the number of cells modified.
    """
    count = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        src = _cell_source(cell)
        for pattern, latex_block in MARKDOWN_ENHANCEMENTS:
            if pattern in src and latex_block.strip() not in src:
                src = src.rstrip() + latex_block
                _set_cell_source(cell, src)
                count += 1
                break  # Each cell matches at most one pattern.
    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_notebook(nb_path: Path, write: bool) -> dict:
    """Process a single notebook.  Returns a summary dict."""
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    summary = {"file": nb_path.name, "outputs_cleared": 0,
               "imports_added": False, "markdown_enhanced": 0}

    summary["outputs_cleared"] = clear_outputs(nb)
    summary["imports_added"] = augment_imports(nb)
    summary["markdown_enhanced"] = enhance_markdown(nb)

    if write:
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")  # trailing newline

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update model-run notebooks with refactored imports "
                    "and LaTeX documentation.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Actually write changes to disk (default is dry-run).",
    )
    args = parser.parse_args()

    if not NOTEBOOKS:
        print(f"ERROR: No notebooks found in {NB_DIR}", file=sys.stderr)
        sys.exit(1)

    mode = "WRITE" if args.write else "DRY-RUN"
    print(f"[{mode}] Processing {len(NOTEBOOKS)} notebooks in {NB_DIR}\n")

    for nb_path in NOTEBOOKS:
        summary = process_notebook(nb_path, write=args.write)
        print(f"  {summary['file']}:")
        print(f"    outputs cleared   : {summary['outputs_cleared']} cells")
        print(f"    imports added     : {summary['imports_added']}")
        print(f"    markdown enhanced : {summary['markdown_enhanced']} cells")

    print(f"\nDone.  {'Changes written.' if args.write else 'Re-run with --write to apply.'}")


if __name__ == "__main__":
    main()
