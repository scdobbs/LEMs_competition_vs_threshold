#!/usr/bin/env python3
"""
Consolidate model-run notebooks: replace inline optimization, save, and
slope-computation code with calls to src/ functions.

Processes all 10 notebooks in notebooks/model_runs/.
"""

import json
import re
import sys
from pathlib import Path


NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks" / "model_runs"


def load_notebook(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_notebook(path: Path, nb: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")


def cell_source(cell: dict) -> str:
    """Join cell source lines into a single string."""
    return "".join(cell.get("source", []))


def set_cell_source(cell: dict, new_source: str) -> None:
    """Replace cell source with new text, splitting into lines."""
    lines = new_source.split("\n")
    # nbformat stores each line ending with \n except possibly the last
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + "\n")
        else:
            result.append(line)
    cell["source"] = result


def is_code_cell(cell: dict) -> bool:
    return cell.get("cell_type") == "code"


# ---------------------------------------------------------------------------
# Pattern matchers
# ---------------------------------------------------------------------------

def is_ad_optimization_cell(src: str) -> bool:
    """Cell 6: AD K,D optimization with misfit_history, solve_advection_diffusion, optimize_KD."""
    return (
        "def optimize_KD" in src
        and "misfit_history" in src
        and "def solve_advection_diffusion" in src
    )


def is_threshold_acrit_optimization_cell(src: str) -> bool:
    """Cell 15: Threshold A_crit optimization (linear diffusion)."""
    return (
        "def solve_partitioned_model(" in src
        and "def optimize_partitioned_model(" in src
        and "solve_partitioned_model_nonlinear" not in src
        and "solve_partitioned_model_all" not in src
    )


def is_nonlinear_acrit_optimization_cell(src: str) -> bool:
    """Cell 25: Nonlinear A_crit optimization."""
    return (
        "def solve_partitioned_model_nonlinear(" in src
        and "def optimize_partitioned_model(" in src
        and "def solve_partitioned_model_all" not in src
    )


def is_full_optimization_cell(src: str) -> bool:
    """Cell 28: Full K,D,A_crit optimization."""
    return (
        "def solve_partitioned_model_all(" in src
        and "def optimize_partitioned_model_all(" in src
    )


def is_save_block_cell(src: str) -> bool:
    """Cells 4, 7, 16, 29: Contains 'if SAVE:' with copy.copy(dem) save chain."""
    return (
        "if SAVE:" in src
        and "copy.copy(dem)" in src
        and "dem_opt.save(" in src
    )


def is_slope_computation_cell(src: str) -> bool:
    """Cell 18: Uses np.gradient for slope computation."""
    return (
        "np.gradient(Z_model)" in src
        and "hillslope_mask" in src
    )


# ---------------------------------------------------------------------------
# Replacement builders
# ---------------------------------------------------------------------------

def replace_ad_optimization(src: str) -> str:
    """Replace AD K,D optimization cell."""
    return (
        "K_star, D_star, mse_star = optimize_advection_diffusion(\n"
        "    advection_diffusion, dem._griddata, U, K0, D0, K_bounds, D_bounds,\n"
        ")\n"
        'print("\\n\\U0001f389 Optimal parameters:")\n'
        'print(f"   K = {K_star:.4e}")\n'
        'print(f"   D = {D_star:.4e}")\n'
        'print(f"   MSE = {mse_star:.4e}")\n'
    )


def replace_threshold_acrit(src: str) -> str:
    """Replace threshold A_crit optimization cell."""
    return (
        "A_crit_0 = 5000\n"
        "A_crit_bounds = (1E2, 1E5)\n"
        "\n"
        "A_crit_star, mse_star = optimize_threshold_acrit(\n"
        "    partitioned_model, dem._griddata, D0, A_crit_0, A_crit_bounds,\n"
        ")\n"
        'print("\\n\\U0001f389 Optimal parameters:")\n'
        'print(f"   A_crit = {A_crit_star:.4e}")\n'
        'print(f"   MSE = {mse_star:.4e}")\n'
    )


def replace_nonlinear_acrit(src: str) -> str:
    """Replace nonlinear A_crit optimization cell."""
    return (
        "A_crit_0 = 5000\n"
        "A_crit_bounds = (1E2, 1E5)\n"
        "\n"
        "A_crit_star, mse_star = optimize_threshold_acrit_nonlinear(\n"
        "    partitioned_model_nonlinear, dem._griddata, D0, Sc, A_crit_0, A_crit_bounds,\n"
        ")\n"
        'print("\\n\\U0001f389 Optimal parameters:")\n'
        'print(f"   A_crit = {A_crit_star:.4e}")\n'
        'print(f"   MSE = {mse_star:.4e}")\n'
    )


def replace_full_optimization(src: str) -> str:
    """Replace full K,D,A_crit optimization cell."""
    return (
        "A_crit_0 = 5000\n"
        "A_crit_bounds = (1E2, 1E5)\n"
        "\n"
        "A_crit_star, D_star, K_star, mse_star = optimize_threshold_full(\n"
        "    dem, area, flow_direction, m, U, dem._griddata,\n"
        "    K0, D0, A_crit_0, K_bounds, D_bounds, A_crit_bounds,\n"
        ")\n"
        'print("\\n\\U0001f389 Optimal parameters:")\n'
        'print(f"   D = {D_star:.4e}")\n'
        'print(f"   K = {K_star:.4e}")\n'
        'print(f"   A_crit = {A_crit_star:.4e}")\n'
        'print(f"   MSE = {mse_star:.4e}")\n'
    )


def extract_save_prefix(src: str) -> str | None:
    """Extract the save prefix string from an if SAVE: block."""
    # Match: prefix = base_save + "AP_1m_best_tile_AD-no-opt"
    m = re.search(r'prefix\s*=\s*base_save\s*\+\s*"([^"]+)"', src)
    if m:
        return m.group(1)
    return None


def replace_save_block(src: str) -> str:
    """Replace inline save block with save_model_outputs call."""
    prefix = extract_save_prefix(src)
    if prefix is None:
        return src  # Can't determine prefix, leave unchanged

    # Keep everything before the if SAVE: block
    lines = src.split("\n")
    new_lines = []
    in_save_block = False
    save_indent = 0

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("if SAVE:"):
            in_save_block = True
            save_indent = len(line) - len(stripped)
            indent = " " * save_indent
            new_lines.append(f"{indent}if SAVE:")
            new_lines.append(f'{indent}    save_model_outputs(dem, Z_model, base_save, "{prefix}")')
            new_lines.append(f"""{indent}    print(f"[saved optimized grids to '{{base_save}}']")""")
            continue

        if in_save_block:
            # Skip lines that are indented more than the if SAVE: line
            if stripped == "" or (len(line) - len(stripped) > save_indent):
                continue
            else:
                in_save_block = False
                new_lines.append(line)
        else:
            new_lines.append(line)

    return "\n".join(new_lines)


def replace_slope_computation(src: str) -> str:
    """Replace np.gradient slope computation with compute_slope_magnitude."""
    # Replace the gradient computation pattern
    src = re.sub(
        r'\[gx,\s*gy\]\s*=\s*np\.gradient\(Z_model\)\n'
        r's\s*=\s*np\.power\(np\.power\(gx,\s*2\)\s*\+\s*np\.power\(gy,\s*2\),\s*1/2\)',
        's = compute_slope_magnitude(Z_model)',
        src,
    )
    return src


def clean_imports(src: str) -> str:
    """Remove no-longer-needed imports from the main import cell."""
    lines = src.split("\n")
    new_lines = []
    for line in lines:
        stripped = line.strip()
        # Remove specific imports that are no longer needed
        if stripped == "from scipy.optimize import minimize":
            continue
        if stripped == "from functools import lru_cache":
            continue
        if stripped == "import copy":
            continue
        new_lines.append(line)
    return "\n".join(new_lines)


def clean_inline_reimports(src: str) -> str:
    """Remove re-imports in optimization cells."""
    lines = src.split("\n")
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped == "from scipy.optimize import minimize":
            continue
        if stripped == "from functools import lru_cache":
            continue
        if stripped in ("import numpy as np", "from scipy.stats import linregress"):
            continue
        new_lines.append(line)
    return "\n".join(new_lines)


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_notebook(path: Path) -> dict:
    """Process a single model-run notebook."""
    nb = load_notebook(path)
    changes = {
        "ad_opt": 0,
        "threshold_acrit": 0,
        "nonlinear_acrit": 0,
        "full_opt": 0,
        "save_blocks": 0,
        "slope_comp": 0,
        "import_clean": 0,
    }

    for cell in nb["cells"]:
        if not is_code_cell(cell):
            continue

        src = cell_source(cell)

        # 1. Replace optimization cells
        if is_ad_optimization_cell(src):
            set_cell_source(cell, replace_ad_optimization(src))
            changes["ad_opt"] += 1
            continue

        if is_full_optimization_cell(src):
            set_cell_source(cell, replace_full_optimization(src))
            changes["full_opt"] += 1
            continue

        if is_nonlinear_acrit_optimization_cell(src):
            set_cell_source(cell, replace_nonlinear_acrit(src))
            changes["nonlinear_acrit"] += 1
            continue

        if is_threshold_acrit_optimization_cell(src):
            set_cell_source(cell, replace_threshold_acrit(src))
            changes["threshold_acrit"] += 1
            continue

        # 2. Replace save blocks
        if is_save_block_cell(src):
            new_src = replace_save_block(src)
            set_cell_source(cell, new_src)
            changes["save_blocks"] += 1
            continue

        # 3. Replace slope computation
        if is_slope_computation_cell(src):
            new_src = replace_slope_computation(src)
            set_cell_source(cell, new_src)
            changes["slope_comp"] += 1
            continue

        # 4. Clean up imports (main import cell)
        if ("from optimization import" in src
            and "from scipy.optimize import minimize" in src):
            new_src = clean_imports(src)
            set_cell_source(cell, new_src)
            changes["import_clean"] += 1

    return nb, changes


def main():
    notebooks = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
    if not notebooks:
        print(f"No notebooks found in {NOTEBOOKS_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(notebooks)} notebooks in {NOTEBOOKS_DIR}\n")

    for path in notebooks:
        print(f"--- {path.name} ---")
        nb, changes = process_notebook(path)
        save_notebook(path, nb)

        total = sum(changes.values())
        if total == 0:
            print("  No changes made (unexpected)")
        else:
            for key, count in changes.items():
                if count > 0:
                    print(f"  {key}: {count}")
            print(f"  Total changes: {total}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
