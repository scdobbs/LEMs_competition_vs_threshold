"""Command-line interface: ``lemcvt``.

Every step is resumable (each invocation writes a new run directory under
``$LEMCVT_RESULTS``) and records its inputs (config snapshot, grid hashes, git
hash, environment lock hash) in ``run.json``.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import click

from lemcvt import __version__
from lemcvt.config import STAGES, TILE_IDS

log = logging.getLogger("lemcvt")


def _setup_logging(verbose: int) -> None:
    level = logging.DEBUG if verbose > 1 else logging.INFO if verbose == 1 else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stderr)
    logging.getLogger("lemcvt").setLevel(logging.INFO if verbose == 0 else level)
    logging.getLogger("rasterio").setLevel(logging.WARNING)


def _parse_tiles(tiles: str) -> list[str]:
    if tiles.lower() == "all":
        return list(TILE_IDS)
    out = []
    for t in tiles.split(","):
        t = t.strip()
        if t.upper() in ("AP", "GM"):
            out += [x for x in TILE_IDS if x.startswith(t.upper())]
        elif t:
            if t not in TILE_IDS:
                raise click.BadParameter(f"unknown tile {t!r}; choose from {TILE_IDS} or AP/GM/all")
            out.append(t)
    return out


def _parse_crop(crop: str | None) -> tuple[int, int, int, int] | None:
    if not crop:
        return None
    parts = [int(p) for p in crop.split(",")]
    if len(parts) != 4:
        raise click.BadParameter("crop must be row0,row1,col0,col1")
    return tuple(parts)  # type: ignore[return-value]


@click.group()
@click.version_option(__version__, prog_name="lemcvt")
@click.option("-v", "--verbose", count=True, help="-v info, -vv debug")
def main(verbose: int) -> None:
    """lemcvt: steady-state LEM calibration (Dobbs & Hilley), v2."""
    _setup_logging(verbose)


# -------------------------------------------------------------------------------- info
@main.command()
def info() -> None:
    """Show environment, data and results locations and whether the data verify."""
    from lemcvt.config import REPO_ROOT, config_root, results_root
    from lemcvt.results import env_lock_sha256, git_hash

    click.echo(f"lemcvt {__version__}  repo={REPO_ROOT}  git={git_hash()}")
    click.echo(f"configs={config_root()}")
    click.echo(f"LEMCVT_DATA={os.environ.get('LEMCVT_DATA', '(unset)')}")
    click.echo(f"LEMCVT_RESULTS={results_root()}")
    click.echo(f"env lock sha256={env_lock_sha256()}")
    click.echo(f"python={sys.version.split()[0]} PYTHONNOUSERSITE={os.environ.get('PYTHONNOUSERSITE', '(unset)')}")


# -------------------------------------------------------------------------------- data
@main.group()
def data() -> None:
    """Grid data: fetch from group storage, verify against data/manifest.csv, build manifest."""


@data.command("verify")
@click.option("--inputs-only", is_flag=True, help="Only the 30 input grids (filled/area/fd).")
def data_verify(inputs_only: bool) -> None:
    from lemcvt.io.manifest import verify

    n_ok, problems = verify(only_inputs=inputs_only)
    click.echo(f"{n_ok} files verified, {len(problems)} problems")
    for p in problems:
        click.echo("  " + p)
    sys.exit(1 if problems else 0)


@data.command("fetch")
@click.option("--source", type=click.Path(path_type=Path), default=None, help="Override configs/data.yaml source.")
@click.option("--inputs-only", is_flag=True)
def data_fetch(source: Path | None, inputs_only: bool) -> None:
    from lemcvt.io.manifest import fetch

    copied, problems = fetch(source=source, only_inputs=inputs_only)
    click.echo(f"copied {copied} files; {len(problems)} problems after verification")
    for p in problems:
        click.echo("  " + p)
    sys.exit(1 if problems else 0)


@data.command("manifest")
@click.option("--root", type=click.Path(path_type=Path), default=None, help="Directory to scan (default LEMCVT_DATA).")
@click.option("--out", type=click.Path(path_type=Path), default=None,
              help="Where to write the manifest (default data/manifest.csv in the repository).")
def data_manifest(root: Path | None, out: Path | None) -> None:
    """Build data/manifest.csv from the grids under LEMCVT_DATA (sha256 of every .tif)."""
    from lemcvt.config import data_root
    from lemcvt.io.grids import sha256_file
    from lemcvt.io.manifest import MANIFEST_PATH, ManifestEntry, write_manifest

    MANIFEST_PATH = out or MANIFEST_PATH  # noqa: N806
    root = root or data_root()
    entries = []
    for p in sorted(root.rglob("*.tif")):
        rel = p.relative_to(root)
        parts = rel.parts
        site, tile = parts[0], parts[1]
        kind = "output" if "outputs" in parts else "input"
        role = p.stem.split("_tile_", 1)[-1]
        entries.append(ManifestEntry(str(rel), sha256_file(p), p.stat().st_size, site, tile, kind, role))
    write_manifest(entries, MANIFEST_PATH)
    click.echo(f"wrote {MANIFEST_PATH} with {len(entries)} entries")


# -------------------------------------------------------------------------------- tile
@main.group()
def tile() -> None:
    """Per-tile calibration (run) and forward evaluation (forward)."""


@tile.command("run")
@click.option("--tile", "tile_id", required=True, help="e.g. GM1")
@click.option("--stages", default=",".join(STAGES), show_default=True)
@click.option("--optimizer", default="v1", show_default=True, type=click.Choice(["v1", "v2"]))
@click.option("--no-grids", is_flag=True, help="Do not write output grids.")
@click.option("--no-derived", is_flag=True, help="Write elevation grids only (skip filled/fd/area/logarea/hillshade).")
@click.option("--exploration", is_flag=True, help="Also run v1's plotted-only exploratory forward runs.")
@click.option("--crop", default=None, help="row0,row1,col0,col1 pixel crop (smoke tests).")
@click.option("--acrit-grid", default="thin", show_default=True, type=click.Choice(["thin", "full"]),
              help="v2 only: every unique drainage area in [1e2, 1e5] m^2 ('full') or the 2000 nearest a uniform log grid ('thin').")
@click.option("--nl-grid", default=200, show_default=True, help="v2 only: log-thinned A_crit values for the nonlinear stage.")
@click.option("--m-acrit-grid", default=200, show_default=True, help="v2 only: log-thinned A_crit values for the m profile.")
@click.option("--run-id", default=None)
def tile_run(tile_id, stages, optimizer, no_grids, no_derived, exploration, crop, acrit_grid, nl_grid, m_acrit_grid, run_id) -> None:
    """v1: the exact port of the four v1 stages. v2 (Phase 4): stages ad, partitioned (= acrit + joint), nonlinear, m_profile."""
    st = [s.strip() for s in stages.split(",")]
    if optimizer == "v2":
        from lemcvt.analyses.optimizer_v2 import run_tile_v2

        rec = run_tile_v2(tile_id, stages=st, acrit_grid=acrit_grid, nl_n=nl_grid, m_acrit_n=m_acrit_grid, save_grids=not no_grids,
                          derived_grids=not no_derived, crop=_parse_crop(crop), run_id=run_id)
    else:
        from lemcvt.analyses.tile_pipeline import run_tile

        rec = run_tile(tile_id, stages=st, optimizer=optimizer, save_grids=not no_grids, derived_grids=not no_derived,
                       exploration_runs=exploration, crop=_parse_crop(crop), run_id=run_id)
    click.echo(f"results: {rec.dir}")


@tile.command("v2-refine")
@click.option("--tile", "tile_id", required=True)
@click.option("--source-run", default=None, help="v2 partitioned run directory or name (default: the latest for the tile).")
@click.option("--max-extra", default=400, show_default=True)
@click.option("--run-id", default=None)
def tile_v2_refine(tile_id, source_run, max_extra, run_id) -> None:
    """Phase 4: evaluate every unique drainage area around the thinned profile's minima (staircase refinement)."""
    from lemcvt.analyses.optimizer_v2 import refine_partitioned

    if source_run is None:
        from lemcvt.results import find_runs

        cands = [r for r in find_runs("tile_run", tile_id) if r.name.endswith("_v2_partitioned")]
        if not cands:
            raise click.ClickException(f"no v2 partitioned run for {tile_id}")
        source_run = str(cands[-1])
    rec = refine_partitioned(tile_id, source_run, max_extra=max_extra, run_id=run_id)
    click.echo(f"results: {rec.dir}")


@tile.command("v2-estimate")
@click.option("--tiles", default="all", show_default=True)
@click.option("--thin-n", default=2000, show_default=True)
def tile_v2_estimate(tiles, thin_n) -> None:
    """Phase 4 budget rule: unique-area counts and the core-hour estimate for the full and thinned A_crit grids."""
    import json as _json

    from lemcvt.analyses.optimizer_v2 import estimate

    est = estimate(_parse_tiles(tiles), thin_n=thin_n)
    click.echo(est.pop("table"))
    click.echo(_json.dumps({k: v for k, v in est.items() if k not in ("rows", "cost_assumptions_s")}, indent=1))
    click.echo(f"cost assumptions (s per evaluation): {_json.dumps(est['cost_assumptions_s'])}")


@tile.command("forward")
@click.option("--tile", "tile_id", required=True)
@click.option("--params", default="v1", show_default=True, help="'v1' or path to a tile_run params.json")
@click.option("--compare", is_flag=True, help="Compare with the saved v1 grids (Phase 2.1 regression).")
@click.option("--keys", default=None, help="Comma list of run keys (ad_no_opt,ad_opt,acrit_opt,nonlinear_opt,joint_opt)")
@click.option("--no-grids", is_flag=True)
@click.option("--derived", is_flag=True, help="Also write the derived grids.")
@click.option("--reference-run", default=None,
              help="Phase 3 gate: compare with this tile_forward run directory ('phase2' = latest Phase 2 run of the tile).")
@click.option("--crop", default=None)
@click.option("--run-id", default=None)
def tile_forward(tile_id, params, compare, keys, no_grids, derived, reference_run, crop, run_id) -> None:
    from lemcvt.analyses.forward import run_forward

    rec = run_forward(tile_id, params=params, compare=compare, save_grids=not no_grids, derived_grids=derived,
                      keys=keys.split(",") if keys else None, crop=_parse_crop(crop), run_id=run_id,
                      reference_run=reference_run)
    click.echo(f"results: {rec.dir}")


# -------------------------------------------------------------------------------- sweeps / analyses
@main.group()
def sweep() -> None:
    """Parameter sweeps."""


@sweep.command("resolution")
@click.option("--sites", default="AP,GM", show_default=True)
@click.option("--routing", default="d8,dinf", show_default=True)
@click.option("--tiles", default=None, help="Restrict to tiles (default all tiles of the sites).")
@click.option("--dx-min", default=5.0, show_default=True)
@click.option("--dx-max", default=100.0, show_default=True)
@click.option("--n-dx", default=25, show_default=True)
@click.option("--run-id", default=None)
def sweep_resolution(sites, routing, tiles, dx_min, dx_max, n_dx, run_id) -> None:
    from lemcvt.analyses.resolution_sweep import run_resolution_sweep

    tile_ids = _parse_tiles(tiles) if tiles else [t for t in TILE_IDS if t[:2] in sites.split(",")]
    rec = run_resolution_sweep(tile_ids, routings=routing.split(","), dx_min=dx_min, dx_max=dx_max, n_dx=n_dx,
                               run_id=run_id)
    click.echo(f"results: {rec.dir}")


@main.group()
def synthetic() -> None:
    """Synthetic planform analyses."""


@synthetic.command("pe")
@click.option("--run-id", default=None)
def synthetic_pe(run_id) -> None:
    """Peclet-number resolution analysis (supp_fig5)."""
    from lemcvt.analyses.synthetic_pe import run_synthetic_pe

    rec = run_synthetic_pe(run_id=run_id)
    click.echo(f"results: {rec.dir}")


@main.command("perron-forward")
@click.option("--tiles", default="all", show_default=True)
@click.option("--run-id", default=None)
def perron_forward(tiles, run_id) -> None:
    """Forward AD and partitioned runs with the regional (Perron et al. 2012) parameters (Fig. 2)."""
    from lemcvt.analyses.perron_forward import run_perron_forward

    rec = run_perron_forward(_parse_tiles(tiles), run_id=run_id)
    click.echo(f"results: {rec.dir}")


@main.command("d8-vs-dinf")
@click.option("--tiles", default="AP1,GM1", show_default=True)
@click.option("--run-id", default=None)
def d8_vs_dinf(tiles, run_id) -> None:
    """D8 against D-infinity comparison at measured parameters (d8_vs_dinf_comparison)."""
    from lemcvt.analyses.d8_vs_dinf import run_d8_vs_dinf

    rec = run_d8_vs_dinf(_parse_tiles(tiles), run_id=run_id)
    click.echo(f"results: {rec.dir}")


@main.command("slope-area")
@click.option("--tiles", default="AP1,GM2", show_default=True)
@click.option("--run-id", default=None)
def slope_area(tiles, run_id) -> None:
    """Binned slope-area regressions of observed and saved model grids (fig_slope_area)."""
    from lemcvt.analyses.slope_area_fig import run_slope_area

    rec = run_slope_area(_parse_tiles(tiles), run_id=run_id)
    click.echo(f"results: {rec.dir}")


# -------------------------------------------------------------------------------- measure
@main.command("measure")
@click.option("--tile", "tiles", required=True, help="Tile id, comma list, AP, GM or all")
@click.option("--metres/--stored-units", default=True, show_default=True,
              help="Measure on metre grids (converts AP from feet regardless of reproduce_v1_units).")
@click.option("--n-boot", default=200, show_default=True)
@click.option("--no-legacy", is_flag=True, help="Skip the re-run of the v1 method (slow on full tiles).")
@click.option("--crop", default=None, help="row0,row1,col0,col1 (smoke tests only)")
@click.option("--run-id", default=None)
def measure(tiles, metres, n_boot, no_legacy, crop, run_id) -> None:
    """Measure theta, k_s, hilltop curvature and D0 = U/C_ht from the tile's own grid."""
    from lemcvt.measure.pipeline import run_measure

    rec = run_measure(_parse_tiles(tiles), force_metres=metres, n_boot=n_boot, run_id=run_id, run_legacy=not no_legacy,
                      crop=_parse_crop(crop))
    click.echo(f"results: {rec.dir}")


@main.command("profile")
@click.option("--tile", "tile_id", required=True)
@click.option("--picard-iters", default=2, show_default=True)
@click.option("--run-id", default=None)
def profile(tile_id, picard_iters, run_id) -> None:
    """Phase 3 step 1: cProfile + line_profiler of every model stage on a full tile."""
    from lemcvt.analyses.profiling import run_profile

    rec = run_profile(tile_id, picard_iters=picard_iters, run_id=run_id)
    click.echo(f"results: {rec.dir}")


# -------------------------------------------------------------------------------- closeout (addendum 2, section 2)
@main.group()
def closeout() -> None:
    """Phase 2 close-out diagnostics (addendum 2): rounding recovery, D-infinity diagnostics."""


@closeout.command("linearized")
@click.option("--tile", "tile_id", required=True)
@click.option("--keys", default="ad_opt,joint_opt", show_default=True)
@click.option("--rel-step", default=1e-6, show_default=True)
@click.option("--forward-run", type=click.Path(path_type=Path), default=None,
              help="tile_forward run directory to take the Phase 2 grids from (default: latest for the tile)")
@click.option("--run-id", default=None)
def closeout_linearized(tile_id, keys, rel_step, forward_run, run_id) -> None:
    """Linearized recovery of the printed-parameter rounding, rounding-interval and difference counts."""
    from lemcvt.analyses.phase2_closeout import run_closeout

    rec = run_closeout(tile_id, keys=tuple(k.strip() for k in keys.split(",")), rel_step=rel_step, run_id=run_id,
                       forward_run=forward_run)
    click.echo(f"results: {rec.dir}")


@closeout.command("dinf")
@click.option("--tiles", default="AP1,GM1", show_default=True)
@click.option("--no-synthetic", is_flag=True)
@click.option("--run-id", default=None)
def closeout_dinf(tiles, no_synthetic, run_id) -> None:
    """Diagnose the D-infinity partitioned model (addendum 2, 2.6); changes no model code."""
    from lemcvt.analyses.dinf_diagnostics import run_dinf_diagnostics

    rec = run_dinf_diagnostics(_parse_tiles(tiles) if tiles else (), synthetic=not no_synthetic, run_id=run_id)
    click.echo(f"results: {rec.dir}")


# -------------------------------------------------------------------------------- audits (addendum 3, sections 3-4)
@main.group()
def audit() -> None:
    """Addendum 3 audits: Picard convergence (section 3), chi estimator and hilltop set (section 4)."""


@audit.command("picard")
@click.option("--tile", "tile_id", required=True)
@click.option("--metres", is_flag=True, help="Load with reproduce_v1_units: false (in memory only; configs unchanged).")
@click.option("--scale-k", "scale_K", default=1.0, show_default=True, help="Multiply K0_partitioned by this factor.")
@click.option("--scale-d", "scale_D", default=1.0, show_default=True, help="Multiply D0 by this factor.")
@click.option("--label", default=None, help="Variant label for the run id (default feet_as_v1 / metres).")
@click.option("--reference-run", default=None,
              help="tile_forward run directory whose nonlinear_opt grid the result is compared with ('step5' = latest Phase 3 gate run).")
@click.option("--max-iter", default=None, type=int, help="Override the Picard cap (default: the config's 50).")
@click.option("--no-grids", is_flag=True)
@click.option("--run-id", default=None)
def audit_picard(tile_id, metres, scale_K, scale_D, label, reference_run, max_iter, no_grids, run_id) -> None:
    """Run the nonlinear model once at the v1 nonlinear optimum with the instrumented Picard loop."""
    from lemcvt.analyses.picard_audit import run_picard_audit

    if reference_run == "step5":
        from lemcvt.results import find_runs

        cands = [r for r in find_runs("tile_forward", tile_id) if "20261109T2143" <= r.name[:13] <= "20261109T2150"]
        reference_run = str(cands[-1]) if cands else None
    rec = run_picard_audit(tile_id, metres=metres, scale_K=scale_K, scale_D=scale_D, label=label, reference_run=reference_run,
                           save_grids=not no_grids, run_id=run_id, max_iter=max_iter)
    click.echo(f"results: {rec.dir}")


@audit.command("measure")
@click.option("--tiles", default="all", show_default=True)
@click.option("--n-boot", default=200, show_default=True)
@click.option("--crop", default=None, help="row0,row1,col0,col1 (smoke tests only)")
@click.option("--run-id", default=None)
def audit_measure(tiles, n_boot, crop, run_id) -> None:
    """chi estimator audit (n_eff, equal-weight and median estimators, truncation) and hilltop exclusion (addendum 3, section 4)."""
    from lemcvt.measure.audits import run_measure_audit

    rec = run_measure_audit(_parse_tiles(tiles), n_boot=n_boot, run_id=run_id, crop=_parse_crop(crop))
    click.echo(f"results: {rec.dir}")


@main.command("measure-diagnostics")
@click.option("--tiles", default="all", show_default=True)
@click.option("--n-boot", default=200, show_default=True)
@click.option("--crop", default=None, help="row0,row1,col0,col1 (smoke tests only)")
@click.option("--run-id", default=None)
def measure_diagnostics(tiles, n_boot, crop, run_id) -> None:
    """Threshold-decision diagnostics (addendum 2, section 6): ladders, scaling break, edge effects, curvature test."""
    from lemcvt.measure.diagnostics import run_measure_diagnostics

    rec = run_measure_diagnostics(_parse_tiles(tiles), n_boot=n_boot, run_id=run_id, crop=_parse_crop(crop))
    click.echo(f"results: {rec.dir}")


# -------------------------------------------------------------------------------- figures / tables
@main.command("figures")
@click.argument("which", default="all")
@click.option("--out", type=click.Path(path_type=Path), default=None, help="Output directory (default figures/).")
def figures(which, out) -> None:
    """Build figures from results only (no model computation)."""
    from lemcvt.figures import build_figures

    paths = build_figures(which, out_dir=out)
    for p in paths:
        click.echo(str(p))


@main.command("tables")
@click.argument("which", default="all")
@click.option("--out", type=click.Path(path_type=Path), default=None, help="Output directory (default tables/).")
def tables(which, out) -> None:
    """Build tables (LaTeX booktabs + CSV) from results only."""
    from lemcvt.tables import build_tables

    paths = build_tables(which, out_dir=out)
    for p in paths:
        click.echo(str(p))


@main.command("regression-report")
@click.option("--out", type=click.Path(path_type=Path), default=None, help="Output markdown path.")
def regression_report(out) -> None:
    """Compare results with the frozen v1 reference and write reports/v2/02_regression_tables.md."""
    from lemcvt.analyses.regression_report import build

    p = build(out_md=out)
    click.echo(str(p))


# -------------------------------------------------------------------------------- slurm
@main.group()
def slurm() -> None:
    """Render and submit Slurm job arrays."""


@slurm.command("submit")
@click.argument("step", type=click.Choice(["tile_run", "tile_forward", "sweep_resolution", "measure", "closeout",
                                           "dinf_diagnostics", "measure_diagnostics", "measure_audit", "profile", "custom"]))
@click.option("--tiles", default="all", show_default=True)
@click.option("--labels", default=None, help="custom step: comma list of array labels (default: one task 'all')")
@click.option("--command", "command_", default=None, help="custom step: command with $LABEL, e.g. \"lemcvt -v slope-area --tiles AP1,GM2\"")
@click.option("--stages", default=",".join(STAGES), show_default=True, help="tile_run only")
@click.option("--optimizer", default="v1", show_default=True)
@click.option("--params", default="v1", show_default=True, help="tile_forward only")
@click.option("--compare", is_flag=True, help="tile_forward only")
@click.option("--reference-run", default=None, help="tile_forward only: Phase 3 gate against this run ('phase2')")
@click.option("--routing", default="d8,dinf", show_default=True, help="sweep_resolution only")
@click.option("--time", "time_", default=None, help="Override wall time (D-HH:MM:SS).")
@click.option("--cpus", default=None, type=int)
@click.option("--mem", default=None)
@click.option("--dry-run", is_flag=True, help="Render and print the script and estimate; do not submit.")
@click.option("--approved", is_flag=True, help="Steve approved a submission above the core-hour limit (300).")
def slurm_submit(step, tiles, labels, command_, stages, optimizer, params, compare, reference_run, routing, time_, cpus,
                 mem, dry_run, approved):
    from lemcvt.slurm import build_spec, estimate_text, render, submit

    labels = _parse_tiles(tiles) if step != "custom" else ([s.strip() for s in labels.split(",")] if labels else ["all"])
    if step == "custom":
        if not command_:
            raise click.BadParameter("custom step needs --command")
        cmd = command_
    elif step == "tile_run":
        cmd = f"lemcvt -v tile run --tile $LABEL --stages {stages} --optimizer {optimizer}"
    elif step == "tile_forward":
        cmd = (f"lemcvt -v tile forward --tile $LABEL --params {params}" + (" --compare" if compare else "")
               + (f" --reference-run {reference_run}" if reference_run else ""))
    elif step == "sweep_resolution":
        cmd = f"lemcvt -v sweep resolution --tiles $LABEL --routing {routing}"
    elif step == "closeout":
        cmd = "lemcvt -v closeout linearized --tile $LABEL"
    elif step == "dinf_diagnostics":
        cmd = f"lemcvt -v closeout dinf --tiles {tiles}"
        labels = ["all"]
    elif step == "measure_diagnostics":
        cmd = f"lemcvt -v measure-diagnostics --tiles {tiles}"
        labels = ["all"]
    elif step == "measure_audit":
        cmd = f"lemcvt -v audit measure --tiles {tiles}"
        labels = ["all"]
    elif step == "profile":
        cmd = "lemcvt -v profile --tile $LABEL"
    else:
        cmd = "lemcvt -v measure --tile $LABEL"
    spec = build_spec(step, labels, cmd, time=time_, cpus=cpus, mem=mem)
    click.echo(estimate_text(spec))
    if dry_run:
        click.echo(render(spec))
    try:
        job, script = submit(spec, dry_run=dry_run, approved=approved)
    except PermissionError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"{'rendered' if dry_run else 'submitted'} {script} -> {job}")


@slurm.command("submit-stages")
@click.option("--tiles", default="GM1,AP5", show_default=True)
@click.option("--stages", default=",".join(STAGES), show_default=True)
@click.option("--optimizer", default="v1", show_default=True)
@click.option("--dry-run", is_flag=True)
@click.option("--approved", is_flag=True)
def slurm_submit_stages(tiles, stages, optimizer, dry_run, approved):
    """Optimizer regression: one serc job per (tile, stage), time = 1.5 x the v1 stage time (addendum 2, section 3)."""
    import json as _json

    from lemcvt.slurm import submit_stages

    try:
        jobs = submit_stages(_parse_tiles(tiles), tuple(s.strip() for s in stages.split(",")), optimizer=optimizer,
                             dry_run=dry_run, approved=approved)
    except (PermissionError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    for j in jobs:
        click.echo(f"{'rendered' if dry_run else 'submitted'} {j['tile']} {j['stage']} time={j['time']} -> {j['job_id']}")
    click.echo(_json.dumps(jobs, indent=1))


if __name__ == "__main__":  # pragma: no cover
    main()
