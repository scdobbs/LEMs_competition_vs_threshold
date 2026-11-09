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
def data_manifest(root: Path | None) -> None:
    """Build data/manifest.csv from the grids under LEMCVT_DATA (sha256 of every .tif)."""
    from lemcvt.config import data_root
    from lemcvt.io.grids import sha256_file
    from lemcvt.io.manifest import MANIFEST_PATH, ManifestEntry, write_manifest

    root = root or data_root()
    entries = []
    for p in sorted(root.rglob("*.tif")):
        rel = p.relative_to(root)
        parts = rel.parts
        site, tile = parts[0], parts[1]
        kind = "output" if "outputs" in parts else "input"
        role = p.stem.split("_tile_", 1)[-1]
        entries.append(ManifestEntry(str(rel), sha256_file(p), p.stat().st_size, site, tile, kind, role))
    write_manifest(entries)
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
@click.option("--run-id", default=None)
def tile_run(tile_id, stages, optimizer, no_grids, no_derived, exploration, crop, run_id) -> None:
    from lemcvt.analyses.tile_pipeline import run_tile

    rec = run_tile(tile_id, stages=[s.strip() for s in stages.split(",")], optimizer=optimizer,
                   save_grids=not no_grids, derived_grids=not no_derived, exploration_runs=exploration,
                   crop=_parse_crop(crop), run_id=run_id)
    click.echo(f"results: {rec.dir}")


@tile.command("forward")
@click.option("--tile", "tile_id", required=True)
@click.option("--params", default="v1", show_default=True, help="'v1' or path to a tile_run params.json")
@click.option("--compare", is_flag=True, help="Compare with the saved v1 grids (Phase 2.1 regression).")
@click.option("--keys", default=None, help="Comma list of run keys (ad_no_opt,ad_opt,acrit_opt,nonlinear_opt,joint_opt)")
@click.option("--no-grids", is_flag=True)
@click.option("--derived", is_flag=True, help="Also write the derived grids.")
@click.option("--crop", default=None)
@click.option("--run-id", default=None)
def tile_forward(tile_id, params, compare, keys, no_grids, derived, crop, run_id) -> None:
    from lemcvt.analyses.forward import run_forward

    rec = run_forward(tile_id, params=params, compare=compare, save_grids=not no_grids, derived_grids=derived,
                      keys=keys.split(",") if keys else None, crop=_parse_crop(crop), run_id=run_id)
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
                                           "dinf_diagnostics", "custom"]))
@click.option("--tiles", default="all", show_default=True)
@click.option("--labels", default=None, help="custom step: comma list of array labels (default: one task 'all')")
@click.option("--command", "command_", default=None, help="custom step: command with $LABEL, e.g. \"lemcvt -v slope-area --tiles AP1,GM2\"")
@click.option("--stages", default=",".join(STAGES), show_default=True, help="tile_run only")
@click.option("--optimizer", default="v1", show_default=True)
@click.option("--params", default="v1", show_default=True, help="tile_forward only")
@click.option("--compare", is_flag=True, help="tile_forward only")
@click.option("--routing", default="d8,dinf", show_default=True, help="sweep_resolution only")
@click.option("--time", "time_", default=None, help="Override wall time (D-HH:MM:SS).")
@click.option("--cpus", default=None, type=int)
@click.option("--mem", default=None)
@click.option("--dry-run", is_flag=True, help="Render and print the script and estimate; do not submit.")
@click.option("--approved", is_flag=True, help="Steve approved a submission above the core-hour limit (300).")
def slurm_submit(step, tiles, labels, command_, stages, optimizer, params, compare, routing, time_, cpus, mem, dry_run,
                 approved):
    from lemcvt.slurm import build_spec, estimate_text, render, submit

    labels = _parse_tiles(tiles) if step != "custom" else ([s.strip() for s in labels.split(",")] if labels else ["all"])
    if step == "custom":
        if not command_:
            raise click.BadParameter("custom step needs --command")
        cmd = command_
    elif step == "tile_run":
        cmd = f"lemcvt -v tile run --tile $LABEL --stages {stages} --optimizer {optimizer}"
    elif step == "tile_forward":
        cmd = f"lemcvt -v tile forward --tile $LABEL --params {params}" + (" --compare" if compare else "")
    elif step == "sweep_resolution":
        cmd = f"lemcvt -v sweep resolution --tiles $LABEL --routing {routing}"
    elif step == "closeout":
        cmd = "lemcvt -v closeout linearized --tile $LABEL"
    elif step == "dinf_diagnostics":
        cmd = f"lemcvt -v closeout dinf --tiles {tiles}"
        labels = ["all"]
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
