#!/usr/bin/env python3
"""Addendum 17 Part B: run `lemcvt.measure.pipeline.run_measure` on a list of tiles.

`lemcvt measure --tile` validates its tile names against the fixed ten v2 tiles (`cli._parse_tiles`), so the new tiles cannot be named
there; `tile run` has no such check. This wrapper calls the same `run_measure` (the `measure` step: rollover A_c, chi fits at A_c* and on
the ladder, slope-area cross-check, hilltop curvature) with the v1 legacy rerun off (its site tables know AP and GM only).
usage: a17b_measure_runs.py --tiles T1,T2,... --label a17b_t<k> [--n-boot 200]
The run id is <UTC timestamp>_measure_<label>; the label carries the array task id so that tasks starting in the same second cannot
collide (the first array of 2026-11-18 did: five tasks shared one directory).
"""
import argparse

from lemcvt.measure.pipeline import run_measure
from lemcvt.results import new_run_id


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tiles", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--n-boot", type=int, default=200)
    a = p.parse_args()
    rec = run_measure([t.strip() for t in a.tiles.split(",") if t.strip()], force_metres=True, n_boot=a.n_boot, run_legacy=False,
                      run_id=new_run_id("measure", a.label))
    print(f"results: {rec.dir}", flush=True)


if __name__ == "__main__":
    main()
