# [2026-11-16 12:45 PST] Addendum 11 (final): the corrections list, then the v2.0.0 release

This addendum overrides the directive and Addenda 1–10 wherever they conflict. Addendum 4 §1 (timestamped titles, one report per checkpoint), Addendum 2 §4–§5 (Slurm, storage, push) and Addendum 7 §0.4 (submit guard) still apply.

## 0. Orientation (you are a new agent instance)

1. **Read** the directive and Addenda 1–10 in `reports/directives/`, this file, and `reports/v2/20261116T0921PST_wrapup.md` (the latest state).
2. **Confirm** that `v2` equals `origin/v2`, that the tests pass on `serc`, and that no job is running.
3. **Where this addendum and the repo disagree,** the repo wins. Report it and stop.

## 1. Decisions

1. **Release approved.** Steve approves merging `v2` into `main` and tagging `v2.0.0`, provided §2 is done first and the gates in §3 pass. No further approval is needed.
2. **The flux-form result stands as "not preferred on GM".** No more nonlinear runs.
3. **CI 3.11:** keep the failure-annotation hook. Do not drop Python 3.11.

## 2. The corrections list

### 2.1 Source files

The manuscript and supplement are in the repo's `papers/` folder, not in `reports/inputs/`. Steve named `Dobbs_Hilley_JGR-ES_Supplement_JGRformatted.docx` plus the manuscript.

1. **List every `.docx` in `papers/`,** with name, size and modified date.
2. **Pick the files.** Use the most recent manuscript and the most recent supplement, the JGR-formatted versions where they exist. State which two you used.
3. **If there is no recognisable manuscript or no recognisable supplement,** stop and report the listing.
4. **Read only.** Do not edit, copy or move them.
5. **Never commit them.** `papers/` must stay untracked; confirm it is in `.gitignore`, and add it if not. The pre-push check of Addendum 2 §5 applies to every push in this session.

### 2.2 Build it

Run `docx_numbers.py` and `make_corrections_table.py` (Addendum 10 §6) on both files, with tracked changes resolved to the current text. Write `reports/v2/corrections_table.md`, plus a CSV beside it.

One row per number printed in the manuscript or supplement, with these columns:
- location: section, paragraph or table, figure caption;
- the printed text;
- v1 value;
- v2 value;
- ratio;
- cause(s);
- source run;
- status.

Status is one of:
- `unchanged`: within printed precision;
- `update`: new value;
- `relabel or drop`: every nonlinear-stage number, marked "centre form, nonstandard";
- `reword`: the number survives but the claim around it changes (see §2.3);
- `no match`: a printed number the tool cannot tie to any v1 or v2 quantity. List these and do not guess.

Where a printed number matches several v1 values, list every candidate and mark the row `ambiguous`. Steve resolves those by hand.

### 2.3 Claims that change in substance (flag, do not rewrite)

Add a short section at the top of `corrections_table.md` listing every sentence that makes one of these claims, quoted with its location:

1. **The partitioned model recovers or retains measured D.** On GM, D is now 1.9–2.8× D₀; on AP it depends on the curvature window.
2. **K is matched to an independent or field measurement.** It agrees with the χ-based estimate from the same channels.
3. **Any nonlinear-stage result.** Centre form, nonstandard.
4. **AP numbers** that were in feet, or that came from the v1 AP measurements.
5. **Optimizer-driven values** that changed by more than 10%: the GM1 AD optimum, the GM2 stage-2 A_crit, and the joint A_crit on GM1, GM3 and GM4.
6. **The D∞ partitioned MSEs.**

Do not edit the manuscript or propose new wording.

## 3. Release

Run only after §2 is committed.

1. **Gates before the merge:**
   - the tests pass on `serc`;
   - the latest CI run on `v2` is green on 3.11 and 3.12;
   - the final build in `$GROUP_HOME/lemcvt_results/v2.0.0/` was made at the current `v2` code. If any code changed since, rebuild and refresh `GIT_HASH` and `SHA256SUMS`;
   - the pre-push check is clean.
2. **Merge** `v2` into `main` with a merge commit, no rebase and no squash.
3. **Tag** `v2.0.0` on the merge commit, as an annotated tag. The message names the final-build hash and this addendum.
4. **Push** `main` and the tag.
5. **Check** that the CI run on `main` is green.
6. **Check whether the Zenodo GitHub integration minted a DOI** for `v2.0.0` under the existing concept DOI. Report it if it did. If it did not (or the integration isn't set up), say so: Steve will publish the release on Zenodo by hand.
7. **If any gate fails,** do not merge. Report it and stop.
8. **Leave `v2`, `v1.0.0` and `v1.0.1` in place.** Delete nothing.

## 4. Report and stop

One report, `reports/v2/<YYYYMMDDTHHMM>PST_release.md`, titled with its timestamp. Fixed outline:

1. State check.
2. Source files used.
3. Corrections list summary: rows by status, the substance flags, and the `ambiguous` and `no match` counts.
4. Gates.
5. Merge, tag and CI on `main`.
6. Zenodo.
7. What is left for Steve.

**Stop.** This ends the v2 refactor directive.
