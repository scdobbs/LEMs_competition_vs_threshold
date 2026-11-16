#!/bin/bash
# Addendum 12: the production submissions (section 3: expected 234 core-hours, run 20261116T223818Z_a12_estimate_10tiles;
# every submission's limits <= 300 core-hours; every array throttled). Run from the repository on a login node.
set -eu
export PYTHONNOUSERSITE=1 MAMBA_ROOT_PREFIX=$GROUP_HOME/micromamba_root LEMCVT_DATA=$GROUP_HOME/lemcvt_data/v1 LEMCVT_RESULTS=$GROUP_HOME/lemcvt_results
LC="$GROUP_HOME/scdobbs-envs/bin/micromamba run -p $GROUP_HOME/scdobbs-envs/lemcvt lemcvt"
ALL=AP1,AP2,AP3,AP4,AP5,GM1,GM2,GM3,GM4,GM5
# section 2 (site m): partitioned variants with the greedy basin halves; AD variants and AD held-out
$LC slurm submit custom --labels $ALL --command 'lemcvt -v a12 equal-part --tile $LABEL' --time 10:00:00 --mem 8G --job-name a12_eqpart
$LC slurm submit custom --labels $ALL --command 'lemcvt -v a12 equal-ad --tile $LABEL' --time 05:00:00 --mem 8G --job-name a12_eqad
# section 1 (m_t and K0(m_t) from the a12_inputs run 20261116T224609Z_a12_inputs_10tiles, already done)
$LC slurm submit custom --labels $ALL --command 'lemcvt -v a12 mt --tile $LABEL --stage ad' --time 03:00:00 --mem 8G --job-name a12_mt_ad
$LC slurm submit custom --labels AP1,AP2,AP3,AP4,AP5 --command 'lemcvt -v a12 mt --tile $LABEL --stage partitioned' --time 1-06:00:00 --mem 8G --job-name a12_mt_part_AP
$LC slurm submit custom --labels GM1,GM2,GM3,GM4,GM5 --command 'lemcvt -v a12 mt --tile $LABEL --stage partitioned' --time 10:00:00 --mem 8G --job-name a12_mt_part_GM
