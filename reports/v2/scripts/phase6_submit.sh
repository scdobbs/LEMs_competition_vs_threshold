#!/bin/bash
# Addendum 6, section 5 (and 3): submit the Phase 6 arrays on serc with the dependencies, every submission <= 300
# core-hours, every array throttled (configs/slurm/sherlock.yaml), and print the job ids. Run from the repository
# root inside the lemcvt environment after the configs are committed. Usage: bash reports/v2/scripts/phase6_submit.sh
set -u
sub() { "$@" 2>&1 | tee -a "$SCRATCH/lemcvt_jobs/phase6_submit.log" | grep -oP "submitted .* -> \K\d+" | tail -1; }
L=lemcvt
echo "== Phase 6 submissions $(date -Is)" | tee -a "$SCRATCH/lemcvt_jobs/phase6_submit.log"
# 3.1 AD profile, all tiles (derived grids on: fig_slope_area reads AD-no-opt / AD-opt filled grids)
J_AD=$(sub $L slurm submit phase6_ad --tiles all --job-name p6_ad)
# 3.2 linear partitioned: thin + refine + 5 % fill, per site (stage 2 at D0_A and D0_B from the same reductions)
J_PT_AP=$(sub $L slurm submit phase6_partitioned --tiles AP --job-name p6_part_AP --time 20:00:00)
J_PT_GM=$(sub $L slurm submit phase6_partitioned --tiles GM --job-name p6_part_GM --time 11:00:00)
# 3.3 nonlinear A_crit profile at D0_A (all) and at D0_B (AP)
J_NL_AP=$(sub $L slurm submit phase6_nonlinear --tiles AP --job-name p6_nl_AP --time 07:00:00)
J_NL_GM=$(sub $L slurm submit phase6_nonlinear --tiles GM --job-name p6_nl_GM --time 11:00:00)
J_NL_APB=$(sub $L slurm submit phase6_nonlinear --tiles AP --d0-variant B --job-name p6_nl_AP_D0B --time 07:00:00)
# 3.4 nonlinear joint (K from the refined partitioned run of the tile) and the K check
J_NLJ_AP=$(sub $L slurm submit phase6_nonlinear_joint --tiles AP --job-name p6_nlj_AP --time 19:00:00 --dependency afterok:$J_PT_AP)
J_NLJ_GM=$(sub $L slurm submit phase6_nonlinear_joint --tiles GM --job-name p6_nlj_GM --time 35:00:00 --dependency afterok:$J_PT_GM)
J_K_AP5a=$(sub $L slurm submit phase6_nonlinear_joint --tiles AP5 --k-factor 0.95 --job-name p6_nlj_AP5_k095 --time 19:00:00 --dependency afterok:$J_PT_AP)
J_K_AP5b=$(sub $L slurm submit phase6_nonlinear_joint --tiles AP5 --k-factor 1.05 --job-name p6_nlj_AP5_k105 --time 19:00:00 --dependency afterok:$J_PT_AP)
J_K_GM1a=$(sub $L slurm submit phase6_nonlinear_joint --tiles GM1 --k-factor 0.95 --job-name p6_nlj_GM1_k095 --time 35:00:00 --dependency afterok:$J_PT_GM)
J_K_GM1b=$(sub $L slurm submit phase6_nonlinear_joint --tiles GM1 --k-factor 1.05 --job-name p6_nlj_GM1_k105 --time 35:00:00 --dependency afterok:$J_PT_GM)
# 3.5 D-infinity joint on AP1 and GM1
J_DINF=$(sub $L slurm submit phase6_dinf_joint --tiles AP1,GM1 --job-name p6_dinf --time 15:00:00)
# 3.6 resolution sweep at the config parameters, D0_A on all tiles and D0_B on AP; supp_fig5 and d8_vs_dinf at the config parameters
J_SW=$(sub $L slurm submit sweep_resolution --tiles all --sweep-params config --job-name p6_sweep --time 01:00:00)
J_SWB=$(sub $L slurm submit sweep_resolution --tiles AP --sweep-params config --d0-variant B --job-name p6_sweep_AP_D0B --time 01:00:00)
J_UTIL=$(sub $L slurm submit custom --command "bash -c 'lemcvt -v synthetic pe && lemcvt -v d8-vs-dinf --tiles AP1,GM1 --params config --units metres --pt-dinf-model both'" --job-name p6_util --time 03:00:00 --mem 64G)
# addendum 6, 3: AP1 seam test
J_SEAM=$(sub $L slurm submit custom --command "lemcvt -v phase6 seam-test --tile AP1" --job-name p6_seamtest --time 18:00:00 --mem 16G)
# post-processing after everything: curvature comparison, slope-area at the new optima, figures, record, tables
DEPS=afterok:$J_AD:$J_PT_AP:$J_PT_GM:$J_NL_AP:$J_NL_GM:$J_NL_APB:$J_NLJ_AP:$J_NLJ_GM:$J_K_AP5a:$J_K_AP5b:$J_K_GM1a:$J_K_GM1b:$J_DINF:$J_SW:$J_SWB:$J_UTIL:$J_SEAM
J_POST=$(sub $L slurm submit custom --command "bash reports/v2/scripts/phase6_post.sh" --job-name p6_post --time 03:00:00 --mem 32G --dependency $DEPS)
echo "jobs: ad=$J_AD part=$J_PT_AP,$J_PT_GM nl=$J_NL_AP,$J_NL_GM,$J_NL_APB nlj=$J_NLJ_AP,$J_NLJ_GM kcheck=$J_K_AP5a,$J_K_AP5b,$J_K_GM1a,$J_K_GM1b dinf=$J_DINF sweep=$J_SW,$J_SWB util=$J_UTIL seam=$J_SEAM post=$J_POST" | tee -a "$SCRATCH/lemcvt_jobs/phase6_submit.log"
