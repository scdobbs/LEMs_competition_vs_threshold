"""Test 1 of the JGR-ES resubmission: does the optimized A_crit of the
threshold-partitioned model follow the Montgomery & Dietrich inverse
area-slope relation across the ten tiles?

Measurement-only package (reports/directives/acrit_slope_test1_directive.md).
It reads the repo's existing filled DEM, D8 area and D8 flow-direction grids,
reuses the flow-graph helpers of ``width_analysis``, and writes everything to
``outputs/acrit_slope/``. No model code, optimization result or figure is
touched.

    python -m acrit_slope.run
"""
