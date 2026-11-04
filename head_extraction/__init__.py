"""Test 2 of the JGR-ES resubmission: channel heads extracted independently of
any drainage-area threshold (Pelletier 2013 contour curvature; DrEICH via
LSDTopoTools), their areas and slopes, and where A_crit falls relative to them.

Directive: reports/directives/head_extraction_test2_directive.md.
Measurement-only package; reads the repo's existing tiles, reuses
``acrit_slope`` and ``width_analysis`` helpers, writes to
``outputs/head_extraction/``.

    python -m head_extraction.run --phase extract    # sections 2-4, then STOP for review
    python -m head_extraction.run --phase analyse    # sections 5-8, needs heads_review.csv
"""
