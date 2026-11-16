# Addendum 1 to the v2 refactor directive (2026-11-09)

This addendum overrides the directive wherever the two conflict.

## Framing

This is a ground-up rework of the analysis, not preparation of a response to reviewers.

- **Drop reviewer- and letter-facing deliverables.** This includes the response-letter paragraph in Phase 6, and the manuscript, response-letter and game-plan rows of `corrections_table.md`.
- **Keep the old-versus-new comparison of every v1 number.** It is the internal record of what changed and why.
- **Judge every choice on scientific grounds**, not on whether it preserves a published value.

## Measured parameters: rebuild from scratch (replaces Phase 5.3 and Phase 0 step 6's consequence)

Steve has placed some old notebooks in `old-notebooks/`. Read them for context on what was done before. They are not a specification, and the v1 hard-coded values are not a target. There is no 5% gate. Design the measurement pipeline independently.

**Requirements**

1. **Purpose.** `lemcvt measure` produces θ, k_s, hilltop curvature C_ht and D₀ = U/C_ht for every tile, from the tile's own 1 m grid (z in metres), using no model output.
2. **Consistency with the models.** k_s must be defined with the same exponent and reference area the models use: m = θ, n = 1, A₀ = 1 m², and the same D8 drainage-area grid. U/k_s is then exactly the model's K₀.
   - Prefer χ–elevation for both θ and k_s (k_s = dz/dχ at the chosen θ; Perron & Royden 2013, Mudd et al. 2014).
   - Report binned slope–area as a cross-check.
3. **Concavity.** Choose θ objectively, for example by χ collinearity across tributaries, or by the best χ–z fit over a θ grid. Report the misfit curve.
   - State and justify the drainage-area threshold for channel pixels.
   - Show the sensitivity of θ and k_s to that threshold.
4. **Hilltop curvature.** Define hilltop pixels explicitly (for example, divides from the D8 network, or a curvature-based ridge definition after Hurst et al. 2012) and document the definition.
   - Curvature depends strongly on the smoothing or fitting scale. Choose the scale by an objective criterion, such as the window at which median hilltop curvature stops changing with window size, and show the curve.
   - Do not simply carry over the v1 block-mean sizes (30 m AP, 10 m GM).
   - Use the same criterion at both sites.
5. **Uncertainty.** Give every measured parameter an uncertainty: a bootstrap over channel segments for θ and k_s, and over hilltop pixels for C_ht. Later fidelity comparisons will need it.
6. **Comparisons, reported but not acted on:**
   - against the v1 hard-coded values (AP values compared after the 3.2808 conversion);
   - against Perron et al. (2012) regional values;
   - against what the `old-notebooks/` methods give, where they can be run.

   Explain the main differences.
7. **Documentation.** Write `docs/measurement_methods.md`: methods, choices, justifications and references, in enough detail for a Methods section.

## Sequencing

- **Measurement work can start in Phase 1 alongside the refactor.** It doesn't depend on reproducing v1.
- **Run it on corrected (metre) AP grids from the start.** There is no reason to measure in feet.
- **Phase 6 uses the new measured values.** The v1 values are used only in Phase 2 regression.
