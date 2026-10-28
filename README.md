# Supplementary material: simulation study of classification before relaxation and evidential fusion

Everything here is self-contained and does not need the platform codebase.

| File | Purpose |
|---|---|
| exact.py | Loop implementation of the satellite path (composite score, classification, Rosenfeld relaxation, certainty) and of the fusion; the nine simulated readings; reproduces Tables 1-3 |
| fast.py, fast_alt.py | Vectorised batch implementation, validated against exact.py on the nine zones (fast.py `__main__`) |
| exp.py | Known-reference evaluation, four pipelines, 9 zones x 300 fields and 600 random zones x 40 fields, noise 0.05-0.30 |
| exp2b.py | Fusion with constant and idealised baselines, K-threshold x sigma_edgar grid, nine-zone Yager counts, confidence sweep |
| exp3.py | Paired bootstrap intervals, confusion matrices, switch-rule comparison (ECE, Brier), baseline scenarios, class-frame variant, noise-model and design-choice sensitivity |
| exp3_lib.py | Shared functions (noise models, pipeline, zone generator) imported by exp3.py and exp4.py |
| exp4.py | Class-frame variant across noise levels and baseline scenarios; comparison with reliability-weighted and fixed averages |
| binsens.py | Bin-edge sensitivity |
| reference/*.json | Reference outputs behind the manuscript tables |
| compare.py | Compares freshly generated results with reference/ |
| run_all.sh, requirements.txt | Run order and package versions |

## Fixed settings
Composite weights 0.40/0.35/0.25; ranges CO [0.020, 0.080], CH4 [1840, 1940], NO2 [1e-6, 8e-5];
bins [30, 60, 80]; centroids 15/45/70/90; initial confidence 0.85; compatibility width 1.5, rows normalised;
Moore neighbourhood, boundary-clipped; at most 20 iterations, tolerance 1e-4; grid 6x6; smoothing width 1.5;
CH4 noise scale 0.3; baseline 35.0; sigma_edgar 0.45; K threshold 0.5.

## Random seeds
Default noise field seed 42. Evaluation fields use seeds 1000+i (exp.py, exp3.py part A), 2000+i (fusion), 3000+i (sweep), 4000+i (noise/design),
shared by all pipelines (paired). Random-zone generators: default_rng(2026) for the 600-zone sets, default_rng(5) for the 400-zone sets.
Idealised baseline noise: default_rng(77). Other generators are seeded in the scripts.

## Provenance of simulated inputs
The nine gas readings in exact.py are the values produced by the platform's deterministic MD5-seeded proxy for 2026-10-27.
They are generated values, not observations. The baseline 35.0 and sigma_edgar = 0.45 are simulation parameters.

## Table-to-script map
| Manuscript tables | Script | Output |
|---|---|---|
| 1-3 | exact.py | printed |
| 4-5 | exp.py | exp1.json |
| 6-10, 13-15, 17, 19-21 | exp3.py | exp3.json |
| 11-12, 22 and the nine-zone Yager counts | exp2b.py | exp2.json |
| 16, 18 | exp4.py | exp4.json |
| bin-edge sensitivity (text, Section 8.2) | binsens.py | printed |

## Reproducing
`sh run_all.sh` regenerates all results and ends with `compare.py`, which reports whether each result file equals the reference. Run time is a few minutes on a laptop. Tested with the versions in requirements.txt.
