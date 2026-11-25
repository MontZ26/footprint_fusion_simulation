RASTER EXPERIMENT SPEC v2 (FROZEN before any raster relaxation result is examined)
Three-way relaxation comparison: probabilistic / possibilistic / evidential. Supersedes RASTER_SPEC.md (v1).
Status rule: after the first validation run nothing in this document changes; any deviation is logged in a dated "Deviations" list at the end.

1 STUDY PERIOD, DATES AND SPLIT
- 236 panel dates of the Madrid LOSO study, 2025-01-01 to 2026-02-28; 24 stations.
- Split by date: train days 1-70 (to 2025-06-04), validation days 71-140 (2025-06-05 to 2025-08-20), final days 141-236 (2025-08-21 to 2026-02-28).
- Evaluation cases = the 5,488 LOSO station-date cases with ground truth (train 1,645; validation 1,657; final 2,186). Same cases, stations and truth for all methods.

2 TROPOMI SOURCE
- Earth Engine COPERNICUS/S5P/OFFL/L3_NO2, band tropospheric_NO2_column_number_density (mol/m2); column to surface: c = column*46.005e6/200 (ug/m3), as in the published Madrid study.
- Scene metadata (5,865 scenes, 2024-12-31 to 2026-02-28): HARP 1.5, L3 processing version 20200414, nominal resolution 5.5x3.5 km2, status "OFFL-processing nominal product".
- Processor 2.8.0 / algorithm 1.7.0 through 2025-11-21 (4,469 scenes); processor 2.9.1 ("02.09.01") / algorithm 1.8.0 from 2025-11-22 (1,395 scenes). One scene 20251221 is "03.00.DV" (algorithm 1.8.0; not a panel date). One scene (orbit starting 20250621T220447) has PRODUCT_QUALITY = DEGRADED.
- Data files (provenance, sha256 first 12 characters): tropomi_station_windows.csv b3dcaa37d1af (1,264,624 bytes, 5,664 rows); tropomi_scene_metadata.csv 2434eee1944c; old station file TROPOMI_NO2_2025_2026.csv 639e6861b287.

3 GRID, STATION WINDOWS, SPATIAL SUPPORT
- Export: one reference native grid (nominal scale 1113.19 m); per date the mean of the valid same-day images; 3x3 window around each station (columns v_dx_dy, dx and dy in {-1,0,1}). Missing stays missing; no interpolation across dates.
- A window is a gridded, oversampled neighbourhood, NOT nine independent measurements (true footprint about 5.5x3.5 km2): hence cautious pooling and the dependence-aware rule.
- Sample flow: 5,664 station-date rows = 19 completely empty windows (the old file has no value for any) + 27 windows with some cells but no old value (23 of the 24 centre-empty windows are among them) + 5,618 comparable rows (one has an empty centre cell). 47 windows with a valid centre have some empty neighbour cells. All 5,488 evaluation cases are among the comparable rows.

4 CENTRE-VALUE PROVENANCE AND ALIGNMENT DIAGNOSTIC
- Primary centre evidence = the old station-level TROPOMI value (mean of the 1-2 same-day images sampled at the station point), exactly as in the published results. The exported v_0_0 is NOT used as centre evidence.
- Diagnostic (exact match to 1e-9 between the old value and the cells of the window, 5,618 comparable rows): centre cell matches (alone 1,307; tied with off-centre cells 2,671) = 3,978 (71%); unique off-centre match 1,188 (21%); off-centre ties with the centre not matching 361 (6%); no cell matches 91 (1.6%, all two-image days).
- For unique off-centre matches the direction is nearly constant within a station (e.g. station 57 east 73/73, station 55 east 64/64, station 47 south 70/70, station 11 west 34/34), but occurs only on a subset of days. Consistent with a day-dependent grid-alignment effect between the images' own grids and the reference grid; the export has no pixel coordinates or identifiers, so the cause is NOT confirmed. Reproduced by alignment_diag.py.
- Terminology: the cells other than the centre are called "the remaining cells of the station-centred 3x3 window", not "eight neighbours", because the matching cell need not be v_0_0.

5 CONFIGURATIONS
- N (Nominal, PRIMARY): centre evidence = old value; spatial evidence = the eight cells other than v_0_0.
- A (Match-aligned, prespecified sensitivity): centre evidence = old value; spatial evidence = the cells other than the date-specific matching cell when it is unique (centre-only matches: v_0_0; unique off-centre: that cell). Ambiguous off-centre ties (361) and no-match cases (91), 452 in total, use centre-only evidence (no spatial update). Rows whose centre matches together with ties: see D.
- D (Duplicate-value exclusion, prespecified sensitivity): where v_0_0 and one or more other cells equal the old value (2,671 rows), all matching cells are excluded from the spatial evidence; other rows as in A.
- N is the only configuration used for model selection and for the headline results; A and D are reported as sensitivity analyses and are never used to pick the configuration with the best final-period score.

6 MISSING DATA AND QUALITY POLICY
- A missing cell contributes the vacuous BBA m(Theta)=1 (no concentration at zero, no artificial class). No minimum count of valid cells is required in the primary analysis; sensitivities: at least 4, at least 6 valid cells in the window.
- Completely empty windows and missing centre values are reported separately in the sample-flow table (section 3).
- Primary quality policy (fixed now): use the old centre values as published (all scenes of the collection, no extra QA filter). Sensitivity: drop station-dates of the panel dates 2025-06-21 and 2025-06-22 (DEGRADED scene) and recompute.

7 CLASSES AND CALIBRATION
- Four classes <15, 15-30, 30-45, >=45 ug/m3; centroids from training days only.
- Satellite-to-surface calibration, days 1-70 only, satellite only (no same-day ground at test time): ground = a + b*c_centre (c_centre from the old centre value); residual RMSE s_cal from the training days.
- Initial pixel evidence: one possibility contour (eta fixed in advance) giving probability / possibility / consonant BBA for the three families.

8 SPATIAL COMPATIBILITY
- R(k,l) = exp(-(c_k - c_l)^2 / (2 s^2)), s = s_cal (not the 6.95 of the fusion study); all parameters from training days only.

9 DEFINITION OF rho_sp (spatial error dependence proxy)
- What it is: a station-level proxy for spatial error dependence from the 24 ground stations; NOT a measurement of error correlation between adjacent TROPOMI pixels.
- Residuals: e(s,t) = truth(s,t) - calibrated satellite value(s,t), calibration out-of-fold (training days split in the ten 7-day blocks of days 1-70, calibration refitted leaving one block out).
- Station pairs: each station paired with its eight nearest stations (neighbours.json of the possibilistic study), unordered pairs de-duplicated; pairs need at least 30 common training days.
- Per-pair Pearson correlation; combined by Fisher-z averaging weighted by (n-3), back-transformed, clipped to [0,1]; missing pairs are dropped (not imputed).
- Uncertainty: block bootstrap of the 7-day blocks (stations kept together); report estimate, 95% interval and number of pairs.
- Use: reported as a diagnostic; also the parameter of the CDD-evidential method, lambda = 1/(1+(k-1)*rho_sp) with k = 1 + number of valid cells (heuristic, exact only for equicorrelated Gaussian errors, not duplicate-invariant). The cautious primary does not need rho_sp.

10 DEFINITION OF rho_proj (projection reliability)
- rho_proj in {0.5, 0.8, 0.95}: reliability of the evidential compatibility projection (discounting of projected neighbour evidence), common to all evidential variants, chosen on the VALIDATION Brier score of BetP.
- There is no formula linking rho_proj to rho_sp; they are never interchanged.

11 METHODS AND COMMON CASES
- No relaxation; Rosenfeld probabilistic; possibilistic (frozen v1 specification, no new tuning); evidential primary (cautious pool + cautious centre, K=4, evid_k.py); evidential comparators: Dempster pool; cautious pool + Dempster centre; CDD-evidential.
- Optional: Frank t-norm family on canonical weights, s in {0, 0.1, 0.5, 1} on validation, only values giving a valid belief function.
- All methods are scored on the same station-date pairs.

12 METRICS AND UNCERTAINTY
- Accuracy, multiclass Brier and ECE of the normalised BetP always with m(empty) reported; coverage vs set size; risk-coverage; conflict frequency; fraction of windows with m(empty) above the validation threshold.
- Paired 7-day block bootstrap keeping all stations of a date together; report number of pairs, unique dates, stations and blocks. The 24 stations are not 24 independent replications.
- Sensitivity analyses: configurations A and D; at least 4 / 6 valid cells; degraded-scene exclusion; Frank family; kriging-based fusion baseline if time allows.

13 LIMITS AND REMAINING UNCERTAINTIES
- Processor change: all 41 panel days from 2025-11-22 are in the final period, so the final evaluation mixes temporal generalisation and a change of the satellite processing chain; the two cannot be separated. A descriptive comparison of satellite-to-ground residual bias and dispersion before/after 2025-11-22, overall and by station, will be reported AFTER the primary results are fixed and will not be used to modify any prediction.
- Alignment cause not confirmed (no pixel coordinates in the export).
- Column vs surface mismatch; daytime overpass vs daily mean; 1.1 km grid vs 3.5x5.5 km footprint; calibration with 24 stations; station-level rho_sp proxy.
- Possibilistic relaxation was a negative result in the earlier study; it enters here with its frozen specification only.

DEVIATIONS (dated list; empty at freezing)
