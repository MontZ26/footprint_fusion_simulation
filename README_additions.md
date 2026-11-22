# Additions for the public repository (real-data and temporal analyses)

Companion to the paper *Evidential Fusion of Dependent Environmental Evidence*.
Add this folder to https://github.com/MontZ26/footprint_fusion_simulation (suggest release v1.1.0, MIT licence).

## Data (not redistributed here, except the derived Madrid table)
| Data | Source | Licence |
|---|---|---|
| ICOS ATC near-real-time CO2, stations SAC, CBW, JFJ, CMN, PRS, LMP, PAL, SSL, IPR (hourly `*.CO2` files) | ICOS Carbon Portal | CC BY 4.0 |
| OCO-2 Lite XCO2, `OCO2_L2_Lite_FP` v11.3r | NASA GES DISC (Earthdata login) | NASA open data |
| Madrid hourly NO2 and Sentinel-5P TROPOMI NO2 | Madrid City Council open data, Copernicus | open |
`madrid/loso.csv` is our derived leave-one-station-out table (5,488 rows).

## Environment
Python 3.11 or later: numpy, pandas, scipy. No other dependencies.

## Reproduce
```
# 1 temporal test (temporal tables of Section 16): put the ICOS *.CO2 files in ./icos_data
cd temporal
ICOS_DIR=../icos_data python temporal.py 30      # window W=30; also try 20 and 45 (supplement table)
python bootstrap_calendar_blocks.py              # 14-day calendar-block bootstrap, 45 blocks, seed 2
# 2 real-data spatial tests
cd ../real_data
ICOS_DIR=../icos_data python pilot.py
OCO2_CSV=/path/to/OCO2_near_ICOS.csv python full_real.py
python misspec.py ; python madrid_evid.py ; python dep_sim.py
```
Fixed seeds are set inside each script. Expected temporal output for W=30: 937 origins, 9 stations; over-confidence Dempster +0.117, cautious rule +0.040, mean of masses -0.026; accuracy about 0.75 for every rule. `temporal/temporal_results_W30.csv` holds the per-origin output.

## Files
- `temporal/temporal.py`: estimators (past, present, forecast), point fusion (mean, BLUE, covariance intersection, BLUE with estimated and shrinkage covariance), evidential rules (Dempster, Yager, cautious, mean of masses, Murphy, independent Bayes), diagnostics (condition number, weights).
- `temporal/bootstrap_calendar_blocks.py`: bootstrap intervals with 14-day blocks shared by all stations.
- `real_data/*`: pilot on OCO-2 vs ICOS (29 station-days), full pipeline on real pixels, misspecified noise, Madrid replication, dependence simulation, and their result tables.
- `colocated_days.csv`: the matched OCO-2 / ICOS station-day table.
The OCO-2 extraction scripts (`oco2_extract.py`, `oco2_extract2.py`) are in the outputs folder of the project and should be added next to these files.

## Not included
The platform function `temporal_dst_3way()` is part of the platform codebase, not of this repository; the paper's temporal results come from `temporal/temporal.py`.

## CCR validation (added)
`temporal/ccr_validation.py` compares the implementation in `temporal/temporal_rules.py` with a general cautious conjunctive rule (commonalities, canonical weights); max |BetP difference| over 20,000 random cases = 0.

## Extended temporal analysis (review 5)
`temporal/temporal_extended.py` (mass vectors, calibration, horizons 1/3/7, per-station, leave-one-station-out; `ICOS_DIR=<folder with nine *.CO2 files>`), `temporal/ccr_full_validation.py` (complete mass-vector check, sequential vs simultaneous), `temporal/rho_sweep.py` (controlled dependence sweep). Outputs: `ext_*.csv`, `ext_output.txt`, `rho_sweep_output.txt`.

`temporal/diagnostics_mapping.py`: Gaussian-mapping diagnostics, cap and Student-t mapping sensitivity, binning sensitivity (output: diagnostics_output.txt, ext_mapping_sensitivity.csv).

## Second dataset (Beijing air quality)
Data: UCI repository #501 "Beijing Multi-Site Air-Quality Data" (Zhang et al. 2017), 12 stations, hourly, 2013-03-01 to 2017-02-28. Put the twelve `PRSA_Data_*.csv` files in `bj_data/`.
```
BJ_DIR=bj_data python temporal/analyze_beijing.py NO2 log     # primary
BJ_DIR=bj_data python temporal/analyze_beijing.py PM2.5 log   # secondary
BJ_DIR=bj_data python temporal/analyze_beijing.py NO2 raw     # sensitivity
```
Outputs: `beijing_*_summary.csv`; console output saved in `beijing_*.txt`. Runtime about 90 s each.

## Round-6 additions
- `temporal/reviewer_stats.py`: direction agreement, decision differences, conflict quantiles, prevalence, per-station error correlations, reliability bins for ICOS and Beijing (NO2, PM2.5); output `reviewer_stats_output.txt`, `corr_by_station_*.csv`. Run: `ICOS_DIR=icos9 BJ_DIR=bj_data python reviewer_stats.py`.
- `temporal/rho_sweep_seeds.py`: controlled sweep over five seeds (output `rho_sweep_seeds_output.txt`).
- `temporal_extended.py` now also stores the signed differences `d0..d2`.
- `temporal/bj_block_sens.py`: Beijing block counts and 7/14/28-day block sensitivity (`bj_block_sens_output.txt`).
- `temporal/cdd_test.py`: correlation-discounted Dempster rule (weights raised to 1/(1+(k-1)*rho_bar)); output `cdd_test_output.txt`.
- `forest/`: heterogeneous-source forest experiment (pipeline tested on synthetic data only; fetch/build scripts untested against live data; see forest/README_forest.md).
