# Forest heterogeneous-source experiment (Paper T) - how to run
The cloud sandbox cannot reach the data services, so steps 1-3 run on YOUR computer. `forest_hetero.py` was tested only on synthetic data (`make_synthetic_forest.py`); `fetch_modis_evi.py` and `build_forest_csv.py` are UNTESTED against the live files, so check the first site by eye.
1. Flux data: download daily (DD) FLUXNET-format files for forest sites (FLUXNET2015 at fluxnet.org, free registration, CC-BY 4.0; or the ICOS Carbon Portal ecosystem L2 product, licence accepted on the portal). Put them in one folder. Never put passwords in scripts or chat.
2. Satellite: `python fetch_modis_evi.py sites.csv 2015 2019` (no account; edit sites.csv to the sites you downloaded; check coordinates).
3. `python build_forest_csv.py <flux folder> <evi folder>` -> `forest_daily.csv`.
4. `python forest_hetero.py forest_daily.csv 1` -> table of rules (Dempster, cautious, mean, CDD), agreement, mean error correlation, block bootstrap; writes `forest_results.csv`. Needs numpy, pandas, scipy and `../temporal/temporal_extended.py`.
Sources: in situ (7-day mean of uptake), satellite (regression on EVI), meteorological (regression on air temperature and radiation), each trained on the trailing 90 days only.
Caveat to state in the paper: FLUXNET temperature/radiation are tower measurements; for an independent driver use reanalysis (e.g. Open-Meteo/ERA5 daily) instead.

## Version used in the paper (v1.1.2)
The paper's forest results use the ICOS FLUXNET product v1.3 r1 (daily tower file FLUXMET_DD and daily ERA5 file ERA5_DD), downloaded with the FLUXNET Shuttle. The satellite (MODIS) source in fetch_modis_evi.py was not used because the NASA/ORNL service was unreachable; the third source is ERA5 weather.
Run: python build_forest_csv.py flux none era5 ; python forest_hetero.py forest_daily.csv 1 > forest_output.txt ; python forest_persite.py forest_results.csv
forest_results.csv.gz holds the origin-level results of the run in the paper (unzip with gunzip -k).
