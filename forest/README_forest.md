# Forest heterogeneous-source experiment (Paper T) - how to run
The cloud sandbox cannot reach the data services, so steps 1-3 run on YOUR computer. `forest_hetero.py` was tested only on synthetic data (`make_synthetic_forest.py`); `fetch_modis_evi.py` and `build_forest_csv.py` are UNTESTED against the live files, so check the first site by eye.
1. Flux data: download daily (DD) FLUXNET-format files for forest sites (FLUXNET2015 at fluxnet.org, free registration, CC-BY 4.0; or the ICOS Carbon Portal ecosystem L2 product, licence accepted on the portal). Put them in one folder. Never put passwords in scripts or chat.
2. Satellite: `python fetch_modis_evi.py sites.csv 2015 2019` (no account; edit sites.csv to the sites you downloaded; check coordinates).
3. `python build_forest_csv.py <flux folder> <evi folder>` -> `forest_daily.csv`.
4. `python forest_hetero.py forest_daily.csv 1` -> table of rules (Dempster, cautious, mean, CDD), agreement, mean error correlation, block bootstrap; writes `forest_results.csv`. Needs numpy, pandas, scipy and `../temporal/temporal_extended.py`.
Sources: in situ (7-day mean of uptake), satellite (regression on EVI), meteorological (regression on air temperature and radiation), each trained on the trailing 90 days only.
Caveat to state in the paper: FLUXNET temperature/radiation are tower measurements; for an independent driver use reanalysis (e.g. Open-Meteo/ERA5 daily) instead.
