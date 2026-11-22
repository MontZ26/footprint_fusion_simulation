# Data
* `icos/` - nine ICOS Atmosphere Thematic Centre near-real-time hourly CO2 files (stations CBW, CMN, IPR, JFJ, LMP, PAL, PRS, SAC, SSL), as downloaded from the ICOS Carbon Portal (https://data.icos-cp.eu). ICOS data are released under CC BY 4.0; please cite the ICOS ATC NRT CO2 product and the station PIs listed in each file header.
* Beijing multi-site air quality (Zhang et al. 2017): NOT stored here. Run `python fetch_beijing.py` from the repository root; it downloads the UCI archive (https://archive.ics.uci.edu/dataset/501) into `bj_data/` (CC BY 4.0). If the download fails, download the zip by hand, unzip it, and place the twelve `PRSA_Data_*.csv` files in `bj_data/`.
* Forest and other experiments: see `forest/README_forest.md`.
