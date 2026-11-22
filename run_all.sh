#!/bin/sh
# Reproduce all temporal results. ICOS *.CO2 files in icos_data/, Beijing PRSA_Data_*.csv files in bj_data/
set -e
cd temporal
ICOS_DIR=../icos_data python temporal.py 30
python bootstrap_calendar_blocks.py
python ccr_validation.py
python ccr_full_validation.py
ICOS_DIR=../icos_data python temporal_extended.py
ICOS_DIR=../icos_data python diagnostics_mapping.py
python rho_sweep.py
BJ_DIR=../bj_data python analyze_beijing.py NO2 log
BJ_DIR=../bj_data python analyze_beijing.py PM2.5 log
BJ_DIR=../bj_data python analyze_beijing.py NO2 raw
