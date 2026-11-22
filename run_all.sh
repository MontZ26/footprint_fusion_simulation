#!/bin/sh
# Reproduce all temporal results from the repository root.
# Needs: ICOS files in data/icos (included), Beijing files in bj_data (run `python fetch_beijing.py` first).
set -e
[ -d bj_data ] && [ "$(ls bj_data/PRSA_Data_*.csv 2>/dev/null | wc -l)" -ge 12 ] || { echo "bj_data/ missing: run 'python fetch_beijing.py' first (see data/README_data.md)"; exit 1; }
cd temporal
export ICOS_DIR=../data/icos
export BJ_DIR=../bj_data
python temporal.py 30
python bootstrap_calendar_blocks.py
python ccr_validation.py
python ccr_full_validation.py
python temporal_extended.py
python diagnostics_mapping.py
python rho_sweep.py
python rho_sweep_seeds.py
python analyze_beijing.py NO2 log
python analyze_beijing.py PM2.5 log
python analyze_beijing.py NO2 raw
python reviewer_stats.py
python bj_block_sens.py
python cdd_test.py
