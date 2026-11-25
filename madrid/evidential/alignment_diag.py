# Alignment diagnostic: where does the old station-level TROPOMI value sit in the exported 3x3 window?
# Usage: python alignment_diag.py tropomi_station_windows.csv TROPOMI_NO2_2025_2026.csv
import csv, sys, collections
import numpy as np
win_path, old_path = sys.argv[1], sys.argv[2]
names = ['v_-1_-1','v_0_-1','v_1_-1','v_-1_0','v_0_0','v_1_0','v_-1_1','v_0_1','v_1_1']
old = collections.defaultdict(list)
for r in csv.DictReader(open(old_path, encoding='utf-8-sig')):
    old[(r['code'], r['date'])].append(float(r['mean']))
flow = collections.Counter(); cat = collections.Counter()
pos = collections.defaultdict(collections.Counter); nomatch_n = collections.Counter()
for r in csv.DictReader(open(win_path, encoding='utf-8')):
    v = np.array([float(r[k]) if r[k] != '' else np.nan for k in names])
    o = old.get((r['code'], r['date'])); flow['rows'] += 1
    if np.isnan(v).all(): flow['empty window'] += 1; continue
    if o is None: flow['cells but no old value'] += 1; continue
    flow['comparable'] += 1
    om = np.mean(o); hit = np.where(np.abs(v - om) / om < 1e-9)[0]
    if len(hit) == 0: c = 'no match'; nomatch_n[r['n']] += 1
    elif 4 in hit and len(hit) == 1: c = 'centre only'
    elif 4 in hit: c = 'centre + ties'
    elif len(hit) == 1: c = 'unique off-centre'; pos[r['code']][names[hit[0]]] += 1
    else: c = 'off-centre ties'
    cat[c] += 1
print(dict(flow)); print(dict(cat)); print('no-match by n images', dict(nomatch_n))
for s in sorted(pos, key=int): print(s, sum(pos[s].values()), pos[s].most_common(2))
