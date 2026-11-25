# Station-level proxy rho_sp for spatial error dependence (RASTER_SPEC_v2, section 9). Training days 1-70 only.
# Usage: python rho_sp.py loso.csv neighbours.json
import csv, sys, json, collections
import numpy as np
loso_path, nb_path = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(loso_path)))
dates = sorted({r['date'] for r in rows}); day = {d: i + 1 for i, d in enumerate(dates)}
tr = [r for r in rows if day[r['date']] <= 70 and r['sat_raw'] not in ('', 'nan')]
print('training days', len({r['date'] for r in tr}), 'last training date', dates[69], 'cases', len(tr))
sat = np.array([float(r['sat_raw']) for r in tr]); tru = np.array([float(r['truth']) for r in tr])
blk = np.array([(day[r['date']] - 1) // 7 for r in tr]); stn = np.array([int(float(r['code'])) for r in tr]); dd = np.array([day[r['date']] for r in tr])
# in-sample calibration (used for s_cal) and out-of-fold calibration (used for residuals)
b1, a1 = np.polyfit(sat, tru, 1); s_cal = float(np.sqrt(np.mean((tru - (a1 + b1 * sat)) ** 2)))
res = np.zeros(len(tr))
for b in np.unique(blk):
    m = blk == b; bb, aa = np.polyfit(sat[~m], tru[~m], 1); res[m] = tru[m] - (aa + bb * sat[m])
print('calibration truth = %.2f + %.3f*sat (in-sample); s_cal %.2f; OOF residual RMSE %.2f' % (a1, b1, s_cal, np.sqrt(np.mean(res ** 2))))
nb = json.load(open(nb_path)); pairs = sorted({tuple(sorted((int(s), int(t)))) for s, v in nb.items() for t in v['n8']})
print('station pairs (each station with its eight nearest, de-duplicated):', len(pairs))
def rho(idx):
    # idx: row indices (with repeats for the bootstrap); residual table station x day
    tab = collections.defaultdict(dict)
    for k in idx: tab[(stn[k], dd[k], k)] = res[k]   # keep repeats distinct through k
    by = collections.defaultdict(list)
    for (s, d, k), e in tab.items(): by[s].append((d, e, k))
    zs, ws, rs = [], [], []
    for s, t in pairs:
        A = collections.defaultdict(list); B = collections.defaultdict(list)
        for d, e, k in by[s]: A[d].append(e)
        for d, e, k in by[t]: B[d].append(e)
        common = [d for d in A if d in B]
        x = [np.mean(A[d]) for d in common]; y = [np.mean(B[d]) for d in common]
        if len(x) < 30: continue
        r = np.corrcoef(x, y)[0, 1]
        if np.isnan(r): continue
        r = min(max(r, -0.999), 0.999); zs.append(np.arctanh(r)); ws.append(len(x) - 3); rs.append(r)
    if not zs: return np.nan, np.nan, 0, np.nan
    z = np.average(zs, weights=ws); return float(np.tanh(z)), float(np.median(rs)), len(zs), float(np.mean(rs))
r0, med, npair, mean_raw = rho(range(len(tr)))
print('rho_sp (Fisher-weighted, before clipping) %.3f; median pair correlation %.3f; pairs used %d' % (r0, med, npair))
rng = np.random.default_rng(1); blocks = np.unique(blk); byb = {b: np.where(blk == b)[0] for b in blocks}; bs = []
for _ in range(1000):
    pick = rng.choice(blocks, len(blocks), replace=True); idx = np.concatenate([byb[b] for b in pick])
    v = rho(idx)[0]
    if not np.isnan(v): bs.append(v)
lo, hi = np.percentile(bs, [2.5, 97.5]); rc = min(max(r0, 0), 1)
print('rho_sp clipped to [0,1]: %.3f ; 95%% block-bootstrap interval (1000 resamples of the %d blocks): [%.3f, %.3f]' % (rc, len(blocks), lo, hi))
for k in (2, 5, 9): print('lambda = 1/(1+(k-1)*rho_sp) for k=%d: %.3f' % (k, 1 / (1 + (k - 1) * rc)))
