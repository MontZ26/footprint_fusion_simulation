# RASTER_SPEC v3 rev 5 (frozen), STAGE 1: data preparation and strict leave-one-station-out fitting.
# Usage: python v3_prep.py loso.csv stations.csv OUTDIR
# Rows of days 141-236 are dropped immediately after reading (C10). Every fitted quantity for held-out station h uses
# training days 1-70 and the 23 other stations only (C2, C15).
import sys, os, json
import numpy as np, pandas as pd
from scipy.optimize import minimize
loso_p, st_p, out = sys.argv[1], sys.argv[2], sys.argv[3]; os.makedirs(out, exist_ok=True)
E = np.array([15.0, 30.0]); K = 3; TRAIN_MAX = 70; VAL_MAX = 140; CONV = 46.005e6 / 200.0

def load(path):
    lo = pd.read_csv(path); lo['code'] = lo.code.astype(int); lo = lo[lo.sat_raw.notna()].copy()
    dates = sorted(lo.date.unique()); lo['day'] = lo.date.map({d: i + 1 for i, d in enumerate(dates)})
    n_all = len(lo); lo = lo[lo.day <= VAL_MAX].copy()                          # final period never kept
    assert lo.day.max() <= VAL_MAX
    return lo, n_all

def mats(lo, codes):
    di = {d: i for i, d in enumerate(sorted(lo.day.unique()))}; ci = {c: j for j, c in enumerate(codes)}
    Y = np.full((len(di), len(codes)), np.nan); S = Y.copy(); days = np.array(sorted(di))
    for r in lo.itertuples(): Y[di[r.day], ci[r.code]] = r.truth; S[di[r.day], ci[r.code]] = r.sat_raw
    return Y, S, days

def idw_loo(Y, D, excl):
    """IDW (power 2) at every station j from the available stations other than j and the excluded set. Returns idw (T x n), dmin (T x n)."""
    T, n = Y.shape; Wm = np.where(D > 0, 1.0 / np.maximum(D, 1e-9) ** 2, 0.0); Wm[:, excl] = 0.0; Wm[np.eye(n, dtype=bool)] = 0.0
    avail = ~np.isnan(Y); Y0 = np.nan_to_num(Y)
    num = (Y0 * avail) @ Wm.T; den = avail.astype(float) @ Wm.T
    with np.errstate(invalid='ignore', divide='ignore'): idw = num / den
    Dm = np.where(Wm > 0, D, np.inf); dmin = np.full((T, n), np.inf)
    for t in range(T): dmin[t] = np.min(np.where(avail[t][None, :], Dm, np.inf), axis=1)
    return idw, dmin

def fit_station(h, lo, codes, X, D):
    """All fitted quantities for held-out station h from training days and the other stations only."""
    j_h = codes.index(h); others = [i for i in range(len(codes)) if i != j_h]
    tr = lo[(lo.day <= TRAIN_MAX) & (lo.code != h)]
    Y, S, days = mats(tr, codes); Y[:, j_h] = np.nan; S[:, j_h] = np.nan
    # satellite calibration, class centroids
    m = ~np.isnan(Y) & ~np.isnan(S); b, a = np.polyfit(S[m], Y[m], 1); s_cal = float(np.sqrt(np.mean((Y[m] - (a + b * S[m])) ** 2)))
    k = np.searchsorted(E, Y[m], side='right'); cent = np.array([Y[m][k == c].mean() for c in range(K)])
    # out-of-fold satellite residuals (ten 7-day blocks of training days)
    blk = (np.arange(len(days)) // 7); esat = np.full(Y.shape, np.nan)
    for bb in np.unique(blk):
        t_in = blk != bb; mm = m & t_in[:, None]; bo, ao = np.polyfit(S[mm], Y[mm], 1)
        sel = blk == bb; esat[sel] = Y[sel] - (ao + bo * S[sel])
    # IDW leave-one-out on training days (station excluded from its own interpolation, h excluded)
    idw, dmin = idw_loo(Y, D, [j_h]); eidw = Y - idw; ok = ~np.isnan(eidw) & np.isfinite(dmin)
    s_comp = float(np.sqrt(np.mean(eidw[ok] ** 2)))                                        # compatibility scale s
    # sigma(d) = s0 + s1*d: Gaussian NLL, s0 >= 0.5, s1 >= 0, L-BFGS-B
    e, d = eidw[ok], dmin[ok] / 1000.0
    nll = lambda p: float(np.sum(np.log(p[0] + p[1] * d) + e ** 2 / (2 * (p[0] + p[1] * d) ** 2)))
    r = minimize(nll, x0=[float(np.sqrt(np.mean(e ** 2))), 0.0], method='L-BFGS-B', bounds=[(0.5, None), (0.0, None)])
    s0, s1 = float(r.x[0]), float(r.x[1])                                                   # d in km
    # rho_ring: eight nearest of the 23 other stations, unordered pairs, >= 30 common days, Fisher-z weighted by n-3
    Eo = eidw.copy(); pairs = set()
    for i in others:
        near = sorted([jj for jj in others if jj != i], key=lambda jj: D[i, jj])[:8]
        for jj in near: pairs.add((min(i, jj), max(i, jj)))
    zs, ws = [], []
    for i, jj in pairs:
        mm = ~np.isnan(Eo[:, i]) & ~np.isnan(Eo[:, jj])
        if mm.sum() >= 30:
            rr = np.corrcoef(Eo[mm, i], Eo[mm, jj])[0, 1]; zs.append(np.arctanh(np.clip(rr, -0.999999, 0.999999))); ws.append(mm.sum() - 3)
    rho_ring = float(np.clip(np.tanh(np.average(zs, weights=ws)), 0, 1)); npairs = len(zs)
    # rho_AB: IDW LOO residual vs out-of-fold calibrated-satellite residual, same station-dates
    mab = ok & ~np.isnan(esat); rho_ab = float(np.clip(np.corrcoef(eidw[mab], esat[mab])[0, 1], 0, 1))
    # B2-strict: inverse-variance fusion of IDW and calibrated satellite
    v_i = float(np.mean(eidw[mab] ** 2)); v_s = float(np.mean(esat[mab] ** 2)); w_idw = (1 / v_i) / (1 / v_i + 1 / v_s)
    # V2 exemplars: (IDW leave-one-out value, true class) and gamma = 1 / mean(d^2) over training pairs
    ex_x = idw[ok]; ex_k = np.searchsorted(E, Y[ok], side='right'); dd = (ex_x[:, None] - ex_x[None, :]) ** 2
    gamma = float(1.0 / dd[np.triu_indices(len(ex_x), 1)].mean())
    par = dict(h=int(h), a=float(a), b=float(b), s_cal=s_cal, cent=cent.tolist(), s_comp=s_comp, sig_s0=s0, sig_s1_per_km=s1, rho_ring=rho_ring, n_pairs=npairs,
               rho_AB=rho_ab, B2_w_idw=w_idw, B2_var_idw=v_i, B2_var_sat=v_s, gamma_ek=gamma, n_train_rows=int(ok.sum()), n_exemplars=int(len(ex_x)))
    return par, ex_x, ex_k

if __name__ == '__main__':
    lo, n_all = load(loso_p); st = pd.read_csv(st_p); codes = sorted(lo.code.unique().tolist())
    st = st.set_index('code').loc[codes]; X = st[['x', 'y']].values; D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    # C1 counts
    tr_n = int((lo.day <= TRAIN_MAX).sum()); va_n = int(((lo.day > TRAIN_MAX) & (lo.day <= VAL_MAX)).sum())
    print('C1 cases read %d of %d; train %d (expect 1645), validation %d (expect 1657), stations %d, max day %d' % (len(lo), n_all, tr_n, va_n, len(codes), lo.day.max()))
    assert (tr_n, va_n) == (1645, 1657) and lo.day.max() == VAL_MAX
    pars = []; ex = {}
    for h in codes:
        p, x, k = fit_station(h, lo, codes, X, D); pars.append(p); ex['x_%d' % h] = x; ex['k_%d' % h] = k
    pd.DataFrame(pars).to_csv(os.path.join(out, 'strict_params.csv'), index=False); np.savez(os.path.join(out, 'exemplars.npz'), **ex)
    P = pd.DataFrame(pars); print(P.describe().loc[['min', 'max']].T.round(4).to_string())
    # C2 / C15: overwrite every row of held-out station h (all days) by random values; the fit must not change
    rng = np.random.default_rng(7); bad = 0
    for h in codes[:4]:
        lo2 = lo.copy(); m = lo2.code == h; lo2.loc[m, 'truth'] = rng.uniform(5, 80, m.sum()); lo2.loc[m, 'sat_raw'] = rng.uniform(1, 80, m.sum()) / (CONV * 0.5)
        p2, x2, k2 = fit_station(h, lo2, codes, X, D); p1 = [q for q in pars if q['h'] == h][0]
        same = all(np.allclose(np.array(p1[kk]), np.array(p2[kk]), atol=1e-12) for kk in p1)
        print('C2/C15 station', h, 'parameters unchanged when its rows are replaced:', same); bad += (not same)
    assert bad == 0
    # C3 IDW with the 23 other stations at the held-out station vs stored idw (days 1-140)
    Yv, Sv, days = mats(lo, codes); worst = 0.0
    for j, c in enumerate(codes):
        Wm = 1.0 / np.maximum(D[j], 1e-9) ** 2; Wm[j] = 0; av = ~np.isnan(Yv); av[:, j] = False
        idw = (np.nan_to_num(Yv) * av) @ Wm / (av @ Wm); sub = lo[lo.code == c]
        di = {d: i for i, d in enumerate(days)}; worst = max(worst, max(abs(idw[di[r.day]] - r.idw) for r in sub.itertuples()))
    print('C3 max |IDW(23 stations) - stored idw| = %.4f (tolerance 0.05, see DEVIATIONS_v3.md D1)' % worst); assert worst < 0.05
    json.dump(dict(spec='RASTER_SPEC_v3 rev5', max_day_read=int(lo.day.max()), train_cases=tr_n, validation_cases=va_n, idw_check=worst), open(os.path.join(out, 'prep_audit.json'), 'w'))
    print('stage 1 done')
