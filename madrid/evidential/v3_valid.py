# RASTER_SPEC v3 rev 5 (frozen, sha256 8bb337cc8635): stages 2-4, VALIDATION ONLY (days 71-140).
# Usage: python v3_valid.py loso.csv stations.csv windows.csv PREPDIR OUTDIR     (PREPDIR = output of v3_prep.py)
# Days 141-236 are dropped when the file is read (C10). Nothing here reads the final period or scores it.
import sys, os, json, time, itertools
import numpy as np, pandas as pd
from scipy.stats import norm
loso_p, st_p, win_p, prep, out = sys.argv[1:6]; os.makedirs(out, exist_ok=True)
EPS = 0.05; E = np.array([15.0, 30.0]); K = 3; CONV = 46.005e6 / 200.0; VAL0, VAL1, TRAIN_MAX = 71, 140, 70
RHO_GRID = (0.5, 0.8, 0.95); GAM_GRID = (0.25, 0.5, 0.75); M_AGG = 3; T_POSS = 3; ALPHA0 = 0.95; KNN = 10
OFF = [(0, 0)] + [(dx, dy) for dy in (-1110.0, 0.0, 1110.0) for dx in (-850.0, 0.0, 850.0) if (dx, dy) != (0.0, 0.0)]   # centre first, then 8 ring positions
VARS = ('V1', 'V2', 'V3')
# ---------- belief-function algebra on K = 3 (masks 0..7, 0 = empty set) ----------
POP = np.array([bin(i).count('1') for i in range(8)]); TH = 7
T3 = np.zeros((8, 8, 8));
for a in range(8):
    for b in range(8): T3[a, b, a & b] = 1
QS = np.array([[1.0 if (b & a) == a else 0.0 for b in range(8)] for a in range(8)])
MW = np.array([[(-(-1.0) ** (POP[b] - POP[a])) if (b & a) == a else 0.0 for b in range(1, 8)] for a in range(7)])
QM = np.array([[1.0 if (c & a) != c else 0.0 for a in range(7)] for c in range(1, 8)])
MO = np.array([[(-1.0) ** (POP[b] - POP[a]) if (b & a) == a else 0.0 for b in range(1, 8)] for a in range(1, 8)])
INV = [0]
def conj(m1, m2): return np.einsum('a,b,abc->c', m1, m2, T3)
def logw(m):
    q = QS[1:] @ m
    return None if q.min() <= 1e-300 else MW @ np.log(q)
def from_logw(lw):
    q = np.exp(QM @ lw); m1 = MO @ q; m = np.r_[1 - m1.sum(), m1]; return m
def _ok(m): return m.min() >= -1e-9
def _clean(m): m = np.where(m < 0, 0.0, m); return m / m.sum()
def cautious(m1, m2):
    w1, w2 = logw(m1), logw(m2)
    if w1 is None or w2 is None: INV[0] += 1; return conj(m1, m2)
    m = from_logw(np.minimum(w1, w2))
    if not _ok(m): INV[0] += 1; return conj(m1, m2)
    return _clean(m)
def cdd(ms, rho):
    lam = 1.0 / (1.0 + (len(ms) - 1) * rho); ws = [logw(m) for m in ms]
    if any(w is None for w in ws): INV[0] += 1; o = ms[0]; [o := conj(o, x) for x in ms[1:]]; return o
    m = from_logw(lam * np.sum(ws, 0))
    if not _ok(m): INV[0] += 1; o = ms[0]; [o := conj(o, x) for x in ms[1:]]; return o
    return _clean(m)
def consonant(c):
    o = np.argsort(-c, kind='stable'); lv = c[o]; m = np.zeros(8)
    for r in range(K):
        nxt = lv[r + 1] if r < K - 1 else 0.0; m[sum(1 << int(o[q]) for q in range(r + 1))] += lv[r] - nxt
    return m
def betp(m):
    s = m[1:].sum()
    if s <= 1e-12: return np.full(K, 1.0 / K), True
    return np.array([sum(m[A] / POP[A] for A in range(1, 8) if A >> l & 1) for l in range(K)]) / s, False
def kernel(cent, s): d = cent[:, None] - cent[None, :]; return np.exp(-d ** 2 / (2 * s ** 2))
def make_CF(R): return {F: consonant(np.max([R[k] for k in range(K) if F >> k & 1], 0)) for F in range(1, 7)}
def project(m, rho, CF):                                   # section 6: empty-set mass carried over, composite sets via upper-envelope contour
    o = np.zeros(8)
    for F in range(1, 7):
        if m[F] > 0: o += rho * m[F] * CF[F]
    o[0] = m[0]; o[7] += 1 - o.sum(); return o
def gcont(x, cent, sig): c = np.exp(-(x - cent) ** 2 / (2 * sig ** 2)); c = c / c.max(); return np.maximum(c, EPS)
def bayes(p): m = np.zeros(8); m[1], m[2], m[4] = (1 - EPS) * p; m[7] = EPS; return m
# ---------- relaxation families ----------
RM = np.array([[1, .5, -.5], [.5, 1, .5], [-.5, .5, 1]])
def prob_relax(p0, ps):
    p = p0.copy(); P = np.array(ps)
    for _ in range(3): q = (P @ RM.T).mean(0); p = p * (1 + q); p /= p.sum()
    return p
def poss_relax(pi0, pis, gam, R):
    sj = np.max(np.minimum(np.array(pis)[:, None, :], R[None]), 2); A = np.sort(sj, 0)[::-1][min(M_AGG, len(pis)) - 1]
    kap = 1 - A.max(); Ab = np.maximum(A, kap); Ab = Ab / Ab.max(); pi = pi0.copy()
    for _ in range(T_POSS): pi = (1 - gam) * pi + gam * np.minimum(pi, Ab); pi = pi / pi.max()
    return pi
def fold(ms, f):
    o = ms[0]
    for x in ms[1:]: o = f(o, x)
    return o
def evid_relax(b0, bs, rho, CF, rho_ring):
    pr = [project(m, rho, CF) for m in bs]; pool = fold(pr, cautious)
    return dict(evid_primary=cautious(b0, pool), evid_cautpool_conjcentre=conj(b0, pool), evid_dempster_all=fold([b0] + pr, conj), evid_cdd=cdd([b0] + pr, rho_ring))
# ---------- reconstruction of m_A from m_rel (section 8) ----------
def lowest(x, cent, sig): return gcont(x, cent, sig).min()
def recon(mrel, xr, cent):
    e = mrel[0]
    if e >= 1 - 1e-9: tgt, flag = EPS, 5
    else: tgt = mrel[7] / (1 - e); flag = 0
    if tgt <= EPS + 1e-12: sig = 0.5; flag = max(flag, 1)
    elif lowest(xr, cent, 200.0) < tgt: sig = 200.0; flag = 2
    elif lowest(xr, cent, 0.5) >= tgt: sig = 0.5; flag = 3
    else:
        lo_, hi_ = 0.5, 200.0; sig = None
        for _ in range(60):
            mid = 0.5 * (lo_ + hi_); f = lowest(xr, cent, mid)
            if abs(f - tgt) < 1e-4: sig = mid; break
            if f < tgt: lo_ = mid
            else: hi_ = mid
        if sig is None: sig = 0.5 * (lo_ + hi_); flag = 4
    con = consonant(gcont(xr, cent, sig)); mA = (1 - e) * con; mA[0] = e
    if e < 1 - 1e-9 and flag in (0,): assert abs(mA[7] - mrel[7]) < 2e-4 and abs(mA[0] - mrel[0]) < 1e-12          # C13 (and C7)
    return mA, flag
# ---------- variants (sections 4) ----------
def v1(x, P):
    c = P['cent']; mu = np.zeros(3)
    if x <= c[0]: mu[0] = 1
    elif x >= c[2]: mu[2] = 1
    else:
        i = 0 if x < c[1] else 1; t = (x - c[i]) / (c[i + 1] - c[i]); mu[i] = 1 - t; mu[i + 1] = t
    pi = np.maximum(mu / mu.max(), EPS); return mu, pi, consonant(pi)
def v2(x, P):
    ex, ek = P['ex'], P['ek']; idx = np.argpartition(np.abs(ex - x), KNN)[:KNN]; m = np.zeros(8); m[7] = 1.0
    for i in idx:
        s = ALPHA0 * np.exp(-P['gamma'] * (ex[i] - x) ** 2); mi = np.zeros(8); mi[1 << int(ek[i])] = s; mi[7] = 1 - s; m = conj(m, mi)
    m = m / (1 - m[0]); m[0] = 0.0
    p, _ = betp(m); pl = np.array([sum(m[A] for A in range(1, 8) if A >> l & 1) for l in range(K)]); pi = np.maximum(pl / pl.max(), EPS); return p, pi, m
def v3(x, P, dkm):
    sig = P['s0'] + P['s1'] * dkm; pi = gcont(x, P['cent'], sig); cdf = norm.cdf((E - x) / sig)
    return np.diff(np.r_[0.0, cdf, 1.0]), pi, consonant(pi)
def evidence(v, x, P, dkm): return {'V1': lambda: v1(x, P), 'V2': lambda: v2(x, P), 'V3': lambda: v3(x, P, dkm)}[v]()
# ---------- load ----------
lo = pd.read_csv(loso_p); lo['code'] = lo.code.astype(int); lo = lo[lo.sat_raw.notna()].copy()
dates = sorted(lo.date.unique()); lo['day'] = lo.date.map({d: i + 1 for i, d in enumerate(dates)}); lo = lo[lo.day <= VAL1].copy()
assert lo.day.max() == VAL1                                                                      # C10: nothing after day 140 is kept
st = pd.read_csv(st_p).set_index('code'); codes = sorted(lo.code.unique().tolist()); X = st.loc[codes, ['x', 'y']].values
PAR = pd.read_csv(os.path.join(prep, 'strict_params.csv')).set_index('h'); EXN = np.load(os.path.join(prep, 'exemplars.npz'))
days = np.array(sorted(lo.day.unique())); di = {d: i for i, d in enumerate(days)}; ci = {c: j for j, c in enumerate(codes)}
Y = np.full((len(days), len(codes)), np.nan)
for r in lo.itertuples(): Y[di[r.day], ci[r.code]] = r.truth
VA = lo[(lo.day >= VAL0)].reset_index(drop=True); assert len(VA) == 1657
VA['k'] = np.searchsorted(E, VA.truth.values, side='right'); win = pd.read_csv(win_p); win['code'] = win.code.astype(int); wv = win.set_index(['code', 'date'])
RING = [c for c in win.columns if c.startswith('v_') and c != 'v_0_0']
HP = {}
for h in codes:
    p = PAR.loc[h]; cent = np.array(eval(p['cent']) if isinstance(p['cent'], str) else p['cent']) if 'cent' in PAR.columns else None
cent_cols = None
# centroids are stored as a list column; recompute from the parameter file via the json-like string
def getcent(p): return np.array(json.loads(str(p['cent']).replace('np.float64', '')))
for h in codes:
    p = PAR.loc[h]; cent = getcent(p); R = kernel(cent, p.s_comp); near4 = [c for c in sorted([c for c in codes if c != h], key=lambda c: np.hypot(*(X[ci[c]] - X[ci[h]])))[:4]]
    HP[h] = dict(a=p.a, b=p.b, s_cal=p.s_cal, cent=cent, R=R, CF=make_CF(R), Rcal=kernel(cent, p.s_cal), CFcal=make_CF(kernel(cent, p.s_cal)), s_comp=p.s_comp, s0=p.sig_s0, s1=p.sig_s1_per_km, rho_ring=p.rho_ring, rho_AB=p.rho_AB,
                 w_idw=p.B2_w_idw, gamma=p.gamma_ek, ex=EXN['x_%d' % h], ek=EXN['k_%d' % h], near4=near4)
# ---------- audits (C4-C9, C12) before the validation loop ----------
aud = []
def chk(name, ok, extra=''): aud.append('%s %s %s' % (name, 'PASS' if ok else 'FAIL', extra)); assert ok, name
rng = np.random.default_rng(3); P0 = HP[codes[0]]
def randbba(emp=True):
    m = rng.dirichlet(np.ones(8) * 0.7);
    if not emp: m[0] = 0; m /= m.sum()
    m[7] = max(m[7], 0.02); return m / m.sum()
ok4 = True
for _ in range(200):
    m = randbba(); pr = project(m, 0.8, P0['CF']); ok4 &= abs(pr.sum() - 1) < 1e-12 and abs(pr[0] - m[0]) < 1e-15 and pr.min() >= -1e-15
for k_ in range(K):
    m = np.zeros(8); m[1 << k_] = 1.0; pr = project(m, 0.8, P0['CF']); ok4 &= np.allclose(pr, 0.8 * consonant(P0['R'][k_]) + 0.2 * np.eye(8)[7])
for F in range(1, 7): ok4 &= abs(np.max([P0['R'][k] for k in range(K) if F >> k & 1], 0).max() - 1) < 1e-12
chk('C4 projection (sums to 1 incl. empty set, empty mass carried over, singleton formula, envelope max 1)', ok4)
ok5 = True
for _ in range(100):
    a, b, c = randbba(False), randbba(False), randbba(False)
    ok5 &= np.allclose(cautious(a, a), a, atol=1e-9) and np.allclose(cautious(a, b), cautious(b, a), atol=1e-9) and np.allclose(cautious(cautious(a, b), c), cautious(a, cautious(b, c)), atol=1e-9)
chk('C5 cautious rule idempotent, commutative, associative (random BBAs, 1e-9)', ok5, '(INV counter %d)' % INV[0])
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from evid_k import Frame
    F3 = Frame(3); okx = True
    for _ in range(50):
        a, b = randbba(False), randbba(False); da = {s: a[s] for s in range(1, 8)}; db = {s: b[s] for s in range(1, 8)}; r_ = F3.cautious(da, db); okx &= all(abs(r_[s] - cautious(a, b)[s]) < 1e-9 for s in range(1, 8))
    chk('C5b cautious rule agrees with the audited evid_k.Frame implementation', okx)
except ImportError: aud.append('C5b skipped (evid_k.py not importable)')
ms = [randbba(False) for _ in range(3)]; o = fold(ms, conj)
chk('C6a CDD with rho = 0 equals the conjunctive rule', np.allclose(cdd(ms, 0.0), o, atol=1e-9))
a = randbba(False); chk('C6b CDD with rho = 1 does not over-count identical copies', np.allclose(cdd([a] * 5, 1.0), a, atol=1e-9))
ok7 = True; sg = []
for _ in range(200):
    xr = rng.uniform(0, 60); mr = np.zeros(8); mr[7] = rng.uniform(0.05, 0.9); mr[1] = 1 - mr[7]; mA, fl = recon(mr, xr, P0['cent'])
    if fl == 0: ok7 &= abs(mA[7] - mr[7]) < 1e-4
    sg.append(fl)
chk('C7 root search hits the target within 1e-4 when no clipping is flagged', ok7, 'flags %s' % np.bincount(sg, minlength=6).tolist())
mr = randbba(); mr[7] = 0.3; mr[0] = 0.2; mr[1:7] *= (1 - 0.5) / mr[1:7].sum(); mA, fl = recon(mr, 20.0, P0['cent']); chk('C13 reconstruction keeps m(empty) and m(Theta)', abs(mA[0] - 0.2) < 1e-12 and abs(mA[7] - 0.3) < 2e-4 and abs(mA.sum() - 1) < 1e-12)
cc = [{'a': 1}]; chk('C14 BetP uses normalised masses', abs(betp(np.array([0.5, 0.25, 0, 0, 0.25, 0, 0, 0]))[0].sum() - 1) < 1e-12)
chk('C10 leakage: maximum day read = %d' % lo.day.max(), lo.day.max() == VAL1)
chk('C1 validation cases = 1657, dates = 70, stations = %d' % VA.code.nunique(), len(VA) == 1657 and VA.day.nunique() == 70)
# ---------- validation loop ----------
recs = []; cent_rows = []; t0 = time.time(); XY = X; S1rows = []
def record(case, variant, family, param, comb, m, flag, extra=None):
    p, cc_ = betp(m); cent = case['cent']; v = float(p @ cent); hard = float(cent[p.argmax()])
    recs.append((case['i'], variant, family, param, comb, v, hard, p[0], p[1], p[2], m[7], m[0], flag, INV[0], int(cc_)))
for n_, r in enumerate(VA.itertuples()):
    h = r.code; P = HP[h]; cent = P['cent']; t = di[r.day]; jh = ci[h]
    av = ~np.isnan(Y[t]); av[jh] = False; idx = np.where(av)[0]; pos = XY[jh][None, :] + np.array(OFF); dd = np.sqrt(((pos[:, None, :] - XY[idx][None, :, :]) ** 2).sum(-1)); w = 1 / dd ** 2
    idw9 = (w * Y[t, idx][None, :]).sum(1) / w.sum(1); dkm = dd.min(1) / 1000.0
    xsat = P['a'] + P['b'] * r.sat_raw; mB = consonant(gcont(xsat, cent, P['s_cal']))
    case = dict(i=n_, cent=cent)
    record(case, 'all', 'B0_idw', '', 'val', np.eye(8)[7], 0); recs[-1] = (n_, 'all', 'B0_idw', '', 'val', float(idw9[0]), np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0, 0, 0)
    recs.append((n_, 'all', 'B1_sat', '', 'val', float(xsat), np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0, 0, 0))
    recs.append((n_, 'all', 'B2_strict', '', 'val', float(P['w_idw'] * idw9[0] + (1 - P['w_idw']) * xsat), np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0, 0, 0))
    # S1 neighbours: nearest four stations with observed classes
    s1k = [int(np.searchsorted(E, Y[t, ci[c]], side='right')) for c in P['near4'] if not np.isnan(Y[t, ci[c]])]
    for v in VARS:
        ev = [evidence(v, idw9[i], P, dkm[i]) for i in range(9)]; p0, pi0, b0 = ev[0]; nb = ev[1:]
        base = {'prob': (lambda p: bayes(p))(p0), 'poss': consonant(pi0), 'evid': b0}
        def finish(family, param, mrel, nbs_note=''):
            p_, c_ = betp(mrel); xr = float(p_ @ cent); mA, fl = recon(mrel, xr, cent)
            record(case, v, family, param, 'conj', conj(mA, mB), fl)
            record(case, v, family, param, 'caut', cautious(mA, mB), fl)
            record(case, v, family, param, 'cddAB', cdd([mA, mB], P['rho_AB']), fl)
            record(case, v, family, param, 'S3', conj(mrel, mB), 0)
        for f_, m_ in base.items(): INV[0] = 0; finish('B3_' + f_, '', m_)
        INV[0] = 0; pr = prob_relax(p0, [x[0] for x in nb]); finish('prob', '', bayes(pr))
        for g in GAM_GRID: INV[0] = 0; finish('poss', g, consonant(poss_relax(pi0, [x[1] for x in nb], g, P['R'])))
        for rho in RHO_GRID:
            INV[0] = 0; res = evid_relax(b0, [x[2] for x in nb], rho, P['CF'], P['rho_ring'])
            for fam, m_ in res.items(): finish(fam, rho, m_)
        # S1: four nearest stations as neighbours (crisp observed classes)
        if s1k:
            INV[0] = 0; oh = [np.eye(3)[c] for c in s1k]; finish('S1_prob', '', bayes(prob_relax(p0, oh)))
            for g in GAM_GRID: INV[0] = 0; finish('S1_poss', g, consonant(poss_relax(pi0, oh, g, P['R'])))
            for rho in RHO_GRID:
                INV[0] = 0; bs = [np.eye(8)[1 << c] for c in s1k]; finish('S1_evid_primary', rho, evid_relax(b0, bs, rho, P['CF'], P['rho_ring'])['evid_primary'])
    if n_ % 200 == 0: print('  case %d of %d  %.0fs' % (n_, len(VA), time.time() - t0), flush=True)
cols = ['i', 'variant', 'family', 'param', 'comb', 'value', 'value_hard', 'pL', 'pM', 'pH', 'm_theta', 'm_empty', 'root_flag', 'invalid_count', 'complete_conflict']
R_ = pd.DataFrame(recs, columns=cols); R_['param'] = R_.param.astype(str)
VA['i'] = np.arange(len(VA)); R_ = R_.merge(VA[['i', 'code', 'date', 'day', 'truth', 'k', 'idw']], on='i')
assert R_.value.notna().all(), 'missing predictions'
print('records', len(R_), '%.0fs' % (time.time() - t0))
# correction-mode defuzzification: x_idw + (soft relaxed - soft unrelaxed), same variant and family type (S2)
b3 = R_[R_.family.str.startswith('B3_')].copy(); b3['ftype'] = b3.family.str[3:]; b3 = b3.set_index(['i', 'variant', 'ftype', 'comb']).value
def ftype(f): return {'prob': 'prob', 'poss': 'poss'}.get(f, 'evid')
nonb = R_[~R_.family.str.startswith(('B3_', 'B0', 'B1', 'B2', 'S1'))].copy(); nonb['ftype'] = nonb.family.map(ftype)
corr = nonb.apply(lambda r: r.idw_stored if False else np.nan, axis=1) if False else None
R_['value_corr'] = np.nan
mask = ~R_.family.str.startswith(('B3_', 'B0', 'B1', 'B2', 'S1'))
keys = list(zip(R_.loc[mask, 'i'], R_.loc[mask, 'variant'], R_.loc[mask, 'family'].map(ftype), R_.loc[mask, 'comb']))
R_.loc[mask, 'value_corr'] = R_.loc[mask, 'idw'].values + R_.loc[mask, 'value'].values - np.array([b3[k] for k in keys])
# idw column of loso is IDW at the station itself: identical to idw9[0] within the C3 tolerance
R_.to_csv(os.path.join(out, 'validation_predictions.csv.gz'), index=False, compression='gzip', float_format='%.6g')
# ---------- metrics ----------
Yoh = np.eye(3)[R_.k.values]; PP = R_[['pL', 'pM', 'pH']].values
R_['sq'] = (R_.value - R_.truth) ** 2; R_['ae'] = (R_.value - R_.truth).abs(); R_['err'] = R_.value - R_.truth
R_['pred_k'] = np.where(R_.pL.notna(), PP.argmax(1), np.searchsorted(E, R_.value.values, side='right')); R_['acc'] = (R_.pred_k == R_.k).astype(float)
R_['brier'] = np.where(R_.pL.notna(), ((PP - Yoh) ** 2).sum(1), np.nan)
R_['sq_hard'] = (R_.value_hard - R_.truth) ** 2; R_['sq_corr'] = (R_.value_corr - R_.truth) ** 2
g = R_.groupby(['variant', 'family', 'param', 'comb'])
S = g.agg(n=('sq', 'size'), RMSE=('sq', lambda s: float(np.sqrt(s.mean()))), MAE=('ae', 'mean'), bias=('err', 'mean'), accuracy=('acc', 'mean'), Brier=('brier', 'mean'), mean_m_empty=('m_empty', 'mean'), mean_m_theta=('m_theta', 'mean'),
          share_complete_conflict=('complete_conflict', 'mean'), share_invalid_decomposition=('invalid_count', lambda s: float((s > 0).mean())), share_root_search_clipped=('root_flag', lambda s: float((s >= 2).mean())),
          share_eps_floor_target=('root_flag', lambda s: float((s == 1).mean())), RMSE_hard=('sq_hard', lambda s: float(np.sqrt(s.mean()))), RMSE_corr=('sq_corr', lambda s: float(np.sqrt(s.mean())))).reset_index()
S.to_csv(os.path.join(out, 'validation_summary_all_configurations.csv'), index=False); pd.set_option('display.width', 250); pd.set_option('display.max_rows', 400)
base_tab = S[(S.comb == 'val') | ((S.comb == 'conj') & S.family.str.startswith('B3_'))]; print(base_tab[['variant', 'family', 'RMSE', 'MAE', 'bias', 'accuracy', 'Brier']].round(3).to_string(index=False))
# ---------- selection on validation RMSE (primary rule: conjunctive combination, soft defuzzification, evidential primary rule) ----------
prim = S[(S.comb == 'conj') & S.family.isin(['prob', 'poss', 'evid_primary'])].copy(); sel = {}
for (v, f), d in prim.groupby(['variant', 'family']): b = d.loc[d.RMSE.idxmin()]; sel['%s|%s' % (v, f)] = dict(param=str(b.param), RMSE=float(b.RMSE))
best = {}
for f in ('prob', 'poss', 'evid_primary'):
    cand = {v: sel['%s|%s' % (v, f)] for v in VARS}; vb = min(cand, key=lambda v: cand[v]['RMSE']); best[f] = dict(variant=vb, **cand[vb])
print('selected (validation RMSE):', json.dumps(sel)); print('best variant per family:', json.dumps(best))
# ---------- paired 7-day block bootstrap of RMSE differences ----------
rng = np.random.default_rng(1); NB = 10; B = 2000; draws = rng.integers(0, NB, (B, NB)); blk = ((VA.day.values - VAL0) // 7).astype(int); assert blk.max() == 9
def sqv(v, f, p, c): d = R_[(R_.variant == v) & (R_.family == f) & (R_.param == p) & (R_.comb == c)].sort_values('i'); return d.sq.values
def sqbase(f): d = R_[R_.family == f].sort_values('i'); return d.sq.values
def bt(a, b):
    sa = np.array([a[blk == k].sum() for k in range(NB)]); sb = np.array([b[blk == k].sum() for k in range(NB)]); nn = np.array([(blk == k).sum() for k in range(NB)])
    ra = np.sqrt(sa[draws].sum(1) / nn[draws].sum(1)); rb = np.sqrt(sb[draws].sum(1) / nn[draws].sum(1)); d = ra - rb
    return float(np.sqrt(a.mean()) - np.sqrt(b.mean())), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
rows = []
def pair(la, a, lb, b): d, lo_, hi_ = bt(a, b); rows.append(dict(a=la, b=lb, RMSE_a_minus_b=d, ci_lo=lo_, ci_hi=hi_))
selv = lambda v, f: sel['%s|%s' % (v, f)]['param']
for v in VARS:
    for f, b3f in (('prob', 'B3_prob'), ('poss', 'B3_poss'), ('evid_primary', 'B3_evid')):
        a = sqv(v, f, selv(v, f), 'conj'); la = '%s %s(%s)' % (v, f, selv(v, f)); b3 = R_[(R_.variant == v) & (R_.family == b3f) & (R_.comb == 'conj')].sort_values('i').sq.values
        pair(la, a, '%s %s (no relaxation, B3)' % (v, b3f), b3); pair(la, a, 'B2_strict', sqbase('B2_strict')); pair(la, a, 'B0_idw', sqbase('B0_idw'))
    for fa, fb in (('evid_primary', 'poss'), ('evid_primary', 'prob'), ('poss', 'prob')): pair('%s %s(%s)' % (v, fa, selv(v, fa)), sqv(v, fa, selv(v, fa), 'conj'), '%s %s(%s)' % (v, fb, selv(v, fb)), sqv(v, fb, selv(v, fb), 'conj'))
for f in ('prob', 'poss', 'evid_primary'):
    for va, vb in (('V1', 'V2'), ('V1', 'V3'), ('V2', 'V3')): pair('%s %s(%s)' % (va, f, selv(va, f)), sqv(va, f, selv(va, f), 'conj'), '%s %s(%s)' % (vb, f, selv(vb, f)), sqv(vb, f, selv(vb, f), 'conj'))
pair('B2_strict', sqbase('B2_strict'), 'B0_idw', sqbase('B0_idw')); pair('B1_sat', sqbase('B1_sat'), 'B0_idw', sqbase('B0_idw'))
BT = pd.DataFrame(rows); BT.to_csv(os.path.join(out, 'validation_paired_bootstrap.csv'), index=False); print(BT.round(3).to_string(index=False))
print('(%d cases, %d dates, %d stations, %d non-overlapping 7-day blocks, B = %d, seed 1)' % (len(VA), VA.day.nunique(), VA.code.nunique(), NB, B))
# ---------- C11 calibration of the variants (centre pixel, before relaxation) ----------
cal = []
def relrows(name, Pm, kk):
    conf = Pm.max(1); acc = (Pm.argmax(1) == kk).astype(float); ece = 0.0; bins = []
    for lo_, hi_ in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        m = (conf > lo_) & (conf <= hi_)
        if m.any(): ece += m.mean() * abs(acc[m].mean() - conf[m].mean()); bins.append(dict(source=name, lo=lo_, hi=hi_, n=int(m.sum()), mean_conf=conf[m].mean(), accuracy=acc[m].mean()))
    return bins, dict(source=name, ECE=ece, Brier=float(((Pm - np.eye(3)[kk]) ** 2).sum(1).mean()), accuracy=float(acc.mean()))
srcs = {'V1_membership': [], 'V1_betp_consonant': [], 'V2_betp': [], 'V3_gaussian': [], 'V3_betp_consonant': []}
for r in VA.itertuples():
    P = HP[r.code]; t = di[r.day]; jh = ci[r.code]; av = ~np.isnan(Y[t]); av[jh] = False; idx = np.where(av)[0]; dd = np.sqrt(((XY[jh][None, :] - XY[idx]) ** 2).sum(-1)); w = 1 / dd ** 2; x = (w * Y[t, idx]).sum() / w.sum(); dk = dd.min() / 1000
    p1, pi1, b1 = v1(x, P); p2, _, b2 = v2(x, P); p3, pi3, b3_ = v3(x, P, dk); srcs['V1_membership'].append(p1); srcs['V1_betp_consonant'].append(betp(b1)[0]); srcs['V2_betp'].append(p2); srcs['V3_gaussian'].append(p3); srcs['V3_betp_consonant'].append(betp(b3_)[0])
tabs = []; sums = []
for k_, v_ in srcs.items(): b_, s_ = relrows(k_, np.array(v_), VA.k.values); tabs += b_; sums.append(s_)
pd.DataFrame(tabs).to_csv(os.path.join(out, 'validation_reliability_bins.csv'), index=False); CAL = pd.DataFrame(sums); CAL.to_csv(os.path.join(out, 'validation_calibration_variants.csv'), index=False); print(CAL.round(4).to_string(index=False))
# ---------- S0: satellite-only relaxation under strict per-station calibration (class accuracy and Brier) ----------
S0 = {}
def s0add(name, Pm, kk): S0.setdefault(name, []).append((Pm, kk))
for r in VA.itertuples():
    P = HP[r.code]; key = (r.code, r.date)
    if key not in wv.index: continue
    w9 = wv.loc[key]; xc = P['a'] + P['b'] * r.sat_raw; xs = [P['a'] + P['b'] * w9[c] * CONV for c in RING if not np.isnan(w9[c])]
    cdf = lambda x: np.diff(np.r_[0.0, norm.cdf((E - x) / P['s_cal']), 1.0]); pi0 = gcont(xc, P['cent'], P['s_cal']); p0 = cdf(xc); b0 = consonant(pi0); k = r.k
    s0add('prob_init', p0, k); s0add('contour_init_betp', betp(b0)[0], k)
    if not xs: continue
    nbp = [cdf(x) for x in xs]; nbc = [gcont(x, P['cent'], P['s_cal']) for x in xs]; s0add('rosenfeld', prob_relax(p0, nbp), k)
    for g_ in GAM_GRID: s0add('poss_g%.2f' % g_, betp(consonant(poss_relax(pi0, nbc, g_, P['Rcal'])))[0], k)
    for rho in RHO_GRID: INV[0] = 0; s0add('evid_primary_rho%.2f' % rho, betp(evid_relax(b0, [consonant(c) for c in nbc], rho, P['CFcal'], 0.0)['evid_primary'])[0], k)
rows = []
for nme, v_ in S0.items(): Pm = np.array([a for a, _ in v_]); kk = np.array([b for _, b in v_]); rows.append(dict(method=nme, n=len(kk), accuracy=float((Pm.argmax(1) == kk).mean()), Brier=float(((Pm - np.eye(3)[kk]) ** 2).sum(1).mean())))
S0T = pd.DataFrame(rows); S0T.to_csv(os.path.join(out, 'validation_S0_satellite_only.csv'), index=False); print(S0T.round(4).to_string(index=False))
# ---------- flags ----------
fl = R_[R_.comb == 'conj'].groupby(['variant', 'family', 'param']).agg(root_eps_floor=('root_flag', lambda s: int((s == 1).sum())), root_upper_clip=('root_flag', lambda s: int((s == 2).sum())), root_lower_clip=('root_flag', lambda s: int((s == 3).sum())),
                                                                       root_nonconverged=('root_flag', lambda s: int((s == 4).sum())), root_complete_conflict_input=('root_flag', lambda s: int((s == 5).sum())), invalid_decomposition_cases=('invalid_count', lambda s: int((s > 0).sum())), complete_conflict_final=('complete_conflict', 'sum')).reset_index()
fl.to_csv(os.path.join(out, 'validation_failure_flags.csv'), index=False); print(fl[(fl[['root_upper_clip', 'root_lower_clip', 'root_nonconverged', 'root_complete_conflict_input', 'invalid_decomposition_cases', 'complete_conflict_final']].sum(1) > 0)].to_string(index=False))
open(os.path.join(out, 'audit_report.txt'), 'w').write('\n'.join(aud) + '\n'); print('\n'.join(aud))
json.dump(dict(spec='RASTER_SPEC_v3 rev5 sha256 8bb337cc8635', selection_grid=sel, best_variant_per_family=best, note='validation days 71-140 only; the final period has not been read'), open(os.path.join(out, 'validation_selection.json'), 'w'), indent=1)
print('done %.0fs' % (time.time() - t0))
