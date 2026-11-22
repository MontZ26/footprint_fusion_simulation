"""Controlled sweep repeated over seeds: mean and SD (across seeds) of accuracy, gap, Brier, ECE. DGP as in rho_sweep.py."""
import numpy as np, sys
from scipy.stats import norm
from temporal_extended import rule_vec, summarize, ece
N=40000; SEEDS=[11,12,13,14,15]; res={}
for sd in SEEDS:
    rng=np.random.default_rng(sd)
    for rho in [0,0.3,0.6,0.8,0.95]:
        C=rho*np.ones((3,3))+(1-rho)*np.eye(3); L=np.linalg.cholesky(C)
        th=rng.normal(0,3,N); u=rng.normal(size=(N,3))@L.T; e=th[:,None]+u; y=(th>0).astype(int)
        P={r:np.zeros(N) for r in ['dempster','mean','ccr']}
        for i in range(N):
            d=e[i]; s=np.clip(2*norm.cdf(np.abs(d))-1,0,.95)
            ms=[((si,0.,1-si) if di>0 else (0.,si,1-si)) for di,si in zip(d,s)]
            for r in P: P[r][i]=summarize(rule_vec(ms,r))['BetP']
        for r,p in P.items():
            dec=p>.5; acc=(dec==y).mean(); gap=np.where(dec,p,1-p).mean()-acc
            res.setdefault((rho,r),[]).append((acc,gap,((p-y)**2).mean(),ece(p,y)))
    print('seed',sd,'done',flush=True)
print('rho rule  acc(mean/sd)  gap  brier  ECE')
for (rho,r),v in res.items():
    v=np.array(v); m=v.mean(0); s=v.std(0,ddof=1)
    print(f'{rho:.2f} {r:9s} '+'  '.join(f'{a:+.3f}±{b:.3f}' for a,b in zip(m,s)))
