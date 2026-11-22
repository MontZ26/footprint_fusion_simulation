"""Controlled test: when is CCR preferable to Dempster and to the mean of masses?
theta~N(0,3^2) latent anomaly, H = theta>0. Three estimators e_i=theta+u_i, u_i~N(0,1), equicorrelated errors rho.
Masses built exactly as in the ICOS experiment (support 2Phi(|d|/sigma)-1 capped at .95). Output: overconfidence, Brier, ECE."""
import numpy as np
from scipy.stats import norm
from temporal_extended import conj_full, rule_vec, summarize, ece
rng=np.random.default_rng(11); N=40000
print('rho  rule      acc   over   brier  ECE')
for rho in [0,0.3,0.6,0.8,0.95]:
    C=rho*np.ones((3,3))+(1-rho)*np.eye(3); L=np.linalg.cholesky(C)
    th=rng.normal(0,3,N); u=rng.normal(size=(N,3))@L.T; e=th[:,None]+u; y=(th>0).astype(int)
    P={r:np.zeros(N) for r in ['dempster','mean','ccr','bayes_ind']}
    for i in range(N):
        d=e[i]; s=np.clip(2*norm.cdf(np.abs(d))-1,0,.95)
        ms=[((si,0.,1-si) if di>0 else (0.,si,1-si)) for di,si in zip(d,s)]
        for r in ['dempster','mean','ccr']: P[r][i]=summarize(rule_vec(ms,r))['BetP']
        pH=norm.cdf(d); od=np.prod(pH/(1-pH+1e-12)); P['bayes_ind'][i]=od/(1+od)
    for r,p in P.items():
        dec=p>.5; acc=(dec==y).mean(); conf=np.where(dec,p,1-p).mean()
        print(f'{rho:.2f} {r:9s} {acc:.3f} {conf-acc:+.3f} {((p-y)**2).mean():.3f} {ece(p,y):.3f}')
