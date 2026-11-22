"""Correlation-discounted Dempster rule (CDD): weights w_i=1-s_i of the simple supports are raised to lambda=1/(1+(k-1)*rho_bar),
rho_bar = mean pairwise error correlation on the trailing window (clipped to [0,1]); then Dempster's rule. lambda=1 -> Dempster.
Usage: ICOS_DIR=icos9 BJ_DIR=bj_data python cdd_test.py ; also sweep with known rho."""
import os, glob, sys, numpy as np, pandas as pd
os.environ.setdefault('ICOS_DIR','icos9')
import temporal_extended as T
from scipy.stats import norm
def dem_prob(d,s,lam):
    ms=[]
    for di,si in zip(d,s):
        sp=1-(1-si)**lam
        ms.append((sp,0.0,1-sp) if di>0 else (0.0,sp,1-sp))
    return T.summarize(T.conj_full(ms))['BetP']
def metrics(p,y):
    dec=p>.5; acc=(dec==y).mean(); gap=np.where(dec,p,1-p).mean()-acc; ic,sl=T.logit_fit(p,y)
    return acc,gap,((p-y)**2).mean(),T.ece(p,y),sl
def boot(R,col_a,col_b,y,B=1000):
    b=((R.date-R.date.min()).dt.days//14); G=R.groupby(b).indices; keys=list(G); rng=np.random.default_rng(5); out=[]
    for _ in range(B):
        idx=np.concatenate([G[k] for k in rng.choice(keys,len(keys))])
        out.append(((R[col_a].values[idx]-y[idx])**2).mean()-((R[col_b].values[idx]-y[idx])**2).mean())
    pt=((R[col_a]-y)**2).mean()-((R[col_b]-y)**2).mean(); return pt,np.percentile(out,[2.5,97.5])
def beijing(var):
    S={}
    for f in sorted(glob.glob(os.path.join(os.environ.get('BJ_DIR','bj_data'),'PRSA_Data_*.csv'))):
        x=pd.read_csv(f); st=x.station.iloc[0]; x['t']=pd.to_datetime(x[['year','month','day','hour']])
        v=x.set_index('t')[var]; g=v.groupby(v.index.normalize()).agg(['mean','count']); g=g[g['count']>=18]['mean']; S[st]=np.log1p(g.asfreq('D'))
    return S
out=[]
def P(*a):
    s=' '.join(str(x) for x in a); print(s,flush=True); out.append(s)
ICOS=dict(T.S)
for name,S in {'ICOS CO2':ICOS,'Beijing NO2':beijing('NO2'),'Beijing PM2.5':beijing('PM2.5')}.items():
    T.S=S; R=T.run(1); y=R.truth.values
    rb=np.clip(R[['rho_pp','rho_pf','rho_cf']].mean(1).values,0,1); lam=1/(1+2*rb)
    P(f'\n=== {name}: n={len(R)}; mean rho_bar {rb.mean():.2f}; mean lambda {lam.mean():.2f}')
    d=R[['d0','d1','d2']].values; sg=R[['sg0','sg1','sg2']].values; s=T.support(d,sg)
    R['cdd_BetP']=[dem_prob(d[i],s[i],lam[i]) for i in range(len(R))]
    for r in ['dempster','ccr','mean','cdd']:
        a,g,b,e,sl=metrics(R[f'{r}_BetP'].values,y); P(f'  {r:9s} acc {a:.3f} gap {g:+.3f} brier {b:.4f} ECE {e:.3f} slope {sl:.2f}')
    for a_,b_ in [('cdd','mean'),('cdd','ccr'),('cdd','dempster')]:
        pt,ci=boot(R,a_+'_BetP',b_+'_BetP',y); P(f'  Brier({a_})-Brier({b_}) = {pt:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]')
# sweep with true rho
P('\n=== sweep (true rho, seed 11, N=20000): gap/Brier/ECE')
rng=np.random.default_rng(11); N=20000
for rho in [0,0.3,0.6,0.8,0.95]:
    C=rho*np.ones((3,3))+(1-rho)*np.eye(3); L=np.linalg.cholesky(C)
    th=rng.normal(0,3,N); e=th[:,None]+rng.normal(size=(N,3))@L.T; y=(th>0).astype(int); lam=1/(1+2*rho)
    p=np.array([dem_prob(e[i],np.clip(2*norm.cdf(np.abs(e[i]))-1,0,.95),lam) for i in range(N)])
    a,g,b,ec,sl=metrics(p,y); P(f'rho {rho:.2f} lambda {lam:.2f}: acc {a:.3f} gap {g:+.3f} brier {b:.4f} ECE {ec:.3f} slope {sl:.2f}')
open('cdd_test_output.txt','w').write('\n'.join(out))
