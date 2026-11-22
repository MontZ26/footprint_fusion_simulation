"""Heterogeneous-source temporal experiment on forest carbon flux (Paper T, section 'Forest sites').
INPUT: one tidy CSV per run, columns: site,date,nee,evi,tair,rad   (daily; nee = net ecosystem exchange gC m-2 d-1, negative = uptake;
       evi = satellite vegetation index (carried forward from the last composite, delayed 16 days, HERE), tair = air temperature, rad = shortwave radiation sum)
EVENT: uptake over the next h days is stronger than the trailing 30-day mean uptake, i.e. H = {-nee_{t+h} > trailing30 mean(-nee)}.
THREE HETEROGENEOUS SOURCES (each uses different data, trained on the trailing TRAIN days only, no look-ahead):
  insitu : mean of the last 7 days of uptake (flux tower history)
  sat    : regression of uptake on today's satellite EVI anomaly  (satellite)
  met    : regression of uptake on today's tair and rad            (meteorological driver model)
Masses and rules as in Paper T (Gaussian support from trailing error SD, cap 0.95), plus CDD with trailing mean pairwise error correlation.
Usage: python forest_hetero.py forest_daily.csv [h]     -> prints table, writes forest_results.csv
"""
import sys, numpy as np, pandas as pd
from scipy.stats import norm
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','temporal'))
os.environ.setdefault('ICOS_DIR','/nonexistent')
import temporal_extended as T
W=30; TRAIN=90; CAP=0.95
def dem_prob(d,s,lam):
    ms=[]
    for di,si in zip(d,s):
        sp=1-(1-si)**lam; ms.append((sp,0.0,1-sp) if di>0 else (0.0,sp,1-sp))
    return ms
def ols(X,y):
    X1=np.c_[np.ones(len(X)),X]; b,*_=np.linalg.lstsq(X1,y,rcond=None); return b
def site_run(df,h):
    df=df.sort_values('date').set_index('date').asfreq('D'); up=-df['nee'].values; n=len(up); rows=[]
    evi=df['evi'].ffill(limit=30).shift(16).values   # last observation only, delayed 16 days (16-day composites): no look-ahead
    ta=df['tair'].values; ra=df['rad'].values
    if 'tair_era' in df:
        satX=np.c_[df['tair_era'].values,df['rad_era'].values,df['vpd_era'].values]
    else: satX=np.c_[evi]
    pred={k:np.full(n,np.nan) for k in ('insitu','sat','met')}
    for t in range(TRAIN+7,n-h):
        w=up[t-6:t+1]
        if np.isnan(w).sum()>2: continue
        pred['insitu'][t]=np.nanmean(w)
        s0=max(7,t-TRAIN-h); idx=np.arange(s0,t-h+1)   # training pairs (x_s, up_{s+h}) with s+h<=t
        y=up[idx+h]; ok=~np.isnan(y)
        for k,X in (('sat',satX[idx]),('met',np.c_[ta[idx],ra[idx]])):
            xk=X[ok]; yk=y[ok]; good=~np.isnan(xk).any(1)
            if good.sum()<30: continue
            b=ols(xk[good],yk[good]); xt=(satX[t][None,:] if k=='sat' else np.c_[ta[t],ra[t]])
            if np.isnan(xt).any(): continue
            pred[k][t]=b[0]+xt[0]@b[1:]
    err={k:pred[k]-np.r_[up[h:],[np.nan]*h] for k in pred}
    for t in range(TRAIN+W+h+7,n-h):
        if any(np.isnan(pred[k][t]) for k in pred) or np.isnan(up[t+h]): continue
        H=np.array([err[k][t-W-h+1:t-h+1] for k in pred]); ok=~np.isnan(H).any(0)
        if ok.sum()<max(10,0.4*W): continue
        H=H[:,ok]; sg=np.sqrt((H**2).mean(1)); base=np.nanmean(up[t-W+1:t+1])
        if np.isnan(base) or (sg==0).any(): continue
        e=np.array([pred[k][t] for k in pred]); d=e-base; s=np.clip(2*norm.cdf(np.abs(d)/sg)-1,0,CAP)
        C=np.corrcoef(H); rb=float(np.clip(C[np.triu_indices(3,1)].mean(),0,1)); lam=1/(1+2*rb)
        r=dict(site=df.index.name or '',date=df.index[t],truth=int(up[t+h]>base),rho=rb,lam=lam)
        ms=[((si,0.,1-si) if di>0 else (0.,si,1-si)) for di,si in zip(d,s)]
        for rl in ['dempster','yager','mean','ccr']:
            out=T.rule_vec(ms,rl)
            if rl=='yager': out=out[0]
            r[rl+'_BetP']=T.summarize(out)['BetP']; 
            r[rl+'_m0']=T.summarize(out)['m0'] if rl!='yager' else 0.0
        r['cdd_BetP']=T.summarize(T.conj_full(dem_prob(d,s,lam)))['BetP']
        r['agree']=int(len(set(np.sign(d)))==1)
        rows.append(r)
    return pd.DataFrame(rows)
if __name__=='__main__':
    f=sys.argv[1]; h=int(sys.argv[2]) if len(sys.argv)>2 else 1
    D=pd.read_csv(f,parse_dates=['date']); R=[]
    for st,g in D.groupby('site'):
        r=site_run(g.copy(),h); r['site']=st; R.append(r); print(st,len(r),flush=True)
    R=pd.concat(R,ignore_index=True); y=R.truth.values
    print(f'\norigins {len(R)}, sites {R.site.nunique()}, prevalence {y.mean():.3f}, sources agree {R.agree.mean():.3f}, mean rho {R.rho.mean():.2f}')
    for rl in ['dempster','ccr','mean','cdd']:
        p=R[rl+'_BetP'].values; dec=p>.5; acc=(dec==y).mean(); gap=np.where(dec,p,1-p).mean()-acc; ic,sl=T.logit_fit(p,y)
        print(f'{rl:9s} acc {acc:.3f} gap {gap:+.3f} brier {((p-y)**2).mean():.4f} ECE {T.ece(p,y):.3f} slope {sl:.2f}')
    b=((R.date-R.date.min()).dt.days//14); G=R.groupby(b).indices; keys=list(G); rng=np.random.default_rng(5)
    for a,c in [('dempster','ccr'),('ccr','mean'),('cdd','mean'),('cdd','ccr'),('dempster','mean')]:
        dd=[]
        for _ in range(2000):
            ii=np.concatenate([G[k] for k in rng.choice(keys,len(keys))]); dd.append(((R[a+'_BetP'].values[ii]-y[ii])**2).mean()-((R[c+'_BetP'].values[ii]-y[ii])**2).mean())
        pt=((R[a+'_BetP']-y)**2).mean()-((R[c+'_BetP']-y)**2).mean(); ci=np.percentile(dd,[2.5,97.5]); print(f'Brier({a})-Brier({c}) = {pt:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]  blocks {len(keys)}')
    R.to_csv('forest_results.csv',index=False)
