import numpy as np, pandas as pd, glob, io, json, sys
W=int(sys.argv[1]) if len(sys.argv)>1 else 30
from scipy.stats import norm, spearmanr
import os
A=os.environ.get('ICOS_DIR','icos_data/')
A=A if A.endswith('/') else A+'/'   # folder with ICOS ATC *.CO2 files
def load(f):
    L=open(f,encoding='utf-8',errors='replace').read().split('\n')
    code=[l for l in L if l.startswith('# STATION CODE')][0].split(': ')[1].strip()
    i=[j for j,l in enumerate(L) if l.startswith('#Site')][0]
    x=pd.read_csv(io.StringIO('\n'.join([L[i][1:]]+L[i+1:])),sep=';')
    x['t']=pd.to_datetime(x[['Year','Month','Day','Hour']].rename(columns=str.lower))
    x=x[(x.co2>0)&(x.Flag!='N')]; x=x[(x.t.dt.hour>=11)&(x.t.dt.hour<=14)]
    g=x.groupby(x.t.dt.normalize()).co2.agg(['mean','count']); g=g[g['count']>=2]['mean']
    return code,g.asfreq('D')
S={}
for f in sorted(glob.glob(A+'*.CO2')):
    c,g=load(f); S[c]=g
if not S: sys.exit(f'No ICOS *.CO2 files found in {A!r}. Put them in data/icos or set ICOS_DIR (see data/README_data.md).')
def comb_masses(ms,rule):
    """ms: list of (mH,mN,mT). frame {H,N}. returns BetP(H)"""
    def dem(a,b,yager=False):
        K=a[0]*b[1]+a[1]*b[0]
        mH=a[0]*b[0]+a[0]*b[2]+a[2]*b[0]; mN=a[1]*b[1]+a[1]*b[2]+a[2]*b[1]; mT=a[2]*b[2]
        if yager: return (mH,mN,mT+K)
        z=1-K
        return (mH/z,mN/z,mT/z) if z>1e-12 else (0,0,1)
    if rule=='dempster':
        r=ms[0]
        for m in ms[1:]: r=dem(r,m)
    elif rule=='yager':
        r=ms[0]
        for m in ms[1:]: r=dem(r,m,True)
    elif rule=='mean':
        r=tuple(np.mean([m[k] for m in ms],0) for k in range(3))
    elif rule=='murphy':
        a=tuple(np.mean([m[k] for m in ms],0) for k in range(3)); r=a
        for _ in range(len(ms)-1): r=dem(r,a)
    elif rule=='ccr':
        wH=min(1-m[0]/(1) if m[0]>0 else 1.0 for m in ms)  # weight of simple support on H: 1-mass
        wN=min(1-m[1] if m[1]>0 else 1.0 for m in ms)
        a=(1-wH,0.0,wH); b=(0.0,1-wN,wN); r=dem(a,b)
    return r[0]+r[2]/2
rows=[]
for st,x in S.items():
    v=x.values; n=len(v)
    err={k:np.full(n,np.nan) for k in ['past','pres','fc']}; pred={k:np.full(n,np.nan) for k in err}
    for t in range(40,n-1):
        w=v[t-6:t+1]
        if np.isnan(w).sum()>2 or np.isnan(v[t]) or np.isnan(v[t+1]): continue
        ep=np.nanmean(w)
        # phi from trailing 60 days of (anomaly_t, x_{t+1}-ep_t) pairs, using only known data
        a=[];b=[]
        for s in range(max(8,t-60),t):
            ws=v[s-6:s+1]
            if np.isnan(ws).sum()>2 or np.isnan(v[s]) or np.isnan(v[s+1]): continue
            e_=np.nanmean(ws); a.append(v[s]-e_); b.append(v[s+1]-e_)
        phi=np.clip(np.dot(a,b)/np.dot(a,a),0,1) if len(a)>=10 and np.dot(a,a)>0 else 0.5
        pred['past'][t]=ep; pred['pres'][t]=v[t]; pred['fc'][t]=ep+phi*(v[t]-ep)
        for k in err: err[k][t]=pred[k][t]-v[t+1]
    ks=['past','pres','fc']
    for t in range(40,n-1):
        if np.isnan(err['past'][t]): continue
        # trailing error history from predictions whose targets are known by t (t' <= t-1)
        H=[err[k][t-W:t] for k in ks]; ok=~np.isnan(H[0])
        if ok.sum()<max(8,W*0.4): continue
        Hm=np.array([h[ok] for h in H]); sig2=(Hm**2).mean(1)
        e=np.array([pred[k][t] for k in ks]); y=v[t+1]
        # --- point fusion
        res=dict(st=st,t=t,y=y)
        iv=1/sig2; res['mean']=e.mean(); var_mean=None
        res['blue_ind']=(iv*e).sum()/iv.sum(); res['var_blue_ind']=1/iv.sum()
        j=sig2.argmin(); res['ci']=e[j]; res['var_ci']=sig2[j]
        C=np.cov(Hm)+1e-6*np.eye(3)*sig2.mean(); Ci=np.linalg.inv(C); w=Ci.sum(1)/Ci.sum()
        res['blue_corr']=w@e; res['var_blue_corr']=1/Ci.sum(); res['cond']=np.linalg.cond(C); res['wmin']=w.min(); res['wneg']=int((w<0).any())
        Cs=0.5*np.cov(Hm)+0.5*np.diag(np.diag(np.cov(Hm)))+1e-6*sig2.mean()*np.eye(3); Cis=np.linalg.inv(Cs); ws=Cis.sum(1)/Cis.sum(); res['blue_shr']=ws@e; res['var_blue_shr']=1/Cis.sum()
        res['var_mean']=(np.ones(3)/3)@C@(np.ones(3)/3)
        # --- evidential binary: H = tomorrow above trailing-30-day mean
        base=np.nanmean(v[t-W+1:t+1]); 
        if np.isnan(base): continue
        res['truth']=int(y>base)
        d=e-base; sg=np.sqrt(sig2)
        s=np.clip(2*norm.cdf(np.abs(d)/sg)-1,0,0.95)
        ms=[((si,0.0,1-si) if di>0 else (0.0,si,1-si)) for di,si in zip(d,s)]
        for r in ['dempster','yager','ccr','mean','murphy']: res['bp_'+r]=comb_masses(ms,r)
        pH=norm.cdf(d/sg); odds=np.prod(pH/(1-pH+1e-12)); res['bp_bayes_ind']=odds/(1+odds); res['bp_bayes_mean']=pH.mean()
        for k,dd,ss in zip(ks,d,sg): res['bp_single_'+k]=norm.cdf(dd/ss)
        res['rho_pf']=np.corrcoef(Hm[0],Hm[2])[0,1]; res['rho_pp']=np.corrcoef(Hm[0],Hm[1])[0,1]
        rows.append(res)
R=pd.DataFrame(rows); 
print(len(R),R.st.nunique(),R.groupby('st').size().to_dict())
# point-estimate table
for m in ['mean','blue_ind','ci','blue_corr','blue_shr']:
    e=R[m]-R.y; var=R['var_'+m] if 'var_'+m in R else R.var_mean
    z=e/np.sqrt(var); print(f'{m:10s} RMSE {np.sqrt((e**2).mean()):.2f}  RMSz {np.sqrt((z**2).mean()):.2f}  cov95 {(np.abs(z)<1.96).mean():.2f}')
for k,c in [('past','past'),('pres','pres'),('fc','fc')]: pass
print('mean error corr past-pres',R.rho_pp.mean().round(2),' past-fc',R.rho_pf.mean().round(2))
# evidential table
y=R.truth.values
def met(col):
    p=R[col].values; dec=(p>0.5).astype(int); acc=(dec==y).mean(); conf=np.where(p>0.5,p,1-p).mean()
    return acc,conf-acc,((p-y)**2).mean()
print('rule acc overconf brier'); 
for c in ['bp_single_past','bp_single_pres','bp_single_fc','bp_dempster','bp_yager','bp_ccr','bp_mean','bp_murphy','bp_bayes_ind','bp_bayes_mean']:
    a,o,b=met(c); print(f'{c:16s} {a:.3f} {o:+.3f} {b:.3f}')
R.to_pickle(f'R{W}.pkl'); print('cond median',R.cond.median().round(0),'p90',R.cond.quantile(.9).round(0),'min w median',R.wmin.median().round(2),'frac neg w',R.wneg.mean().round(2))
