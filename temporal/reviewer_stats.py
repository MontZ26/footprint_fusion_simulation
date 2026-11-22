"""Agreement, decision-difference, conflict, prevalence and error-correlation statistics (reviewer round 6).
Usage: ICOS_DIR=icos9 BJ_DIR=bj_data python reviewer_stats.py   -> writes reviewer_stats_output.txt"""
import os, sys, glob, numpy as np, pandas as pd
os.environ.setdefault('ICOS_DIR','icos9')
import temporal_extended as T
def beijing(var):
    S={}
    for f in sorted(glob.glob(os.path.join(os.environ.get('BJ_DIR','bj_data'),'PRSA_Data_*.csv'))):
        x=pd.read_csv(f); st=x.station.iloc[0]; x['t']=pd.to_datetime(x[['year','month','day','hour']])
        v=x.set_index('t')[var]; g=v.groupby(v.index.normalize()).agg(['mean','count']); g=g[g['count']>=18]['mean']
        S[st]=np.log1p(g.asfreq('D'))
    return S
ICOS=dict(T.S); sets={'ICOS CO2':ICOS,'Beijing NO2':beijing('NO2'),'Beijing PM2.5':beijing('PM2.5')}
rules=['dempster','yager','mean','ccr']; out=[]
def P(*a):
    s=' '.join(str(x) for x in a); print(s); out.append(s)
for name,S in sets.items():
    T.S=S; R=T.run(1); y=R.truth.values; n=len(R)
    sd=np.sign(R[['d0','d1','d2']].values); agree=(sd==sd[:,[0]]).all(1)
    P(f'\n=== {name}: origins {n}, base rate {y.mean():.3f}, accuracy of always-majority {max(y.mean(),1-y.mean()):.3f}, Brier(base rate) {y.mean()*(1-y.mean()):.3f}')
    P(f'direction agreement (all 3 sources same side): {agree.mean():.3f}; two-vs-one: {(~agree).mean():.3f}')
    dec={r:(R[f'{r}_BetP'].values>.5) for r in rules}
    for a,b in [('dempster','ccr'),('ccr','mean'),('dempster','mean'),('dempster','yager')]:
        P(f'  decision differs {a} vs {b}: all {(dec[a]!=dec[b]).mean():.4f}; among agreeing {(dec[a]!=dec[b])[agree].mean():.4f}; among disagreeing {(dec[a]!=dec[b])[~agree].mean():.4f}')
    anyd=np.zeros(n,bool)
    for r in rules: anyd|=dec[r]!=dec['mean']
    P(f'  any rule decision differs from mean: {anyd.mean():.4f}')
    acc_ag=(dec['mean']==y)[agree].mean(); acc_dis=(dec['mean']==y)[~agree].mean() if (~agree).any() else np.nan
    P(f'  accuracy (mean rule) agreeing {acc_ag:.3f}, disagreeing {acc_dis:.3f}')
    for r in rules:
        m0=R[f'{r}_m0'].values; p=R[f'{r}_BetP'].values
        P(f'  {r}: conflict m(empty) mean {m0.mean():.3f} median {np.median(m0):.3f} q90 {np.quantile(m0,.9):.3f} q99 {np.quantile(m0,.99):.3f}; share with conflict>0.1: {(m0>.1).mean():.3f}; accuracy {((p>.5)==y).mean():.3f}')
    P('  error correlations (mean over origins): past-pres %.2f past-fc %.2f pres-fc %.2f'%(R.rho_pp.mean(),R.rho_pf.mean(),R.rho_cf.mean()))
    g=R.groupby('st')[['rho_pp','rho_pf','rho_cf']].mean(); P('  per-station range: pp %.2f-%.2f pf %.2f-%.2f cf %.2f-%.2f'%(g.rho_pp.min(),g.rho_pp.max(),g.rho_pf.min(),g.rho_pf.max(),g.rho_cf.min(),g.rho_cf.max()))
    g.round(2).to_csv(f'corr_by_station_{name.replace(" ","_")}.csv')
    R.to_pickle(f'rs_{name.replace(" ","_")}_h1.pkl')
    # reliability bins (mean rule and dempster)
    for r in ['dempster','mean']:
        p=R[f'{r}_BetP'].values; b=np.minimum((p*10).astype(int),9)
        P('  reliability',r,[(k,int((b==k).sum()),round(float(p[b==k].mean()),2),round(float(y[b==k].mean()),2)) for k in range(10) if (b==k).any()])
open('reviewer_stats_output.txt','w').write('\n'.join(out))
