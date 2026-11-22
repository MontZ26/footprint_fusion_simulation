"""Second dataset: Beijing multi-site air quality (UCI #501, Zhang et al. 2017), 12 stations, hourly, 2013-03 to 2017-02.
Same protocol as the ICOS study: daily series -> three estimators (past 7-day mean, persistence, damped anomaly) -> masks on H = 'next value above trailing 30-day mean'.
Usage: BJ_DIR=<folder with PRSA_Data_*.csv> python analyze_beijing.py NO2 log     (variable, transform: log|raw)"""
import sys, os, glob, numpy as np, pandas as pd
from scipy import stats
os.environ.setdefault('ICOS_DIR','/nonexistent')
import temporal_extended as T
VAR=sys.argv[1] if len(sys.argv)>1 else 'NO2'; TR=sys.argv[2] if len(sys.argv)>2 else 'log'
D=os.environ.get('BJ_DIR','bj_data/')
def load(f):
    x=pd.read_csv(f); st=x.station.iloc[0]
    x['t']=pd.to_datetime(x[['year','month','day','hour']])
    v=x.set_index('t')[VAR]
    g=v.groupby(v.index.normalize()).agg(['mean','count']); g=g[g['count']>=18]['mean']
    g=g.asfreq('D')
    if TR=='log': g=np.log1p(g)
    return st,g
S={}
for f in sorted(glob.glob(os.path.join(D,'PRSA_Data_*.csv'))):
    st,g=load(f); S[st]=g
T.S=S
print(f'variable {VAR} ({TR}); stations {len(S)}; daily points per station {[int(g.notna().sum()) for g in S.values()]}')
rules=['dempster','yager','mean','ccr']
def analyse(h):
    R=T.run(h); y=R.truth.values
    print(f'\n=== {VAR} {TR} horizon {h}: origins {len(R)}, stations {R.st.nunique()}, base rate {y.mean():.3f}, Brier(base rate) {y.mean()*(1-y.mean()):.3f}')
    print('mean error corr past-pres %.2f  past-fc %.2f  pres-fc %.2f; per-station past-fc range %.2f-%.2f'%(R.rho_pp.mean(),R.rho_pf.mean(),R.rho_cf.mean(),R.groupby('st').rho_pf.mean().min(),R.groupby('st').rho_pf.mean().max()))
    R['blk']=((R.date-R.date.min()).dt.days//14); G=R.groupby('blk').indices; keys=list(G); rng=np.random.default_rng(5)
    tab=[]
    for rl in rules:
        p=R[f'{rl}_BetP'].values; dec=p>.5; acc=(dec==y).mean(); conf=np.where(dec,p,1-p).mean(); ic,sl=T.logit_fit(p,y)
        tab.append(dict(h=h,rule=rl,acc=acc,gap=conf-acc,brier=((p-y)**2).mean(),ECE=T.ece(p,y),slope=sl,mT=R[f'{rl}_mT'].mean(),m0=R[f'{rl}_m0'].mean(),Bel=R[f'{rl}_BelH'].mean(),Pl=R[f'{rl}_PlH'].mean()))
    pairs=[('dempster','ccr'),('ccr','mean'),('dempster','mean')]; bs={p:[] for p in pairs}
    for _ in range(2000):
        idx=np.concatenate([G[k] for k in rng.choice(keys,len(keys))]); yy=y[idx]
        for a,b in pairs: bs[(a,b)].append(((R[a+'_BetP'].values[idx]-yy)**2).mean()-((R[b+'_BetP'].values[idx]-yy)**2).mean())
    for a,b in pairs:
        pt=((R[a+'_BetP']-y)**2).mean()-((R[b+'_BetP']-y)**2).mean(); ci=np.percentile(bs[(a,b)],[2.5,97.5]); ci2=np.percentile(bs[(a,b)],[0.83,99.17])
        print(f'Brier({a})-Brier({b}) = {pt:+.4f}  95% [{ci[0]:+.4f},{ci[1]:+.4f}]  Bonferroni 98.3% [{ci2[0]:+.4f},{ci2[1]:+.4f}]')
    out=pd.DataFrame(tab); print(out.round(3).to_string(index=False)); print('blocks',len(keys))
    return R,out
res=[]
for h in (1,3,7):
    R,out=analyse(h); res.append(out)
    if h==1:
        z=np.concatenate([R.z0,R.z1,R.z2]); print('standardised errors: sd %.2f skew %.2f exkurt %.2f cov95 %.3f'%(z.std(),stats.skew(z),stats.kurtosis(z),(abs(z)<1.96).mean()))
        st=R.groupby('st').apply(lambda g:pd.Series({'n':len(g),'gap_D':np.where(g.dempster_BetP>.5,g.dempster_BetP,1-g.dempster_BetP).mean()-((g.dempster_BetP>.5)==g.truth).mean(),'gap_C':np.where(g.ccr_BetP>.5,g.ccr_BetP,1-g.ccr_BetP).mean()-((g.ccr_BetP>.5)==g.truth).mean(),'gap_M':np.where(g.mean_BetP>.5,g.mean_BetP,1-g.mean_BetP).mean()-((g.mean_BetP>.5)==g.truth).mean()}))
        print(st.round(3).to_string())
        for mp,cap in [('gauss',0.99),('t5',0.95)]:
            T.MAP=mp; T.CAP=cap; Q=T.run(1); yy=Q.truth.values
            for rl in ['dempster','ccr','mean']:
                p=Q[f'{rl}_BetP'].values; dec=p>.5; a=(dec==yy).mean(); c=np.where(dec,p,1-p).mean()
                print(f'  mapping {mp} cap {cap} {rl}: gap {c-a:+.3f} brier {((p-yy)**2).mean():.3f} ECE {T.ece(p,yy):.3f}')
        T.MAP='gauss'; T.CAP=0.95
pd.concat(res).to_csv(f'beijing_{VAR}_{TR}_summary.csv',index=False)
