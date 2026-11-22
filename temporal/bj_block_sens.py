"""Beijing: number of blocks and sensitivity of paired Brier differences to block length (7/14/28 days). Usage: BJ_DIR=bj_data python bj_block_sens.py"""
import os, glob, numpy as np, pandas as pd
os.environ.setdefault('ICOS_DIR','/nonexistent')
import temporal_extended as T
for var in ('NO2','PM2.5'):
    S={}
    for f in sorted(glob.glob(os.path.join(os.environ.get('BJ_DIR','bj_data'),'PRSA_Data_*.csv'))):
        x=pd.read_csv(f); st=x.station.iloc[0]; x['t']=pd.to_datetime(x[['year','month','day','hour']])
        v=x.set_index('t')[var]; g=v.groupby(v.index.normalize()).agg(['mean','count']); g=g[g['count']>=18]['mean']; S[st]=np.log1p(g.asfreq('D'))
    T.S=S; R=T.run(1); y=R.truth.values; rng=np.random.default_rng(5)
    cov=R.groupby('st').size()
    print(f'{var}: origins per station min {cov.min()} max {cov.max()}')
    for L in (7,14,28):
        b=((R.date-R.date.min()).dt.days//L); G=R.groupby(b).indices; keys=list(G); sz=[len(G[k]) for k in keys]
        res={}
        for a,c in [('dempster','ccr'),('ccr','mean'),('dempster','mean')]:
            d=[]
            for _ in range(2000):
                idx=np.concatenate([G[k] for k in rng.choice(keys,len(keys))])
                d.append(((R[a+'_BetP'].values[idx]-y[idx])**2).mean()-((R[c+'_BetP'].values[idx]-y[idx])**2).mean())
            pt=((R[a+'_BetP']-y)**2).mean()-((R[c+'_BetP']-y)**2).mean(); ci=np.percentile(d,[2.5,97.5]); res[a+'-'+c]=f'{pt:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]'
        print(f'  block {L}d: blocks {len(keys)}, origins/block min {min(sz)} median {int(np.median(sz))} max {max(sz)}; '+'; '.join(f'{k} {v}' for k,v in res.items()))
