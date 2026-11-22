import pandas as pd, numpy as np, io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import temporal as T
R=pd.read_pickle('R30.pkl')
R['date']=[T.S[s].index[t] for s,t in zip(R.st,R.t)]
R['blk']=((R.date-R.date.min()).dt.days//14)   # calendar blocks shared by all stations
y=R.truth.values; cols=['bp_dempster','bp_yager','bp_murphy','bp_bayes_ind','bp_ccr','bp_mean']
groups=R.groupby('blk').indices; keys=list(groups); rng=np.random.default_rng(2)
out={c:[] for c in cols}; bd=[]; bm=[]
for _ in range(2000):
    idx=np.concatenate([groups[k] for k in rng.choice(keys,len(keys))]); yy=y[idx]
    for c in cols:
        p=R[c].values[idx]; d=p>.5; out[c].append(np.where(d,p,1-p).mean()-(d==yy).mean())
    bd.append(((R.bp_dempster.values[idx]-yy)**2).mean()-((R.bp_ccr.values[idx]-yy)**2).mean())
    bm.append(((R.bp_ccr.values[idx]-yy)**2).mean()-((R.bp_mean.values[idx]-yy)**2).mean())
print('calendar blocks',len(keys))
for c in cols: print(c,np.percentile(out[c],[2.5,97.5]).round(3))
print('Brier D-CCR',np.percentile(bd,[2.5,97.5]).round(4),' CCR-mean',np.percentile(bm,[2.5,97.5]).round(4), (R.bp_ccr.sub(y)**2).mean()-(R.bp_mean.sub(y)**2).mean())
# other threshold: h = above trailing mean + 0 is current; skip. horizon note only.
