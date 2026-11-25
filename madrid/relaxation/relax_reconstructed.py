import os
HERE=os.path.dirname(os.path.abspath(__file__))
# RECONSTRUCTED reference implementation of the Rosenfeld probabilistic relaxation (original code unavailable)
import pandas as pd, numpy as np, json, itertools
from scipy.stats import norm
lo=pd.read_csv(os.path.join(HERE,'..','loso.csv'))
st=pd.read_csv(os.path.join(HERE,'stations.csv')); xy={r.code:(r.x,r.y) for r in st.itertuples()}
EDGES=np.array([15,30,45]); K=4
def cls(v): return np.searchsorted(EDGES,v,side='right')   # 0:<15,1:15-30,2:30-45,3:>=45
lo['tcls']=cls(lo.truth.values)
dates=sorted(lo.date.unique())
# --- frozen choices ---
NTRAIN=70                      # first 70 days: training period for centroids only
SIG_FALLBACK=6.5; WARM=10      # sigma of fusion error from earlier days only
tr=lo[lo.date.isin(dates[:NTRAIN])]
cent=np.array([tr.truth[tr.tcls==k].mean() for k in range(K)])
print('centroids (train mean per class):',cent.round(2), 'train counts',np.bincount(tr.tcls,minlength=4))
# predefined compatibility matrices R in [-1,1] (Rosenfeld 1976), by ordinal distance 0,1,2,3
RS={'weak':[1,.25,-.25,-.5],'medium':[1,.5,-.5,-1],'strong':[1,.75,-.75,-1]}
def Rmat(v): return np.array([[v[abs(i-j)] for j in range(K)] for i in range(K)])
# sigma expanding
err=(lo.truth-lo.fuse_var).groupby(lo.date).apply(lambda s:(s**2).sum()).reindex(dates)
cnt=lo.groupby('date').size().reindex(dates)
sig={}
for t,d in enumerate(dates):
    if t<WARM: sig[d]=SIG_FALLBACK
    else: sig[d]=np.sqrt(err.iloc[:t].sum()/cnt.iloc[:t].sum())
def init_p(z,s):
    c=norm.cdf((np.r_[EDGES,np.inf]-z)/s)-norm.cdf((np.r_[-np.inf,EDGES]-z)/s)
    return c/c.sum()
def relax(p0,nbr_cls,R,T):
    p=p0.copy(); oh=np.eye(K)[nbr_cls]          # neighbours observed: one-hot (fixed)
    for _ in range(T):
        q=(oh@R.T).mean(0)                       # q(l)=mean_j sum_l' r(l,l') p_j(l')
        p=p*(1+q); p/=p.sum()
    return p
rows=[]
for d,g in lo.groupby('date'):
    g=g.reset_index(drop=True)
    for i,r in g.iterrows():
        o=g.drop(i); dd=[np.hypot(xy[c][0]-xy[r.code][0],xy[c][1]-xy[r.code][1]) for c in o.code]
        idx=np.argsort(dd)[:4]; nb=o.iloc[idx]
        rows.append((d,int(r.code),r.truth,r.fuse_var,sig[d],nb.code.tolist(),nb.tcls.values,r.tcls))
res={}
def run(name,T):
    R=Rmat(RS[name]); out=[]
    for d,c,tru,fz,s,nbc,nbk,tk in rows:
        p0=init_p(fz,s); p=relax(p0,nbk,R,T)
        out.append((p0@cent,p@cent,p0.argmax(),p.argmax()))
    return np.array(out)
tc=np.array([r[7] for r in rows]); truth=np.array([r[2] for r in rows]); fz=np.array([r[3] for r in rows])
dts=np.array([r[0] for r in rows])
rm=lambda e:np.sqrt(np.mean(e**2))
print('n=',len(rows),'RMSE fuse_var %.3f'%rm(truth-fz))
tab=[]
for name in RS:
    for T in (1,3,10):
        o=run(name,T)
        tab.append((name,T,rm(truth-o[:,0]),rm(truth-o[:,1]),(o[:,2]==tc).mean(),(o[:,3]==tc).mean()))
t=pd.DataFrame(tab,columns=['matrix','iters','RMSE_class','RMSE_relax','acc_class','acc_relax']); print(t.round(3).to_string())
t.to_csv('step2_grid.csv',index=False)
# primary (frozen before running): medium, 3 iterations
o=run('medium',3)
pd.DataFrame({'date':dts,'heldout_station':[r[1] for r in rows],'observed_NO2':truth,'fuse_var':fz,
 'class_est':o[:,0],'relaxed_estimate':o[:,1],'neighbour_codes':[' '.join(map(str,r[5])) for r in rows]}).to_csv('step2_predictions.csv',index=False)
# paired block bootstrap (7-day blocks over consecutive days)
ud=np.array(dates); rng=np.random.default_rng(1)
def boot(a,b,B=2000,L=7):
    idx={d:np.where(dts==d)[0] for d in ud}; nb=int(np.ceil(len(ud)/L)); res=[]
    for _ in range(B):
        st_=rng.integers(0,len(ud)-L+1,nb); sel=np.concatenate([idx[d] for s in st_ for d in ud[s:s+L]])
        res.append(rm(truth[sel]-b[sel])-rm(truth[sel]-a[sel]))
    return np.percentile(res,[2.5,97.5])
print('primary RMSE class %.3f relax %.3f'%(rm(truth-o[:,0]),rm(truth-o[:,1])))
print('relax - class  %.3f CI'%(rm(truth-o[:,1])-rm(truth-o[:,0])),boot(o[:,0],o[:,1]).round(3))
print('class - fuse   %.3f CI'%(rm(truth-o[:,0])-rm(truth-fz)),boot(fz,o[:,0]).round(3))
late=dts>=ud[NTRAIN]
print('later period only (days>=71): fuse %.3f class %.3f relax %.3f n=%d'%(rm(truth[late]-fz[late]),rm(truth[late]-o[late,0]),rm(truth[late]-o[late,1]),late.sum()))
