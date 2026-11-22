"""Extended temporal analysis (reviewer 5): full mass vectors incl. empty set, normalisation effects,
calibration (ECE, slope/intercept, reliability bins), Brier differences with calendar-block bootstrap,
base-rate reference, per-station error correlations, horizons h in {1,3,7}.
Usage: ICOS_DIR=icos9 python temporal_extended.py   (writes ext_*.csv)"""
import numpy as np, pandas as pd, glob, io, os, sys
from scipy.stats import norm
from scipy.optimize import minimize
A=os.environ.get('ICOS_DIR','icos_data/'); A=A if A.endswith('/') else A+'/'
W=30
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
def conj(a,b):  # a,b = (H,N,T); returns (H,N,T,empty)
    return (a[0]*b[0]+a[0]*b[2]+a[2]*b[0], a[1]*b[1]+a[1]*b[2]+a[2]*b[1], a[2]*b[2], a[0]*b[1]+a[1]*b[0])
def combine(ms,rule):
    """returns full unnormalised vector (H,N,T,empty) for the rule"""
    if rule in('dempster','yager'):
        r=(*ms[0],0.0)
        for m in ms[1:]: r=conj(r[:3],m) if r[3]==0 else conj(r[:3],m)  # sequential on unnormalised comps
        # conjunctive rule is associative on unnormalised masses: recompute properly
        r=(*ms[0],0.0)
        for m in ms[1:]:
            h,n,t,e=conj(r[:3],m); r=(h,n,t,r[3]+e*1.0)  # placeholder, replaced below
        return None
def conj_full(ms):
    # unnormalised conjunctive combination over {H,N,T,empty}: empty is absorbing
    cur=(ms[0][0],ms[0][1],ms[0][2],0.0)
    for m in ms[1:]:
        h,n,t,e=conj(cur[:3],m); cur=(h,n,t,cur[3]+e)
    return cur
def rule_vec(ms,rule):
    if rule=='dempster': return conj_full(ms)
    if rule=='yager':
        h,n,t,e=conj_full(ms); return (h,n,t+e,0.0), e
    if rule=='mean': return tuple(np.mean([m[k] for m in ms],0) for k in range(3))+(0.0,)
    if rule=='ccr':
        wH=min(1-m[0] if m[0]>0 else 1.0 for m in ms); wN=min(1-m[1] if m[1]>0 else 1.0 for m in ms)
        return conj((1-wH,0.0,wH),(0.0,1-wN,wN))
def summarize(vec):
    h,n,t,e=vec; z=1-e
    return dict(mH=h,mN=n,mT=t,m0=e,BelH=h/z,PlH=(h+t)/z,mTn=t/z,BetP=(h+t/2)/z,BelH_un=h,PlH_un=h+t)
CAP=0.95; MAP='gauss'
def support(d,sg):
    from scipy.stats import t as tdist
    u=np.abs(d)/sg
    if MAP=='t5': u=u/np.sqrt(5/3)
    return np.clip((2*(tdist.cdf(u,5)-0.5) if MAP=='t5' else 2*norm.cdf(u)-1),0,CAP)
def run(h):
    rows=[]
    for st,x in S.items():
        v=x.values; n=len(v); ks=['past','pres','fc']
        err={k:np.full(n,np.nan) for k in ks}; pred={k:np.full(n,np.nan) for k in ks}
        for t in range(40,n-h):
            w=v[t-6:t+1]
            if np.isnan(w).sum()>2 or np.isnan(v[t]) or np.isnan(v[t+h]): continue
            ep=np.nanmean(w); a=[];b=[]
            for s in range(max(8,t-60-h+1),t-h+1):
                ws=v[s-6:s+1]
                if np.isnan(ws).sum()>2 or np.isnan(v[s]) or np.isnan(v[s+h]): continue
                e_=np.nanmean(ws); a.append(v[s]-e_); b.append(v[s+h]-e_)
            phi=np.clip(np.dot(a,b)/np.dot(a,a),0,1) if len(a)>=10 and np.dot(a,a)>0 else 0.5
            pred['past'][t]=ep; pred['pres'][t]=v[t]; pred['fc'][t]=ep+phi*(v[t]-ep)
            for k in ks: err[k][t]=pred[k][t]-v[t+h]
        for t in range(40,n-h):
            if np.isnan(err['past'][t]): continue
            Hh=[err[k][t-W-h+1:t-h+1] for k in ks]; ok=~np.isnan(Hh[0])
            if ok.sum()<max(8,W*0.4): continue
            Hm=np.array([q[ok] for q in Hh]); sg=np.sqrt((Hm**2).mean(1))
            e=np.array([pred[k][t] for k in ks]); y=v[t+h]; base=np.nanmean(v[t-W+1:t+1])
            if np.isnan(base): continue
            d=e-base; s=support(d,sg)
            ms=[((si,0.0,1-si) if di>0 else (0.0,si,1-si)) for di,si in zip(d,s)]
            r=dict(st=st,t=t,date=x.index[t],truth=int(y>base),rho_pf=np.corrcoef(Hm[0],Hm[2])[0,1],rho_pp=np.corrcoef(Hm[0],Hm[1])[0,1],rho_cf=np.corrcoef(Hm[1],Hm[2])[0,1],z0=(pred['past'][t]-y)/sg[0],z1=(pred['pres'][t]-y)/sg[1],z2=(pred['fc'][t]-y)/sg[2],sg0=sg[0],sg1=sg[1],sg2=sg[2],d0=d[0],d1=d[1],d2=d[2])
            for rl in ['dempster','yager','mean','ccr']:
                out=rule_vec(ms,rl)
                if rl=='yager': vec,c=out; sm=summarize(vec); sm['m0']=c   # report conflict moved to Theta
                else: sm=summarize(out)
                for k,val in sm.items(): r[f'{rl}_{k}']=val
            rows.append(r)
    return pd.DataFrame(rows)
def ece(p,y,bins=10):
    b=np.minimum((p*bins).astype(int),bins-1); e=0
    for k in range(bins):
        m=b==k
        if m.any(): e+=m.mean()*abs(p[m].mean()-y[m].mean())
    return e
def logit_fit(p,y):
    z=np.log(np.clip(p,1e-3,1-1e-3)/(1-np.clip(p,1e-3,1-1e-3)))
    f=lambda b:-(y*(b[0]+b[1]*z)-np.log1p(np.exp(b[0]+b[1]*z))).sum()
    r=minimize(f,[0,1],method='BFGS'); return r.x   # intercept, slope
if __name__=='__main__':
    if not S: sys.exit(f'No ICOS *.CO2 files found in {A!r}. Put them in data/icos or set ICOS_DIR (see data/README_data.md).')
    rules=['dempster','yager','mean','ccr']; res_all={}
    for h in [1,3,7]:
        R=run(h); R.to_pickle(f'ext_R_h{h}.pkl'); y=R.truth.values
        print(f'\n=== horizon {h}: origins {len(R)}, stations {R.st.nunique()}, base rate {y.mean():.3f}, Brier(base rate) {y.mean()*(1-y.mean()):.3f}')
        R['blk']=((R.date-R.date.min()).dt.days//14); G=R.groupby('blk').indices; keys=list(G); rng=np.random.default_rng(5)
        tab=[]
        for rl in rules:
            p=R[f'{rl}_BetP'].values; dec=p>.5; acc=(dec==y).mean(); conf=np.where(dec,p,1-p).mean()
            ic,sl=logit_fit(p,y)
            tab.append(dict(h=h,rule=rl,acc=acc,over=conf-acc,brier=((p-y)**2).mean(),ECE=ece(p,y),slope=sl,intercept=ic,
                 m0=R[f'{rl}_m0'].mean(),mH=R[f'{rl}_mH'].mean(),mN=R[f'{rl}_mN'].mean(),mT=R[f'{rl}_mT'].mean(),
                 Bel=R[f'{rl}_BelH'].mean(),Pl=R[f'{rl}_PlH'].mean(),mTn=R[f'{rl}_mTn'].mean(),Bel_un=R[f'{rl}_BelH_un'].mean(),Pl_un=R[f'{rl}_PlH_un'].mean()))
        T=pd.DataFrame(tab)
        # pairwise Brier differences with calendar-block bootstrap
        pairs=[('dempster','ccr'),('ccr','mean'),('dempster','mean')]; bs={pr:[] for pr in pairs}
        for _ in range(2000):
            idx=np.concatenate([G[k] for k in rng.choice(keys,len(keys))]); yy=y[idx]
            for a,b in pairs: bs[(a,b)].append(((R[f'{a}_BetP'].values[idx]-yy)**2).mean()-((R[f'{b}_BetP'].values[idx]-yy)**2).mean())
        for a,b in pairs:
            pt=((R[f'{a}_BetP']-y)**2).mean()-((R[f'{b}_BetP']-y)**2).mean(); ci=np.percentile(bs[(a,b)],[2.5,97.5])
            print(f'Brier({a})-Brier({b}) = {pt:+.4f}  95% [{ci[0]:+.4f},{ci[1]:+.4f}]')
        print(T.round(3).to_string(index=False)); res_all[h]=T
        if h==1:
            # reliability bins
            rb=[]
            for rl in rules:
                p=R[f'{rl}_BetP'].values; b=np.minimum((p*10).astype(int),9)
                for k in range(10):
                    m=b==k
                    if m.sum()>0: rb.append(dict(rule=rl,bin=k,n=int(m.sum()),conf=p[m].mean(),freq=y[m].mean()))
            pd.DataFrame(rb).to_csv('ext_reliability_h1.csv',index=False)
            # per-station correlations
            st=R.groupby('st').agg(n=('t','size'),rho_pp=('rho_pp','mean'),rho_pf=('rho_pf','mean'),rho_cf=('rho_cf','mean'))
            for rl in ['dempster','ccr','mean']:
                st['over_'+rl]=[ (np.where(g[f'{rl}_BetP']>.5,g[f'{rl}_BetP'],1-g[f'{rl}_BetP']).mean()-((g[f'{rl}_BetP']>.5)==g.truth).mean()) for _,g in R.groupby('st')]
            st.to_csv('ext_per_station_h1.csv'); print(st.round(2).to_string())
            # leave-one-station-out stability of the overconfidence ordering
            for out in R.st.unique():
                g=R[R.st!=out]; yy=g.truth.values
                o={rl:(np.where(g[f'{rl}_BetP']>.5,g[f'{rl}_BetP'],1-g[f'{rl}_BetP']).mean()-((g[f'{rl}_BetP']>.5)==yy).mean()) for rl in rules}
                print('LOSO excl',out,{k:round(v,3) for k,v in o.items()})
    pd.concat(res_all.values()).to_csv('ext_summary.csv',index=False)
