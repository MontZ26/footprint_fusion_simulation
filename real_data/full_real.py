import numpy as np, pandas as pd, sys, json
sys.argv=['x']
import pilot as pl
import os
A=os.environ.get('OCO2_CSV','OCO2_near_ICOS.csv')   # output of oco2_extract.py
D=pd.read_csv(A); D=D[D.qflag==0].copy(); D['t']=pd.to_datetime(D.time_utc); D['day']=D.t.dt.normalize()
CELL=0.1; BINS=[30,60,80]; CENT=np.array([15,45,70,90.]); SC=1.5
Rm=np.exp(-(np.subtract.outer(range(4),range(4))**2)/SC**2); Rm=Rm/Rm.sum(1,keepdims=True)
def score(x,L,H): return 100*np.clip((np.asarray(x)-L)/(H-L),0,1)
def cls(s): return np.digitize(s,BINS)
def p0(k,c=0.85):
    p=np.array([1/(1+abs(l-k)) if l!=k else 0 for l in range(4)]); p=p/p.sum()*(1-c); p[k]=c; return p
def relax(P,nb,iters=20):
    P=P.copy()
    for _ in range(iters):
        Q=P.copy()
        for i in range(len(P)):
            if not nb[i]: continue
            s=sum(Rm@P[j] for j in nb[i])/len(nb[i])   # sum_l R(k,l)P_j(l), equal weights
            q=P[i]*s; Q[i]=q/q.sum()
        ch=np.abs(Q-P).max(); P=Q
        if ch<1e-4: break
    return P
def evid(s1,sg1,s2,sg2,kthr=0.5):
    K=(1-sg1)*(1-sg2); m1=(1-sg1)*sg2; m2=(1-sg2)*sg1; mt=sg1*sg2
    if K<kthr: z=1-K; m1,m2,mt=m1/z,m2/z,mt/z
    else: mt=mt+K
    b1=m1+mt/2; b2=m2+mt/2; return b1*s1+b2*s2, K>=kthr
def run(L=420,H=436,emax=8.0):
    rows=[]
    c=pl.c
    for i,r in c.iterrows():
        d=r.day; o=c.day!=d
        bias=(c.sat-c.y)[o].mean()
        sub=D[(D.station==r.st)&(D.day==d)].copy()
        sub['x']=sub.xco2-bias
        sub['ci']=np.floor(sub.lat/CELL).astype(int); sub['cj']=np.floor(sub.lon/CELL).astype(int)
        g=sub.groupby(['ci','cj']).x.mean().reset_index()
        n=len(g); sc=score(g.x.values,L,H); k=cls(sc)
        pos=list(zip(g.ci,g.cj)); idx={p:j for j,p in enumerate(pos)}
        nb=[[idx[(a+da,b+db)] for da in(-1,0,1) for db in(-1,0,1) if (da or db) and (a+da,b+db) in idx] for a,b in pos]
        P0=np.array([p0(x) for x in k]); P1=relax(P0,nb)
        zs_pix=sc.mean(); Pm0=P0.mean(0); Pm1=P1.mean(0)
        zs_cls=Pm0@CENT; zs_rel=Pm1@CENT
        cert=lambda P:1-(-(P[P>0]*np.log(P[P>0])).sum())/np.log(4)
        sg_sat=1-cert(Pm1)
        y=r.y; st=score(y,L,H)
        base=score(r.base,L,H)
        eb=pl.rms(pl.eB[pl.eB.index.get_level_values(0)!=d]); sg_b=min(.9,eb/emax)
        e_sat=pl.rms((c.sat-c.y-bias)[o]); sg_s_h=min(.9,e_sat/emax)
        res=dict(st=r.st,day=d,cells=n,truth=st,truth_cls=int(cls(st)),pix=zs_pix,cls=zs_cls,rel=zs_rel,base=base,sg_sat=sg_sat,sg_b=sg_b)
        res['ev_P3']=evid(zs_rel,sg_sat,base,sg_b)[0]; res['ev_hist']=evid(zs_rel,sg_s_h,base,sg_b)[0]
        res['ev_P3_y']=evid(zs_rel,sg_sat,base,sg_b)[1]
        w=np.array([1-sg_sat,1-sg_b]); res['rw']=(w*np.array([zs_rel,base])).sum()/w.sum()
        res['mean2']=(zs_rel+base)/2
        iv=np.array([1/e_sat**2,1/eb**2]); res['blue']=(iv*np.array([zs_rel,base])).sum()/iv.sum()
        res['ev_pix']=evid(zs_pix,sg_s_h,base,sg_b)[0]
        rows.append(res)
    return pd.DataFrame(rows)
M=['pix','cls','rel','base','ev_P3','ev_hist','rw','mean2','blue','ev_pix']
def boot(R,B=4000):
    rng=np.random.default_rng(3); days=R.day.unique(); ix={d:np.where(R.day==d)[0] for d in days}
    err={m:(R[m]-R.truth).values for m in M}
    f=lambda sel:{m:np.sqrt(np.mean(err[m][np.concatenate([ix[d] for d in sel])]**2)) for m in M}
    pt=f(days); bs=pd.DataFrame([f(rng.choice(days,len(days))) for _ in range(B)]); return pt,bs
if __name__=='__main__':
    out={}
    for (L,H) in [(420,436),(418,438),(422,434)]:
        R=run(L,H); pt,bs=boot(R)
        print(f'\nL={L} H={H}  n={len(R)} days={R.day.nunique()}  cells median {R.cells.median()} (min {R.cells.min()}, max {R.cells.max()})  Yager used: {R.ev_P3_y.mean():.2f}')
        print(pd.DataFrame({'RMSE':pt,'lo':bs.quantile(.025),'hi':bs.quantile(.975)}).round(1).T.to_string())
        for a,b in [('rel','pix'),('rel','cls'),('cls','pix'),('ev_P3','rw'),('ev_P3','mean2'),('ev_P3','blue'),('ev_P3','rel'),('ev_P3','base'),('ev_hist','blue')]:
            dd=bs[a]-bs[b]; print(f'  {a}-{b}: {pt[a]-pt[b]:+.1f} [{dd.quantile(.025):+.1f},{dd.quantile(.975):+.1f}]')
        acc={m:float((cls(R[m])==R.truth_cls).mean()) for m in ['pix','cls','rel','base','ev_P3','rw','mean2','blue']}
        print('  class accuracy',{k:round(v,2) for k,v in acc.items()})
        out[f'{L}_{H}']=dict(rmse=pt,acc=acc)
        if (L,H)==(420,436): R.to_csv('full_real_results.csv',index=False)
