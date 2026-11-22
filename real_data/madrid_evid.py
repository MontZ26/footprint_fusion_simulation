import os
import numpy as np, pandas as pd
r=pd.read_csv(os.environ.get('LOSO_CSV','../madrid/loso.csv'),parse_dates=['date']).dropna(subset=['truth','krig','sat_bias','sn']).reset_index(drop=True)
EMAX=30.0   # ug/m3 scale for sigma = eps/EMAX (sensitivity below)
def evid(s1,sg1,s2,sg2,kthr=.5):
    K=(1-sg1)*(1-sg2); m1=(1-sg1)*sg2; m2=(1-sg2)*sg1; mt=sg1*sg2
    if K<kthr: z=1-K; m1,m2,mt=m1/z,m2/z,mt/z
    else: mt+=K
    return (m1+mt/2)*s1+(m2+mt/2)*s2, K>=kthr
def one(emax=EMAX,kthr=.5,mode='hist'):
    # leave-one-day-out error scales
    ek=(r.krig-r.truth)**2; es=(r.sat_bias-r.truth)**2
    day=r.date; sk=ek.groupby(day).sum(); ss=es.groupby(day).sum(); n=r.groupby('date').size(); N=len(r)
    out=[]
    Ek=(sk.sum()-sk)/(N-n); Es=(ss.sum()-ss)/(N-n)
    epk=np.sqrt(day.map(Ek)); eps_=np.sqrt(day.map(Es))
    sgk=np.minimum(.9,epk/emax); sgs=np.minimum(.9,eps_/emax)
    if mode=='krigvar': sgk=np.minimum(.9,r.sn.clip(.02,1))
    ev=[];yg=[]
    for i in range(N):
        v,y=evid(r.krig[i],sgk[i],r.sat_bias[i],sgs[i],kthr); ev.append(v); yg.append(y)
    w1=1-sgk; w2=1-sgs
    o=pd.DataFrame(dict(date=r.date,truth=r.truth,krig=r.krig,sat=r.sat_bias,mean2=(r.krig+r.sat_bias)/2,
        rw=(w1*r.krig+w2*r.sat_bias)/(w1+w2),blue=(r.krig/epk**2+r.sat_bias/eps_**2)/(1/epk**2+1/eps_**2),
        evid=ev,student_fuse=r.fuse_var)); o['yager']=yg; return o
M=['krig','sat','mean2','rw','blue','evid','student_fuse']
def boot(o,B=3000):
    rng=np.random.default_rng(5); days=o.date.unique(); ix={d:np.where(o.date==d)[0] for d in days}
    err={m:(o[m]-o.truth).values for m in M}
    f=lambda sel:{m:np.sqrt(np.mean(err[m][np.concatenate([ix[d] for d in sel])]**2)) for m in M}
    pt=f(days); bs=pd.DataFrame([f(rng.choice(days,len(days))) for _ in range(B)]); return pt,bs
for mode in ['hist','krigvar']:
    o=one(mode=mode); pt,bs=boot(o)
    print(f'\nMadrid mode={mode}: n={len(o)} days={o.date.nunique()} Yager share={o.yager.mean():.2f}')
    print(pd.DataFrame({'RMSE':pt,'lo':bs.quantile(.025),'hi':bs.quantile(.975)}).round(2).T.to_string())
    for a,b in [('evid','krig'),('evid','mean2'),('evid','rw'),('evid','blue'),('blue','krig'),('rw','mean2')]:
        d=bs[a]-bs[b]; print(f'  {a}-{b}: {pt[a]-pt[b]:+.3f} [{d.quantile(.025):+.3f},{d.quantile(.975):+.3f}]')
    if mode=='hist': o.to_csv('madrid_evid_results.csv',index=False)
for em in [15,30,60]:
    for kt in [.5,.8]:
        o=one(em,kt); print('emax',em,'kthr',kt,{m:round(float(np.sqrt(((o[m]-o.truth)**2).mean())),3) for m in ['krig','rw','blue','evid']},'yager',round(o.yager.mean(),2))
