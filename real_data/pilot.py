import pandas as pd, numpy as np, glob, io, json, sys
import os
A=os.environ.get('ICOS_DIR','icos_data/')
A=A if A.endswith('/') else A+'/'   # folder with ICOS ATC *.CO2 files
S26=['CBW','JFJ','PRS','LMP','PAL','SSL','MHD']
def icos(code):
    f=[x for x in glob.glob(A+'*.CO2') if f'_{code}_' in x][0]
    L=open(f,encoding='utf-8',errors='replace').read().split('\n')
    i=[j for j,l in enumerate(L) if l.startswith('#Site')][0]
    x=pd.read_csv(io.StringIO('\n'.join([L[i][1:]]+L[i+1:])),sep=';')
    x['t']=pd.to_datetime(x[['Year','Month','Day','Hour']].rename(columns=str.lower))
    return x[(x.co2>0)&(x.Flag!='N')][['t','co2']]
# daily afternoon (11-14 UTC) mean per station
pm={}
for s in S26:
    x=icos(s); x=x[(x.t.dt.hour>=11)&(x.t.dt.hour<=14)]
    g=x.groupby(x.t.dt.normalize()).co2.agg(['mean','count']); g=g[g['count']>=2]['mean']
    pm[s]=g
P=pd.DataFrame(pm).sort_index(); P=P[P.index>='2026-04-01']
W=10
def base(s,d):
    v=P[s]; m=(abs((v.index-d).days)<=W)&(v.index!=d); u=v[m].dropna()
    return u.mean() if len(u)>=5 else np.nan
# BASE and NET for every station-day
B=pd.DataFrame({s:[base(s,d) for d in P.index] for s in S26},index=P.index)
ANOM=P-B   # station anomaly vs own baseline
NET=pd.DataFrame(index=P.index,columns=S26,dtype=float)
for s in S26:
    oth=[k for k in S26 if k!=s]
    cnt=ANOM[oth].notna().sum(axis=1)
    NET[s]=(B[s]+ANOM[oth].mean(axis=1)).where(cnt>=2)
# overpass station-days
c=pd.read_csv('../colocated_days.csv',parse_dates=['day']); c=c[c.st.isin(S26)].copy()
c['y']=c.apply(lambda r:P.loc[r.day,r.st] if r.day in P.index and not np.isnan(P.loc[r.day,r.st]) else np.nan,axis=1)
c['base']=c.apply(lambda r:B.loc[r.day,r.st],axis=1); c['net']=c.apply(lambda r:NET.loc[r.day,r.st],axis=1)
c=c.dropna(subset=['y','base','net','sat']).reset_index(drop=True)
# network/baseline error scale from all station-days (excluding target day)
eB=(B-P).stack(); eN=(NET-P).stack().dropna()
def rms(x): return float(np.sqrt(np.mean(np.square(x))))
def run(emax=8.0,kthr=0.5,sat_unc_extra=False):
    out=[]
    for i,r in c.iterrows():
        d=r.day; o=c.day!=d
        bias=(c.sat-c.y)[o].mean(); e_sat=rms((c.sat-c.y-bias)[o])
        eb=rms(eB[eB.index.get_level_values(0)!=d]); en=rms(eN[eN.index.get_level_values(0)!=d])
        src=np.array([r.sat-bias, r.net, r.base]); eps=np.array([e_sat,en,eb])
        sig=np.minimum(0.9,eps/emax)
        res=dict(st=r.st,day=d,y=r.y,sat=src[0],net=src[1],base=src[2],eps_sat=e_sat,eps_net=en,eps_base=eb)
        res['mean3']=src.mean()
        w=1-sig; res['rw3']=(w*src).sum()/w.sum()
        iv=1/eps**2; res['ivw3']=(iv*src).sum()/iv.sum()
        # evidential (simple supports on mutually exclusive hypotheses)
        def evid(idx):
            sg=sig[idx]; n=len(sg)
            ms=np.array([(1-sg[i])*np.prod(np.delete(sg,i)) for i in range(n)]); mt=np.prod(sg)
            K=1-ms.sum()-mt
            if K<kthr: ms=ms/(1-K); mt=mt/(1-K)
            else: mt=mt+K                    # Yager
            bet=ms+mt/n
            return float((bet*src[idx]).sum()), K, K>=kthr
        res['evid3'],res['K3'],res['yager3']=evid([0,1,2])
        res['evid2'],res['K2'],res['yager2']=evid([0,2])   # P3-style: satellite + baseline
        w2=1-sig[[0,2]]; res['rw2']=(w2*src[[0,2]]).sum()/w2.sum()
        res['mean2']=src[[0,2]].mean()
        out.append(res)
    return pd.DataFrame(out)
M=['sat','net','base','mean3','rw3','ivw3','evid3','mean2','rw2','evid2']
def boot(R,B_=4000,seed=1):
    rng=np.random.default_rng(seed); days=R.day.unique()
    idx={d:np.where(R.day==d)[0] for d in days}
    err={m:(R[m]-R.y).values for m in M}
    def stat(sel):
        ii=np.concatenate([idx[d] for d in sel]); return {m:np.sqrt(np.mean(err[m][ii]**2)) for m in M}
    pt=stat(days); bs=[stat(rng.choice(days,len(days))) for _ in range(B_)]
    return pt,pd.DataFrame(bs)
if __name__=='__main__':
    R=run(); R.to_csv('pilot_results_emax8.csv',index=False)
    print('n station-days',len(R),'days',R.day.nunique(),'| mean eps sat/net/base',R[['eps_sat','eps_net','eps_base']].mean().round(2).to_dict())
    pt,bs=boot(R)
    tab=pd.DataFrame({'RMSE':pt,'lo':bs.quantile(.025),'hi':bs.quantile(.975)}).round(2); print(tab.to_string())
    for a,b in [('evid3','rw3'),('evid3','ivw3'),('evid3','net'),('evid3','mean3'),('evid2','rw2'),('evid2','sat'),('evid2','base'),('rw3','net')]:
        dd=(bs[a]-bs[b]); print(f'{a}-{b}: {pt[a]-pt[b]:+.2f}  [{dd.quantile(.025):+.2f},{dd.quantile(.975):+.2f}]')
    print('Yager share 3-src',R.yager3.mean().round(2),'2-src',R.yager2.mean().round(2),' mean K3',R.K3.mean().round(2))
    for em in [5,8,12]:
        for kt in [0.5,0.8]:
            Rr=run(em,kt); print('emax',em,'kthr',kt,{m:round(rms(Rr[m]-Rr.y),2) for m in ['rw3','evid3','evid2','rw2']})
