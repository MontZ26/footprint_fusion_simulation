import os
HERE=os.path.dirname(os.path.abspath(__file__))
import pandas as pd, numpy as np, json, sys
from scipy.stats import norm
lo=pd.read_csv(os.path.join(HERE,'..','loso.csv')); st=pd.read_csv(os.path.join(HERE,'stations.csv'))
xy={r.code:(r.x,r.y) for r in st.itertuples()}
E=np.array([15,30,45]); K=4; cls=lambda v:np.searchsorted(E,v,side='right')
lo['k']=cls(lo.truth.values); dates=sorted(lo.date.unique())
TR,VA,FI=dates[:70],dates[70:140],dates[140:]
# --- cases with neighbours (once) ---
cases=[]
for d,g in lo.groupby('date'):
    g=g.reset_index(drop=True)
    for i,r in g.iterrows():
        o=g.drop(i); dd=[np.hypot(xy[c][0]-xy[r.code][0],xy[c][1]-xy[r.code][1]) for c in o.code]
        idx=np.argsort(dd)[:4]; cases.append(dict(date=d,code=int(r.code),truth=r.truth,fz=r.fuse_var,k=int(r.k),nk=o.iloc[idx].k.values,nc=o.iloc[idx].code.values))
cs=pd.DataFrame(cases)
# --- training quantities ---
trm=cs[cs.date.isin(TR)]
cent=np.array([lo[lo.date.isin(TR)&(lo.k==k)].truth.mean() for k in range(K)])
s_phys=np.sqrt(((trm.truth-trm.fz)**2).mean())
P=np.ones((K,K)); D=[set() for _ in range(K)]
for r in trm.itertuples():
    for kj in r.nk: P[kj,r.k]+=1; D[kj].add(r.date)
pe=P/P.sum(1,keepdims=True); H=-(pe*np.log(pe)).sum(1)/np.log(4); Dk=np.array([len(x) for x in D])
Cphys=np.exp(-(cent[:,None]-cent[None,:])**2/(2*s_phys**2))   # [l,k]
Cemp=(pe/pe.max(1,keepdims=True)).T                            # [l,k]
def Chyb(kind,tau=None):
    if kind=='P1': C=Cphys
    elif kind=='P2': C=Cemp
    elif kind=='P3': C=.5*Cphys+.5*Cemp
    else:
        w=Dk/(Dk+tau)*(1-H); C=(1-w)[None,:]*Cphys+w[None,:]*Cemp
    return C/C.max(0,keepdims=True)
# sigma expanding
byd=lo.assign(e2=(lo.truth-lo.fuse_var)**2).groupby('date').e2.agg(['sum','count']).reindex(dates)
sig={d:(6.5 if t<10 else np.sqrt(byd['sum'].iloc[:t].sum()/byd['count'].iloc[:t].sum())) for t,d in enumerate(dates)}
ETA,M,T=0.5,3,3
def poss(r,C,gam,relax=True):
    S=np.exp(-(r.fz-cent)**2/(2*sig[r.date]**2)); S/=S.max(); pi=1-ETA+ETA*S
    if not relax: return pi,0.
    sj=np.array([C[:,k] for k in r.nk])             # 4 x K
    A=np.sort(sj,0)[::-1][M-1]; kap=1-A.max(); Ab=np.maximum(A,kap); Ab/=Ab.max()
    for _ in range(T):
        pi=(1-gam)*pi+gam*np.minimum(pi,Ab); pi/=pi.max()
    return pi,kap
EDG=np.r_[-np.inf,E,np.inf]
def p_init(z,s):
    c=norm.cdf((EDG[1:]-z)/s)-norm.cdf((EDG[:-1]-z)/s); return c/c.sum()
Rm=np.array([[[1,.5,-.5,-1][abs(i-j)] for j in range(K)] for i in range(K)])
def prob_relax(r,relax=True):
    p=p_init(r.fz,sig[r.date])
    if not relax: return p
    oh=np.eye(K)[r.nk]
    for _ in range(3):
        q=(oh@Rm.T).mean(0); p=p*(1+q); p/=p.sum()
    return p
rm=lambda e:np.sqrt(np.mean(e**2))
THS=np.r_[np.linspace(0.02,1,50)]
def curve_poss(PI,kt):   # coverage vs mean size over theta
    sz=[];cv=[]
    for th in THS:
        m=PI>=th; sz.append(m.sum(1).mean()); cv.append(m[np.arange(len(kt)),kt].mean())
    return np.array(sz),np.array(cv)
def curve_prob(PP,kt):
    sz=[];cv=[]; o=np.argsort(-PP,1)
    for lev in np.r_[np.linspace(0.3,0.99,50)]:
        cum=np.cumsum(np.take_along_axis(PP,o,1),1); n=(cum<lev).sum(1)+1
        inset=np.array([kt[i] in o[i,:n[i]] for i in range(len(kt))]); sz.append(n.mean()); cv.append(inset.mean())
    return np.array(sz),np.array(cv)
def cov_at(sz,cv,target=2.0):
    o=np.argsort(sz); return float(np.interp(target,sz[o],cv[o]))
def aurc(score,correct):
    o=np.argsort(-score,kind='stable'); c=correct[o]; return float(np.mean(np.cumsum(1-c)/np.arange(1,len(c)+1)))
def boot_diff(truth,a,b,dts,ud,B=1000,L=7,seed=3):
    rng=np.random.default_rng(seed); idx={d:np.where(dts==d)[0] for d in ud}; nb=int(np.ceil(len(ud)/L)); out=[]
    for _ in range(B):
        s=rng.integers(0,len(ud)-L+1,nb); sel=np.concatenate([idx[d] for q in s for d in ud[q:q+L]])
        out.append(rm(truth[sel]-b[sel])-rm(truth[sel]-a[sel]))
    return np.percentile(out,[2.5,97.5])
def evaluate(dset,configs=None):
    sub=cs[cs.date.isin(dset)].reset_index(drop=True); kt=sub.k.values; truth=sub.truth.values; fz=sub.fz.values; dts=sub.date.values; ud=np.array(sorted(dset))
    rows=[]
    def summarize(name,PI,kind,extra={}):
        if kind=='poss':
            w=PI/PI.sum(1,keepdims=True); zh=w@cent; sz,cv=curve_poss(PI,kt)
            top=np.argmax(PI*1e3+w,1)  # tie-break by normalised weight
            srt=np.sort(PI,1); nec=1-srt[:,-2]; sc=nec
        else:
            zh=PI@cent; sz,cv=curve_prob(PI,kt); top=PI.argmax(1); sc=PI.max(1)
        corr=(top==kt).astype(float)
        lo_,hi_=boot_diff(truth,fz,zh,dts,ud)
        rows.append(dict(model=name,RMSE=rm(truth-zh),dRMSE_vs_fuse=rm(truth-zh)-rm(truth-fz),dR_lo=lo_,dR_hi=hi_,cov_at_2=cov_at(sz,cv),acc=corr.mean(),AURC=aurc(sc,corr),**extra))
    summarize('prob_init',np.array([prob_relax(r,False) for r in sub.itertuples()]),'prob')
    summarize('prob_relax(medium,3)',np.array([prob_relax(r) for r in sub.itertuples()]),'prob')
    summarize('poss_init(no relax)',np.array([poss(r,Chyb('P1'),0.5,False)[0] for r in sub.itertuples()]),'poss')
    for kind,tau in [('P1',None),('P2',None),('P3',None),('P4',5),('P4',15),('P4',30)]:
        C=Chyb(kind,tau)
        for g in (.25,.5,.75):
            out=[poss(r,C,g) for r in sub.itertuples()]; PI=np.array([o[0] for o in out]); kap=np.array([o[1] for o in out])
            summarize(f'{kind}'+(f'_tau{tau}' if tau else '')+f'_g{g}',PI,'poss',dict(mean_kappa=kap.mean()))
    return pd.DataFrame(rows),(rm(truth-fz),len(sub))
if __name__=='__main__':
    print('cent',cent.round(2),'s_phys %.2f'%s_phys,'Dk',Dk,'H',H.round(3))
    print('omega tau5/15/30:',[np.round(Dk/(Dk+t)*(1-H),3).tolist() for t in (5,15,30)])
    which=sys.argv[1]; df,(rf,n)=evaluate(VA if which=='val' else FI)
    print(which,'n=',n,'plain fusion RMSE %.3f'%rf)
    df.to_csv(f'{which}_results.csv',index=False); pd.set_option('display.width',250); print(df.round(3).to_string())
