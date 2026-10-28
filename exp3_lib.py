import numpy as np, json, sys
import fast
from fast import *
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
LO=np.array([CO_R[0],CH_R[0],NO_R[0]]); HI=np.array([CO_R[1],CH_R[1],NO_R[1]])
BINS=[30,60,80]
def mkR(sc): 
    r=np.array([[np.exp(-((k-l)**2)/sc**2) for l in range(4)] for k in range(4)]); return r/r.sum(1,keepdims=True)
def compw(co,ch,no,w=(0.40,0.35,0.25)):
    return (w[0]*nrm(co,*CO_R)+w[1]*nrm(ch,*CH_R)+w[2]*nrm(no,*NO_R))*100
def field(seeds,smooth=True,off=0):
    out=[]
    for sd in seeds:
        rng=np.random.RandomState(int(sd)+off); raw=rng.randn(6,6)
        if smooth: sm=gaussian_filter(raw,1.5); out.append(sm/(np.abs(sm).max()+1e-10))
        else: out.append(np.clip(raw/2.5,-1,1))
    return np.array(out)
def gen(M,seeds,noise,kind='base'):
    f=field(seeds,smooth=(kind!='white'))*noise
    if kind=='indep':
        f2=field(seeds,off=7919)*noise; f3=field(seeds,off=15401)*noise
        fco,fch,fno=f,f2,f3
    else: fco=fch=fno=f
    bc,bh,bn=(1.05,1.005,1.05) if kind=='bias' else (1.0,1.0,1.0)
    co=M[:,0,None,None]*bc*(1+fco); ch=M[:,1,None,None]*bh*(1+0.3*fch); no=M[:,2,None,None]*bn*(1+fno)
    if kind=='outlier':
        rng=np.random.default_rng(int(seeds[0])+5); om=rng.random(co.shape)<0.05; fac=np.where(om,rng.choice([0.4,2.5],co.shape),1.0)
        co=co*fac; ch=ch*np.where(om,fac**0.3,1); no=no*fac
    co=np.clip(co,0.005,0.085); ch=np.clip(ch,1800,1950); no=np.clip(no,1e-7,8e-5)
    mask=np.ones(co.shape,bool)
    if kind=='cloud':
        cf=field(seeds,off=31337); thr=np.quantile(cf.reshape(len(cf),-1),0.30,axis=1)[:,None,None]; mask=cf>thr
    return co,ch,no,mask
def pipe(M,seeds,noise=0.10,kind='base',conf=0.85,w=(0.40,0.35,0.25),sigc=1.5,iters=20,bins=BINS):
    fast.R=mkR(sigc)
    co,ch,no,mask=gen(M,seeds,noise,kind); score=compw(co,ch,no,w); lab=np.clip(np.digitize(score,bins),0,3)
    P0=p0_labels_b(lab,conf); P0[~mask]=0.25
    out={}
    mk=mask[...,None]; n=mask.sum((1,2))
    def agg(P):
        avg=(P*mk).sum((1,2))/n[:,None]; H=-(avg*np.log(avg+1e-12)).sum(-1)
        zs=((P*LV).sum(3)*mask).sum((1,2))/n
        return zs,(1-H/np.log(4))*100,avg
    Pr=fast.relax_b(P0,iters)
    out['relax']=agg(Pr); out['cls']=agg(P0)
    ms=(score*mask).sum((1,2))/n; out['pm']=ms
    d=np.where(mask[...,None],np.exp(-((score[...,None]-LV)**2)/(2*8.0**2))+1e-12,1.0); d=d/d.sum(-1,keepdims=True)
    out['raw']=agg(fast.relax_b(d,iters))
    return out
def preds(o,bins=BINS):
    cl=lambda s:np.clip(np.digitize(s,bins),0,3)
    return {'relax':o['relax'][2].argmax(1),'raw':o['raw'][2].argmax(1),'cls':o['cls'][2].argmax(1),'pm':cl(o['pm'])}
def scores(o): return {'relax':o['relax'][0],'raw':o['raw'][0],'cls':o['cls'][0],'pm':o['pm']}
NAMES={'relax':'Classify, then relax (platform)','raw':'Raw soft initialisation, then relax','cls':'Classify only','pm':'Pixel-mean score'}
def zones(n,seed):
    rng=np.random.default_rng(seed); u=rng.uniform(0.05,0.95,n); dev=rng.normal(0,0.06,(n,3))
    return LO+np.clip(u[:,None]+dev,0.02,0.98)*(HI-LO)
def setup(means,ns,off=1000):
    Z=len(means); return np.repeat(means,ns,axis=0),np.tile(np.arange(off,off+ns),Z)
def boot(v,Z,ns,B=2000,seed=1):  # v per-pair array -> cluster bootstrap over zones
    zm=v.reshape(Z,ns).mean(1); rng=np.random.default_rng(seed); idx=rng.integers(0,Z,(B,Z)); bs=zm[idx].mean(1)
    return zm.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5)
