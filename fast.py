import numpy as np
from scipy.ndimage import gaussian_filter
from exact import *
def nbsum(P):
    pad=np.pad(P,((0,0),(1,1),(1,1),(0,0)))
    S=np.zeros_like(P)
    for di in (-1,0,1):
        for dj in (-1,0,1):
            if di==0 and dj==0: continue
            S+=pad[:,1+di:7+di,1+dj:7+dj]
    return S
def relax_b(P,it=20,eps=1e-4):
    P=P.copy(); active=np.ones(len(P),bool)
    for t in range(it):
        Q=nbsum(P)@R.T; Pn=P*Q; Pn/=Pn.sum(-1,keepdims=True)
        delta=np.abs(Pn-P).max(axis=(1,2,3))
        P=np.where(active[:,None,None,None],Pn,P)
        active=active&(delta>=eps)
        if not active.any(): break
    return P
def aggregate_b(P):
    avg=P.mean((1,2)); H=-(avg*np.log(avg+1e-12)).sum(-1)
    return (P*LV).sum(3).mean((1,2)),(1-H/np.log(4))*100,avg
def p0_labels_b(lab,conf):
    B=lab.shape[0]; P=np.zeros((B,6,6,4)); ks=np.arange(4)
    for k in range(4):
        w=np.array([1/(1+abs(l-k)) if l!=k else 0.0 for l in range(4)]); w=w/w.sum()*(1-conf)
        m=(lab==k); P[m]=w; P[m,k]=conf
    return P
def smooth_fields(seeds):
    out=[]
    for sd in seeds:
        rng=np.random.RandomState(int(sd)); raw=rng.randn(6,6); sm=gaussian_filter(raw,1.5); out.append(sm/(np.abs(sm).max()+1e-10))
    return np.array(out)
def run_batch(means,seeds,noise_sigma=0.10,conf=0.85,mode='classify',relax_on=True):
    # means: (B,3) ; seeds: (B,)
    sm=smooth_fields(seeds)*noise_sigma
    co=np.clip(means[:,0,None,None]*(1+sm),0.005,0.085); ch=np.clip(means[:,1,None,None]*(1+sm*0.3),1800,1950); no=np.clip(means[:,2,None,None]*(1+sm),1e-7,8e-5)
    score=comp(co,ch,no); lab=np.clip(np.digitize(score,[30,60,80]),0,3)
    P=p0_labels_b(lab,conf) if mode=='classify' else (lambda d:d/d.sum(-1,keepdims=True))(np.exp(-((score[...,None]-LV)**2)/(2*8.0**2))+1e-12)
    Pf=relax_b(P) if relax_on else P
    zs,cert,avg=aggregate_b(Pf)
    return dict(zs=zs,cert=cert,avg=avg,score=score)
def true_class(means): return np.clip(np.digitize(comp(means[:,0],means[:,1],means[:,2]),[30,60,80]),0,3)
if __name__=='__main__':
    # validation against loop implementation
    m=np.array(list(ZONES.values())); b=run_batch(m,np.full(len(m),42))
    for i,z in enumerate(ZONES):
        r=zone_sat(*ZONES[z]); assert abs(r['cert']-b['cert'][i])<1e-6 and abs(r['zone_score']-b['zs'][i])<1e-6,(z,r['cert'],b['cert'][i])
    print('batch implementation matches the exact loop implementation on all nine zones')
