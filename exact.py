import numpy as np
from scipy.ndimage import gaussian_filter
LV=np.array([15.,45.,70.,90.])
CO_R=(0.020,0.080); CH_R=(1840,1940); NO_R=(1e-6,8e-5)
def nrm(v,lo,hi): return np.clip((v-lo)/(hi-lo),0,1)
def comp(co,ch,no): return (0.40*nrm(co,*CO_R)+0.35*nrm(ch,*CH_R)+0.25*nrm(no,*NO_R))*100
sc=1.5
R=np.array([[np.exp(-((k-l)**2)/sc**2) for l in range(4)] for k in range(4)]); R=R/R.sum(1,keepdims=True)
NB={(i,j):[(r,c) for r in range(max(0,i-1),min(6,i+2)) for c in range(max(0,j-1),min(6,j+2)) if (r,c)!=(i,j)] for i in range(6) for j in range(6)}
def grids(co,ch,no,seed=42,noise_sigma=0.10):
    rng=np.random.RandomState(seed); raw=rng.randn(6,6); sm=gaussian_filter(raw,sigma=1.5); sm=sm/(np.abs(sm).max()+1e-10)
    nz=noise_sigma*sm
    return np.clip(co*(1+nz),0.005,0.085),np.clip(ch*(1+nz*0.3),1800,1950),np.clip(no*(1+nz),1e-7,8e-5)
def p0_from_labels(lab,conf=0.85):
    P=np.zeros((6,6,4))
    for i in range(6):
        for j in range(6):
            k=lab[i,j]; w=np.array([1/(1+abs(l-k)) if l!=k else 0.0 for l in range(4)])
            if w.sum()>0: w=w/w.sum()*(1-conf)
            P[i,j]=w; P[i,j,k]=conf
    return P
def p0_raw(score,noise_std=8.0):
    d=np.exp(-((score[...,None]-LV)**2)/(2*noise_std**2))+1e-12
    return d/d.sum(-1,keepdims=True)
def relax(P,it=20,eps=1e-4):
    for t in range(it):
        Pn=np.zeros_like(P)
        for i in range(6):
            for j in range(6):
                Q=sum(R@P[a,b] for a,b in NB[(i,j)]); x=P[i,j]*Q; Pn[i,j]=x/x.sum()
        if np.abs(Pn-P).max()<eps: return Pn,t+1
        P=Pn
    return P,it
def aggregate(P):
    avg=P.mean((0,1)); H=-(avg*np.log(avg+1e-12)).sum()
    return (P*LV).sum(2).mean(),(1-H/np.log(4))*100
def zone_sat(co,ch,no,seed=42,conf=0.85,mode='classify',noise_sigma=0.10,iters=20):
    g=grids(co,ch,no,seed,noise_sigma); score=comp(*g); lab=np.clip(np.digitize(score,[30,60,80]),0,3)
    P=p0_from_labels(lab,conf) if mode=='classify' else p0_raw(score)
    if mode=='norelax_classify': Pf=P
    elif mode=='norelax_raw': Pf=P
    else: Pf,_=relax(P,iters)
    zs,cert=aggregate(Pf); return dict(zone_score=zs,cert=cert,sigma=1-cert/100,score=score,lab=lab,P=Pf)
def fuse(zs,sig_s,edgar=35.0,sig_e=0.45):
    ms=1-sig_s; me=1-sig_e; K=ms*me
    if K<0.5: den=1-K+1e-10; a=ms*sig_e/den; b=sig_s*me/den; t=sig_s*sig_e/den; rule='Dempster'
    else: a=ms*sig_e; b=sig_s*me; t=sig_s*sig_e+K; rule='Yager'
    bs=a+t/2; be=b+t/2
    return dict(K=K,rule=rule,fused=bs*zs+be*edgar,cert=(1-t)*100,bs=bs)
ZONES={'Tunis':(0.055789,1864.3,5.03e-05),'Sfax':(0.044924,1884.1,4.88e-05),'Sousse':(0.050067,1851.3,5.09e-05),
'Bizerte':(0.043936,1906.4,5.54e-05),'Gabes':(0.045756,1883.8,4.99e-05),'Beja':(0.045025,1868.6,3.52e-05),
'Nabeul':(0.047586,1872.8,4.46e-05),'Kairouan':(0.043072,1889.0,4.94e-05),'Gafsa':(0.039719,1861.6,2.96e-05)}
if __name__=='__main__':
    print('zone      scalar_score  sigma_sat  K       rule      post-cert  fused')
    for z,(co,ch,no) in ZONES.items():
        r=zone_sat(co,ch,no); f=fuse(r['zone_score'],r['sigma'])
        print('%-9s %8.2f   %7.3f   %.4f  %-8s  %6.1f   %6.2f'%(z,comp(co,ch,no),r['sigma'],f['K'],f['rule'],f['cert'],f['fused']))
