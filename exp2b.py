import numpy as np, json
from fast import *
OUT={}
rng=np.random.default_rng(2026)
LO=np.array([CO_R[0],CH_R[0],NO_R[0]]); HI=np.array([CO_R[1],CH_R[1],NO_R[1]])
u=rng.uniform(0.05,0.95,600); dev=rng.normal(0,0.06,(600,3)); rand=LO+np.clip(u[:,None]+dev,0.02,0.98)*(HI-LO)
nine=np.array(list(ZONES.values()))
def fuse_b(zs,sig_s,base,sig_e,thr=0.5):
    sig_s=np.clip(sig_s,1e-6,1-1e-6); ms=1-sig_s; me=1-sig_e; K=ms*me
    den=1-K+1e-10
    aD=ms*sig_e/den; bD=sig_s*me/den; tD=sig_s*sig_e/den
    aY=ms*sig_e; bY=sig_s*me; tY=sig_s*sig_e+K
    y=K>=thr; a=np.where(y,aY,aD); b=np.where(y,bY,bD); t=np.where(y,tY,tD)
    bs=a+t/2; be=b+t/2
    return bs*zs+be*base, y, K
# ---------- Experiment 3: fusion
nseed=40; M=np.repeat(rand,nseed,axis=0); seeds=np.tile(np.arange(2000,2000+nseed),len(rand))
A=run_batch(M,seeds,0.10,0.85,'classify',True); ts=np.repeat(comp(rand[:,0],rand[:,1],rand[:,2]),nseed)
sig=1-A['cert']/100; zs=A['zs']; pm=A['score'].mean((1,2))
print('Fusion experiment: 600 random zones x 40 seeds, noise 0.10; error vs true composite score')
print('  satellite zone_score only            MAE %.2f'%np.abs(zs-ts).mean())
print('  pixel-mean score (no classification) MAE %.2f'%np.abs(pm-ts).mean())
print('  constant baseline 35 only            MAE %.2f'%np.abs(35-ts).mean())
f,y,K=fuse_b(zs,sig,35.0,0.45); OUT['fusion']=dict(sat=float(np.abs(zs-ts).mean()),pm=float(np.abs(pm-ts).mean()),base=float(np.abs(35-ts).mean()),plat=float(np.abs(f-ts).mean()),yf=float(y.mean()),fixed=float(np.abs(0.5*zs+17.5-ts).mean())); print('  platform fusion (baseline 35)        MAE %.2f   Yager fraction %.2f'%(np.abs(f-ts).mean(),y.mean()))
print('  fixed 50/50 with baseline 35         MAE %.2f'%np.abs(0.5*zs+17.5-ts).mean())
base_sim=np.clip(ts+np.random.default_rng(77).normal(0,12,len(ts)),0,100)
f2,y2,_=fuse_b(zs,sig,base_sim,0.45); OUT['fusion_sim']=dict(plat=float(np.abs(f2-ts).mean()),yf=float(y2.mean()),base=float(np.abs(base_sim-ts).mean())); print('  platform fusion, SIMULATED zone-specific baseline (truth+N(0,12)): MAE %.2f  Yager %.2f   baseline-only MAE %.2f'%(np.abs(f2-ts).mean(),y2.mean(),np.abs(base_sim-ts).mean()))
print('\nK-threshold x sigma_edgar sweep (MAE of fused score; constant-35 baseline | simulated baseline); Yager fraction in brackets')
OUT['grid']={}
print('%-8s'%'thr\\sig_e'+'  '.join('%-27s'%('sig_e=%.2f'%s) for s in (0.15,0.30,0.45,0.60,0.75)))
for thr in (0.0,0.3,0.4,0.5,0.6,0.7,1.01):
    cells=[]
    for se in (0.15,0.30,0.45,0.60,0.75):
        fa,ya,_=fuse_b(zs,sig,35.0,se,thr); fb,yb,_=fuse_b(zs,sig,base_sim,se,thr)
        OUT['grid'][f'{thr}|{se}']=[float(np.abs(fa-ts).mean()),float(np.abs(fb-ts).mean()),float(ya.mean())]
        cells.append('%5.2f | %5.2f [%.2f]      '%(np.abs(fa-ts).mean(),np.abs(fb-ts).mean(),ya.mean()))
    print('%-8s'%('%.2f'%thr)+'  '.join(cells))
# ---------- nine-zone version of the K-threshold effect on the real zones
OUT['nine_yager']={}
print('\nNine zones (seed 42): number using Yager vs threshold at sigma_edgar=0.45')
r=[zone_sat(*ZONES[z]) for z in ZONES]; sg=np.array([x['sigma'] for x in r]); zz=np.array([x['zone_score'] for x in r])
for thr in (0.3,0.4,0.5,0.6,0.7):
    _,yy,_=fuse_b(zz,sg,35.0,0.45,thr); OUT['nine_yager'][str(thr)]=int(yy.sum()); print('  thr %.1f -> %d zones Yager'%(thr,yy.sum()))
# ---------- Experiment 4: confidence sweep and the dip
confs=np.round(np.concatenate([np.arange(0.50,0.701,0.01),np.arange(0.75,0.951,0.05)]),2)
def sweep(means,nseeds,label):
    Z=len(means); Mm=np.repeat(means,nseeds,axis=0); sd=np.tile(np.arange(3000,3000+nseeds),Z)
    C=np.array([run_batch(Mm,sd,0.10,c,'classify',True)['cert'].reshape(Z,nseeds) for c in confs])  # (nconf,Z,ns)
    print('\nConfidence sweep, %s: mean certainty (%%)'%label)
    print(' '.join('%.2f:%.1f'%(c,C[i].mean()) for i,c in enumerate(confs)))
    i50,i55,i60=list(confs).index(0.5),list(confs).index(0.55),list(confs).index(0.6)
    dip=(C[i55]<np.minimum(C[i50],C[i60])); print('  fraction of (zone,seed) pairs with certainty(0.55) < min(cert(0.50),cert(0.60)): %.3f'%dip.mean())
    sub=confs<=0.70
    argmin=confs[sub][C[sub].argmin(0)]; vals,cnt=np.unique(argmin,return_counts=True); print('  location of minimum certainty within [0.50,0.70]:',dict(zip(vals.tolist(),np.round(cnt/cnt.sum(),3).tolist())))
    best=confs[C.argmax(0)]; vals,cnt=np.unique(best,return_counts=True); print('  confidence maximising certainty:',{k:round(v,3) for k,v in zip(vals.tolist(),(cnt/cnt.sum()).tolist()) if v/cnt.sum()>0.03})
    OUT.setdefault('sweep',{})[label]=dict(confs=confs.tolist(),mean=C.mean((1,2)).tolist(),dipfrac=float(dip.mean()))
    return C
sweep(nine,100,'nine zones x 100 noise fields'); sweep(rand[:200],10,'200 random zones x 10 noise fields')
json.dump(OUT,open('exp2.json','w'))
