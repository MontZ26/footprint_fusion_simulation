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
R={}
# ---------------- A: paired comparisons with CIs ----------------
rand=zones(600,2026); nine=np.array(list(ZONES.values()))
A={}
for noise in (0.05,0.10,0.20,0.30):
    for pop,means,ns in (('rand',rand,40),('nine',nine,300)):
        M,sd=setup(means,ns); Z=len(means)
        y=np.repeat(np.clip(np.digitize(compw(means[:,0],means[:,1],means[:,2]),BINS),0,3),ns); ts=np.repeat(compw(means[:,0],means[:,1],means[:,2]),ns)
        o=pipe(M,sd,noise); pr=preds(o); sc=scores(o)
        d={}
        for k in NAMES:
            d[k]=dict(acc=boot((pr[k]==y).astype(float),Z,ns),mae=boot(np.abs(sc[k]-ts),Z,ns))
        pd={}
        for a,b in (('relax','cls'),('relax','pm'),('relax','raw')):
            pd[a+'-'+b]=dict(acc=boot((pr[a]==y).astype(float)-(pr[b]==y),Z,ns),mae=boot(np.abs(sc[a]-ts)-np.abs(sc[b]-ts),Z,ns))
        # auc, pooled and cluster-bootstrap
        corr=(pr['relax']==y); cert=o['relax'][1]
        auc=roc_auc_score(corr,cert) if 0<corr.mean()<1 else np.nan
        rng=np.random.default_rng(3); aucs=[]
        if pop=='rand' and 0<corr.mean()<1:
            for _ in range(200):
                zi=rng.integers(0,Z,Z); ii=(zi[:,None]*ns+np.arange(ns)).ravel(); c=corr[ii]
                if 0<c.mean()<1: aucs.append(roc_auc_score(c,cert[ii]))
        d['auc']=(auc,np.percentile(aucs,2.5) if aucs else None,np.percentile(aucs,97.5) if aucs else None)
        A['%s|%.2f'%(pop,noise)]=dict(m=d,pd=pd)
        if noise==0.10 and pop=='rand':
            P=pr['relax']; cm=np.zeros((4,4),int)
            for t,p in zip(y,P): cm[t,p]+=1
            R['confusion_platform']=cm.tolist()
            P2=pr['pm']; cm2=np.zeros((4,4),int)
            for t,p in zip(y,P2): cm2[t,p]+=1
            R['confusion_pm']=cm2.tolist()
            # accuracy by distance to nearest bin edge
            dist=np.min(np.abs(ts[:,None]-np.array(BINS)[None]),1); bins_=[(0,3),(3,8),(8,100)]
            R['edgedist']={ '%g-%g'%b:{k:float((pr[k]==y)[(dist>=b[0])&(dist<b[1])].mean()) for k in NAMES}|{'n':int(((dist>=b[0])&(dist<b[1])).sum())} for b in bins_}
R['A']=A
print('A done',flush=True)
# ---------------- C/D: fusion rules and baselines ----------------
def fuse_r(zs,sig_s,base,sig_e,yager):
    sig_s=np.clip(sig_s,1e-6,1-1e-6); ms=1-sig_s; me=1-sig_e; K=ms*me; den=1-K+1e-10
    aD=ms*sig_e/den; bD=sig_s*me/den; tD=sig_s*sig_e/den
    aY=ms*sig_e; bY=sig_s*me; tY=sig_s*sig_e+K
    a=np.where(yager,aY,aD); b=np.where(yager,bY,bD); t=np.where(yager,tY,tD)
    return (a+t/2)*zs+(b+t/2)*base,(1-t)
def ece(p,c,nb=10):
    e=0; edges=np.linspace(0,1,nb+1)
    for i in range(nb):
        m=(p>=edges[i])&(p<edges[i+1]+(1e-9 if i==nb-1 else 0))
        if m.any(): e+=m.mean()*abs(p[m].mean()-c[m].mean())
    return e
Z=600; ns=40; M,sd=setup(rand,ns,2000); tsz=np.repeat(compw(rand[:,0],rand[:,1],rand[:,2]),ns); yz=np.clip(np.digitize(tsz,BINS),0,3)
o=pipe(M,sd,0.10); zs,cert,avg=o['relax']; sig=1-cert/100; pm=o['pm']
rng=np.random.default_rng(77)
sig_e=0.45; K=(1-sig)*(1-sig_e)
rules={'Always Dempster':np.zeros(len(zs),bool),'Always Yager':np.ones(len(zs),bool),'Hybrid, K >= 0.5 (platform)':K>=0.5}
base_or=np.clip(tsz+rng.normal(0,12,len(zs)),0,100)
disc=np.abs(zs-base_or)/100
rules['Disagreement-aware, |score gap| > 0.25']=disc>0.25
def rule_table(base):
    rows={}
    for nm,yg in rules.items():
        f,pc=fuse_r(zs,sig,base,sig_e,yg); cls=np.clip(np.digitize(f,BINS),0,3); cor=(cls==yz).astype(float)
        rows[nm]=dict(mae=float(np.abs(f-tsz).mean()),acc=float(cor.mean()),ece=float(ece(pc,cor)),brier=float(((pc-cor)**2).mean()),meancert=float(pc.mean()),yfrac=float(yg.mean()))
    return rows
R['rules_const']=rule_table(np.full(len(zs),35.0)); R['rules_oracle']=rule_table(base_or)
# disagreement-aware rule under constant baseline uses its own gap
disc35=np.abs(zs-35.0)/100; yg=disc35>0.25; f,pc=fuse_r(zs,sig,35.0,sig_e,yg); cls=np.clip(np.digitize(f,BINS),0,3); cor=(cls==yz).astype(float)
R['rules_const']['Disagreement-aware, |score gap| > 0.25']=dict(mae=float(np.abs(f-tsz).mean()),acc=float(cor.mean()),ece=float(ece(pc,cor)),brier=float(((pc-cor)**2).mean()),meancert=float(pc.mean()),yfrac=float(yg.mean()))
R['sat_only']=dict(mae=float(np.abs(zs-tsz).mean()),acc=float((np.clip(np.digitize(zs,BINS),0,3)==yz).mean()),ece=float(ece(cert/100,(avg.argmax(1)==yz).astype(float))))
R['pm_only']=dict(mae=float(np.abs(pm-tsz).mean()),acc=float((np.clip(np.digitize(pm,BINS),0,3)==yz).mean()))
# baseline scenarios
def fusion_mae(base): 
    f,_=fuse_r(zs,sig,base,sig_e,K>=0.5); return float(np.abs(f-tsz).mean()),float(np.abs(base-tsz).mean())
sc={}
sc['Constant national value 35.0 (platform)']=fusion_mae(np.full(len(zs),35.0))
sc['Idealised: true score + N(0,12)']=fusion_mae(base_or)
sc['Biased: true score + 10 + N(0,12)']=fusion_mae(np.clip(tsz+10+rng.normal(0,12,len(zs)),0,100))
sc['Correlated with satellite error']=fusion_mae(np.clip(tsz+0.8*(zs-tsz)+rng.normal(0,8,len(zs)),0,100))
miss=np.repeat(rng.random(Z)<0.30,ns); b_m=np.where(miss,35.0,base_or); sc['30% of zones missing (fall back to 35.0)']=fusion_mae(b_m)
reps=[]
for r in range(200):
    a=rng.uniform(0.6,1.4); b=rng.normal(0,10); reps.append(fusion_mae(np.clip(a*tsz+b+rng.normal(0,12,len(zs)),0,100)))
reps=np.array(reps); sc['Scale/offset mapping error (200 draws; mean, 95th pct)']=(float(reps[:,0].mean()),float(reps[:,1].mean()),float(np.percentile(reps[:,0],95)))
R['scen']=sc
print('CD done',flush=True)
# state-frame variant
def state_frame(avg,base_cls,sig_e):
    # satellite: Bayesian mass on classes = zone label distribution; inventory: class of baseline with mass 1-sig_e, rest on Theta
    m=avg.copy()*((1-sig_e)*np.eye(4)[base_cls]+sig_e)  # unnormalised conjunction
    K=(1-sig_e)*(1-avg[np.arange(len(avg)),base_cls]); m=m/m.sum(1,keepdims=True)
    return m,K
yS=(avg.argmax(1)==yz).mean()
b35=np.zeros(len(zs),int); m35,K35=state_frame(avg,b35,sig_e)
bor=np.clip(np.digitize(base_or,BINS),0,3); mor,Kor=state_frame(avg,bor,sig_e)
R['state']=dict(sat=float(yS),const_acc=float((m35.argmax(1)==yz).mean()),oracle_acc=float((mor.argmax(1)==yz).mean()),Kconst=float(K35.mean()),Koracle=float(Kor.mean()),
  src_const=R['rules_const']['Hybrid, K >= 0.5 (platform)']['acc'],src_oracle=R['rules_oracle']['Hybrid, K >= 0.5 (platform)']['acc'],base_or_acc=float((bor==yz).mean()),
  Kcorr_correct=float(np.corrcoef(Kor,(mor.argmax(1)==yz))[0,1]),Ksrc_corr=float(np.corrcoef(K,(np.clip(np.digitize(fuse_r(zs,sig,base_or,sig_e,K>=0.5)[0],BINS),0,3)==yz))[0,1]))
print('state done',flush=True)
# ---------------- E: noise-model sensitivity ----------------
E={}
Mr,sdr=setup(zones(400,5),20,4000); Zr=400; zm=zones(400,5); tse=np.repeat(compw(zm[:,0],zm[:,1],zm[:,2]),20); ye=np.clip(np.digitize(tse,BINS),0,3)
for kind in ('base','indep','white','bias','outlier','cloud'):
    for noise in (0.10,0.20):
        o=pipe(Mr,sdr,noise,kind); pr=preds(o); s_=scores(o)
        E['%s|%.2f'%(kind,noise)]={k:(float((pr[k]==ye).mean()),float(np.abs(s_[k]-tse).mean())) for k in NAMES}
zm=zones(400,5); R['bias_shift']=float((compw(zm[:,0]*1.05,zm[:,1]*1.005,zm[:,2]*1.05)-compw(zm[:,0],zm[:,1],zm[:,2])).mean())
R['E']=E; print('E done',flush=True)
# ---------------- F: design choices ----------------
F={}
def run_cfg(name,**kw):
    w=kw.get('w',(0.40,0.35,0.25)); ts_=np.repeat(compw(zm[:,0],zm[:,1],zm[:,2],w),20); y_=np.clip(np.digitize(ts_,BINS),0,3)
    o=pipe(Mr,sdr,0.10,'base',**kw); pr=preds(o); s_=scores(o); F[name]={k:(float((pr[k]==y_).mean()),float(np.abs(s_[k]-ts_).mean())) for k in NAMES}
run_cfg('Default (weights 0.40/0.35/0.25, sigma_c 1.5, 20 iterations)')
run_cfg('Equal weights 1/3 each',w=(1/3,1/3,1/3)); run_cfg('Weights 0.50/0.30/0.20',w=(0.5,0.3,0.2))
run_cfg('sigma_c = 0.75',sigc=0.75); run_cfg('sigma_c = 3.0',sigc=3.0)
run_cfg('5 iterations',iters=5); run_cfg('50 iterations',iters=50)
R['F']=F
json.dump(R,open('exp3.json','w'),default=float); print('saved')
