import numpy as np, json
from exp3_lib import *
BINS=[30,60,80]
def fuse_h(zs,sig,base,sig_e,mode):
    sig=np.clip(sig,1e-6,1-1e-6); ms=1-sig; me=1-sig_e; K=ms*me; den=1-K+1e-10
    aD=ms*sig_e/den; bD=sig*me/den; tD=sig*sig_e/den
    aY=ms*sig_e; bY=sig*me; tY=sig*sig_e+K
    y=(K>=0.5) if mode=='hybrid' else np.zeros(len(zs),bool)
    a=np.where(y,aY,aD); b=np.where(y,bY,bD); t=np.where(y,tY,tD)
    return (a+t/2)*zs+(b+t/2)*base
def pool(zs,sig,base,sig_e):
    w=(1-sig)/((1-sig)+(1-sig_e)); return w*zs+(1-w)*base
def state(avg,base_cls,sig_e):
    m=avg*((1-sig_e)*np.eye(4)[base_cls]+sig_e); return (m/m.sum(1,keepdims=True)).argmax(1)
rand=zones(600,2026); Z=600; ns=40; M,sd=setup(rand,ns,2000)
tsz=np.repeat(compw(rand[:,0],rand[:,1],rand[:,2]),ns); yz=np.clip(np.digitize(tsz,BINS),0,3); n=len(tsz)
R={'pool':{},'frame':{}}
def scenarios(zs,reps=40):
    rng=np.random.default_rng(77); base_or=np.clip(tsz+rng.normal(0,12,n),0,100)
    out={'Constant 35.0':[np.full(n,35.0)],'Idealised oracle-informed':[base_or],
         'Biased (+10)':[np.clip(tsz+10+np.random.default_rng(78).normal(0,12,n),0,100)],
         'Correlated with satellite error':[np.clip(tsz+0.8*(zs-tsz)+np.random.default_rng(79).normal(0,8,n),0,100)]}
    miss=np.repeat(np.random.default_rng(80).random(Z)<0.30,ns); out['30% zones missing (fall back to 35.0)']=[np.where(miss,35.0,base_or)]
    r=np.random.default_rng(81); out['Unknown scale and offset']=[np.clip(r.uniform(0.6,1.4)*tsz+r.normal(0,10)+r.normal(0,12,n),0,100) for _ in range(reps)]
    return out
for noise in (0.05,0.10,0.20,0.30):
    o=pipe(M,sd,noise); zs,cert,avg=o['relax']; sig=1-cert/100; pm=o['pm']
    sc=scenarios(zs); sig_e=0.45
    for name,bl in sc.items():
        row={k:[] for k in ('sat','dst_hyb','dst_dem','pool','fixed','pm','acc_sat','acc_src','acc_cls','acc_base')}
        for base in bl:
            row['sat'].append(np.abs(zs-tsz).mean()); row['pm'].append(np.abs(pm-tsz).mean())
            row['dst_hyb'].append(np.abs(fuse_h(zs,sig,base,sig_e,'hybrid')-tsz).mean()); row['dst_dem'].append(np.abs(fuse_h(zs,sig,base,sig_e,'dem')-tsz).mean())
            row['pool'].append(np.abs(pool(zs,sig,base,sig_e)-tsz).mean()); row['fixed'].append(np.abs(0.5*zs+0.5*base-tsz).mean())
            row['acc_sat'].append((avg.argmax(1)==yz).mean())
            row['acc_src'].append((np.clip(np.digitize(fuse_h(zs,sig,base,sig_e,'hybrid'),BINS),0,3)==yz).mean())
            row['acc_cls'].append((state(avg,np.clip(np.digitize(base,BINS),0,3),sig_e)==yz).mean())
            row['acc_base'].append((np.clip(np.digitize(base,BINS),0,3)==yz).mean())
        R['pool'].setdefault('%.2f'%noise,{})[name]={k:float(np.mean(v)) for k,v in row.items()}
json.dump(R,open('exp4.json','w'))
for nz in ('0.10',):
    for name,v in R['pool'][nz].items(): print(name,{k:round(x,3) for k,x in v.items()})
for nz in ('0.05','0.20','0.30'):
    print(nz,{n:(round(v['acc_sat'],3),round(v['acc_src'],3),round(v['acc_cls'],3)) for n,v in R['pool'][nz].items()})
