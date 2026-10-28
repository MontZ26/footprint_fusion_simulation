import numpy as np, json
from fast import *
from sklearn.metrics import roc_auc_score
rng=np.random.default_rng(2026)
LO=np.array([CO_R[0],CH_R[0],NO_R[0]]); HI=np.array([CO_R[1],CH_R[1],NO_R[1]])
nine=np.array(list(ZONES.values()))
# random zones: correlated normalised levels
u=rng.uniform(0.05,0.95,600); dev=rng.normal(0,0.06,(600,3))
rand=LO+np.clip(u[:,None]+dev,0.02,0.98)*(HI-LO)
def evaluate(means,nseeds,noise,conf=0.85):
    Z=len(means); M=np.repeat(means,nseeds,axis=0); seeds=np.tile(np.arange(1000,1000+nseeds),Z)
    y=np.repeat(true_class(means),nseeds); ts=np.repeat(comp(means[:,0],means[:,1],means[:,2]),nseeds)
    out={}
    A=run_batch(M,seeds,noise,conf,'classify',True); B=run_batch(M,seeds,noise,conf,'raw',True); C=run_batch(M,seeds,noise,conf,'classify',False)
    rows={}
    for nm,r in (('Classify+relax (platform)',A),('Raw soft-init+relax',B),('Classify, no relax',C)):
        lab=r['avg'].argmax(1); corr=(lab==y)
        auc=roc_auc_score(corr,r['cert']) if 0<corr.mean()<1 else np.nan
        rows[nm]=dict(acc=corr.mean(),mae=np.abs(r['zs']-ts).mean(),auc=auc,cert=r['cert'].mean())
    ms=A['score'].mean((1,2)); rows['Raw pixel mean (no classif./relax)']=dict(acc=(np.clip(np.digitize(ms,[30,60,80]),0,3)==y).mean(),mae=np.abs(ms-ts).mean(),auc=np.nan,cert=np.nan)
    return rows,A,y,ts
def show(title,rows):
    print('\n'+title); print('%-38s %6s %7s %7s %8s'%('method','acc','MAE','AUROC','cert%'))
    for k,v in rows.items(): print('%-38s %6.3f %7.2f %7.3f %8.1f'%(k,v['acc'],v['mae'],v['auc'],v['cert']))
res={}
for noise in (0.05,0.10,0.20,0.30):
    r9,_,_,_=evaluate(nine,300,noise); show('NINE ZONES x 300 seeds, noise amplitude %.2f'%noise,r9)
    rr,_,_,_=evaluate(rand,40,noise); show('600 RANDOM ZONES x 40 seeds, noise amplitude %.2f'%noise,rr)
    res[str(noise)]=dict(nine=r9,rand=rr)
json.dump(res,open('exp1.json','w'),default=float)
