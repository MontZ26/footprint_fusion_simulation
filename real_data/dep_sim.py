import numpy as np, pandas as pd
rng=np.random.default_rng(11); BINS=[30,60,80]; cls=lambda s:np.digitize(s,BINS)
def comb(k1,s1,k2,s2,rule):
    """returns BetP vector over 4 classes. simple supports on {k}, uncertainty sigma on Theta"""
    m=np.zeros(4); mt=0.0
    if k1==k2:
        sg=s1*s2 if rule=='dempster' else max(s1,s2)   # cautious: min weight -> idempotent
        if rule=='average': sg=(s1+s2)/2
        m[k1]=1-sg; mt=sg
    else:
        if rule=='average':
            m[k1]=(1-s1)/2; m[k2]=(1-s2)/2; mt=(s1+s2)/2
        else:
            K=(1-s1)*(1-s2); z=1-K
            m[k1]=(1-s1)*s2/z; m[k2]=s1*(1-s2)/z; mt=s1*s2/z
    return m+mt/4
def sim(rho,sd=12,n=40000,sg=None):
    t=rng.uniform(0,100,n); c=rng.normal(0,sd,n)
    e1=np.sqrt(rho)*c+np.sqrt(1-rho)*rng.normal(0,sd,n); e2=np.sqrt(rho)*c+np.sqrt(1-rho)*rng.normal(0,sd,n)
    x1=np.clip(t+e1,0,100); x2=np.clip(t+e2,0,100); y=cls(t); k1=cls(x1); k2=cls(x2)
    acc1=(k1==y).mean(); sg1=1-acc1                       # each source is calibrated on its own
    out={}
    for rule in ['dempster','cautious','average']:
        conf=np.empty(n); hit=np.empty(n)
        for i in range(n):
            b=comb(k1[i],sg1,k2[i],sg1,rule); k=b.argmax(); conf[i]=b[k]; hit[i]=(k==y[i])
        out[rule]=(hit.mean(),conf.mean())
    return acc1,out
rows=[]
for rho in [0,.3,.6,.9]:
    a1,o=sim(rho)
    for r,(a,cf) in o.items(): rows.append(dict(rho=rho,rule=r,accuracy=round(a,3),mean_conf=round(cf,3),overconf=round(cf-a,3),single_acc=round(a1,3)))
T=pd.DataFrame(rows); print(T.to_string(index=False)); T.to_csv('dep_sim_results.csv',index=False)
