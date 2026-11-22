import numpy as np, pandas as pd
import full_real as fr, pilot as pl
D0=fr.D.copy()
def trial(bias_extra=0.0,out_frac=0.0,out_mag=8.0,sig_scale=1.0,seed=0):
    rng=np.random.default_rng(seed)
    D=D0.copy(); D['xco2']=D.xco2+bias_extra
    if out_frac>0:
        m=rng.random(len(D))<out_frac; D.loc[m,'xco2']+=out_mag
    fr.D=D
    R=fr.run()
    # sigma misspecification on the evidential rule only
    c=pl.c; ev=[]
    for _,r in R.iterrows():
        s=min(.95,r.sg_sat*sig_scale); ev.append(fr.evid(r.rel,s,r.base,r.sg_b)[0])
    R['ev_s']=ev
    e=lambda m:float(np.sqrt(np.mean((R[m]-R.truth)**2)))
    return {m:e(m) for m in ['pix','rel','base','ev_s','rw','blue']}
rows=[]
for name,kw in [('none',{}),('sat bias +2 ppm',dict(bias_extra=2)),('sat bias +4 ppm',dict(bias_extra=4)),
                ('10% outliers +8 ppm',dict(out_frac=.1)),('25% outliers +8 ppm',dict(out_frac=.25)),
                ('sigma_sat x0.25 (overtrust)',dict(sig_scale=.25)),('sigma_sat x4 (undertrust)',dict(sig_scale=4)),
                ('bias +4 ppm and sigma x0.25',dict(bias_extra=4,sig_scale=.25))]:
    r=trial(**kw); r['case']=name; rows.append(r)
T=pd.DataFrame(rows).set_index('case').round(1); print(T.to_string()); T.to_csv('misspec_results.csv')
