"""Gaussian-mapping diagnostics, cap/mapping sensitivity and binning sensitivity (h=1). Usage: ICOS_DIR=icos9 python diagnostics_mapping.py"""
import numpy as np, pandas as pd
from scipy import stats
import temporal_extended as T
R=T.run(1)
z=np.concatenate([R.z0,R.z1,R.z2]); 
print('standardised errors: n',len(z),'mean %.2f sd %.2f skew %.2f exkurt %.2f'%(z.mean(),z.std(),stats.skew(z),stats.kurtosis(z)))
print('coverage |z|<1.645: %.3f (nominal .90)  |z|<1.96: %.3f (.95)'%((abs(z)<1.645).mean(),(abs(z)<1.96).mean()))
print('KS vs N(0,1): stat %.3f p %.3g ; Shapiro p %.3g'%(*stats.kstest(z,'norm'),stats.shapiro(z[:5000])[1]))
for k in ['z0','z1','z2']: print(k,'sd %.2f cov95 %.3f'%(R[k].std(),(abs(R[k])<1.96).mean()))
# heteroscedasticity: |z| vs trailing sigma (rank corr), and by station
print('spearman(|z|, sigma) past: %.2f'%stats.spearmanr(abs(R.z0),R.sg0)[0],' fc: %.2f'%stats.spearmanr(abs(R.z2),R.sg2)[0])
print(R.groupby('st')[['z0','z1','z2']].std().round(2).T.to_string())
# sensitivity
rows=[]
for mp,cap in [('gauss',0.90),('gauss',0.95),('gauss',0.99),('t5',0.95)]:
    T.CAP=cap; T.MAP=mp; Q=T.run(1); y=Q.truth.values
    for rl in ['dempster','ccr','mean']:
        p=Q[f'{rl}_BetP'].values; dec=p>.5; acc=(dec==y).mean(); conf=np.where(dec,p,1-p).mean()
        rows.append(dict(mapping=mp,cap=cap,rule=rl,acc=acc,gap=conf-acc,brier=((p-y)**2).mean(),ECE10=T.ece(p,y,10)))
S=pd.DataFrame(rows); print(S.round(3).to_string(index=False)); S.to_csv('ext_mapping_sensitivity.csv',index=False)
T.CAP=0.95; T.MAP='gauss'; y=R.truth.values
for rl in ['dempster','ccr','mean']:
    p=R[f'{rl}_BetP'].values; print(rl,'ECE bins 5/10/20:',[round(T.ece(p,y,b),3) for b in (5,10,20)])
d=pd.read_csv('ext_reliability_h1.csv'); print(d.groupby('rule').n.agg(['min','median','max','count']).to_string())
