import sys, numpy as np, pandas as pd
R=pd.read_csv(sys.argv[1],parse_dates=['date']); y=R.truth.values; rules=['dempster','ccr','mean','cdd']
rows=[]
for st,g in R.groupby('site'):
    r=dict(site=st,origins=len(g),rho=g.rho.mean(),agree=g.agree.mean(),prevalence=g.truth.mean())
    for k in rules: r['brier_'+k]=((g[k+'_BetP']-g.truth)**2).mean()
    m=min(r['brier_'+k] for k in rules); r['best']='+'.join(k for k in rules if abs(r['brier_'+k]-m)<5e-5); rows.append(r)
P=pd.DataFrame(rows); pd.set_option('display.width',200); print(P.round(4).to_string(index=False)); P.to_csv('forest_persite.csv',index=False)
print('\nsites where each rule has the lowest Brier (ties counted for every tied rule):',{k:int(P.best.str.contains(k).sum()) for k in rules},'of',len(P))
print('equal-weighted mean over sites of Brier:',{k:round(float(P['brier_'+k].mean()),4) for k in rules})
E=P[['brier_'+k for k in rules]].mean()
print('equal-weighted differences: dempster-ccr %+.4f, dempster-cdd %+.4f, ccr-mean %+.4f, cdd-mean %+.4f, cdd-ccr %+.4f'%(E.brier_dempster-E.brier_ccr,E.brier_dempster-E.brier_cdd,E.brier_ccr-E.brier_mean,E.brier_cdd-E.brier_mean,E.brier_cdd-E.brier_ccr))
b=((R.date-R.date.min()).dt.days//14); G=R.groupby(b).indices; keys=list(G); rng=np.random.default_rng(5)
for a,c in [('dempster','ccr'),('dempster','cdd'),('dempster','mean'),('ccr','mean'),('cdd','mean'),('cdd','ccr')]:
    d=[]
    for _ in range(2000):
        ii=np.concatenate([G[k] for k in rng.choice(keys,len(keys))]); d.append(((R[a+'_BetP'].values[ii]-y[ii])**2).mean()-((R[c+'_BetP'].values[ii]-y[ii])**2).mean())
    pt=((R[a+'_BetP']-y)**2).mean()-((R[c+'_BetP']-y)**2).mean(); ci=np.percentile(d,[2.5,97.5]); print(f'Brier({a})-Brier({c}) = {pt:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]')
sites=sorted(R.site.unique()); S={st:np.where(R.site.values==st)[0] for st in sites}; rng=np.random.default_rng(11)
print('\nEqual-site-weighted paired Brier differences (block bootstrap over dates, 2000 replicates):')
for a,c in [('dempster','ccr'),('dempster','cdd'),('dempster','mean'),('ccr','mean'),('cdd','mean'),('cdd','ccr')]:
    da=(R[a+'_BetP'].values-y)**2-(R[c+'_BetP'].values-y)**2; d=[]
    for _ in range(2000):
        ii=np.concatenate([G[k] for k in rng.choice(keys,len(keys))]); w=np.zeros(len(R)); np.add.at(w,ii,1)
        v=[(w[S[st]]*da[S[st]]).sum()/w[S[st]].sum() for st in sites if w[S[st]].sum()>0]; d.append(np.mean(v))
    pt=np.mean([da[S[st]].mean() for st in sites]); ci=np.percentile(d,[2.5,97.5]); print(f'site-weighted Brier({a})-Brier({c}) = {pt:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]')
