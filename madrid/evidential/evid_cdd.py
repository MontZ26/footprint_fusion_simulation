# CDD on canonical weights for general (non-dogmatic) BBAs: weights w -> w^lambda, then conjunctive combination (weights multiply). Keeps m(empty).
import numpy as np
from evid_k import Frame, kernel, popc
F=Frame(4); R=kernel([9.68,22.09,37.20,54.10],8.0)      # s=8 placeholder (toy only)
def from_weights(w):
    q={c:np.prod([w[a] for a in w if c&a!=c]) for c in F.S}
    return {a:sum((-1)**(popc(b)-popc(a))*q[b] for b in F.S if b&a==a) for a in F.S}
def cdd_combine(ms,rho_bar):
    k=len(ms); lam=1/(1+(k-1)*rho_bar); ws=[F.weights(m) for m in ms]
    return from_weights({a:np.prod([w[a]**lam for w in ws]) for a in ws[0]}),lam
centre={1:.1,2:.25,4:.12,8:.05,F.TH:.48}
nA={1:.7,2:.1,4:.05,F.TH:.15}; nC={8:.7,4:.15,F.TH:.15}
def show(tag,m): print(f'  {tag:28s} BetP={np.round(F.betp(m),2)} m(Θ)={m[F.TH]:.2f} m(∅)={F.empty(m):.2f}')
rho=.8
for nm,nb in (('agree (8 x class1)',[nA]*8),('disagree (4+4)',[nA]*4+[nC]*4)):
    print(nm)
    P=[F.project(x,rho,R) for x in nb]
    show('cautious primary',F.cautious(centre,F.fold(P,F.cautious)))
    for rb in (0,.3,.6,.9,1.0):
        m,lam=cdd_combine([centre]+P,rb); show(f'CDD rho_bar={rb} (lambda={lam:.2f})',m)
print('repeat test, 8 identical neighbours vs 1, rho_bar=1 (lambda=1/9):')
m1,_=cdd_combine([centre,F.project(nA,rho,R)],1.0); m8,_=cdd_combine([centre]+[F.project(nA,rho,R)]*8,1.0)
print('  1 neighbour BetP',np.round(F.betp(m1),3),'| 8 neighbours BetP',np.round(F.betp(m8),3))
m0,_=cdd_combine([centre,F.project(nA,rho,R)],0.0); print('  rho_bar=0, k=2 equals unnormalised conjunctive (Dempster before normalisation): max diff %.1e'%max(abs(m0[s]-F.conj(centre,F.project(nA,rho,R))[s]) for s in F.S))
