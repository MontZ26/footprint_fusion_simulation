"""Validation of the cautious conjunctive rule (CCR) implementation used in temporal.py.
General CCR (Denoeux 2008): canonical decomposition of non-dogmatic masses via commonalities,
w12(A) = min(w1(A), w2(A)), then conjunctive combination of the resulting generalised simple supports.
Frame {H, N} (N = not H), focal sets H, N, Theta (empty set only in the output)."""
import itertools, numpy as np
from temporal_rules import comb_ccr_shortcut   # the shortcut used in temporal.py
SETS=['H','N','T']            # focal sets; T = Theta
def commonality(m):           # m = dict over 'H','N','T'; q(A)=sum_{B superset of A} m(B)
    return {'H':m['H']+m['T'],'N':m['N']+m['T'],'T':m['T']}
def weights(m):               # canonical weights on proper subsets A != Theta
    q=commonality(m)          # w(A)=prod_{B>=A} q(B)^{(-1)^{|B|-|A|+1}}; for singletons B in {A,T}
    return {'H':q['T']/q['H'] if q['H']>0 else np.nan,'N':q['T']/q['N'] if q['N']>0 else np.nan}
def conj(a,b):                # unnormalised conjunctive rule on {H,N,T}; returns dict incl. empty set
    out={'H':0,'N':0,'T':0,'0':0}
    inter={('H','H'):'H',('H','T'):'H',('T','H'):'H',('N','N'):'N',('N','T'):'N',('T','N'):'N',('T','T'):'T',('H','N'):'0',('N','H'):'0'}
    for x in SETS:
        for y in SETS: out[inter[(x,y)]]+=a[x]*b[y]
    return out
def ccr_general(ms):
    ws=[weights(m) for m in ms]
    wH=min(w['H'] for w in ws); wN=min(w['N'] for w in ws)
    a={'H':1-wH,'N':0,'T':wH}; b={'H':0,'N':1-wN,'T':wN}   # simple supports with the min weights
    return conj(a,b)
def betp(m):
    k=1-m['0']; return (m['H']+m['T']/2)/k
if __name__=='__main__':
    m1={'H':0.8,'N':0.0,'T':0.2}; m2={'H':0.0,'N':0.7,'T':0.3}
    g=ccr_general([m1,m2]); print('reviewer example, general CCR (unnormalised):',{k:round(v,3) for k,v in g.items()},'BetP(H)=%.4f'%betp(g))
    s=comb_ccr_shortcut([(0.8,0.0,0.2),(0.0,0.7,0.3)]); print('implementation in temporal.py, BetP(H)=%.4f'%s)
    rng=np.random.default_rng(0); worst=0; n=0
    for _ in range(20000):
        ms=[]
        for _i in range(rng.integers(2,5)):
            s_=rng.uniform(0.02,0.95); side=rng.integers(0,2)
            ms.append({'H':s_ if side==0 else 0,'N':s_ if side==1 else 0,'T':1-s_})
        a=betp(ccr_general(ms)); b=comb_ccr_shortcut([(m['H'],m['N'],m['T']) for m in ms]); worst=max(worst,abs(a-b)); n+=1
    print('random mixed-side simple supports, %d cases, max |BetP general - BetP implementation| = %.2e'%(n,worst))
    # idempotency and contradictory tests
    m={'H':0.6,'N':0.0,'T':0.4}; print('idempotent (H^0.4 with itself):',{k:round(v,3) for k,v in ccr_general([m,m]).items()})
    print('contradictory equal supports H 0.7 vs N 0.7: m=',{k:round(v,3) for k,v in ccr_general([{'H':.7,'N':0,'T':.3},{'H':0,'N':.7,'T':.3}]).items()})
