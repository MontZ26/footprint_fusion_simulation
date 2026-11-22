"""Complete mass-vector validation of the CCR implementation (reviewer 5, comment 3).
(a) implementation rule_vec('ccr') vs general CCR (commonalities -> canonical weights -> min -> conjunction), full vectors incl. empty set
(b) effect of Dempster normalisation on Bel, Pl, m(Theta), conflict
(c) three sources: simultaneous min weights vs sequential pairwise application (normalised intermediate)"""
import numpy as np
from ccr_validation import ccr_general
from temporal_extended import rule_vec
rng=np.random.default_rng(3)
def rand_ms(k):
    ms=[]
    for _ in range(k):
        s=rng.uniform(0.02,0.95); side=rng.integers(0,2); ms.append((s if side==0 else 0.,s if side==1 else 0.,1-s))
    return ms
def vec(g): return np.array([g['H'],g['N'],g['T'],g['0']])
for k in (2,3,4):
    mx=0; seq=0; n=5000; worst_seq=0
    for _ in range(n):
        ms=rand_ms(k)
        a=vec(ccr_general([{'H':m[0],'N':m[1],'T':m[2]} for m in ms])); b=np.array(rule_vec(ms,'ccr'))
        mx=max(mx,np.abs(a-b).max())
        # sequential: pairwise, normalise intermediate (Dempster normalisation), then combine with next source
        cur=ms[0]
        for m in ms[1:]:
            g=ccr_general([{'H':cur[0],'N':cur[1],'T':cur[2]},{'H':m[0],'N':m[1],'T':m[2]}]); z=1-g['0']; cur=(g['H']/z,g['N']/z,g['T']/z)
        z=1-b[3]; d=np.abs(np.array(cur)-b[:3]/z).max(); worst_seq=max(worst_seq,d); seq+=d>1e-9
    print(f'{k} sources: max |mass vector difference| implementation vs general CCR (incl. empty set) = {mx:.1e}; sequential-normalised vs simultaneous differ in {seq}/{n} cases (max {worst_seq:.3f})')
# effect of normalisation on a contradictory example
g=ccr_general([{'H':.8,'N':0,'T':.2},{'H':0,'N':.7,'T':.3}]); z=1-g['0']
print('unnormalised',{k:round(v,3) for k,v in g.items()},'| normalised m(H),m(N),m(T)=',round(g['H']/z,3),round(g['N']/z,3),round(g['T']/z,3),'Bel,Pl,BetP=',round(g['H']/z,3),round((g['H']+g['T'])/z,3),round((g['H']+g['T']/2)/z,3))
