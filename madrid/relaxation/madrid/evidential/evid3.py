# evidential relaxation toy, revised: composite focal sets, normalised BetP, full-empty-set audit, parametrised compatibility
import numpy as np
TH=7; names={1:'L',2:'M',4:'H',3:'LM',6:'MH',5:'LH',7:'Θ'}
A={1:.75,2:.10,4:.05,7:.10}; B={1:.20,2:.25,4:.10,7:.45}; C={1:.05,2:.15,4:.70,7:.10}
def Rmat(a): return np.array([[a**((k-l)**2) for l in range(3)] for k in range(3)])   # ordinal Gaussian kernel, adjacent compatibility a
def consonant(contour):
    order=np.argsort(-contour,kind='stable'); lv=contour[order]; m={}
    for r in range(3):
        nxt=lv[r+1] if r<2 else 0.; mask=sum(1<<int(order[q]) for q in range(r+1)); w=lv[r]-nxt
        if w>1e-12: m[mask]=m.get(mask,0)+w
    return m
def project(mj,rho,R):                 # composite focal sets: contour = upper envelope of compatibility rows
    out={}
    for F,mass in mj.items():
        if F==TH or mass==0: continue
        ks=[k for k in range(3) if F>>k&1]; contour=np.max([R[k] for k in ks],0)
        for s,w in consonant(contour).items(): out[s]=out.get(s,0)+rho*mass*w
    out[TH]=out.get(TH,0)+1-sum(out.values()); return out
def conj(m1,m2):
    o=np.zeros(8)
    for a,x in m1.items():
        for b,y in m2.items(): o[a&b]+=x*y
    return o
def dempster(m1,m2): o=conj(m1,m2); K=o[0]; return {s:o[s]/(1-K) for s in range(1,8)}
def q_of(m): return {a:sum(v for s,v in m.items() if s&a==a) for a in range(1,8)}
def weights(m):
    q=q_of(m); w={}
    for a in range(0,7):
        w[a]=np.exp(sum(-((-1)**(bin(b).count('1')-bin(a).count('1')))*np.log(q[b]) for b in range(1,8) if b&a==a))
    return w
def cautious(m1,m2):
    w1,w2=weights(m1),weights(m2); w={a:min(w1[a],w2[a]) for a in w1}
    q={c:np.prod([w[a] for a in w if c&a!=c]) for c in range(1,8)}
    return {a:sum((-1)**(bin(b).count('1')-bin(a).count('1'))*q[b] for b in range(1,8) if b&a==a) for a in range(1,8)}
def fold(ms,f):
    m=ms[0]
    for x in ms[1:]: m=f(m,x)
    return m
def empty(m): return 1-sum(m.values())
def betp(m):
    s=sum(m.values()); return [sum(v/bin(k).count('1') for k,v in m.items() if k&b)/s for b in (1,2,4)]
def primary(centre,nbrs,rho,R):
    pool=fold([project(x,rho,R) for x in nbrs],cautious); return cautious(centre,pool),pool
if __name__=='__main__':
    R=Rmat(.3); x,y,z=[project(u,.8,R) for u in (A,B,C)]
    full=lambda m:{s:m.get(s,0) for s in range(0,8)}
    def full_close(a,b):
        fa,fb=dict(full(a)),dict(full(b)); fa[0]=empty(a); fb[0]=empty(b); return max(abs(fa[s]-fb[s]) for s in range(8))
    print('AUDIT incl. empty set: idempotent %.1e commutative %.1e associative %.1e'%(full_close(cautious(x,x),x),full_close(cautious(x,y),cautious(y,x)),full_close(cautious(cautious(x,y),z),cautious(x,cautious(y,z)))))
    sing=project({1:1.0},.8,R); comp=project({3:1.0},.8,R)
    print('projection sums',round(sum(sing.values()),12),round(sum(comp.values()),12),'| singleton L ->',{names[k]:round(v,3) for k,v in sing.items()},'| {L,M} ->',{names[k]:round(v,3) for k,v in comp.items()})
    print('\nSENSITIVITY: adjacent-class compatibility a x reliability rho. Top class (BetP) / m(Θ) / m(∅) / pool conflict')
    for rho in (.5,.8,.95):
        for a in (.1,.2,.3,.45,.6):
            R=Rmat(a); cells=[]
            for nm,nb in (('agree',[A]*8),('disagree',[A]*4+[C]*4),('uncert',[B]*8)):
                m,pool=primary(B,nb,rho,R); b=betp(m); cells.append(f"{nm}: {'LMH'[int(np.argmax(b))]} ({b[0]:.2f},{b[1]:.2f},{b[2]:.2f}) Θ={m[TH]:.2f} ∅={empty(m):.2f} pool∅={empty(pool):.2f}")
            m1,_=primary(B,[A],rho,R); m8,_=primary(B,[A]*8,rho,R)
            print(f'rho={rho} a={a}: '+' | '.join(cells)+f' | n=1 vs 8 max diff {max(abs(m1[s]-m8[s]) for s in range(1,8)):.1e}')
    print('\nMadrid-style kernel (class centroids 9.68,22.09,37.20,54.10; s=6.95): adjacent compat L-M %.2f, M-H %.2f, H-VH %.2f'%tuple(np.exp(-(np.diff([9.68,22.09,37.20,54.10]))**2/(2*6.95**2))))
