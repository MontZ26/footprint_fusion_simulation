# Evidential relaxation, K-class implementation (Theta = 2^K-1). Cautious rule with empty-set weight.
import numpy as np, itertools
def popc(x): return bin(x).count('1')
class Frame:
    def __init__(self,K): self.K=K; self.TH=(1<<K)-1; self.S=range(1,self.TH+1)
    def consonant(self,contour):
        order=np.argsort(-contour,kind='stable'); lv=contour[order]; m={}
        for r in range(self.K):
            nxt=lv[r+1] if r<self.K-1 else 0.; mask=sum(1<<int(order[q]) for q in range(r+1)); w=lv[r]-nxt
            if w>1e-12: m[mask]=m.get(mask,0)+w
        return m
    def project(self,mj,rho,R):
        out={}
        for F,mass in mj.items():
            if F==self.TH or mass==0: continue
            ks=[k for k in range(self.K) if F>>k&1]; contour=np.max([R[k] for k in ks],0)
            for s,w in self.consonant(contour).items(): out[s]=out.get(s,0)+rho*mass*w
        out[self.TH]=out.get(self.TH,0)+1-sum(out.values()); return out
    def conj(self,m1,m2):
        o=np.zeros(self.TH+1)
        for a,x in m1.items():
            for b,y in m2.items(): o[a&b]+=x*y
        return o
    def dempster(self,m1,m2): o=self.conj(m1,m2); return {s:o[s]/(1-o[0]) for s in self.S}
    def q_of(self,m): return {a:sum(v for s,v in m.items() if s&a==a) for a in self.S}
    def weights(self,m):
        q=self.q_of(m); w={}
        for a in range(0,self.TH):
            w[a]=np.exp(sum(-((-1)**(popc(b)-popc(a)))*np.log(q[b]) for b in self.S if b&a==a))
        return w
    def cautious(self,m1,m2):
        w1,w2=self.weights(m1),self.weights(m2); w={a:min(w1[a],w2[a]) for a in w1}
        q={c:np.prod([w[a] for a in w if c&a!=c]) for c in self.S}
        return {a:sum((-1)**(popc(b)-popc(a))*q[b] for b in self.S if b&a==a) for a in self.S}
    def fold(self,ms,f):
        m=ms[0]
        for x in ms[1:]: m=f(m,x)
        return m
    def empty(self,m): return 1-sum(m.values())
    def betp(self,m):
        s=sum(m.values()); return np.array([sum(v/popc(k) for k,v in m.items() if k>>l&1)/s for l in range(self.K)])
    def primary(self,centre,nbrs,rho,R):
        pool=self.fold([self.project(x,rho,R) for x in nbrs],self.cautious); return self.cautious(centre,pool),pool
def kernel(cent,s): c=np.asarray(cent,float); return np.exp(-(c[:,None]-c[None,:])**2/(2*s**2))
if __name__=='__main__':
    rng=np.random.default_rng(0); F=Frame(4); R=kernel([9.68,22.09,37.20,54.10],8.0)   # s=8 PLACEHOLDER for the toy only
    def rand_bba():
        m={a:rng.random()**3 for a in F.S}; m[F.TH]+=0.3; t=sum(m.values()); return {a:v/t for a,v in m.items()}
    def dist(a,b): return max(abs(a.get(s,0)-b.get(s,0)) for s in F.S)+abs(F.empty(a)-F.empty(b))
    d1=d2=d3=0
    for _ in range(20):
        x,y,z=rand_bba(),rand_bba(),rand_bba()
        d1=max(d1,dist(F.cautious(x,x),x)); d2=max(d2,dist(F.cautious(x,y),F.cautious(y,x)))
        d3=max(d3,dist(F.cautious(F.cautious(x,y),z),F.cautious(x,F.cautious(y,z))))
    print('K=4 AUDIT on 20 random non-dogmatic BBAs incl. empty set: idempotent %.1e commutative %.1e associative %.1e'%(d1,d2,d3))
    # projection sums to 1 incl. composite focal sets
    mj={0b0001:.3,0b0011:.2,0b0110:.1,0b1000:.15,F.TH:.25}; p=F.project(mj,.8,R); print('projection sum',round(sum(p.values()),12),{bin(k):round(v,3) for k,v in p.items()})
    # K=4 toy: centre uncertain, 8 neighbours
    cl=lambda i,p,th=.1:{1<<i:p,F.TH:1-p}
    centre={1:.1,2:.25,4:.12,8:.05,F.TH:.48}
    for name,nb in (('8 neighbours class 1 (agree)',[{1:.7,2:.1,4:.05,F.TH:.15}]*8),('4 class1 + 4 class4 (disagree)',[{1:.7,2:.1,4:.05,F.TH:.15}]*4+[{8:.7,4:.15,F.TH:.15}]*4)):
        for rho in (.5,.8,.95):
            m,pool=F.primary(centre,nb,rho,R); print(f'{name:32s} rho={rho}: BetP={np.round(F.betp(m),2)} m(Θ)={m[F.TH]:.2f} m(∅)={F.empty(m):.2f} pool∅={F.empty(pool):.2f}')
    m1,_=F.primary(centre,nb[:1],.8,R); m8,_=F.primary(centre,[nb[0]]*8,.8,R); print('repeat n=1 vs 8 max diff %.1e'%dist(m1,m8))
