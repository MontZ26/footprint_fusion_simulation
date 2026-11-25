exec(open('evid2.py').read().split("if __name__")[0])
def full(m): # add m(empty)
    d=dict(m); d[0]=1-sum(m.values()); return d
def close(a,b): return max(abs(a.get(s,0)-b.get(s,0)) for s in range(1,8))
P=lambda x:project(x,.8)
# 1 consonant contour check incl. rho on contour == rho*cons+(1-rho)Theta
for k in range(3):
    c=consonant(R[k]); mine=project({1<<k:1.0},.8)
    c2=consonant(np.append(.8*R[k],0)[:3]) # contour rho*R (max .8)
    alt={}
    for s,w in c2.items(): alt[s]=alt.get(s,0)+w
    alt[TH]=alt.get(TH,0)+1-sum(alt.values())
    print('class',k,'focal sets',{names[s]:round(w,2) for s,w in mine.items() if w>1e-12},'| equals contour rho*R construction:',close(mine,alt)<1e-12)
# 2 cautious properties
x,y,z=P(A),P(C),P(B)
print('cautious idempotent',close(cautious(x,x),x),'commutative',close(cautious(x,y),cautious(y,x)),'associative',close(cautious(cautious(x,y),z),cautious(x,cautious(y,z))))
print('centre alone: cautious(B,B)=B',close(cautious(B,B),B))
# 3 primary = cautious(centre, cautious-pool(neighbours)); conflict m(empty)
print('\nPRIMARY (cautious pool + cautious centre) m(Θ), m(∅) , BetP(normalised)')
for rho in (.5,.8):
    for sn,nb in {'agree 8xA':[A]*8,'disagree 4A4C':[A]*4+[C]*4,'uncertain 8xB':[B]*8}.items():
        pool=fold([project(x,rho) for x in nb],cautious); m=cautious(B,pool); e=1-sum(m.values())
        bet=[sum(v/bin(k).count('1') for k,v in m.items() if k&b)/(1-e) for b in (1,2,4)]
        pe=1-sum(pool.values())
        print(f' rho={rho} {sn:14s} m(Θ)={m[TH]:.2f} m(∅)={e:.2f} (pool alone m(∅)={pe:.2f}) BetP={bet[0]:.2f},{bet[1]:.2f},{bet[2]:.2f}')
