import numpy as np, fast_alt as fa, exact
rng=np.random.default_rng(5)
LO=np.array([exact.CO_R[0],exact.CH_R[0],exact.NO_R[0]]); HI=np.array([exact.CO_R[1],exact.CH_R[1],exact.NO_R[1]])
u=rng.uniform(0.05,0.95,400); dev=rng.normal(0,0.06,(400,3)); rand=LO+np.clip(u[:,None]+dev,0.02,0.98)*(HI-LO)
for name,edges,cent in (('default [30,60,80]',[30,60,80],[15,45,70,90]),('equal-width [25,50,75]',[25,50,75],[12.5,37.5,62.5,87.5]),('shifted [35,60,85]',[35,60,85],[17.5,47.5,72.5,92.5])):
    fa.EDGES[:]=edges; fa.LV=np.array(cent,float); exact.LV=fa.LV
    Z=len(rand); ns=20; M=np.repeat(rand,ns,axis=0); sd=np.tile(np.arange(500,500+ns),Z)
    y=np.repeat(np.clip(np.digitize(exact.comp(rand[:,0],rand[:,1],rand[:,2]),edges),0,3),ns)
    out=[]
    for mode,rl in (('classify',True),('raw',True),('classify',False)):
        r=fa.run_batch(M,sd,0.10,0.85,mode,rl); out.append((r['avg'].argmax(1)==y).mean())
    pm=fa.run_batch(M,sd,0.10,0.85,'classify',True)['score'].mean((1,2)); out.append((np.clip(np.digitize(pm,edges),0,3)==y).mean())
    print('%-24s classify+relax %.3f | raw+relax %.3f | classify only %.3f | pixel mean %.3f'%(name,*out))
