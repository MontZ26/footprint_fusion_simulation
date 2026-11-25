import os
HERE=os.path.dirname(os.path.abspath(__file__))
import numpy as np, pandas as pd
from poss_relax import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
def feats(sub,nb):
    z=sub.fz.values[:,None]; mn=np.array([cent[r.nk].mean() for r in sub.itertuples()])[:,None]
    return np.hstack([z,mn]) if nb else z
tr=cs[cs.date.isin(TR)]; va=cs[cs.date.isin(VA)]
for nb in (False,True):
    m=LogisticRegression(C=1e4,max_iter=2000).fit(feats(tr,nb),tr.k); p=m.predict_proba(feats(va,nb))
    print('neighbour classes used' if nb else 'fused estimate only','| val logloss %.4f acc %.3f'%(log_loss(va.k,p,labels=[0,1,2,3]),(p.argmax(1)==va.k).mean()))
