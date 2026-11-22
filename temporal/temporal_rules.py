import numpy as np
def comb_ccr_shortcut(ms):
    """CCR as implemented in temporal.py: minimum canonical weight on each side, then Dempster-normalised
    conjunction of the two resulting simple supports; returns BetP(H). ms: list of (mH, mN, mT)."""
    wH=min(1-m[0] if m[0]>0 else 1.0 for m in ms); wN=min(1-m[1] if m[1]>0 else 1.0 for m in ms)
    a=(1-wH,0.0,wH); b=(0.0,1-wN,wN)
    K=a[0]*b[1]+a[1]*b[0]; z=1-K
    mH=a[0]*b[0]+a[0]*b[2]+a[2]*b[0]; mT=a[2]*b[2]
    return (mH+mT/2)/z if z>1e-12 else 0.5
