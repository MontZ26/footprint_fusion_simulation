"""Synthetic forest-like daily data to TEST the pipeline only (not for results): uptake follows a seasonal cycle driven by temperature/radiation plus
AR noise; EVI follows a smoothed seasonal state; so the three sources carry partly different information."""
import numpy as np, pandas as pd
rng=np.random.default_rng(1); rows=[]
for s in range(4):
    d=pd.date_range('2015-01-01','2019-12-31'); n=len(d); doy=d.dayofyear.values
    tair=10-12*np.cos(2*np.pi*(doy-10)/365)+rng.normal(0,3,n)+np.cumsum(rng.normal(0,.3,n))*0.1
    rad=150+120*(-np.cos(2*np.pi*(doy-10)/365))+rng.normal(0,40,n)
    state=np.zeros(n); 
    for i in range(1,n): state[i]=0.9*state[i-1]+0.1*(0.15*tair[i]+0.01*rad[i])+rng.normal(0,.05)
    uptake=np.clip(state,0,None)*2+rng.normal(0,.4,n)+0.02*(tair-tair.mean())
    evi=0.3+0.15*state+rng.normal(0,.02,n); evi=pd.Series(evi).where(rng.random(n)<0.15).values  # sparse satellite observations
    rows.append(pd.DataFrame(dict(site=f'S{s}',date=d,nee=-uptake,evi=evi,tair=tair,rad=rad)))
pd.concat(rows).to_csv('forest_daily_synthetic.csv',index=False)
