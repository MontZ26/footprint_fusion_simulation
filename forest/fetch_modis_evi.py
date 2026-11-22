try:
    import truststore; truststore.inject_into_ssl()
except ImportError:
    pass
"""Download MODIS MOD13Q1 EVI (250 m, 16-day) at each site from the ORNL DAAC MODIS REST service (no account needed).
RUN ON YOUR OWN COMPUTER (the cloud sandbox cannot reach it). UNTESTED against the live service: check the JSON field names on the first site.
Usage: python fetch_modis_evi.py sites.csv 2015 2019   -> evi_<site>.csv (date,evi)"""
import sys, json, urllib.request, pandas as pd, datetime as dt
BASE='https://modis.ornl.gov/rst/api/v1/MOD13Q1'
def get(url):
    with urllib.request.urlopen(url,timeout=120) as r: return json.load(r)
sites=pd.read_csv(sys.argv[1]); y0,y1=int(sys.argv[2]),int(sys.argv[3])
for _,s in sites.iterrows():
    d=get(f"{BASE}/dates?latitude={s.lat}&longitude={s.lon}")['dates']
    ds=[x['modis_date'] for x in d if y0<=int(x['modis_date'][1:5])<=y1]
    rows=[]
    for i in range(0,len(ds),10):   # service limits the number of dates per request
        u=f"{BASE}/subset?latitude={s.lat}&longitude={s.lon}&band=250m_16_days_EVI&startDate={ds[i]}&endDate={ds[min(i+9,len(ds)-1)]}&kmAboveBelow=0&kmLeftRight=0"
        for rec in get(u)['subset']: rows.append((rec['calendar_date'],rec['data'][0]*0.0001))
    out=pd.DataFrame(rows,columns=['date','evi']); out.loc[out.evi<-0.2,'evi']=float('nan')
    out.to_csv(f'evi_{s.site}.csv',index=False); print(s.site,len(out))
