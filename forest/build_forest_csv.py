"""Build forest_daily.csv (site,date,nee,evi,tair,rad) from (1) FLUXNET-format daily files (*_FLUXNET_DD_*.csv or *_DD_*.csv from FLUXNET2015 or the ICOS ecosystem L2 product,
columns TIMESTAMP, NEE_VUT_REF, TA_F, SW_IN_F, optional NEE_VUT_REF_QC) and (2) evi_<site>.csv from fetch_modis_evi.py.
Usage: python build_forest_csv.py <folder with FLUXNET daily files> <folder with evi_*.csv>
Site code is taken from the file name (e.g. FLX_DE-Hai_FLUXNET2015_FULLSET_DD_2000-2012...). Days with NEE QC < 0.5 (mostly gap-filled) are set to missing when the QC column exists."""
import sys, glob, os, re, pandas as pd
rows=[]
for f in sorted(glob.glob(os.path.join(sys.argv[1],'*_DD_*.csv'))):
    m=re.search(r'([A-Z]{2}-[A-Za-z0-9]{3})',os.path.basename(f)); st=m.group(1)
    x=pd.read_csv(f,na_values=['-9999']); x['date']=pd.to_datetime(x['TIMESTAMP'].astype(str),format='%Y%m%d')
    nee=x['NEE_VUT_REF'].where(x['NEE_VUT_REF_QC']>=0.5) if 'NEE_VUT_REF_QC' in x else x['NEE_VUT_REF']
    d=pd.DataFrame(dict(site=st,date=x['date'],nee=nee,tair=x['TA_F'],rad=x['SW_IN_F']*0.0864))  # W m-2 daily mean -> MJ m-2 d-1
    ef=os.path.join(sys.argv[2],f'evi_{st}.csv')
    if len(sys.argv)>3:
        g=glob.glob(os.path.join(sys.argv[3],f'*{st}*ERA5_DD_*.csv'))
        if g:
            e=pd.read_csv(g[0],na_values=['-9999']); e['date']=pd.to_datetime(e['TIMESTAMP'].astype(str),format='%Y%m%d')
            e=pd.DataFrame(dict(date=e['date'],tair_era=e['TA_ERA'],rad_era=e['SW_IN_ERA']*0.0864,vpd_era=e['VPD_ERA']))
            d=d.merge(e,on='date',how='left')
    if os.path.exists(ef):
        e=pd.read_csv(ef,parse_dates=['date']); d=d.merge(e,on='date',how='left')
    else: d['evi']=float('nan')
    rows.append(d)
D=pd.concat(rows); cols=['site','date','nee','evi','tair','rad']+[c for c in ('tair_era','rad_era','vpd_era') if c in D]; D=D[cols]; D.to_csv('forest_daily.csv',index=False)
print(D.groupby('site').agg(n=('nee','size'),nee_valid=('nee','count'),evi_valid=('evi','count')))
