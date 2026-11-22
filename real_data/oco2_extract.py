"""
Extract OCO-2 XCO2 soundings near the ICOS stations (NASA GES DISC, Earthdata login required).

Setup (once, on your computer):   pip install earthaccess netCDF4 numpy pandas
Run:   python oco2_extract.py              -> 2026 window only (7 stations overlap here)
       python oco2_extract.py --all        -> also 2023, 2024-25 windows (large download)
Output: OCO2_near_ICOS.csv  (one row per sounding within BOX degrees of a station)
Files are downloaded one at a time and deleted after extraction.
"""
import argparse, os, sys, glob, tempfile, datetime as dt
import numpy as np, pandas as pd, netCDF4 as nc, earthaccess

BOX = 1.0   # degrees (~100 km) around each station
STATIONS = {  # code: (lat, lon, start, end)
 "SAC": (48.7227,  2.1420, "2024-04-01", "2025-03-27"),
 "CBW": (51.9703,  4.9264, "2026-04-01", "2026-08-10"),
 "JFJ": (46.5475,  7.9851, "2026-04-01", "2026-08-10"),
 "MHD": (53.3261, -9.9036, "2026-07-07", "2026-08-10"),
 "CMN": (44.1936, 10.6999, "2023-04-01", "2024-01-08"),
 "PRS": (45.9300,  7.7000, "2026-04-01", "2026-08-09"),
 "LMP": (35.5181, 12.6322, "2026-04-01", "2026-08-09"),
 "PAL": (67.9733, 24.1157, "2026-04-01", "2026-08-10"),
 "SSL": (47.9167,  7.9166, "2026-04-01", "2026-08-10"),
 "IPR": (45.8147,  8.6360, "2023-04-01", "2023-08-07"),
}
ap = argparse.ArgumentParser(); ap.add_argument("--all", action="store_true")
args = ap.parse_args()
wins = [("2026-04-01", "2026-08-10")]
if args.all: wins += [("2023-04-01", "2024-01-08"), ("2024-04-01", "2025-03-27")]

earthaccess.login(strategy="interactive", persist=True)
rows = []
tmp = tempfile.mkdtemp()
for (a, b) in wins:
    res = earthaccess.search_data(short_name="OCO2_L2_Lite_FP", temporal=(a, b))
    print(f"{a}..{b}: {len(res)} granules found", flush=True)
    for i, g in enumerate(res):
        try:
            f = earthaccess.download([g], tmp)[0]
        except Exception as e:
            print("download failed", i, e); continue
        try:
            d = nc.Dataset(str(f))
            lat = d["latitude"][:]; lon = d["longitude"][:]
            t = d["time"][:]
            day = dt.datetime.utcfromtimestamp(float(np.ma.mean(t))).date()
            for code, (sl, so, s0, s1) in STATIONS.items():
                if not (s0 <= str(day) <= s1): continue
                m = (np.abs(lat - sl) <= BOX) & (np.abs(lon - so) <= BOX)
                if not m.any(): continue
                idx = np.where(m)[0]
                for k in idx:
                    rows.append(dict(station=code, time_utc=dt.datetime.utcfromtimestamp(float(t[k])).isoformat(),
                        lat=float(lat[k]), lon=float(lon[k]), xco2=float(d["xco2"][k]),
                        xco2_unc=float(d["xco2_uncertainty"][k]), qflag=int(d["xco2_quality_flag"][k])))
            d.close()
        finally:
            try: os.remove(str(f))
            except Exception: pass
        if i % 20 == 0: print(f"  {i}/{len(res)} rows so far {len(rows)}", flush=True)
pd.DataFrame(rows).to_csv("OCO2_near_ICOS.csv", index=False)
print("done:", len(rows), "soundings -> OCO2_near_ICOS.csv")
