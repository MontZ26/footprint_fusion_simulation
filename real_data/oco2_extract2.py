"""
Resumable OCO-2 extraction near ICOS stations. Safe to stop and restart.
Output: OCO2_near_ICOS.csv (appended after every granule); done.txt lists finished granules.
"""
import argparse, os, csv, datetime as dt, tempfile, sys
import numpy as np, netCDF4 as nc, earthaccess

BOX = 1.0
STATIONS = {
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

OUT, DONE = "OCO2_near_ICOS.csv", "done.txt"
FIELDS = ["station","time_utc","lat","lon","xco2","xco2_unc","qflag","granule"]
done = set(open(DONE).read().split()) if os.path.exists(DONE) else set()
if not os.path.exists(OUT):
    with open(OUT, "w", newline="") as f: csv.writer(f).writerow(FIELDS)

try:
    earthaccess.login(strategy="netrc")
except Exception:
    earthaccess.login(strategy="interactive", persist=True)

tmp = tempfile.mkdtemp()
total = 0
for (a, b) in wins:
    res = earthaccess.search_data(short_name="OCO2_L2_Lite_FP", temporal=(a, b))
    print(f"{a}..{b}: {len(res)} granules found", flush=True)
    for i, g in enumerate(res):
        gid = g["umm"]["GranuleUR"]
        if gid in done: continue
        f = None
        try:
            f = earthaccess.download([g], tmp)[0]
            d = nc.Dataset(str(f))
            lat = d["latitude"][:]; lon = d["longitude"][:]; t = d["time"][:]
            day = dt.datetime.utcfromtimestamp(float(np.ma.mean(t))).date()
            new = []
            for code, (sl, so, s0, s1) in STATIONS.items():
                if not (s0 <= str(day) <= s1): continue
                m = (np.abs(lat - sl) <= BOX) & (np.abs(lon - so) <= BOX)
                for k in np.where(m)[0]:
                    new.append([code, dt.datetime.utcfromtimestamp(float(t[k])).isoformat(),
                        float(lat[k]), float(lon[k]), float(d["xco2"][k]),
                        float(d["xco2_uncertainty"][k]), int(d["xco2_quality_flag"][k]), gid])
            d.close()
            with open(OUT, "a", newline="") as fo: csv.writer(fo).writerows(new)
            with open(DONE, "a") as fd: fd.write(gid + "\n")
            total += len(new)
        except Exception as e:
            print("FAILED", i, gid, e, flush=True)
        finally:
            if f is not None:
                try: os.remove(str(f))
                except Exception: pass
        print(f"{i+1}/{len(res)} {gid} rows added so far {total}", flush=True)
print("finished. See", OUT)
