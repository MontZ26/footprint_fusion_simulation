"""Compares freshly generated result files with the reference outputs behind the manuscript tables."""
import json, os, sys
ok = True
for f in ("exp1.json", "exp2.json", "exp3.json", "exp4.json"):
    if not os.path.exists(f):
        print("MISSING", f); ok = False; continue
    same = json.load(open(f)) == json.load(open(os.path.join("reference", f)))
    print("%-10s %s" % (f, "reproduces the reference" if same else "DIFFERS"))
    ok &= same
sys.exit(0 if ok else 1)
