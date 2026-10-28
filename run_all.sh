#!/bin/sh
# Regenerates every numerical result of the paper. Run from this directory.
python3 fast.py        # check: batch implementation equals the loop implementation on the nine zones
python3 exact.py       # Tables 1-3: nine-zone pipeline output (default seed 42)
python3 exp.py         # Tables 4-5: accuracy/MAE by noise level            -> exp1.json
python3 exp2b.py       # Tables 11-12, nine-zone Yager counts, confidence sweep (Table 22) -> exp2.json
python3 exp3.py        # Tables 6-10, 13-15, 17, 19-21: paired CIs, confusion, switch rules,
                       # baseline scenarios, class frame, noise models, design choices -> exp3.json
python3 exp4.py        # Tables 16 and 18: class frame across noise/baselines; comparison with reliability-weighted pooling -> exp4.json
python3 binsens.py     # bin-edge sensitivity quoted in Section 8.2
python3 compare.py     # checks that the regenerated results equal reference/*.json (the values behind the manuscript tables)
