# Relaxation study (reconstructed reference implementation)
- relaxation/relax_reconstructed.py : probabilistic Rosenfeld relaxation reconstructed after the original code was lost (not the original source). Run: python relaxation/relax_reconstructed.py
- relaxation/poss_relax.py          : possibilistic relaxation, frozen spec in SPEC_frozen.md; `python relaxation/poss_relax.py val` (validation days 71-140 only; the final period was NOT run).
- relaxation/headroom.py            : do neighbour classes add information beyond the fused estimate?
- evidential/                       : evidential relaxation toy (evid_k.py = K-class implementation), specifications, Earth Engine export (untested draft).
Needs madrid/loso.csv (in this folder). Python 3 with numpy, pandas, scipy, scikit-learn.
