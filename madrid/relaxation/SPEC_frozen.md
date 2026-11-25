FROZEN BEFORE VALIDATION RUN (reconstructed reference implementation; original code unavailable)
Data: loso.csv (5488 LOSO cases, 236 days). Train days 1-70, validation days 71-140, final days 141-236 (NOT touched until selection frozen).
Fixed: 4 nearest available neighbours (held-out excluded), eta=0.5, m=3 (third largest), T=3, class edges 15/30/45, centroids/C_emp/s/D_k/H_k from train days only,
sigma_i = expanding RMSE of fuse_var on earlier days (fallback 6.5 for first 10 days), s (physical scale) = train RMSE of fuse_var.
Compatibility variants: P1 physical only; P2 empirical only; P3 global hybrid alpha=0.5; P4 class-conditional omega_k=D_k/(D_k+tau)*(1-H_k), tau in {5,15,30}. gamma in {0.25,0.5,0.75}. => 6 x 3 = 18 configs.
Update: pi <- (1-gamma)*pi + gamma*min(pi, Abar), renormalise by max; Abar = max(A,kappa)/max(.), kappa=1-max A.
Baselines (same cases): plain fusion; initial possibility pi0 (no relaxation); reconstructed soft probabilistic relaxation (medium matrix, 3 iterations); probabilistic initial classification.
SELECTION RULE (validation only):
 1. Filter: non-inferiority vs plain fusion: upper 95% limit (7-day block bootstrap) of RMSE(model)-RMSE(fuse) <= 0.05 ug/m3.
 2. Primary: highest coverage of true class by possible set P_theta={l: pi(l)>=theta} at mean set size 2.0 (linear interpolation of coverage-size curve).
 3. If within 0.005 coverage of the best: lower AURC (risk-coverage, 0/1 error of top class, ranking by necessity of top class; higher = keep).
 4. Then lower RMSE.
All 18 configs are recorded. If the selected configuration does not beat the probabilistic credible-set baseline on validation, that is reported; the final period is still run once.
