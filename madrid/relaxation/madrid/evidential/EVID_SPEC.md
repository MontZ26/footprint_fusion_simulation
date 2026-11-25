SPECIFICATION v2 (draft, to be frozen before real data): evidential relaxation, third method of the three-way comparison
Frame {classes}; toy: {L,M,H}. Same classes, same neighbourhood, same data split and bootstrap as the probabilistic and possibilistic methods.
1 Initial evidence of a pixel: consonant bba of the possibility contour pi0(l)=1-eta+eta*S~(l) (same eta, sigma, centroids as the possibilistic method; m(Theta)=1-eta at least).
2 Neighbour j with observed/assessed class bba m_j: for each class k, consonant bba with contour rho*R(k,.) (ties in R merged, zero-weight sets dropped), mixed with weights m_j(k); m_j(Theta) goes to Theta. [checked: equals rho*consonant(R_k)+(1-rho)Theta]
3 Pool the neighbours with the CAUTIOUS rule (same image, dependent errors). Canonical weights include the empty-set weight.
4 PRIMARY: combine pooled neighbours with the centre bba with the CAUTIOUS rule (centre not assumed distinct). Comparators: cautious pool + Dempster centre; Dempster all; Yager (order-dependent, applied in distance order, flagged); no relaxation; probabilistic Rosenfeld; possibilistic.
5 Conflict m(empty) retained, never silently normalised. Report (a) neighbour-pool conflict and (b) centre-vs-pool conflict separately. Pignistic probability normalised by 1-m(empty) for scoring only. Abstention thresholds from validation only.
6 Parameters: rho in {0.5,0.8,0.95}, chosen on validation by Brier of BetP; eta as in the possibilistic method. No other tuning.
7 Evaluation: temporal train/validation/final split; accuracy, Brier and ECE of BetP, coverage vs set size ({l: Pl(l)>=theta}), risk-coverage with m(empty) or Pl-Bel ranking, conflict distributions; paired 7-day block bootstrap.
8 Caveat to state in the paper: toy results show behaviour for constructed examples, not a general guarantee of ignorance preservation.

--- REVISIONS v2 (after code audit) ---
9 Compatibility is PREDECLARED, not chosen for outcome: R(k,l)=exp(-(c_k-c_l)^2/(2 s^2)) with class centroids c and s = training RMSE of the fused estimate (same as the physical compatibility of the possibilistic method). Madrid values give adjacent compatibility 0.20 / 0.09 / 0.05. Toy sensitivity (evid3.py) is reported as sensitivity only; toy value a=0.3 is NOT frozen.
10 Projection handles composite focal sets: contour pi_F(l)=max_{k in F} R(k,l); neighbour -> sum_F m_j(F)[rho Con(pi_F)+(1-rho) delta_Theta]. Singleton case unchanged (checked: sums to 1).
11 rho grid {0.5,0.8,0.95} on validation. Toy: agreement gives low as top class only if rho>=0.8 and adjacent compatibility <=0.3 (rho=0.5 never reaches it, tie at a=0.1).
12 Pignistic values are normalised by 1-m(empty) and ALWAYS reported together with m(empty); high m(empty) means the evidence is conflicting (flag/abstain), not that a class is supported. Conflict and ignorance are separate quantities.
13 Audit includes the empty set (idempotent 7e-17, commutative 0, associative 2e-16).
