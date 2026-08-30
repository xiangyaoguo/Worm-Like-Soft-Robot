# Final Conclusions and Open Questions

Date: 2026-08-04  
Research boundary: observations remain `[delta_theta, theta_dot]` per joint and actions remain `[K1, K2]` per joint. This study changed neither observations, actions, physics, PPO, nor frozen policies.

## I. Evidence Scope

1. Formal frozen Rroll baseline: five training seeds x 20 deterministic evaluation initial states = 100 episodes.
2. Causal-completion experiment: 113 conditions x five training seeds x 20 episodes = 11,300 episodes.
3. Same-seed, same-initial-state matched C00: 100 episodes.
4. Archived legacy mechanism experiment: 5,900 episodes.
5. Final joint analysis: 17,300 episodes and 865 condition-seed cells.
6. Local actor analysis: 105 trajectories and 105,000 time steps.
7. Closed-loop physical-propagation analysis: 160 paired intervention comparisons.

Terms such as necessary, timing-critical, and sufficient are operational causal conclusions under the current frozen contract, not universal mathematical theorems over every policy, terrain, and initial state.

## II. Established Conclusions

### A. Research Objective and Network Architecture

1. The adviser asked for more than a rolling video: the study must explain how local observations form reproducible distributed rolling through per-joint K1/K2, which signs, timing, and joint roles are irreplaceable, and how effects propagate through body and contact.
2. The policy contains eight local 2-to-2 networks with no sharing: `[K1_j,K2_j]=f_j(delta_theta_j,theta_dot_j)`.
3. The direct `16 x 16` policy Jacobian is strictly block diagonal: 32 within-joint entries and 224 structural cross-joint zeros.
4. Maximum error between analytic Jacobian and autograd is `4.97 x 10^-14`; maximum error against finite differences is `2.96 x 10^-9`.
5. A joint's K does not directly read another joint's observation at the same step. Cross-joint coupling comes from adjacent-angle definitions and closed-loop action -> dynamics/contact -> other local observation -> other local policy output propagation.

### B. Reproducibility of Rolling and Reward Effectiveness

6. Full frozen Rroll policies succeeded in 99/100 episodes, `[19,20,20,20,20]` by seed, robust in all 5/5 seeds.
7. Matched C00 succeeded in 46/100, `[18,0,9,0,19]` by seed; only 2/5 seeds met the single-seed threshold.
8. Legacy C00 succeeded in 50/100, `[20,0,11,0,19]` by seed. It is close to matched C00, but final paired comparisons must use matched C00.
9. Rroll reward materially increased robustness across seeds. It did not create rolling from complete absence; it stabilized a mechanism that was otherwise highly seed-dependent.
10. Legacy s0 rolled well at an intermediate checkpoint, but later K magnitude continued growing while rolling degraded. Larger K is therefore not sufficient for rolling.

### C. Per-K Intervention Conclusions

11. Six channels met the strong-necessity threshold under single-channel zeroing: J01-K1, J01-K2, J02-K1, J03-K1, J04-K1, and J05-K1.
12. Their zeroed success rates were respectively 47%, 26%, 13%, 1%, 0%, and 5%.
13. Sign-flipping each of those six strongly necessary channels reduced success to 0%, demonstrating that correct sign/action direction is central.
14. J01-K2 was the only channel meeting the strict dynamic timing-critical threshold. Static-mean replacement yielded 27% and time permutation 37%, drops of 72 and 62 percentage points from the 99% baseline, each degrading 4/5 seeds.
15. J02-K2, J03-K2, J04-K2, J05-K2, both J06 channels, both J07 channels, and both J08 channels did not meet the single-channel strong-necessity threshold. Failure to meet the threshold does not mean absence of effect.
16. J06-K1 showed the strongest cross-seed heterogeneity: zeroing reduced aggregate success from 99% to 44%, but seed outcomes were `[1,20,20,3,0]`, so only 3/5 seeds degraded and the preregistered rule does not label it necessary.
17. Success remained 100% after zeroing J05-K2 or J07-K2, but continuous motion metrics failed the strict equivalence bounds, so neither may be called redundant.
18. No single-K zeroing condition among the 16 channels met strict equivalent/redundant criteria.
19. Scaling by 0.5 or 1.5 usually preserved high success, while zeroing or sign reversal often caused catastrophic failure. Within this tested amplitude range, sign and direction matter more than magnitude.
20. The amplitude study establishes tolerance near 0.5-1.5x only; it identifies neither an optimal magnitude nor a safe extrapolation limit.
21. Failures typically degraded pulse count, net rotation, direction ratio, forward distance, and pulse interval together rather than merely reducing speed.

### D. Whole-Joint K1+K2 Roles

22. Zeroing both K1+K2 at J1, J2, or J3 reduced success to 0%; all three joints are strongly necessary.
23. Whole-pair zeroing yielded 53% for J4 and 65% for J5; both are necessary but not strongly necessary.
24. J6 yielded 71% with only 3/5 seeds degrading, J7 yielded 93%, and J8 yielded 99%; these joints did not meet whole-joint necessity criteria.
25. No whole-joint K1+K2 zeroing met strict equivalence. Even J8 at 99% was not equivalent on continuous performance.
26. Transplanting one Rroll joint's K1+K2 onto matched C00 produced J1-J8 success rates of 38%, 35%, 40%, 35%, 26%, 40%, 33%, and 43%, all below C00's 46%.
27. No single joint independently creates rolling; rolling is a distributed multijoint cooperation result.
28. The current evidence does not identify a minimally sufficient multijoint subset.

### E. Local K1/K2 Response and Mechanical Roles

29. The most cross-seed-consistent local velocity-response directions include negative J01 `K1<-theta_dot`, positive J01 `K2<-theta_dot`, positive J02 `K1<-theta_dot`, positive J02 `K2<-theta_dot`, positive J03 `K2<-theta_dot`, negative J04 `K2<-theta_dot`, and negative J05 `K1<-theta_dot`.
30. The three largest absolute responses are J01 `K2<-theta_dot` (20.85), J01 `K1<-theta_dot` (16.04), and J02 `K1<-theta_dot` (10.62).
31. Many `dK/d(delta_theta)` entries and some other derivatives change sign across training seeds. A two-dimensional map from one seed cannot be generalized as a universal rule.
32. Exact Shapley allocation reconstructs clipped total torque with maximum error `1.78 x 10^-15`.
33. K1 has a positive power proxy at J02-J08 in 5/5 seeds, supporting the interpretation that K1 predominantly maintains positive active work/propulsion.
34. K2 injects power in most seeds at J01/J02 and absorbs or modulates power in most seeds at J03 and J04-J07, forming distributed front injection and midbody modulation/absorption roles.
35. Total-torque saturation rates are about 0.95 at J01 and 0.92 at J02, so numerical K growth need not cause proportional executed-torque growth.
36. Correlation between K and instantaneous contact is usually small (most `|r| <= 0.08`); current evidence does not support an interpretation that a K directly switches contact at the same step.
37. The support center moves from about 4.126 in q1 to 5.175 in q5, a shift of about 1.049 joint positions, evidencing support transfer through the rolling cycle.
38. `phi x theta_dot` is only a boundary power proxy and must not be presented as exact energy across the simulator's ten physics substeps.

### F. Closed-Loop Cross-Joint Propagation

39. At intervention lag 0, the maximum K difference at all other joints is exactly zero, ruling out direct cross-joint actor reads.
40. Typical K1-zero propagation is slower: source observation around step 3, adjacent observation step 4, distant observation step 6.5, adjacent K step 6.5, distant K step 7.5, contact step 11, rotation step 12.5, and support step 15.
41. K2 sign flips propagate fastest: typical first statistical separation is about one step for adjacent/distant observations and K, two steps for support, 6.5 for rotation, and nine for contact.
42. For observations, K, and torque, all four intervention families showed distant separation later than adjacent separation in 5/5 seeds; position propagation followed this pattern in most or all seeds.
43. Within 50 steps, K2 sign flips propagated throughout the body in 100% of observation comparisons, 97% of K comparisons, 100% of torque comparisons, and 100% of position comparisons.
44. There is no universal fixed order in which contact changes precede rotation; for K2 sign flips, rotation separated first in 26/40 cases.
45. Every first-separation time is a statistical detection time requiring three consecutive steps beyond 0.5 baseline SD; it is not automatically the first engineering-significant physical change.
46. Some propagation has a long tail (rotation can separate as late as step 232), so median lag is not a fixed sequence for every trajectory.

### G. Historical Training and Overall Mechanism

47. Legacy four-condition success C00/C10/C01/C11 was 50%/40%/39%/99%, with interaction +0.70, showing K1 and K2 must cooperate and cannot be interpreted in isolation.
48. Legacy global K1 sign forcing showed `[+,-,-,-,-,-,-,-]` retained 100/100 success, while all-positive, all-negative, alternating, and mirrored patterns failed; the front sign boundary of positive J01 and negative J02 is especially important.
49. Legacy global K2 magnitude results were 41% at zero, 100% at 0.5x, 99% at 1x, and 94% at 1.5x. Moderate scaling is tolerated, but complete removal disrupts the mechanism.
50. Legacy K2 static-mean, open-loop-template, and time-permuted interventions achieved only 1%, 8%, and 5%, showing that static K distributions cannot replace dynamic closed-loop regulation.
51. The final mechanism picture is that anterior J1-J3 establish launch and directional boundaries, J4-J5 contribute necessarily to sustained rolling, and posterior J6-J8 provide seed-dependent distributed regulation. K1 emphasizes active power and propulsion, K2 emphasizes velocity-related injection/absorption and temporal shaping, and both form rolling through body-and-contact feedback.

## III. Unresolved Questions

1. What is the minimally sufficient multijoint subset, and which second-, third-, or higher-order K combinations are synergistic or compensatory?
2. What phase-specific causal role does each K play in launch, sustained rolling, recovery, and cycle reset? Current interventions persist from step 0 through the entire episode and cannot separate these roles.
3. What are the continuous optimal magnitude, stable interval, and safety boundary for every K? Current evidence has only zero, 0.5x, 1x, 1.5x, and sign-flipped points.
4. Are short-horizon propagation lags robust over multiple initial states and evaluation seeds? Current propagation diagnostics use only one fixed-evaluation-seed trajectory per condition.
5. How do particle-level contact order, normal impulse, and friction impulse produce support transfer?
6. What are exact actuator torque, mechanical work, and energy flow within the ten physics substeps? Current data contain only control-boundary proxies.
7. What are the success/failure counterfactuals near the same state, and how do small off-manifold observation perturbations change K and rolling?
8. How robust is the mechanism to random initial states, observation noise, different friction, slopes, obstacles, and domain randomization?
9. Does the simulated mechanism transfer to hardware, and how do K scale, saturation, latency, and contact-model error affect transfer?
10. The earliest s0-s2 results lack step-level torque, contact, and full-state traces matching the formal experiment specification; their mechanism analysis cannot be promoted to the same causal-evidence level as seeds 9201-9205.
11. Do the present necessity conclusions generalize to other networks, rewards, and terrains? At present they hold only for five frozen policies, deterministic flat-ground evaluation, and current thresholds.

## IV. Concise Answer

Forward rolling is not caused by one large K. It arises from correct signs, dynamic timing, and multijoint closed-loop cooperation. The most critical anterior and mid-anterior individual channels are J01-K1, J01-K2, J02-K1, J03-K1, J04-K1, and J05-K1. J01-K2 is also the only channel strictly shown to be timing-critical and therefore neither static-replaceable nor time-permutable. Whole joints J1-J3 are strongly necessary and J4-J5 are necessary, but no individual K or single joint is sufficient.
