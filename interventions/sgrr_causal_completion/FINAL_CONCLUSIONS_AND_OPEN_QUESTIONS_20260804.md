# Final Conclusions and Open Questions

Date: 2026-08-04  
Research boundary: the observation channels permanently remain `[delta_theta, theta_dot]` per joint, and the action channels permanently remain `[K1, K2]` per joint. This study did not modify the observations, actions, physics, PPO, or frozen policies.

## I. Scope of Evidence

1. Formal frozen Rroll baseline: 5 training seeds x 20 deterministic evaluation initial states, for 100 episodes.
2. Additional causal experiments: 113 conditions x 5 training seeds x 20 episodes, for 11,300 episodes.
3. Matched C00 with the same seeds and initial states: 100 episodes.
4. Sealed legacy mechanism study: 5,900 episodes.
5. Total included in the final joint analysis: 17,300 episodes and 865 condition-seed units.
6. Local actor analysis: 105 trajectories and 105,000 time steps.
7. Closed-loop physical-propagation analysis: 160 paired intervention comparisons.

Terms such as "necessary," "temporally critical," and "sufficient" below are operational causal conclusions under the current frozen contract, not mathematically universal theorems covering every policy, terrain, and initial state.

## II. Conclusions Reached

### A. Research Objective and Network Structure

1. The supervisor's actual requirement was not merely to obtain a video of rolling, but to answer how local observations form reproducible distributed rolling through per-joint K1/K2; which signs, temporal patterns, and joint roles are irreplaceable; and how those effects propagate through the body and contacts.
2. The policy comprises eight independent, nonshared local 2-to-2 networks: `[K1_j,K2_j]=f_j(delta_theta_j,theta_dot_j)`.
3. The 16 x 16 direct policy Jacobian is strictly block diagonal: 32 within-joint entries, while all 224 cross-joint entries are structural zeros.
4. The maximum error between the analytic Jacobian and automatic differentiation is `4.97 x 10^-14`; the maximum error against finite differences is `2.96 x 10^-9`.
5. Thus, a joint's K values do not directly read observations from other joints within the same step. Cross-joint coupling arises from the definitions of adjacent angles and from the closed-loop chain "action -> dynamics/contact -> other local observations -> other local policy outputs."

### B. Reproducibility of Rolling and Effectiveness of the Reward

6. The complete frozen Rroll policies succeed in 99/100 episodes, with per-seed counts `[19,20,20,20,20]`; all 5/5 seeds are robust.
7. Matched C00 succeeds in 46/100 episodes, with per-seed counts `[18,0,9,0,19]`; only 2/5 seeds meet the per-seed threshold.
8. The legacy C00 result is 50/100, with per-seed counts `[20,0,11,0,19]`. It is close to matched C00, but the final paired comparison must use matched C00.
9. The conclusion is that the Rroll reward markedly increases robustness across seeds. It does not create rolling from behavior that was entirely absent; it stabilizes a rolling mechanism that had previously been highly seed-dependent.
10. Legacy s0 rolled well at some intermediate checkpoints, but later K magnitudes continued to grow while rolling deteriorated. Thus, "larger K" is not sufficient for rolling.

### C. Intervention Conclusions for Each K

11. In the single-channel zeroing experiments, six K channels meet the "strong necessity" threshold: J01-K1, J01-K2, J02-K1, J03-K1, J04-K1, and J05-K1.
12. Their respective success rates after zeroing are 47%, 26%, 13%, 1%, 0%, and 5%.
13. Sign-flipping each of these six strongly necessary channels reduces success to 0%, showing that the correct sign/effect direction is a core condition.
14. J01-K2 is the only channel that meets the strict "dynamic temporal criticality" threshold: success is 27% after static-mean replacement and 37% after time permutation. Relative to the 99% baseline, these are drops of 72 and 62 percentage points, respectively, with deterioration in 4/5 seeds in both cases.
15. J02-K2, J03-K2, J04-K2, J05-K2, J06-K1/K2, J07-K1/K2, and J08-K1/K2 do not meet the strong-necessity threshold for a single channel, but "not meeting the threshold" does not mean "having no effect."
16. J06-K1 is the most pronounced cross-seed heterogeneous channel: zeroing lowers overall success from 99% to 44%, but the per-seed counts are `[1,20,20,3,0]`. Only 3/5 seeds deteriorate, so the preregistered rule does not permit labeling it necessary.
17. Success remains 100% after zeroing J05-K2 or J07-K2, but the continuous movement metrics do not satisfy the strict equivalence bounds, so neither channel may be called redundant.
18. Among the 16 channels, no single-K zeroing condition can be strictly classified as equivalent/redundant.
19. Scaling by 0.5 x or 1.5 x usually preserves high success, whereas zeroing or sign reversal frequently causes catastrophic decline. Within the currently tested finite magnitude range, sign and direction matter more than magnitude.
20. This magnitude experiment shows only that a tolerance region exists near 0.5-1.5 x. It cannot determine the optimal magnitude or extrapolate a safe upper bound.
21. Failure often involves simultaneous deterioration of pulse count, net rotation, direction ratio, forward distance, and pulse interval, rather than merely reduced speed.

### D. Whole-Joint K1+K2 Roles

22. Simultaneously zeroing K1+K2 at J1, J2, or J3 reduces success to 0% in every case, making these strongly necessary joints.
23. Whole-pair zeroing yields 53% at J4 and 65% at J5; both are necessary but not strongly necessary joints.
24. J6 yields 71% with deterioration in only 3/5 seeds, J7 yields 93%, and J8 yields 99%; these joints do not meet the whole-joint necessity threshold.
25. No whole-joint K1+K2 zeroing condition achieves strict equivalence. Even J8, which remains at 99%, is not equivalent in continuous performance.
26. After transplanting the K1+K2 pair from a single Rroll joint into matched C00, success rates for J1-J8 are 38%, 35%, 40%, 35%, 26%, 40%, 33%, and 43%, respectively; all are below the C00 value of 46%.
27. Therefore, no single joint can independently produce rolling; rolling results from distributed multijoint coordination.
28. Current evidence does not yet identify a minimal sufficient multijoint subset.

### E. Local K1/K2 Responses and Mechanical Roles

29. The most stable local velocity-response directions across all five seeds include: J01 `K1<-theta_dot` negative, J01 `K2<-theta_dot` positive, J02 `K1<-theta_dot` positive, J02 `K2<-theta_dot` positive, J03 `K2<-theta_dot` positive, J04 `K2<-theta_dot` negative, and J05 `K1<-theta_dot` negative.
30. The three strongest absolute responses are J01 `K2<-theta_dot` (20.85), J01 `K1<-theta_dot` (16.04), and J02 `K1<-theta_dot` (10.62).
31. Many `partial K / partial delta_theta` derivatives and some other derivatives change sign across training seeds; a two-dimensional map from one seed must not be generalized as a universal rule.
32. Exact Shapley allocation reconstructs the clipped total torque with a maximum reconstruction error of `1.78 x 10^-15`.
33. The K1 power proxy is positive at J02-J08 in 5/5 seeds, supporting the interpretation that "K1 primarily maintains active positive work/propulsion."
34. K2 injects power at J01/J02 in most seeds and absorbs or modulates power at J03 and J04-J07 in most seeds, creating distributed roles of "front-end injection and middle-section modulation/absorption."
35. Total-torque saturation rates at J01/J02 are approximately 0.95/0.92. Larger numerical K values therefore do not necessarily produce a proportional increase in actual torque.
36. Correlations between K and instantaneous contact are generally small (most `|r|<=0.08`). Current evidence does not support the claim that a particular K directly switches contact within the same step.
37. The support center moves from approximately 4.126 in q1 to 5.175 in q5, a shift of about 1.049 joint positions. This is evidence of support transfer during the rolling cycle.
38. `phi x theta_dot` may be described only as a boundary power proxy; it must not be presented as exact energy over the simulator's ten physics substeps.

### F. Cross-Joint Closed-Loop Propagation

39. At lag 0, when a single K intervention is applied, the maximum difference in other joints' K values is exactly 0. This rules out direct cross-joint reading inside the actor.
40. Propagation after K1 zeroing is typically slower: source observation at about 3 steps, neighboring-joint observation at 4 steps, distal observation at 6.5 steps, neighboring-joint K at 6.5 steps, distal K at 7.5 steps, contact at 11 steps, rotation at 12.5 steps, and support at 15 steps.
41. Propagation after a K2 sign flip is fastest: the typical first statistical separation for neighboring/distal observations and K is about 1 step, support about 2 steps, rotation 6.5 steps, and contact 9 steps.
42. For observations, K values, and torque, all four interventions show later distal than neighboring responses in 5/5 seeds. Position propagation follows the same ordering in most or all seeds.
43. Within 50 steps, the proportions of K2-sign-flip effects propagating across the whole body are: observations 100%, K 97%, torque 100%, and position 100%.
44. There is no universal fixed sequence in which "contact changes first and rotation changes next." Under K2 sign flip, rotation instead separates first in 26/40 comparisons.
45. Every first-separation time is a statistical detection time defined as "exceeding 0.5 baseline SD for three consecutive steps." It must not automatically be interpreted as the first physically meaningful engineering change.
46. Some propagation distributions have very long tails; for example, rotation may separate as late as 232 steps. The median lag is therefore not a fixed sequence for every trajectory.

### G. Historical Training and the Overall Mechanism

47. Legacy four-condition success rates C00/C10/C01/C11 are 50%/40%/39%/99%, with an interaction effect of +0.70. This shows that K1 and K2 must cooperate and cannot be interpreted in isolation.
48. Legacy global K1 sign-forcing experiments show that `[+,-,-,-,-,-,-,-]` preserves 100/100, while all-positive, all-negative, alternating, and mirrored schemes fail. The front-end sign boundary with positive J01 and negative J02 is especially important.
49. Legacy global K2 magnitude results are: zero 41%, 0.5 x 100%, 1 x 99%, and 1.5 x 94%. Again, moderate scaling is tolerated but complete removal disrupts the mechanism.
50. Legacy K2 static mean, open-loop template, and time permutation achieve only 1%, 8%, and 5%, respectively, showing that static K distributions cannot replace dynamic closed-loop regulation.
51. The final mechanism picture is as follows: front-end J1-J3 provide initiation and directional boundaries; J4-J5 make necessary contributions to sustained rolling; posterior J6-J8 provide seed-dependent distributed regulation. K1 is biased toward active power and propulsion, whereas K2 is biased toward velocity-related injection/absorption and temporal shaping. Together, they create rolling through the body/contact closed loop.

## III. Questions That Remain Unresolved

1. What is the minimal sufficient multijoint subset, and which second-order, third-order, and higher-order K combinations exhibit synergy or compensation?
2. What is the phase-specific causal role of each K during initiation, sustained rolling, recovery, and cycle reset? Current interventions persist from step 0 through the whole episode and cannot separate phase responsibilities.
3. What are the continuous optimal magnitude, stable interval, and safety boundary for each K? The current study tests only a finite set of points: 0, 0.5, 1, 1.5, and sign reversal.
4. Are short-term propagation lags robust across multiple initial states and evaluation seeds? Current propagation diagnostics use only one trajectory with a fixed evaluation seed per condition.
5. How do particle-level contact order, normal impulse, and frictional impulse complete the support transfer?
6. What are the exact actuator torque, mechanical work, and energy flow during the ten physics substeps? Only control-boundary proxies are currently available.
7. What are the success/failure counterfactuals near the same state, and how do small off-manifold observation perturbations change K and rolling?
8. How robust is the mechanism to randomized initial states, observation noise, different friction, slopes, obstacles, and domain randomization?
9. Can the simulated mechanism transfer to real hardware, and what are the effects of K magnitude, saturation, delay, and contact-model error?
10. The earliest s0-s2 results lack step-by-step torque, contact, and full-state trajectories matching the formal experimental specification. Their mechanism analyses therefore cannot be elevated to the same level of formal causal evidence as seeds 9201-9205.
11. Do the current "necessity" conclusions generalize to other networks, rewards, and terrains? At present, they apply only to the five frozen policies, deterministic flat-terrain evaluation, and current thresholds.

## IV. Concise Answer

The robot's forward rolling is not caused by any one "large K," but by correct signs, dynamic timing, and multijoint closed-loop coordination. The most important front and mid-front single channels are J01-K1, J01-K2, J02-K1, J03-K1, J04-K1, and J05-K1. J01-K2 is also the only channel currently shown under the strict criteria to be neither statically replaceable nor time-permutable. J1-J3 are strongly necessary as whole joints, and J4-J5 are necessary, but no single K or single joint is sufficient.
