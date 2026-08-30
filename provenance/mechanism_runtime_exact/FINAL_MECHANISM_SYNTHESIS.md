# Joint-Level K1/K2 Mechanism Analysis of Forward Robot Rolling

## Final conclusion

Under the current simulation, formal v2.1 checkpoints, and fixed success thresholds, stable forward rolling is not produced by a set of fixed constants. It is produced by a state-dependent closed-loop K trajectory. The most stable spatial structure is:

\[
\operatorname{sign}(K1_{J01\ldots J08})=[+,-,-,-,-,-,-,-]
\]

That is, K1 is positive at the first tail-end joint, J01, and negative at J02–J08; the policy still determines the absolute magnitude dynamically from each joint's current `[Δθ, theta_dot]`. When the learned dynamic magnitudes were retained and only this sign template was enforced, all 100/100 episodes across the five training seeds satisfied the complete five-part rolling threshold. Making J02 positive as well; making the first three or four joints positive; making every joint positive or every joint negative; alternating signs; or mirroring the positive endpoint to J08 all yielded 0/100.

K2 is not an ordinary coefficient that “only regulates speed and does not affect initiation.” K2 is dynamic velocity feedback. The current ablations are consistent with roles in tail-end initiation, energy injection/dissipation, and phase organisation, and they show that K2 can turn occasional rolling into stable rolling across training seeds. With dynamic K1 retained, zeroing K2 yielded only 41/100; scaling K2 to 0.5× yielded 100/100; the original 1.0× yielded 99/100; and 1.5× yielded 94/100. Retaining K2 only at the tail-end J01–J02 still yielded 89/100, whereas retaining only J03–J08 yielded 23/100. The current evidence therefore supports the statement that “J01–J02 are the dominant K2 region, while the contribution of J03–J08 is more training-seed dependent.”

The mean K values below must not be presented directly as a fixed controller. The policy re-emits K1/K2 at every control step; the raw gains have no finite bounds, while the final active torque is clipped to `[-9,9]`. In the saturation proxy recomputed at the control boundary from the current observation, the C11 proportions for J01 and J02 were approximately 95.4% and 92.2%, respectively. The environment then recomputes torque over the following 10 physics substeps using the latest state, so these values are not actual substep saturation fractions. Evidence for sign, relative spatial structure, and closed-loop timing is therefore stronger than evidence for mean absolute magnitude.

## What this analysis actually did

- It used the 10 formal `checkpoint_1500.pt` files from training seeds 9201–9205; no intermediate checkpoint was loaded and no retraining was performed.
- Observations always remained per-joint `[Δθ, theta_dot]`, and actions always remained per-joint `[K1,K2]`; the reward, PPO, physics, and success thresholds were unchanged.
- A matrix of 59 channel-transplant, per-joint, sign, region, magnitude, and timing conditions was frozen in advance. Every condition used the same 20 new initial states and 1000 steps per episode.
- The study comprised 5 trained policies × 59 conditions × 20 episodes = 5900 episodes.
- Success required all of the following simultaneously: at least 4 rolling pulses, at least 360° of desired-direction net rotation, a direction ratio of at least 0.70, at least 1 body length of forward progress, and a mean pulse interval no greater than 250 steps.
- One episode from each of the 295 policy–condition combinations was also extracted for independent stepwise replay. Observations, transformed actions, and all five metrics matched the main results across all 295,000 steps.
- A further 400 paired comparisons between the native R0 environment and the Rroll environment were completed. Each environment was run once per pair, giving 800 actual rollouts; the maximum stepwise error in position, velocity, observation, and action was 0.

The experimental matrix was frozen before the main results were observed. The detailed statistical summariser and extended figures were implemented transparently after result generation, so intervals and rankings in the tables should be treated as descriptive analyses rather than additional preregistered significance tests.

## Control law and spatial joint positions

The policy does not output torque directly. Each per-joint action is first multiplied by 100 and then enters:

\[
K_{1,i}=100u_{1,i},\qquad K_{2,i}=100u_{2,i}
\]

\[
\Delta_i=q_{i+1}-q_{i-1},\qquad
\tau_i^*=K_{1,i}\Delta_i+K_{2,i}\dot\theta_i
\]

\[
\tau_i=\operatorname{clip}(\tau_i^*,-9,9)
\]

Here, `Δθ` is the head-side neighbour angle minus the tail-side neighbour angle, not the joint's own angle. J01–J08 are fixed material coordinates:

```text
rear/tail                                                        front/head
P0 — J01 — J02 — J03 — J04 — J05 — J06 — J07 — J08 — P9  → direction of travel
```

K1 is spatial-curvature-gradient feedback: reversing the sign of K1 reverses the chirality/propagation tendency of this spatial coupling. K2 is velocity feedback: considered in isolation, `K2>0` acts in the same direction as angular velocity and injects energy into the motion, whereas `K2<0` opposes angular velocity and increases dissipation. Actual torque is jointly determined by K1, K2, passive elasticity, passive damping, contact, and clipping, so an individual sign cannot be mechanically equated with an upward or downward direction in world coordinates.

## First, a historical conclusion must be corrected

The old formal summary scored R0 as 0/20 because the R0 environment did not export the newer contact/support-event fields and the evaluator treated missing fields as failure. The cross-environment audit proved that the two environments generated exactly identical physical trajectories. When the shared trajectory-based metrics were recomputed, C00 under R0 produced the following results across the five seeds:

| seed | R0/C00 | Rroll/C11 |
|---|---:|---:|
| 9201 | 20/20 | 19/20 |
| 9202 | 0/20 | 20/20 |
| 9203 | 11/20 | 20/20 |
| 9204 | 0/20 | 20/20 |
| 9205 | 19/20 | 20/20 |
| Total | 50/100, 3/5 seeds | 99/100, 5/5 seeds |

The rolling reward therefore did not create rolling from an absolute “never rolls” baseline. Instead, it converted a behaviour that depended on the training seed under R0 into one reproduced by every formal training seed. This correction does not weaken the conclusion that Rroll is robust, but it changes the thesis question from “does the reward produce rolling?” to “why does K1/K2 co-adaptation turn occasional rolling into robust rolling?”

## Overall causal relationship between K1 and K2

The four channel-transplant conditions are C00 = R0 K1 + R0 K2, C10 = Rroll K1 + R0 K2, C01 = R0 K1 + Rroll K2, and C11 = Rroll K1 + Rroll K2.

| Condition | Successful episodes by seed | Overall success rate | Seeds meeting criterion |
|---|---|---:|---:|
| C00 | 20, 0, 11, 0, 19 | 50% | 3/5 |
| C10 | 20, 0, 0, 0, 20 | 40% | 2/5 |
| C01 | 20, 0, 0, 0, 19 | 39% | 2/5 |
| C11 | 19, 20, 20, 20, 20 | 99% | 5/5 |

Against the other R0 channel, transplanting K1 or K2 alone did not improve success; transplanting the pair caused seeds 9202–9204 to develop robust rolling abruptly. The non-additive interaction on the success-rate scale is `C11−C10−C01+C00=+0.70`. This supports K1/K2 co-adaptation rather than a simple division of labour in which “K1 independently determines direction and K2 independently adjusts speed.” Because there are only five independent training seeds, the result cannot establish that this is the unique mechanism or that population-level significance has been demonstrated.

## Which K1 distributions produce forward rolling

### Components established causally

| K1 condition (K2 always from Rroll) | Success rate | Mean progress | Mean desired rotation | Mean pulses |
|---|---:|---:|---:|---:|
| Original dynamic C11 | 99% | 3.331 body lengths | 937.8° | 14.13 |
| J01 positive and J02–J08 negative; dynamic magnitudes retained | 100% | 3.484 body lengths | 957.6° | 14.38 |
| All K1 set to 0 | 0% | −0.150 body lengths | 2.9° | 0 |
| J01–J02 positive, all others negative | 0% | −0.032 body lengths | 2.9° | 0 |
| J01–J03 positive, all others negative | 0% | −0.154 body lengths | 4.7° | 0 |
| J01–J04 positive, all others negative | 0% | 0.034 body lengths | 3.2° | 0 |
| All positive / all negative / alternating / mirrored | 0% | all below 0.29 body lengths | all below 22° | all no greater than 0.30 |

Therefore:

1. Within the canonical dynamic sign templates tested here, positive K1 at J01 is part of the tail-end sign boundary.
2. J02 must switch immediately to negative. This J01/J02 sign boundary is the strongest current single-joint causal evidence for K1.
3. Among the templates tested here, only the canonical template with J03–J08 collectively negative succeeded. The frozen design did not flip J03–J08 individually under the complete C11 background and did not enumerate all sign combinations. It therefore cannot establish general necessity of this collectively negative region, still less absolute necessity of any individual joint.
4. The present data reject the specific half-positive patterns “first four consecutive joints positive” and “four alternating joints positive.” Not all 70 four-positive/four-negative combinations were tested, so the result cannot be generalised into a complete rejection of every distribution with approximately half the joints positive.

### Dynamic K1 envelope observed in successful policies

The following values are episode means for successful C11 policies, not fixed settings:

| Joint | Pooled 5-seed mean | Fraction of time with K1>0 | Range of seed-level episode means | Interpretation |
|---|---:|---:|---:|---|
| J01 | +69.75 | 95.64% | +44.75 to +93.75 | Strong positive tail-end initiation term |
| J02 | −88.02 | 0.06% | −126.52 to −46.58 | Strongest negative term; forms the sign boundary with J01 |
| J03 | −56.56 | 0% | −76.47 to −29.23 | Stable negative body-segment term |
| J04 | −54.75 | 0.76% | −88.67 to −27.44 | Stable negative body-segment term |
| J05 | −49.71 | 0.07% | −74.09 to −22.76 | Stable negative body-segment term |
| J06 | −39.31 | 0% | −80.15 to −5.37 | Negative term with relatively large magnitude freedom |
| J07 | −31.44 | 4.08% | −59.74 to −7.79 | Gradually weakens towards the head |
| J08 | −30.07 | 9.45% | −73.92 to −0.38 | Least stable and weakest negative endpoint |

As a shape summary, the pooled means can be normalised by `|J02|=1`:

\[
K1_{shape}\approx[+0.79,-1.00,-0.64,-0.62,-0.56,-0.45,-0.36,-0.34]
\]

This is only a spatial profile of the training result and must not be used as an open-loop fixed gain. The intervention that achieved 100% retained the policy's original time-varying dynamic magnitudes.

## Role of K2 at each joint

The pooled episode means of K2 under C11 are approximately:

\[
[+32.17,+16.00,-24.67,-7.70,-25.70,-26.68,-11.74,-32.92]
\]

The episode means for J01 and J02 were positive in every seed; those for J03 and J08 were negative in every seed; and the seed-level means for J04–J07 changed sign. Replacing one Rroll K2 in complete C11 with the same seed's R0 K2 gives the following “dependence on the Rroll source.” This is not K2 zeroing and therefore cannot be interpreted as strict necessity of K2 at that joint:

| Joint | Success after Rroll→R0 replacement | Success-rate decrease from C11 | Seeds with a decrease | Mechanistic interpretation consistent with the result |
|---|---:|---:|---:|---|
| J01 | 61% | 38 pp | 2/5 | Consistent with a tail-end initiation/energy-injection contribution, with clear policy dependence |
| J02 | 46% | 53 pp | 4/5 | Largest mean dependence on the Rroll source; consistent with the J01–J02 tail-end module evidence |
| J03 | 78% | 21 pp | 2/5 | Predominantly negative velocity feedback; smaller replacement effect with training-seed dependence |
| J04 | 50% | 49 pp | 3/5 | Consistent with a mid-body phase/contact-transfer contribution, with clear training-seed dependence |
| J05 | 65% | 34 pp | 2/5 | Predominantly negative feedback; consistent with a training-seed-dependent corrective role |
| J06 | 63% | 36 pp | 5/5 | The only joint whose replacement worsened every seed; consistent with a stable cross-policy contribution |
| J07 | 97% | 2 pp | 2/5 | Most replaceable under this Rroll→R0 substitution |
| J08 | 80% | 19 pp | 3/5 | Predominantly negative feedback; large replacement effects in a minority of seeds |

Dependence on the Rroll source cannot be ranked only by the mean decrease: J02 has the largest mean decrease, whereas J06 has the greatest directional consistency across seeds. These represent, respectively, “a stronger but policy-dependent replacement effect” and “a moderate replacement effect with stable direction across policies”; neither is equivalent to strict physical necessity.

## K2 magnitude, spatial region, and timing

| Condition | Success rate | Seeds meeting criterion | Mean progress | Mean desired rotation | Conclusion |
|---|---:|---:|---:|---:|---|
| K2=0 | 41% | 3/5 | 1.842 body lengths | 343.7° | Some policies can still roll, but not robustly |
| 0.5× dynamic K2 | 100% | 5/5 | 4.067 body lengths | 985.7° | Best in this grid; the original magnitude is not exactly optimal |
| 1.0× dynamic K2 | 99% | 5/5 | 3.331 body lengths | 937.8° | Robust baseline |
| 1.5× dynamic K2 | 94% | 5/5 | 2.511 body lengths | 733.6° | Still robust, but speed/persistence declines |
| J01–J02 K2 only | 89% | 4/5 | 3.779 body lengths | 822.1° | Tail-end module is dominant |
| J03–J08 K2 only | 23% | 1/5 | 0.583 body lengths | 139.3° | Robust initiation is difficult without tail-end K2 |
| All K2 positive | 0% | 0/5 | 1.428 body lengths | 19.6° | Translation can occur, but it is not rolling |
| All K2 negative | 2% | 0/5 | 0.152 body lengths | 34.5° | Excessive dissipation; almost no rolling |
| Static-mean K2 | 1% | 0/5 | 0.334 body lengths | 57.2° | A fixed mean cannot replace closed-loop control |
| Fixed temporal-template K2 | 8% | 0/5 | 0.675 body lengths | 114.3° | Open-loop timing cannot adapt to new initial states |
| Time-shuffled-template K2 | 5% | 0/5 | 0.452 body lengths | 65.0° | Further degradation after timing is disrupted |

The description supported by the current data is therefore as follows: dynamic K2 at J01–J02 is consistent with appropriately timed positive feedback/energy injection, while the remaining joints contribute to phase, contact transitions, and dissipation through velocity feedback whose sign can change. Closed-loop state dependence is more important than the static mean. A 0.5× scale is best within the current four-point dose grid, but the evidence does not establish that 0.5 is a global optimum in the continuous sense.

## How far can the necessity of each K1 be stated?

Per-joint K1 necessity in the frozen matrix was defined on the C10 background: every other K1 came from Rroll while K2 came from R0. This background itself achieved only 40% success and was driven mainly by seeds 9201/9205. Replacing individual K1 channels changed mean success by only −1 to +3 percentage points; single-joint K1 sufficiency was only 32%–48%, reaching criterion in just 2/5 seeds.

This indicates that the exact Rroll K1 value at one joint is neither independently sufficient nor universally irreplaceable; the current result is more consistent with a distributed spatial sign structure. However, this experiment cannot replace individual K1 zeroing/flipping at J03–J08 under the complete C11 background. The thesis must therefore distinguish the conditional causal conclusion about the J01/J02 sign boundary from the observation that “J03–J08 are collectively negative in the canonical successful template.”

## The research the supervisor actually requested

The supervisor's materials did not ask only for “another video of a trained robot that rolls.” They required the following falsifiable mechanism study:

1. Plot the spatial distribution of K1 across the material-joint index and determine whether it corresponds to propagation direction.
2. Systematically test positive/negative K1 combinations, with particular attention to the conjecture that “approximately half are positive.”
3. Distinguish the roles of K1 and K2 and test the claim that “K2 only regulates speed and does not determine initiation.”
4. Use reward only for training; deployment must still use the original `[Δθ,theta_dot]→[K1,K2]` mapping and must not add a global rolling state.
5. Use multiple training seeds, shared initial states, and a unified success criterion; do not substitute a single video or trajectory for evidence.

The present analysis has rejected the tested versions of two early intuitions: neither the first four consecutive K1 values being positive nor four alternating K1 values being positive produced rolling, and K2 is not merely a speed dial. A more accurate and conservative thesis claim is: among the tested sign templates, the dynamic K1 template with one positive tail-end joint and all remaining joints negative was the most robust; the results for state-dependent K2 are consistent with tail-end energy injection and whole-body phase regulation, and K1/K2 co-adaptation turns occasional rolling into robust rolling across seeds.

## Specific next steps

### Stage 1: Complete the full-C11 causal loop for per-joint K1

Continue evaluating only the existing checkpoints; do not train. Freeze the second-stage matrix before running:

- For each of J01–J08, test `K1=0`, `K1=-original value`, `0.5×K1`, and `1.5×K1`, while keeping every other channel at complete C11.
- For each of J03–J08, test “force only this joint positive” to answer directly whether negative K1 is necessary in each body segment.
- Add adjacent two-joint flips J02–J03, J03–J04, …, J07–J08 to test local redundancy.
- Retain seeds 9201–9205, the 20 fixed new initial states, 1000 steps, and the joint five-part threshold; do not select an intermediate checkpoint.
- Primary outputs should be the per-joint success-rate decrease, forward body lengths, desired rotation, pulse count, direction ratio, pulse interval, and change in torque clipping.

This is the necessary step for advancing the observation that “J03–J08 are collectively negative in the canonical successful template” into evidence about the contribution of each individual joint.

### Stage 2: Refine the optimal K2 magnitude interval

Freeze the dose grid `0.25, 0.50, 0.75, 1.00, 1.25, 1.50` and test:

- proportional scaling of every K2;
- scaling only J01–J02;
- scaling only J03–J08;
- separate scaling of J02 and J06.

The primary objective is to maximise the five-part success rate; forward body lengths are only a secondary objective. Energy/power and torque saturation must also be reported, and parameters must not be selected on speed alone. The current data support prioritising the 0.5–1.0 interval rather than increasing K2 indiscriminately.

### Stage 3: If retraining is undertaken, change only the reward

Observations and actions must remain permanently locked. The reward may consider:

- adding a weak K1 sign-consistency term during effective rolling phases: encourage J01 to be positive and J02–J08 to be negative, but reward only the sign/soft margin and not unbounded growth in absolute magnitude;
- avoiding any fixed all-positive or all-negative reward for K2; retain task rewards based on initiation, directional rotation, progress, and continuous pulses so that the policy learns phase-dependent sign switching;
- adding a mild active-torque saturation or control-power penalty to prevent unbounded K gains from pressing persistently against the `±9` limit;
- because this is a substantive reward change, first conducting an independent paired pilot with entirely new seeds before deciding whether to begin formal training; neither the present nor any historical checkpoint may be loaded.

A safe candidate soft constraint is:

\[
r_{sign}=w_s\left[\tanh(K1_{J01}/\kappa)+\frac{1}{7}\sum_{j=2}^{8}\tanh(-K1_j/\kappa)\right]
\]

It should be enabled only after a valid initiation/rolling precursor has appeared, preventing the policy from exploiting a large static gain for reward. `w_s` and `κ` must be frozen through a new paired pilot and must not be selected post hoc from the present results and then declared formally effective.

## Evidence boundaries

- There are only five independent training seeds. The smallest possible two-sided exact sign-flip-test p-value is 0.0625; the 20 evaluation initial states are repeated conditions for the same frozen policy, not 20 independent trained policies.
- After Holm multiplicity correction, the per-joint comparisons do not reach the conventional 0.05 threshold. Joint rankings should be interpreted as descriptive evidence based on effect magnitude and consistency across seeds.
- K1/K2 values are in simulation units. The source code is insufficient to convert them into physical motor N·m, stiffness, or damping.
- The results apply to the present robot, ground, physics parameters, and PPO policy. Transfer to hardware or other friction conditions requires separate validation.
- The 100% result for the K1 sign template retained the original policy's dynamic magnitudes and does not show that a set of static K1 constants can also produce rolling.

## Primary evidence files

- `analysis/RESULTS_REPORT.md`: automatically generated results report.
- `analysis/condition_summary.csv`: summary of the 59 conditions.
- `analysis/contrasts.csv`: prespecified success-rate contrasts and cross-seed effects.
- `analysis/continuous_mechanism_contrasts.csv`: continuous results for progress, rotation, pulses, direction, intervals, power, and related metrics.
- `analysis/baseline_joint_profile.csv`: per-joint K1/K2 distributions under C11.
- `analysis/RECOMMENDED_K_ENVELOPE.csv`: per-joint summary used in this document.
- `VALIDATION_PASS.json`: integrity of the 5900 episodes and recomputation of the five-part threshold.
- `INDEPENDENT_TRACE_AUDIT_PASS.json`: independent stepwise replay audit of 295 episodes.
- `CROSS_ENVIRONMENT_IDENTITY_AUDIT_PASS.json`: same-trajectory audit across the R0/Rroll environments.
- `VALIDATOR_SEMANTIC_AMENDMENT.md`: deduplication semantics for shared pulse boundaries and the validator correction record.
