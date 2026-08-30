# Semantics and Boundaries of the 59 Executed Conditions

This file records the semantics actually executed by the frozen runner so that interpretation does not exceed the implementation.

- A (4 conditions): C00/C10/C01/C11 are the 2 x 2 combinations of K1 and K2 channel sources proposed by the R0 and Rroll policies in the same current state.
- B (32 conditions): single-joint K1 sufficiency and necessity are both evaluated against an R0-K2 background. Single-joint K2 sufficiency uses an R0-K1 background, while K2 necessity uses an Rroll-K1 background. These are conditional effects with explicit backgrounds, not unconditional physical constants.
- C (13 conditions): the four K1 subsets use R0-K2; the nine sign/spatial conditions retain Rroll-K2. `K1_SIGN_MIRROR_CANONICAL_J08_POS` is actually a reversed fixed-sign template (J01-J07 negative and J08 positive, with magnitudes still taken from `abs(Rroll K1)` at each target joint). It is not a mirror of the joint-magnitude map and does not swap any observations.
- D (10 conditions): seven K2 magnitude/sign/region conditions, plus a calibrated static mean, a calibrated time template, and a fixed time-permuted template. The time templates come from identity-calibration episodes whose initial states are disjoint from the main test set. Templates use the global control-step clock and therefore do not have the same information set as the original local-feedback policy.

The following conditions were not executed in this study and therefore cannot be claimed as answered by these results:

- a true spatial mirror of joint K magnitudes or policy functions;
- closed-loop K2 delay;
- K2 time reversal;
- any general rule that "half the joints being positive" is sufficient (only two specific templates were tested: the first four positive, and alternating positive).

After the main run started, the frozen runner, condition matrix, checkpoints, initial states, episode counts, and success thresholds were not changed. The validator and statistical analyzer were transparent post-result implementations; see `ANALYSIS_TIMING_AND_STATUS.json`.
