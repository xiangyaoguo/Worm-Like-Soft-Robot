# Semantics and Boundaries of the 59 Conditions Actually Executed

This file records the semantics actually executed by the frozen runner so that the results are not interpreted beyond the code.

- A (4 conditions): C00/C10/C01/C11 are the 2 x 2 combinations of the K1 and K2 channel sources proposed by the R0 and Rroll policies from the same current state.
- B (32 conditions): Both K1 single-joint sufficiency and necessity are evaluated against the R0-K2 background. K2 single-joint sufficiency uses the R0-K1 background, whereas K2 necessity uses the Rroll-K1 background. These are condition effects with explicitly defined backgrounds, not unconditional physical constants.
- C (13 conditions): The four K1-subset conditions use R0-K2; the nine sign/spatial conditions retain Rroll-K2. `K1_SIGN_MIRROR_CANONICAL_J08_POS` is actually a "reversed fixed-sign template" (J01-J07 negative and J08 positive, with each target joint's magnitude still taken from its own `abs(Rroll K1)`). It is not a spatial mirror of the joint-amplitude map, and no observations are exchanged.
- D (10 conditions): Seven K2 magnitude/sign/region conditions, plus three calibrated conditions: static mean, temporal template, and a fixed time-permuted template. The temporal templates come from identity calibration episodes whose initial states are separate from those of the main test. The template uses the global control-step clock and does not have the same information set as the original local-feedback policy.

The following conditions were not executed in this study and therefore cannot be claimed as answered by its results:

- A true spatial mirror of joint K magnitudes / policy functions.
- Closed-loop K2 delay.
- K2 time reversal.
- Any general rule that "half the joints are positive" (only the specific first-four-positive and alternating-positive templates were tested).

After the main run began, the frozen runner, condition matrix, checkpoints, initial states, number of episodes, and success threshold were not modified. The subsequent validators and statistical analyzers were implemented transparently after the results; see `ANALYSIS_TIMING_AND_STATUS.json`.
