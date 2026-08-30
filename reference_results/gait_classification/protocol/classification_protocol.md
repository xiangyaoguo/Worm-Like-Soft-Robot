# Formal offline gait-classification protocol

## Scope

This secondary analysis uses the locked six-configuration endpoint archive: six configurations, five independent training runs per configuration, twenty paired reset states per frozen checkpoint-1500 policy, 1,000 control steps, and 600 trajectories in total. No policy was retrained and no locked endpoint file was modified.

## Formal rolling gate

Formal rolling is copied without alteration from the thesis Equation 3.9: desired-direction net rotation at least 360 degrees, desired active-rotation fraction at least 0.70, and forward displacement at least one initial body length. The legacy direction-independent rotation-span flag is used only to prevent a full-rotation trajectory from being labelled crawling.

## Automated crawling-candidate gate

A trajectory is labelled a crawling candidate only if it is not formal rolling, its rotation span is below 180 degrees, and its full-episode forward displacement is at least 1.0 initial body length. After discarding the first 100 settling steps, it must advance by at least 0.75 body length. Steps 100--1000 are divided into 4 equal windows: at least 3/4 windows must advance by more than 0.05 body length, at least 3/4 must have circular joint-shape RMS amplitude of at least 20 degrees, and at least 3/4 must have a 5th--95th percentile geometric support-index span of at least 1.5 material indices. A support window is observable only if it contains at least 10 valid geometric-contact samples.

Shape periodicity, detected period, cycle progress, and support autocorrelation are reported only as descriptive gait-subtype features. They are deliberately not hard crawling gates because irregular but sustained crawling would otherwise be systematically excluded. The final thesis claim must distinguish automated candidates from independent human blind confirmation.

## Statistical unit

The independent training run is the configuration-level inference unit. The twenty reset trajectories are nested paired observations within one frozen policy. Results are therefore summarised first as counts per 20 rollouts for each run, then across the five independent runs. No rollout-level p-value is reported as if 100 rollouts were 100 independent training replications.
