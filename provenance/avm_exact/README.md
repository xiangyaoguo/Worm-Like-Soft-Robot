# HPR O1-sham formal extension: operator guide

This package is designed to be deployed at:

```text
E:\finalproject\result\formal_hpr_o1_sham_vs_o2_20260811\_control\code_snapshot
```

The original ten formal runs remain untouched.  The five new training products
belong to the same formal study tree but are segregated under:

```text
<parent>\formal\extensions\formal_o1_sham_hpr_20260811
```

## Pre-training checks

Run with the project Python and site-packages on `PYTHONPATH`:

```powershell
python verify_extension.py
python test_actor_observation_sham.py
python launch_formal_o1_sham.py --phase commands
python launch_formal_o1_sham.py --phase freeze
python launch_formal_o1_sham.py --phase preflight
```

`preflight` constructs each seed-paired actor, critic and optimiser, compares
all locked hashes, and exits before collector iteration.  It is not a training
run.

## Explicit approval marker

Do not create this marker until the frozen manifest and all five preflight
audits have been independently inspected.  Its location is:

```text
E:\finalproject\result\formal_hpr_o1_sham_vs_o2_20260811\_control\FORMAL_APPROVAL.json
```

Required content:

```json
{
  "statement": "approved_for_formal_hpr_o1_sham_extension_from_scratch",
  "frozen_manifest_sha256": "REPLACE_WITH_PRINTED_FROZEN_MANIFEST_SHA256"
}
```

Only then may the operator deliberately run:

```powershell
python launch_formal_o1_sham.py --phase launch
```

The launcher permits a pre-created run directory only when it is strictly
empty.  It refuses to overwrite a non-empty run directory or any existing
initialisation audit.  No automatic resume, retry, seed replacement, checkpoint
selection or budget extension is implemented.

## Status

```powershell
python launch_formal_o1_sham.py --phase status
```

Paper-facing identifiers are `run 0--4`; seeds `9201--9205` remain internal
reproducibility identifiers.
