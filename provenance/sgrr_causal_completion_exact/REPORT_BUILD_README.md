# Final English DOCX Report Build Guide

## Purpose

After all three completion analyses have been sealed, `build_final_report.py` generates a read-only English master report of approximately 25-40 pages:

- output: `K1K2_Per_Joint_Causal_Mechanism_and_Rolling_Interpretation_Report_20260804.docx`;
- manifest: `FINAL_REPORT_MANIFEST.json`;
- design: `narrative_proposal` preset with `editorial_cover`;
- page: portrait US Letter, 1-inch margins, Calibri 11 pt body text, Microsoft YaHei as the East Asian fallback; and
- tables: 9,360 DXA body width, 120 DXA indentation, with identical `tblW`, `tblGrid`, and per-cell `tcW` geometry.

The script modifies no training log, checkpoint, trace, CSV, or analysis output, and it refuses to read derived results that have not passed their completion gates.

## Input Gates

All of the following must exist and pass:

1. `analysis_causal_completion/analysis_manifest.json` and `classification.json`;
2. `analysis_local_actor/ANALYSIS_MANIFEST.json` and `JACOBIAN_VALIDATION_PASS.json`;
3. `analysis_physical_propagation/ANALYSIS_AUDIT.json` with `status=complete` and `inputs_unchanged=true`;
4. `matched_c00/MATCHED_C00_COMPLETE.json`, confirming the same seeds and evaluation seeds 20264401-20264420, per-seed successes `[18,0,9,0,19]/20`, total `46/100`, and unchanged checkpoint/policy artifacts;
5. `matched_c00/results/seed9201..9205/C00.json`, each matching the completion gate's hash;
6. `FINAL_EVIDENCE_SEAL.json`, `VALIDATION_PASS.json`, and `FINAL_MECHANISM_SYNTHESIS.md` from the legacy mechanism study; and
7. `ANALYSIS_COMPLETE.json` from the historical K1/K2 atlas.

If any gate is missing, the script fails closed and does not emit a partial report.

## Runtime Environment

Use the bundled Python returned by the Codex workspace dependency loader, not system Python, global packages, or a project-local installation. That Python must import:

- `python-docx`;
- `pandas`; and
- `Pillow`.

Replace `<BUNDLED_PYTHON>` below with the executable returned by the loader:

```powershell
& '<BUNDLED_PYTHON>' '.\build_final_report.py' --validate-only
& '<BUNDLED_PYTHON>' '.\build_final_report.py'
```

Optional paths:

```powershell
& '<BUNDLED_PYTHON>' '.\build_final_report.py' `
  --output '.\K1K2_Per_Joint_Causal_Mechanism_and_Rolling_Interpretation_Report_20260804.docx' `
  --manifest '.\FINAL_REPORT_MANIFEST.json'
```

Report and manifest paths are forcibly restricted to this study root.

## Report Structure

The report dynamically reads the three new analysis directories, the legacy `FINAL_MECHANISM_SYNTHESIS.md`, and historical atlas metrics. Its structure is fixed:

1. executive summary;
2. contract and the key correction that the architecture contains eight local 2-to-2 actors;
3. project-chat and evidence-evolution timeline (s0 observation -> formal reproduction -> 59 conditions -> 113 conditions -> architecture correction);
4. numbered validated conclusions in evidence-chain order;
5. 16 K mechanism cards (K1/K2 for J01-J08);
6. event-stage patterns in the local Jacobian;
7. Shapley allocation of clipped torque, power proxy, contact, and physical propagation;
8. per-K zeroing, sign flip, dose, static replacement, and time permutation;
9. whole-joint K1+K2 necessity and same-seed/same-initial-state sufficiency;
10. the research question actually requested by the adviser;
11. priority-numbered unresolved questions;
12. evidence boundary and shortest reproduction path; and
13. an appendix containing only three key historical figures.

The script does not duplicate all 710 historical atlas figures. New causal-completion, local-Jacobian, Shapley, and physical-propagation figures take priority.

## Mandatory Render QA After Generation

Generating the DOCX is not completion. Use the documents skill's `render_docx.py` with the same bundled Python, render every page, and inspect at 100% zoom:

```powershell
& '<BUNDLED_PYTHON>' `
  'C:\Users\PUBLIC_USER\.codex\plugins\cache\openai-primary-runtime\documents\26.802.11031\skills\documents\render_docx.py' `
  '.\K1K2_Per_Joint_Causal_Mechanism_and_Rolling_Interpretation_Report_20260804.docx' `
  --output_dir '.\report_qa' `
  --emit_pdf
```

Check every page for:

- approximately 25-40 pages;
- correct rendering of all English text, mathematics, subscripts, and minus signs;
- figures, captions, and interpretations kept together;
- uncropped tables, no fixed row heights, and sufficient cell padding;
- all 16 mechanism cards complete;
- consistent heading hierarchy, headers, footers, and page numbering;
- no excessive blank areas, widows/orphans, broken page transitions, or undersized figures; and
- whether the table-of-contents field must be updated in Word or LibreOffice.

After any layout issue, edit `build_final_report.py`, regenerate, rerender, and reinspect every page. Update `qa_status` in `FINAL_REPORT_MANIFEST.json` only after the final render passes.

## Preset Audit Points

- Body: Calibri/Microsoft YaHei, 11 pt, justified, 8 pt after, 320-twip line spacing (1.333x).
- H1: 16 pt, `#2E74B5`, 18 pt before/10 pt after.
- H2: 13 pt, `#2E74B5`, 12 pt before/6 pt after.
- H3: 12 pt, `#1F4D78`, 8 pt before/4 pt after.
- Lists: true Word numbering; marker 261 DXA, body 540 DXA, hanging 279 DXA, 290-twip line spacing.
- Tables: 9,360 DXA, 120 DXA indent, single-line grid, `#F4F6F9` header fill.
- Cell margins: 80 DXA vertically and 120 DXA horizontally.
- First page: `editorial_cover`, generous whitespace, no decorative table.
- Body pages: restrained header rule and right-aligned page number.

## Evidence-Wording Boundary

Manual QA must preserve these statements:

- 100 episodes are not 100 independent policies; the inference units are five training seeds.
- The direct cross-joint actor Jacobian is structurally zero; cross-joint effects arise from closed-loop physical propagation.
- `phi x theta_dot` is only a control-boundary `power proxy`, never exact physical energy.
- Whole-joint sufficiency must use `matched::C00`, paired by training seed, Rroll environment, and evaluation seed 20264401-20264420.
- The 16 legacy single-channel sufficiency labels still come from within-study pairings in the legacy 59-condition study and must not be presented as matched-C00 results from this study.
- Any K that does not meet a frozen threshold remains unresolved and must not be relabeled necessary or useless.
- Historical response plots only describe what a policy may output; they do not replace frozen causal interventions.
