# Experiment diary

One entry per experiment (or per evaluation that led to a decision), newest last. This is
the only place in `docs/` where results are written down. Everywhere else, results stay
with their run outputs ([PIPELINE § How these docs are organised](../PIPELINE.md#how-these-docs-are-organised)).

An entry records what was run, what came out, how it compares with earlier runs, and
what the most likely explanation is. It is a dated snapshot: entries are not updated when
the code changes later, so a symbol or path in an old entry may no longer exist.

## Writing an entry

- File name: `YYYY-MM-DD-<short-slug>.md`, dated by when the run finished.
- Start with the commit the run used and the run directories (`TrainingOutputs/…`,
  `Outputs/…`) holding the full logs and figures. The entry quotes only the numbers its
  argument needs.
- Keep observations (from logs) apart from hypotheses (not yet tested), and name the check
  that would confirm or reject each hypothesis.
- Record when a result was read from `val`. The next entry that changes the design in
  response has to say so ([conventions § Split role naming](../reference/conventions.md#split-role-naming)).

## Entries

| Date | Entry | Outcome |
|---|---|---|
| 2026-09-25 | [First six-split IIoTset run](2026-09-25-iiotset-six-split-first-run.md) | AD far below the three-split run; thresholds do not transfer from `ad_calib` to later periods. The stage-1 follow-up (2026-09-26) points to the split, not the fine-tune |
| 2026-10-07 | [IIoTset session split (second run)](2026-10-07-iiotset-session-split.md) | The session split fixes the short-window failure and the FPR overshoot. The thresholds now overshoot the other way (TPR 0.09%), and the long-window bin and pooled AUROC drop. The structure fine-tune collapsed to rank ~2. Follow-up (2026-10-08): the threshold is set by the day-36 traffic burst, which falls into `ad_calib` but not `ad_fit` |
| 2026-10-12 | [IIoTset session split, 90-minute gap (third run)](2026-10-12-iiotset-90min-session-split.md) | With the λ 0.01 fine-tune: Mahalanobis pooled AUROC 0.998, TPR 99.85% at 0.46% FPR on `val`; thresholds transfer again. The stage-1 follow-up scores the same, so the split, not the fine-tune, is the cause. Long windows can't be checked against length-matched normals |
| 2026-10-13 | [CICAPT-IIoT retrain on the strict split](2026-10-13-cicapt-retrain.md) | One continuous recording, so the strict split is kept. The retrain reproduces the first run at the byte level; the λ 0.01 fine-tune shapes structure (silhouette 0.03 → 0.73) but its `_best` is epoch 1, so E85 was exported. No AD (no attack captures) |
