# 2026-09-25 · First six-split IIoTset run

**Outcome.** The pipeline retrained end to end on the six-split IIoTset layout, and every
stage ran cleanly. Anomaly detection came out far below the earlier three-split run:
the focus detector (Mahalanobis) scored a pooled AUROC of 0.878 against 0.996 before, and
its thresholds do not transfer to `val` (realised FPR 36% at a 1% target). The most
likely cause is the evaluation protocol. The old split sampled `val` from the same
period and traffic mix as the data the detectors were fitted on. The new split holds out
a later period whose traffic the short AD fitting and calibration windows do not cover.
The model and the attack side changed little. See [§ Most likely cause](#most-likely-cause).

**Follow-up (2026-09-26):** the same evaluation of the six-split *stage-1* model
confirms that the split, not the fine-tune, is the main cause. Stage 1 also misses the
FPR target by a wide margin (24% on `val` at a 1% target) and fails on short windows.
See [§ Follow-up: stage-1 model on the six-split](#follow-up-stage-1-model-on-the-six-split).

**Later finding (2026-10-05): the capture is four recording sessions.** A per-day packet
profile of `NormalMerged.parquet` shows traffic only on days 0–2 (16.7% of packets),
day 28 (0.07%), day 36 (39.8%) and day 44 (43.4%) of the 45-day span. The "bursts" this
entry describes are those sessions. In this run, the AEs trained on days 0–36, and
`ad_fit`, `ad_calib`, `val` and `late` all came from the day-44 session. The
three-split run drew every split from all sessions. This sharpens the explanation below:
the AD splits were not just short. They all came from a recording session the AEs never
trained on, and `ad_fit` was a 23-minute slice of it. The next run cuts each session separately
([02 § Sessions](../components/02-temporal-split.md#sessions)).

`val` has been read **twice** in this experiment: once for the fine-tuned model
(2026-09-25) and once for the stage-1 model (2026-09-26). Any design change made in
response is informed by `val` and must be recorded as such.

## What ran

Code: commit `12d58d8`, plus `EmbeddingADSuite.py · RUN_NAME` pointed at the fine-tuned
model (committed together with this entry). Split: `SplitFlowsDF` with `SIX_ROLE_SPLITS`
at the same fractions as CICAPT.

| Stage | Run / output |
|---|---|
| Packet AE | `TrainingOutputs/PacketAE_IIoTset_d128` |
| Latent cache | `flow_split/latents_PacketAE_IIoTset_d128_best/{train,test}` |
| Sequence AE (stage 1) | `TrainingOutputs/SeqAE_IIoTset_d128_Mamba_s512` |
| Structure fine-tune | `TrainingOutputs/SeqAEStruct_IIoTset_d128_Mamba_s512_pair_l010` |
| Export | `Outputs/Embeddings/SequenceEmbeddings_SeqAEStruct_IIoTset_d128_Mamba_s512_pair_l010` |
| AD | `Outputs/AD/SeqAEStruct_IIoTset_d128_Mamba_s512_pair_l010_eval-val` |

Since 2026-10-05, all of these (split, cache, runs, embeddings, AD outputs, including
the stage-1 follow-up) are archived under `archive/2026-09-25-sixsplit/` at the same
relative paths.

The comparison run is the archived three-split evaluation
`archive/…/Outputs/AD/SeqAE_IIoTset_d128_Mamba_s512_eval-val` (2026-08-18). Its logs are
in `archive/` until that directory is deleted.

## Results next to the three-split run

Mahalanobis is the focus detector, fixed in advance. `seq_len_only` is the length
baseline.

| | Three-split (old) | Six-split (new) |
|---|---|---|
| Model evaluated | stage-1 `SeqAE_…_s512` | structure fine-tune `SeqAEStruct_…_pair_l010` |
| Roles | fit `train`, calib `test`, eval `val` | fit `ad_fit`, calib `ad_calib`, eval `val` |
| Mahalanobis pooled AUROC | 0.996 | 0.878 |
| Mahalanobis TPR @ q0.99 | 1.000 | 0.866 |
| Mahalanobis realised FPR on `val` @ q0.99 (target 1%) | 5.7% | 36.4% |
| Mahalanobis AUROC, bins 2–8 / 9–32 / 33–64 | 0.976 / 0.978 / 0.994 | 0.641 / 0.995 / 0.978 |
| `seq_len_only` pooled AUROC | 0.900 | 0.831 |
| Mahalanobis realised FPR on `late` @ q0.99 | — | 5.9% |

Other detectors show the same pattern. Their operating points break in both directions:
Mahalanobis, PCA reconstruction and LOF overshoot the target FPR on `val`, while
isolation forest ranks well (AUROC 0.993) but its threshold sits above almost everything
(TPR 0.8%).

## What differs between the two runs

| | Three-split (old) | Six-split (new) |
|---|---|---|
| Splitter | old flow-selection splitter: whole conversations assigned to splits, so every split spans the full ~45 days | strict temporal split: every flow is cut at the boundaries |
| Data the detectors are fitted on | `train`: the AE's own training data (88,505 conversations across the whole capture; 20,000 exported windows) | `ad_fit`: 16,379 windows from **23 minutes** at the tail of the capture |
| Calibration data | `test`: 20,034 conversations across the whole capture (20,000 windows) | `ad_calib`: **4.8 h** directly after `ad_fit` |
| `val` | a held-out *population* from the same 45 days | a held-out *period* (4.2 h) after `ad_calib` |
| Normal window length (median) | `train`/`test`/`val` all 64 | `ad_fit` 63, `ad_calib` 62, `val` **36**, `late` 33 |
| Model | stage 1 (effective rank of `z` ≈ 35) | stage 1 + structure fine-tune (effective rank 29 → 13) |
| Attack windows | 116,741 | 116,741 (the same captures and exports, differing only in the model) |
| Upstream reconstruction (on each run's own `test`) | packet AE non-pad 0.596; sequence AE non-pad 0.554 | packet AE non-pad 0.831; sequence AE non-pad 0.769 |

## Most likely cause

**The main cause is that the evaluation became honest, and the AD splits are too narrow
for that.** In the old run, the detectors were fitted, calibrated and evaluated on three
random draws from the same 45 days of traffic, so `val` was essentially drawn from the
same distribution as the fitting data. That is why the thresholds transferred almost
perfectly (FPR 5.7% at a 1% target). It is also why the old numbers were optimistic by
construction: this is the leak the temporal splitter was introduced to remove
([02 § How it works](../components/02-temporal-split.md#how-it-works)). In the new run,
the detectors see 23 minutes of traffic and the thresholds see the next 4.8 hours. `val`
is a different period with a different make-up. Its windows are much shorter (median 36
against 62–63), and `test` in the same layout shows that later periods can be dominated by
host pairs that barely appear earlier (half of `test`'s flows come from one endpoint pair
absent from `train`; only 50.5% of `test` flows fall in a group seen in `train`).
Mahalanobis measures distance from the fitting distribution, so normal `val` traffic that
differs from those 23 minutes is scored as anomalous, which produces the 36% FPR.

The evidence that the representation still separates attacks where the normals are
comparable: within the 9–32 and 33–64 length bins, Mahalanobis stays at 0.995 and 0.978,
the same level as before, and it beats `seq_len_only` there by a wide margin. The loss is
concentrated in the pooled number (driven by the calibration and length mismatch) and in
the 2–8 bin (0.976 → 0.641).

**Contributing factor, not tested: the structure fine-tune.** The two runs also differ in
model. The fine-tune halved the effective rank of `z` in its first epoch, and a
lower-rank `z` gives Mahalanobis a sharper covariance, so it may amplify the
distribution shift above. There is no AD run of the new stage-1 model to separate the
two effects. *(Tested in the follow-up: not the main cause, and its effect is mixed.)*

**Ruled out as causes:**
- **Worse upstream models.** Reconstruction is higher than before at both levels.
  (The two runs' `test` sets differ, so this is not a like-for-like comparison, but
  nothing points to a weaker encoder.)
- **The attack side.** The attack windows are identical in count and source.
- **The kNN inversion** (pooled AUROC ≈ 0.17–0.21). It was already present in the old
  stage-1 run (0.161 / 0.168), so it is a length artefact of the pooled metric and was
  not introduced by the fine-tune.

**Unexplained so far:** `late` comes after `val` and is just as short (median 33), yet
its Mahalanobis FPR is only 5.9%. Window length alone therefore does not explain `val`.
This fits `val`'s period containing a traffic regime that `ad_fit` and `ad_calib` do not,
which `late` does not share. *(Resolved by the follow-up: the low FPR on `late` is
specific to the fine-tuned model. Stage 1 overshoots on `late` as much as on `val`.)*

## Follow-up: stage-1 model on the six-split

Run 2026-09-26: `SeqAE_IIoTset_d128_Mamba_s512` (best checkpoint, epoch 43) exported
and evaluated with the same `DATA_ROLES`, detectors and settings. Outputs are in
`Outputs/Embeddings/SequenceEmbeddings_SeqAE_IIoTset_d128_Mamba_s512` and
`Outputs/AD/SeqAE_IIoTset_d128_Mamba_s512_eval-val`. Role sizes are identical to the
fine-tuned run.

| Mahalanobis (focus) | Three-split, stage 1 | Six-split, stage 1 | Six-split, fine-tuned |
|---|---|---|---|
| Pooled AUROC | 0.996 | 0.941 | 0.878 |
| TPR @ q0.99 | 1.000 | 0.885 | 0.866 |
| Realised FPR on `val` @ q0.99 (target 1%) | 5.7% | 24.3% | 36.4% |
| Realised FPR on `late` @ q0.99 | — | 21.9% | 5.9% |
| AUROC, bins 2–8 / 9–32 / 33–64 | 0.976 / 0.978 / 0.994 | **0.398** / 0.986 / 0.987 | 0.641 / 0.995 / 0.978 |
| kNN-1 / kNN-20 pooled AUROC | 0.161 / 0.168 | 0.248 / 0.312 | 0.209 / 0.167 |

What it shows:

- **The split is the main cause.** Without the fine-tune, the stage-1 model on the
  six-split still misses the FPR target by a factor of about 24 on `val` and 22 on
  `late`, against 5.7 in the three-split run. Both later periods are miscalibrated
  alike. The consistent explanation is that any period after the 5-hour
  `ad_fit`/`ad_calib` window looks different from it, rather than a regime specific to
  `val`.
- **Short windows fail in both models.** The 2–8 bin falls below chance for stage 1
  (0.398) and is weak for the fine-tune (0.641), against 0.976 in the three-split run.
  Short normal windows in the later periods score as anomalous as short attacks. The
  longer bins stay at 0.98–0.99 in every run.
- **The fine-tune is not the main cause, and its effect is mixed.** Compared with
  stage 1 on the same split, it has a lower pooled AUROC and a higher FPR on `val`, but
  a better 2–8 bin and a much lower FPR on `late`. It shifts which normal period looks
  off rather than making things uniformly worse.
- **The kNN inversion is present in every run**, including stage 1 on both splits, which
  confirms it is a length artefact of the pooled metric.

Other detectors show the same miscalibration on stage 1. For example, PCA reconstruction
has a pooled AUROC of 0.983 but a realised FPR of 17.7% on `val` at the 1% target.
Choosing a detector from these rows would select on `val`, so the focus stays on
Mahalanobis.

## Checks that would separate the explanations

Each one that reads `val` spends it again.

1. ~~**Split out the fine-tune**~~ **Done 2026-09-26**, see the follow-up above. The
   split is the main cause.
2. **Per-bin realised FPR on `val` and `late`:** available in this run's
   `metrics.parquet` (the `negatives` column). It shows whether the FPR gap lives in the
   short bins. Reads only existing outputs.
3. **Composition of the AD splits:** the endpoint-pair profile (`InspectFlowGroups`-style)
   for `ad_fit`, `ad_calib` and `late`, without labels. Checks whether `ad_fit` covers the
   host mix of the later periods. Profiling `val` itself would be another look at the
   held-out set.
4. **Wider AD windows:** place the boundaries after `test` so that `ad_fit`/`ad_calib`
   cover more time. This is a splitting decision, and the new `val` would be a fresh
   period, but the change is made in response to this run's `val`.
