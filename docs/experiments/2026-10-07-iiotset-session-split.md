# 2026-10-07 · IIoTset session split (second run)

**Outcome.** The second full IIoTset run used a session split
([02 § Sessions](../components/02-temporal-split.md#sessions)), so every split draws from
every recording session instead of a strict split that handed whole sessions to roles.
For the stage-1 model this **fixed the two failures of the first run**:

- short windows are no longer scored as anomalous (Mahalanobis AUROC in the 2–8 bin went
  from 0.398 to 0.950);
- the false-positive overshoot is gone (from 24.3% to 0.06% realised at a 1% target).

It **introduced two new problems**:

- the operating point now overshoots the other way: the thresholds are so high that almost
  no attack passes them (TPR 0.09%);
- long windows got worse (the 33–64 bin went from 0.987 to 0.863), and so did the pooled
  AUROC (0.941 → 0.773).

The structure fine-tune collapsed the embedding to an effective rank of about 2. It was
not evaluated.

This tests the hypothesis from the
[first entry](2026-09-25-iiotset-six-split-first-run.md): that the split's effect on
window length and session coverage drove the first run's failures. The short-window part
is confirmed. The calibration problem changed direction but did not go away.

`val` of this new split was read **once**, for the stage-1 model. The split design itself
(sessions, `active_time`, dropping `late`) was chosen after the first run's `val` results
and is therefore informed by them.

## What ran

Code: committed together with this entry. Split: `SplitFlowsDF` with `session_gap_s =
86400`, `min_session_share = 0.01`, `boundary_mode = "active_time"`, `pause_cap_s = 60`,
and fractions `train` .40 / `test` .10 / `ad_fit` .20 / `ad_calib` .20 / `val` .10
(no `late`). The first run's artefacts are under `archive/2026-09-25-sixsplit/`. Since
2026-10-11 this run's artefacts (split, cache, runs, embeddings, AD outputs) are under
`archive/2026-10-07-sessionsplit/`, at the same relative paths.

| Stage | Run / output | Result |
|---|---|---|
| Split | `flow_split/` (with a `session` column) | 3 of 4 sessions kept; day 28 (7,020 packets) dropped |
| Packet AE | `TrainingOutputs/PacketAE_IIoTset_d128` | test CE 0.0453, non-pad 0.817 |
| Latent cache | `latents_PacketAE_IIoTset_d128_best/{train,test}` | 96,542 / 19,193 flows; all checks passed |
| Sequence AE | `TrainingOutputs/SeqAE_IIoTset_d128_Mamba_s512` (37 epochs) | val total 0.177, length accuracy 0.996, non-pad 0.793 of ceiling 0.820 |
| Structure fine-tune | `TrainingOutputs/SeqAEStruct_IIoTset_d128_Mamba_s512_pair_l010` | **collapsed**, not exported (see below) |
| Export (stage 1) | `Outputs/Embeddings/SequenceEmbeddings_SeqAE_IIoTset_d128_Mamba_s512` | 19 sets, 216,741 windows |
| AD (stage 1) | `Outputs/AD/SeqAE_IIoTset_d128_Mamba_s512_eval-val` | see below |

## How the split came about

Two dry runs ruled out simpler fixes first:

- **Packet fractions** .40/.10/.20/.20/.10 still put `test` into a 23-minute burst and `val`
  into 1.9 hours.
- **Wall-clock fractions** over the whole capture failed: the `test` window fell into a gap
  with no traffic at all.

A per-day and then per-hour profile of `NormalMerged.parquet` showed why. The capture is
four separate recording sessions with very different traffic:

| Session | Days | Packets | Shape |
|---|---|---|---|
| 0 | 0–2 | 1.62M | steady ~58k/h, three chunks separated by overnight pauses |
| 1 | 28 | 7k | dropped (below 1%) |
| 2 | 36 | 3.88M | bursty, up to 1.8M/h, ~5 h |
| 3 | 44 | 4.23M | bursty, with pauses of 1–3 h |

Time quantiles *within* a session also failed: session 3's `test` window fell into one of
its internal pauses. The active clock (pauses capped at 60 s) fixed this.

The resulting splits draw from all three kept sessions, but with different mixes:

| Split | Packets | Share of packets | Session 0 / 2 / 3 packets |
|---|---|---|---|
| `train` | 5,345,055 | 54.9% | 817k / 1,578k / 2,950k |
| `test` | 377,280 | 3.9% | 136k / 182k / 60k |
| `ad_fit` | 947,484 | 9.7% | 260k / 251k / 437k |
| `ad_calib` | 1,891,479 | 19.4% | 277k / 1,108k / 507k |
| `val` | 1,174,965 | 12.1% | 136k / 763k / 276k |

Session coverage, measured by the structure fine-tune: 99.7% of `test` flows fall in an
endpoint-pair group seen in `train`, against 50.5% in the first run.

## Results next to the first run

All columns are the stage-1 model; Mahalanobis is the focus detector, fixed in advance.

| | Three-split (old) | First six-split | Session split |
|---|---|---|---|
| Median window length `ad_fit` / `ad_calib` / `val` | 64 / 64 / 64 (`train`/`test`/`val`) | 63 / 62 / 36 | 18 / 34 / 18 |
| Mahalanobis pooled AUROC | 0.996 | 0.941 | 0.773 |
| AUROC, bins 2–8 / 9–32 / 33–64 | 0.976 / 0.978 / 0.994 | 0.398 / 0.986 / 0.987 | **0.950 / 0.9997 / 0.863** |
| Realised FPR on `val` @ q0.99 (target 1%) | 5.7% | 24.3% | **0.06%** |
| TPR @ q0.99 | 1.000 | 0.885 | **0.0009** |
| `seq_len_only` pooled AUROC | 0.900 | 0.831 | 0.814 |
| kNN-1 / kNN-20 pooled AUROC | 0.161 / 0.168 | 0.248 / 0.312 | 0.173 / 0.150 |

Other detectors show the same reversal of the operating point. The realised FPR on `val`
at the 1% target lies between 0.005% (LOF) and 0.5% (isolation forest) for every
detector. Among the learned detectors, TPR ranges from 0.006% (kNN-1) to 35.9% (PCA
reconstruction). Only the length baseline keeps a usable operating point (TPR 73%, FPR
0.4%).

## What it shows

**Observations from the logs:**

- **Matching the window lengths fixed the short bin.** `ad_fit` and `val` now have the same
  median window length (18). The 2–8 bin recovered for every density-based detector:
  Mahalanobis 0.950, kNN 0.92–0.94, GMM 0.90.
- **The thresholds no longer transfer from `ad_calib` to `val`, in the opposite direction.**
  `ad_calib` scores have a much heavier upper tail than `val`. The realised FPR is
  roughly 1/16 of the target, and the threshold sits above almost all attacks.
- **The long bin got worse.** Long normal windows are now rare: `ad_fit`'s median is 18,
  and only `ad_calib` has many long windows. Mahalanobis drops to 0.863 in the 33–64 bin,
  and kNN collapses there (0.012 / 0.005), which is where its pooled inversion comes from.

**First hypothesis for the new calibration problem (refined by the 2026-10-08 follow-up
below, which places the cause in a traffic burst inside session 2):** the session
mix differs between the AD splits. `ad_calib` is 59% session 2 and has longer windows
(median 34); `ad_fit` is 27% session 2 (median 18); `val` is 65% session 2 but has short
windows (median 18). Taking equal shares of *active time* from bursty sessions gives each
role a different traffic mix. `ad_calib` then contains traffic that is far from `ad_fit`
yet absent from `val`, which inflates its score quantiles.

## Follow-up (2026-10-08): where the `ad_calib` tail comes from

Check 1 below, run without reading `val`. It joins the Mahalanobis scores of the `ad_fit`
and `ad_calib` rows in this run's `scores.parquet` to the `session` column of the split
parquets through `flow_key`. The `val` rows are filtered out when the file is scanned.
Only 17 flow keys occur in more than one session, and these were assigned to the
earliest one.

| Split · session | Windows | Median length | Median score | 99th percentile | Share > 10,000 |
|---|---|---|---|---|---|
| `ad_fit` · 0 | 6,335 | 16 | 184 | 1,571 | 0% |
| `ad_fit` · 2 | 6,576 | 18 | 199 | 1,361 | 0.02% |
| `ad_fit` · 3 | 7,089 | 31 | 410 | 2,373 | 0% |
| `ad_calib` · 0 | 5,119 | 16 | 196 | 8,915 | 0.5% |
| `ad_calib` · 2 | 8,233 | 63 | 13,491 | 19,961 | **74.5%** |
| `ad_calib` · 3 | 6,648 | 34 | 488 | 1,884 | 0.02% |

The `ad_calib` threshold at the 99th percentile is 18,993.

**Finding: the whole calibration tail comes from session 2's piece of `ad_calib`.**
Sessions 0 and 3 calibrate normally: their `ad_calib` pieces score like their `ad_fit`
pieces. Session 2's `ad_calib` piece is the day-36 traffic burst (about 1.1M packets in
~35 minutes of active time). Its windows are long (median 63, against 18 in session 2's
`ad_fit` piece) and three quarters of them score above 10,000, in the attacks' range. On
the active clock, `ad_fit`'s piece of session 2 falls in the quieter stretch before the
burst, and `ad_calib`'s piece falls in the burst. Nothing like the burst traffic is in
`ad_fit`, so the threshold is set by traffic the detector never saw. That is why almost
no attack passes it.

This replaces the earlier hypothesis that the *session mix* differs between the AD splits.
The cause is a change in traffic over time **within** session 2. It also connects to
window length: the burst's dense traffic is what produces the long windows.

**Likely but not checked:** the normal clusters that score in the attack range in this
run's projection figure (the upper-middle group, `val` and `ad_calib` points with almost
no `ad_fit` among them) are session-2 burst windows. Confirming it would mean breaking
down `val` by session, which is another look at the held-out set, so it was not done.

Options under consideration (none chosen yet):
- packet quantiles within each session, so every role gets a proportional slice of the
  burst;
- treating the burst as its own session (a shorter gap, or an explicit rule on traffic
  rate);
- keeping the split and documenting the burst as a regime `ad_fit` does not cover.

## Follow-up (2026-10-11): how the capture was recorded

A timing-only profile of `NormalMerged.parquet` (timestamps, no payload, no labels)
split the traffic wherever two packets are more than 5 minutes apart:

| Segment | Hours since start | Duration | Packets | Packets/min | Pause before | Session (1-day gap) |
|---|---|---|---|---|---|---|
| 0 | 0.0 | 503 min | 754,658 | 1,500 | — | 0 |
| 1 | 8.6 | 133 min | 129,287 | 970 | 10.6 min | 0 |
| 2 | 27.8 | 436 min | 394,593 | 906 | 17.1 h | 0 |
| 3 | 52.2 | 192 min | 190,185 | 993 | 17.1 h | 0 |
| 4 | 56.7 | 163 min | 156,137 | 958 | 80.3 min | 0 |
| 5 | 685.6 | 18 min | 7,020 | 387 | 26.1 days | 1 (dropped) |
| 6 | 876.2 | 27 min | 1,143,909 | **42,576** | 7.9 days | 2 |
| 7 | 876.8 | 75 min | 759,557 | 10,202 | 9.4 min | 2 |
| 8 | 878.9 | 31 min | 170,393 | 5,564 | 50.6 min | 2 |
| 9 | 881.1 | 42 min | 1,807,820 | **42,920** | 98.0 min | 2 |
| 10 | 1067.2 | 12 min | 494,599 | **42,727** | 7.7 days | 3 |
| 11 | 1069.1 | 31 min | 1,308,193 | **42,955** | 98.9 min | 3 |
| 12 | 1074.0 | 24 min | 1,070,474 | **44,693** | 263.5 min | 3 |
| 13 | 1074.5 | 72 min | 155,887 | 2,166 | 8.1 min | 3 |
| 14 | 1077.0 | 136 min | 1,200,571 | 8,804 | 80.1 min | 3 |

**What this says about the dataset.** The normal capture was not recorded as one
experiment. It was produced in a series of separate runs, sometimes hours apart within a
day and sometimes days or weeks apart. The settings most likely changed by hand between
runs: the packet rate jumps between runs and is roughly constant within one.

- Some runs are alike, at least in packet rate. Five segments (6, 9, 10, 11, 12) run at
  ~43,000 packets/min in two sessions a week apart, which points to one high-rate setting
  being used repeatedly.
- Other runs are entirely different: ~1,000 packets/min on days 0–2, and 2,000–10,000
  packets/min in segments 7, 8, 13 and 14.
- A 1-day gap groups runs with very different settings into one session.

This was inferred from timing alone. No dataset README or label file was read.

**Consequence for the session split.** In the 2026-10-07 run, `train` got the high-rate
segments 6, 10, 11 and 12, and `ad_fit` got none. `ad_calib` got the first ~24 minutes of
segment 9 and `val` the rest of it, which explains the calibration failure in the previous
follow-up. With `session_gap_s` between 80.3 and 98.0 minutes, segment 9 becomes its own
session and every role gets a piece of it. The margin is thin (the 98.0-minute pause
before segment 9 against 80.3 and 80.1 minutes inside other sessions), and high-rate
segments 6 and 12 stay grouped with lower-rate runs. The next run uses a 90-minute gap.
A rule based on packet rate would not depend on this margin; it was noted as an
alternative and not pursued.

## Structure fine-tune: collapse

At `LAMBDA_STRUCT = 0.1`, the effective rank of `z` fell from 26.9 to 3.0 within the first
epoch and ended at 1.9. Silhouette on supported groups rose from 0.15 to 0.88, and byte
reconstruction stayed unchanged (non-pad 0.793). This is the collapse the fine-tune's
docstring warns about. The first run's fine-tune only reached rank 13. With the session
split, `train` holds every session's endpoint pairs, so the contrastive term has many
well-populated groups to separate, and it does so in very few directions. The model was
not exported or evaluated; the stage-1 model was used instead.

## Checks that would separate the explanations

None has been run. Each one that reads `val` spends it again.

1. ~~**Per-session score distributions**~~ **Done 2026-10-08 for `ad_fit` and
   `ad_calib`, without reading `val`.** The tail comes from session 2's burst; see the
   follow-up above.
2. **Calibrate per session, or weight `ad_calib` to `val`'s session mix:** a change to the
   detector protocol, so it needs a decision first.
3. **Packet quantiles within each session** instead of active time. This gives every role
   the same session mix by packet count, at the cost of very short pieces of the bursty
   sessions. It is a splitting decision and would need a new full run.
4. **Fine-tune at a much lower `LAMBDA_STRUCT`** (e.g. 0.01), watching the effective rank
   per epoch.
