# 2026-10-12 · IIoTset session split with a 90-minute gap (third run)

**Outcome.** With sessions cut at pauses longer than 90 minutes, and the structure
fine-tune at `LAMBDA_STRUCT = 0.01`, anomaly detection works for the first time on a
held-out split of this capture:

- Mahalanobis (the focus detector, fixed in advance) reaches a pooled AUROC of 0.998.
- At the threshold calibrated on `ad_calib` for a 1% false-positive rate, it detects
  99.85% of attack windows at a realised FPR of 0.46% on `val`.
- Every one of the 14 attack classes has an AUROC of 1.00 and a TPR of at least 0.97.
- The thresholds transfer from `ad_calib` to `val` for the first time since the
  three-split run.

Two changes separate this run from the previous one: the gap (90 minutes instead of a
day) and the model (the λ 0.01 fine-tune instead of the stage-1 model).
**Follow-up (2026-10-12): the stage-1 model on the same split scores the same**
(Mahalanobis pooled AUROC 0.9984, TPR 0.9995 at 0.45% FPR), so the improvement comes
from the split, not from the fine-tune
([§ Follow-up](#follow-up-stage-1-model-on-the-same-split)). Long windows still cannot be
checked against length-matched normals, because `val` has almost none (see
[§ Caveats](#caveats)).

`val` of this split has been read **twice**: once for the λ 0.01 fine-tuned model and
once for the stage-1 model. The 90-minute gap
was chosen from a timing profile of the capture and from the `ad_fit`/`ad_calib` score
breakdown of the previous run
([2026-10-07 entry, follow-ups](2026-10-07-iiotset-session-split.md)), so the split design
is informed by earlier `val` results.

## What ran

Code: uncommitted on top of `c31d0a4`. Split: `SplitFlowsDF` with `session_gap_s =
5400`, `min_session_share = 0.01`, `boundary_mode = "active_time"`, `pause_cap_s = 60`,
and fractions `train` .40 / `test` .10 / `ad_fit` .20 / `ad_calib` .20 / `val` .10.
Earlier generations: `archive/2026-09-25-sixsplit/` and `archive/2026-10-07-sessionsplit/`.

| Stage | Run / output | Result |
|---|---|---|
| Split | `flow_split/` | 9 sessions, 8 kept; the day-28 run (7,020 packets) dropped |
| Packet AE | `TrainingOutputs/PacketAE_IIoTset_d128` | test CE 0.0779, non-pad 0.713 |
| Latent cache | `latents_PacketAE_IIoTset_d128_best/{train,test}` | 126,122 / 45,913 flows; all checks passed |
| Sequence AE | `TrainingOutputs/SeqAE_IIoTset_d128_Mamba_s512` (28 epochs) | val total 0.039, length accuracy 0.997, non-pad 0.658 of ceiling 0.660 |
| Structure fine-tune | `TrainingOutputs/SeqAEStruct_IIoTset_d128_Mamba_s512_pair_l001` (λ 0.01, 15 epochs) | best epoch 10; see below |
| Export | `Outputs/Embeddings/SequenceEmbeddings_SeqAEStruct_IIoTset_d128_Mamba_s512_pair_l001` | 19 sets, 216,741 windows |
| AD | `Outputs/AD/SeqAEStruct_IIoTset_d128_Mamba_s512_pair_l001_eval-val` | see below |

The packet AE's and sequence AE's `test` numbers are not comparable with earlier runs:
`test` is a different and larger set (0.77M packets, high-rate traffic from five
sessions), and its windows are short (median flow length 16).

## The split

| Session | Start (h) | Span | Packets | Share | Traffic |
|---|---|---|---|---|---|
| 0 | 0 | 10.8 h | 884k | 9.1% | low rate (segments 0–1) |
| 1 | 27.8 | 7.3 h | 395k | 4.0% | low rate (segment 2) |
| 2 | 52.2 | 7.2 h | 346k | 3.6% | low rate (segments 3–4) |
| 3 | 685.6 | 18 min | 7k | 0.07% | dropped |
| 4 | 876.2 | 3.2 h | 2.07M | 21.3% | high-rate segment 6, then medium-rate 7–8 |
| 5 | 881.1 | 42 min | 1.81M | 18.6% | high-rate burst (segment 9) |
| 6 | 1067.2 | 12 min | 495k | 5.1% | high rate (segment 10) |
| 7 | 1069.1 | 30 min | 1.31M | 13.4% | high rate (segment 11) |
| 8 | 1074.0 | 5.3 h | 2.43M | 24.9% | high-rate segment 12, then medium-rate 13–14 |

| Split | Packets | Share of packets | From the high-rate-only sessions 5 / 6 / 7 | Median window length |
|---|---|---|---|---|
| `train` | 4,626,981 | 47.5% | 716k / 234k / 522k | 30 |
| `test` | 767,379 | 7.9% | 182k / 6k / 132k | 16 |
| `ad_fit` | 1,673,208 | 17.2% | 358k / 91k / 261k | 16 |
| `ad_calib` | 1,851,806 | 19.0% | 373k / 108k / 262k | 16 |
| `val` | 816,889 | 8.4% | 179k / 56k / 131k | 16 |

The three AD splits now hold matching shares of the high-rate mode (40–45% of their
packets), and matching window lengths.

## Structure fine-tune at λ 0.01

| | Stage 1 | After epoch 1 | End (epoch 15) |
|---|---|---|---|
| Effective rank of `z` | 29.7 | 9.5 | 5.7 (flat from epoch 5) |
| Silhouette, all / supported groups | 0.023 / 0.053 | 0.194 / 0.348 | 0.456 / 0.707 |
| Byte accuracy, non-pad | 0.6576 | 0.6576 | 0.6577 |

The rank still falls steeply, but it settles at 5.7 rather than 1.9 (λ 0.1 in the previous
run), and reconstruction is untouched. The model was exported and evaluated regardless of
its rank: a low effective rank alone was not taken as grounds to discard it. 78.4% of
`test` flows fall in an endpoint-pair group with at least 8 `train` flows.

## Results

Mahalanobis is the focus detector. The previous runs' columns are stage-1 models.

| | First six-split | Session split, 1-day gap | **90-min gap, λ 0.01 fine-tune** |
|---|---|---|---|
| Pooled AUROC | 0.941 | 0.773 | **0.998** |
| AUROC, bins 2–8 / 9–32 / 33–64 | 0.398 / 0.986 / 0.987 | 0.950 / 0.9997 / 0.863 | **0.985 / 0.9999 / n/a** |
| Realised FPR on `val` @ q0.99 (target 1%) | 24.3% | 0.06% | **0.46%** |
| TPR @ q0.99 | 0.885 | 0.0009 | **0.9985** |
| Partial AUROC, FPR ≤ 1% | 0.592 | 0.532 | **0.922** |
| `seq_len_only` pooled AUROC | 0.831 | 0.814 | 0.805 |

Other detectors:
- **PCA reconstruction:** pooled 0.999, TPR 0.90.
- **GMM:** pooled 0.995, TPR 0.86.
- **Isolation forest:** pooled 0.987, but TPR only 0.08.
- **kNN-1 / kNN-20:** inverted when pooled (0.45 / 0.45), though strong within bins
  (0.91–0.99).
- **LOF:** pooled 0.65.

Per attack class, Mahalanobis reaches AUROC 1.00 on all 14 classes, and TPR at least
0.97 (the lowest are MITM and OS fingerprinting at 0.97, and Ransomware at 0.98).

## What it shows

**Observations:**
- **The thresholds transfer again.** `ad_calib` and `val` now have the same session mix,
  so the score quantiles of `ad_calib` describe `val`. The realised FPR is 0.46% at a 1%
  target, slightly conservative. This supports the previous entry's finding that the
  calibration failure came from the high-rate burst being missing from `ad_fit`.
- **The short-window bin stays fixed** (0.985), and ranking is near perfect in the 9–32 bin.
- **The kNN inversion persists** across all runs, and this run shows where it comes from:
  the single-packet floods. kNN scores the ICMP, UDP and SYN floods at AUROC 0.01–0.23,
  against 0.91–0.99 in the 2–8 and 9–32 bins. Those windows are almost all one packet
  long, and the length-1 bin has no normals to compare them with. So the inversion is how
  a nearest-neighbour score treats one-packet windows, which lie close to each other, and
  it does not depend on the split.

## Follow-up: stage-1 model on the same split

Run 2026-10-12: `SeqAE_IIoTset_d128_Mamba_s512` (best checkpoint, epoch 28) exported and
evaluated with the same `DATA_ROLES`, detectors and settings. Outputs are in
`Outputs/Embeddings/SequenceEmbeddings_SeqAE_IIoTset_d128_Mamba_s512` and
`Outputs/AD/SeqAE_IIoTset_d128_Mamba_s512_eval-val`.

| 90-minute split | Stage 1 | Fine-tune λ 0.01 |
|---|---|---|
| Mahalanobis pooled AUROC | 0.9984 | 0.9984 |
| Mahalanobis AUROC, bins 2–8 / 9–32 | 0.982 / 0.9999 | 0.985 / 0.9999 |
| Mahalanobis TPR / realised FPR @ q0.99 | 0.9995 / 0.45% | 0.9985 / 0.46% |
| Mahalanobis partial AUROC, FPR ≤ 1% | 0.920 | 0.922 |
| PCA reconstruction pooled AUROC / TPR | 0.9992 / 0.993 | 0.9986 / 0.896 |
| GMM pooled AUROC / TPR | 0.998 / 0.895 | 0.995 / 0.863 |
| kNN-1 / kNN-20 pooled AUROC | 0.470 / 0.611 | 0.451 / 0.448 |
| kNN-1 / kNN-20, 2–8 bin | 0.633 / 0.804 | 0.908 / 0.974 |
| LOF, 2–8 bin | 0.477 | 0.579 |

**What it shows:**
- **The split is the cause of the improvement.** For the focus detector, stage 1 and the
  fine-tune are indistinguishable. Stage 1's operating point is even slightly better
  (TPR 0.9995 against 0.9985 at the same FPR).
- **The fine-tune helps the local-density detectors on short windows, and nothing else.**
  kNN and LOF in the 2–8 bin improve markedly. PCA reconstruction and GMM lose a little.
  Its lower effective rank (5.7 against 29.7) did not hurt Mahalanobis.
- The kNN pooled inversion is present for both models, as in every earlier run.

## Caveats

- ~~**The model and the split changed together.**~~ Resolved by the follow-up above: the
  stage-1 model on the same split gives the same Mahalanobis result.
- **Long windows are not checked against length-matched normals.** The 33–64 bin is n/a:
  `val` has fewer than `MIN_BIN_N` normal windows of 33 packets or more. The 90-minute
  gap cuts flows at many more session edges, so normal windows are short (median 16 in
  every AD split). Long-window attacks (Password attacks and the vulnerability scanner,
  median 64) are therefore compared only with shorter normals. `seq_len_only` gives
  them AUROC 0.01 and 0.24 (length alone says they look normal), while Mahalanobis gives
  1.00. Whether Mahalanobis detects their content, or simply "longer than normal windows
  ever are", cannot be told from this run.
- **Short normal windows are partly a product of the split.** Cutting every session into
  five roles, at active-time quantiles, truncates flows at many boundaries. The normal
  traffic the detectors see is shorter than the traffic as recorded.

## Checks that would separate the explanations

Each one that reads `val` spends it again.

1. ~~**Stage-1 model on this split**~~ **Done 2026-10-12**, see the follow-up: the split
   is the cause.
2. **Long-window attacks against long `ad_calib` windows:** `ad_calib` holds more long
   windows than `val`. Comparing Mahalanobis scores of long attack windows with long
   `ad_calib` windows tests the length confound without touching `val`.
3. **Fewer sessions or longer active-time pieces:** a split that cuts flows less often
   would give long normal windows, so the 33–64 bin could be evaluated. This is a
   splitting decision.
