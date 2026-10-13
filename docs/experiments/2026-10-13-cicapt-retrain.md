# 2026-10-13 · CICAPT-IIoT retrain on the strict split

**Outcome.** A timing profile shows that CICAPT Phase 1, unlike IIoTset, is one continuous
recording with a steady traffic rate. So the strict six-split it already had is the right
design, and it was kept. Every stage was retrained on it with the current code, up to the
embedding export.

- The packet AE and the sequence AE reproduce the first CICAPT run (2026-09-02) at the
  byte level.
- The structure fine-tune at λ 0.01 shaped the embedding strongly (silhouette on
  supported groups 0.03 → 0.73) without costing reconstruction.
- Its best-by-reconstruction checkpoint was the first epoch, so the last epoch (E85) was
  exported instead.

CICAPT has no attack captures, so no anomaly detection was run and `val` was not read
by any detector.

## What ran

Code: uncommitted on top of `7b0536f`. Split: the existing strict split from 2026-09-01
(`SplitFlowsDF`, `boundary_mode = "packets"`, no sessions), with fractions `train` .55 /
`test` .10 / `ad_fit` .10 / `ad_calib` .10 / `val` .10 / `late` .05. The first
generation's cache and runs are in `archive/2026-09-02-cicapt-sixsplit/`, together with a
copy of the split report. The split itself is shared with this run.

| Stage | Run / output | Result |
|---|---|---|
| Packet AE | `TrainingOutputs/PacketAE_CICAPT_d128` | best epoch 1: test CE 0.1566, non-pad 0.624 (epoch 2: CE 0.1654, non-pad 0.627) |
| Latent cache | `latents_PacketAE_CICAPT_d128_best/{train,test}` | 21,829 / 11,600 flows; all checks passed |
| Sequence AE | `TrainingOutputs/SeqAE_CICAPT_d128_Mamba_s512` (160 epochs) | non-pad 0.6061 of ceiling 0.6159; length accuracy 0.9985 |
| Group profile | `InspectFlowGroups` | see below |
| Structure fine-tune | `TrainingOutputs/SeqAEStruct_CICAPT_d128_Mamba_s512_pair_l001` (λ 0.01, 86 epochs) | see below |
| Export (E85) | `Outputs/Embeddings/SequenceEmbeddings_SeqAEStruct_CICAPT_d128_Mamba_s512_pair_l001_E85` | 6 sets, 107,357 windows |

The fine-tune's epoch count was scaled to the IIoTset step budget: 86 epochs × 86 steps ≈
7.4k steps, against 15 × 493 on IIoTset.

## Timing profile of the capture

Read from timestamps and flow keys only:

- **One continuous recording.** 12,103,661 packets over 95.9 hours, with no pause
  longer than a few seconds. A gap-based session split does not apply.
- **A steady rate.** About 122,000 packets/hour and about 1,850 distinct flows/hour
  throughout.
- **A 6-hour cycle.** Every 6 hours (hours 5, 11, 17, …, 89) a peak of about 133,000
  packets and about 2,100 flows (+9%), most likely a scheduled job in the simulated
  environment.
- **One short burst.** Hour 22 has 216,000 packets (+77%) at a normal flow count, so it
  is a burst of packets within existing flows.
- **Mild elevations** in hours 44–55 (127,000–140,000 packets/hour).

Because the rate is near constant, packet-count and wall-clock boundaries nearly coincide:
every AD split spans about 9.5 hours, and `late` about 4.9 hours.

## Results next to the first CICAPT run

| | Retrain (2026-10-13) | First run (2026-09-02) |
|---|---|---|
| Packet AE test CE / non-pad (best, epoch 1 in both runs) | 0.1566 / 0.624 | 0.1514 / 0.626 |
| Sequence AE byte accuracy, all / non-pad | 0.9744 / 0.6061 | 0.9745 / 0.6073 |
| Packet-AE ceiling on the same windows | 0.9750 / 0.6159 | 0.9751 / 0.6170 |
| Sequence AE length accuracy | 0.9985 | 0.9987 |
| Reconstruction loss / per-position baseline | 0.294 / 0.530 (55%) | 0.166 / 0.470 (35%) |

**At the byte level the retrain reproduces the first run** (98.4% of the packet-AE
ceiling in both). In latent space it is relatively worse: 55% against 35% of its
per-position baseline. The two runs reconstruct the latents of different packet AEs, so
the absolute losses are not comparable. Both packet AEs' best checkpoints are from epoch
1, so that is not the explanation. The new packet AE's test CE is slightly higher (0.1566
against 0.1514), which may leave latents that are harder to predict. Not checked.

CICAPT is much harder to reconstruct than IIoTset at the packet level (non-pad 0.62
against 0.71–0.82), and its windows are long: median flow length 44, 42% of `test` flows
reach 64 packets, and windows are 87% full on average, against about 26–29% on the
IIoTset session splits.

## Group profile (`InspectFlowGroups`)

CICAPT has real protocol variety. `train` holds ModbusTCP (14,101 flows), DNS (5,039),
plain UDP (1,649), HTTP (631), HTTPS (313), MQTT (8 flows), and some ARP, mDNS and ICMP.

`endpoint_pair` grouping in `train`: 134 groups, but heavily skewed.

| Endpoint pair | Flows | Windows | Share of windows | Supported (≥ 8 flows) |
|---|---|---|---|---|
| 172.16.64.128 ↔ .66.128 (ModbusTCP) | 14,102 | 52,916 | 52.9% | yes |
| 172.16.64.128 ↔ .64.2 (MQTT) | 3 | 35,749 | 35.7% | no |
| 172.16.65.1 ↔ .65.128 (DNS) | 4,920 | 4,939 | 4.9% | yes |

- **Coverage of the contrastive term.** Support is counted in flows, so the MQTT pair
  (three very long flows) is excluded, and the contrastive term acts on 60.7% of the
  windows.
- **Alternative keyings are no better.** `host_lo` puts 88.6% of windows in one group,
  and `host_hi` is close to `endpoint_pair`.
- **A second level is possible here.** `endpoint_pair × app_proto` splits each pair into
  about 2.25 sub-groups in `train` (1.36 in `test`), so the two-level objective the
  fine-tune's docstring anticipates would have something to nest on this capture.
- **Group coverage** from `train` to `test` is 99.9% of flows.
- **The decision:** keep the fine-tune as configured (`endpoint_pair`, support in flows),
  so it stays comparable with IIoTset.

## Structure fine-tune (λ 0.01, 86 epochs)

| Epoch | Val total loss | Byte accuracy, non-pad | Effective rank | Silhouette, all / supported |
|---|---|---|---|---|
| Stage 1 | 0.3008 | 0.6060 | 8.85 | 0.187 / 0.029 |
| 1 | **0.3014** (`_best`) | 0.6059 | 8.43 | 0.242 / 0.129 |
| 10 | 0.3042 | 0.6058 | 5.14 | 0.473 / 0.552 |
| 50 | 0.3077 | 0.6057 | 3.01 | 0.580 / 0.714 |
| 86 | 0.3037 | 0.6059 | 2.88 | 0.580 / 0.729 |

- **The structure term works on CICAPT without hurting reconstruction.** Silhouette on
  supported groups rises from 0.03 to 0.73, while non-pad byte accuracy stays at 0.606.
- **CICAPT's stage-1 embedding already starts at a low effective rank** (8.85, against
  29.7 on IIoTset), and the fine-tune lowers it to 2.88. This is reported as a diagnostic
  only; a low rank is not grounds to discard a model.
- **`_best` is the first epoch.** The fine-tune selects `_best` by validation
  reconstruction + length loss, which never went below its first-epoch value. Exporting
  `_best` would have given a model with one epoch of structure training (silhouette
  0.13). The export was pointed at the last epoch instead (`SequenceEmbeddingAD.py ·
  CKPT_SUFFIX = "E85"`, output directory suffixed `_E85`). On IIoTset this did not come
  up, because `_best` landed at epoch 10 of 15.

## Export

`SeqAEStruct_CICAPT_d128_Mamba_s512_pair_l001`, checkpoint E85. `train` and `test` come
from the cache; `ad_fit`, `ad_calib`, `val` and `late` are encoded from bytes. The
`attack` role resolves to nothing, because CICAPT has no attack captures.

| Set | Windows | Median length |
|---|---|---|
| `train` | 20,000 (subsample) | 64 |
| `test` | 19,213 | 64 |
| `ad_fit` | 19,064 | 64 |
| `ad_calib` | 19,062 | 64 |
| `val` | 19,183 | 64 |
| `late` | 10,835 | 55 |

The cross-path check passed (3.74e-04), and the run log has one `DATA ACCESS` line per
read.

## Open

- **No anomaly detection on CICAPT** until attack captures exist for it
  ([07 § Known gaps](../components/07-anomaly-detection.md#known-gaps)). The export is
  ready for any detector, or for cross-capture use.
- **Fine-tune checkpoint selection.** Selecting `_best` by reconstruction alone can pick
  a checkpoint before the structure objective has had any effect. A selection rule that
  takes the structure metrics into account would be a protocol change.
- **The relatively higher latent reconstruction loss** of the retrained sequence AE is
  unexplained (see above).
