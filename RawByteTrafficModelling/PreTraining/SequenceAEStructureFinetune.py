"""
Stage-2 structure fine-tune of the sequence-level autoencoder.

Stage 1 (SequenceLevelAutoEncoder.py) trains the flow bottleneck vector `z` with one
objective: reconstruct all P packet latents from it. Nothing in that objective says
where a flow should sit relative to other flows, so the space is shaped only by what
reconstruction happens to need -- and what it mostly needs is flow length (the length
head scores ~0.978, which is why EmbeddingADSuite pins a `seq_len_only` baseline
detector). This stage adds the missing constraint: a supervised contrastive term that
pulls flows sharing an endpoint pair together and pushes different pairs apart.

Why endpoint identity is worth supervising: PreTrainingDatasetHandler.apply_mask
replaces every endpoint-identifying byte with <EndPointMasking> -- MACs, IPv4, IPv6 and
ARP addresses alike -- so the packet encoder has never seen an address. Endpoint
supervision therefore injects information the model provably cannot read off the bytes,
and satisfying it requires learning a behavioural device fingerprint rather than
memorising an IP.

Why there is only one level, when the intent was a protocol hierarchy inside each
endpoint cluster: this capture has no protocol variation to nest. InspectFlowGroups.py
reports 84,882 of 88,505 train flows as MQTT over TCP (95.9%; test is 99.7%), and four
of the five largest endpoint pairs carry exactly one port and one protocol. A
"protocol within endpoint" level cuts its parent into ~1 piece across the bulk of the
data, so it would be the coarse clustering again at a second temperature. The two-level
form is a two-line change once a multi-protocol capture exists -- see StructureLosses.

Two consequences of the group profile are wired into this script rather than left to be
discovered in the numbers:

  - 18 groups carry 98.2% of the training windows and the largest is 27.1% of them, so
    supcon averages per group before averaging across groups (balance_groups). A plain
    anchor mean would hand one endpoint pair a quarter of the loss.
  - The pair holding 65% of *test* windows is present in train as four flows, three of
    them ModbusTCP against 13,351 MQTT flows in test -- SplitFlowsDF assigns whole
    conversations chronologically and that workload starts late in the capture. Pooled
    structure metrics on test would therefore be dominated by a group the contrastive
    term never shaped, so every structure metric is also reported over the supported
    groups alone (`*_sup`).

Reconstruction and the length head stay in the loss at full weight. They are the
anti-collapse anchor: a contrastive term over ~18 effective classes will happily flatten
`z` onto a low-rank simplex, which reads as beautiful separation while destroying the
local geometry the downstream kNN / Mahalanobis / LOF detectors consume.
`struct_eff_rank` in the metrics CSV is the alarm for exactly that.

Before running this, profile the groups:
    python -m RawByteTrafficModelling.PreTraining.InspectFlowGroups

It needs the same packet-latent cache stage 1 uses, and the trained stage-1 checkpoint.
Run from the repo root:
    python -m RawByteTrafficModelling.PreTraining.SequenceAEStructureFinetune
"""
from RawByteTrafficModelling.ModelComponents.ModelDefinitions import (
    SequenceAutoencoder,
    PacketAutoencoder,
    load_AE_Checkpoint,
    load_SeqAE_checkpoint,
    save_checkpoint,
    baseline_mses,
    byte_level_reconstruction,
)
from RawByteTrafficModelling.ModelComponents.DataUtils import (
    CachedLatentSequenceHandler,
    flow_group_ids,
    group_support,
    load_latent_cache,
)
from RawByteTrafficModelling.ModelComponents.StructureLosses import (
    effective_rank,
    group_distance_stats,
    knn_group_agreement,
    supcon,
)
from RawByteTrafficModelling.PreTraining.SeqByteEval import build_byte_eval_set
from RawByteTrafficModelling.PreTraining.RunConfig import (
    DATASETS,
    TRAINING_OUTPUT_ROOT,
    MetricsCsv,
    cosine_warmup_lambda,
    plot_curves,
    resolve_device,
    setup_run,
)
from sklearn.metrics import silhouette_score
import numpy as np
import torch
import torch.nn.functional as F
import math

### The stage-1 run this fine-tunes
STAGE1_RUN = "SeqAE_IIoTset_d128_Mamba_s512"
STAGE1_CKPT = (f"{TRAINING_OUTPUT_ROOT}/{STAGE1_RUN}/"
               f"SequenceLevelAutoEncoder_{STAGE1_RUN}_best.ckpt")

# The run name carries the keying, so a sweep over COARSE_KEY / LAMBDA_STRUCT lands in
# separate directories. Checkpoints keep stage 1's `SequenceLevelAutoEncoder_` prefix
# on purpose: SequenceEmbeddingAD.py builds its path from that prefix, so exporting
# this model's embeddings needs only its RUN_NAME constant changed.
RUN_NAME = "SeqAEStruct_IIoTset_d128_Mamba_s512_pair_l010"
DEVICE_INDEX = 1               # leaves cuda:0 free for a concurrent lambda sweep

### Data -- must match stage 1 and CachePacketLatents
DATASET = DATASETS["IIoTset-Ferrag"]
CACHE_TAG = "PacketAE_d128_best"
PACKET_AE_CKPT = ("RawByteTrafficModelling/PreTraining/TrainingOutputs/PacketAE_IIoTset_d128/"
                  "PacketLevelAutoEncoder_PacketAE_IIoTset_d128_best.ckpt")
TRAIN_LATENT_CACHE = DATASET.latent_cache(CACHE_TAG, "train")
TEST_LATENT_CACHE = DATASET.latent_cache(CACHE_TAG, "test")
VAL_SPLIT_FILE = DATASET.test  # byte eval needs the raw packets the test cache was built from

PACKETS_PER_SEQUENCE = 65      # P, including the slot the seq-CLS overwrites

### Structure objective
# The unordered IP pair from the direction-normalised flow_key: 112 groups in train, of
# which 18 hold 98.2% of the windows. "host_lo"/"host_hi" are the alternatives (52/53
# groups) -- coarser still, so the pair is the best of the three here.
COARSE_KEY = "endpoint_pair"
TAU = 0.2                      # higher temperature -> looser cluster
LAMBDA_STRUCT = 0.1            # the sweep knob. 0.3 collapsed effective rank 34.7 -> 7.5
                               # and pooled silhouette peaked at epoch 0 then decayed,
                               # so the sweep runs downward from there, not up.
BALANCE_GROUPS = True          # equal weight per group, not per anchor
MIN_TRAIN_FLOWS = 8            # a group needs this many train flows to count as supported

### Optimisation -- a fine-tune, not a fresh run
Epochs = 15
batch_size = 256
val_batch_size = 256
learning_rate = 5e-5
weight_decay = 1e-2
WARMUP_STEPS = 200
GRAD_CLIP = 1.0
AMP_DTYPE = torch.bfloat16     # None disables autocast
# 16 runs per batch. The first smoke run used batch_size // 4 and produced only 3-4
# distinct groups per batch with 98 positives per anchor -- a contrastive loss starved
# of negatives while drowning in positives. With 18 groups carrying the mass, trading
# surplus positives for negatives is the right side of that trade.
MAX_GROUP_RUN = batch_size // 16

log_every_n_batches = 50
BYTE_EVAL_WINDOWS = 256        # fixed subset of val windows scored byte-by-byte each epoch
BYTE_EVAL_CHUNK = 64           # packets per frozen-decoder forward
STRUCT_EVAL_WINDOWS = 4000     # fixed subset the structure metrics are computed on
KNN_K = 10
# Restated from EmbeddingADSuite.SEQ_LEN_BINS -- that file is a top-to-bottom script,
# so importing it would run the whole AD suite. Keep the two in step.
SEQ_LEN_BINS = ((1, 1, "1"), (2, 8, "2_8"), (9, 32, "9_32"), (33, 64, "33_64"))
MAX_STEPS_PER_EPOCH = None     # set to a small int for a wiring smoke test
RESUME_FROM = None
SEED = 42

# Wiring test: same code path in a couple of minutes, into its own output_dir so a
# smoke checkpoint can never be mistaken for a trained one.
SMOKE = False
if SMOKE:
    RUN_NAME = f"{RUN_NAME}_smoke"
    Epochs = 2
    MAX_STEPS_PER_EPOCH = 20
    log_every_n_batches = 10
    WARMUP_STEPS = 5
    BYTE_EVAL_WINDOWS = 64
    STRUCT_EVAL_WINDOWS = 512

output_dir, logger = setup_run(RUN_NAME, "SequenceAEStructureFinetune.log")
logger.info(f"Run {RUN_NAME} (SMOKE={SMOKE}) -> {output_dir}")
logger.info(f"Fine-tuning {STAGE1_CKPT}")
logger.info(f"Structure: key={COARSE_KEY} tau={TAU} lambda={LAMBDA_STRUCT} "
            f"balance_groups={BALANCE_GROUPS}")

torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)
device = resolve_device(DEVICE_INDEX)

### Load the cached packet latents
# Same sha256 pin stage 1 uses: retraining the packet model, even back to the same
# filename, fails loudly here instead of fine-tuning on stale latents.
train_latents, train_flow_offsets, train_meta = load_latent_cache(TRAIN_LATENT_CACHE, PACKET_AE_CKPT)
val_latents, val_flow_offsets, val_meta = load_latent_cache(TEST_LATENT_CACHE, PACKET_AE_CKPT)
logger.info(f"Train cache: {train_meta['num_rows']} packets in {train_meta['num_flows']} flows")
logger.info(f"Val cache:   {val_meta['num_rows']} packets in {val_meta['num_flows']} flows")

TrainHandler = CachedLatentSequenceHandler(train_latents, train_flow_offsets, PACKETS_PER_SEQUENCE)
ValHandler = CachedLatentSequenceHandler(val_latents, val_flow_offsets, PACKETS_PER_SEQUENCE)

### Group labels, straight off the flow keys
# The val split is encoded against the train vocabulary, so an id means the same group
# in both -- which is what group_support then needs to ask how much training signal a
# val group actually received.
train_groups, _f, group_vocab, _fv = flow_group_ids(TrainHandler.flow_keys, coarse=COARSE_KEY)
val_groups, _f, group_vocab, _fv = flow_group_ids(ValHandler.flow_keys, coarse=COARSE_KEY,
                                                  coarse_vocab=group_vocab)
val_supported = group_support(train_groups, val_groups, MIN_TRAIN_FLOWS)
logger.info(f"Groups: {len(group_vocab)} across both splits, "
            f"{len(set(train_groups.tolist()))} in train; "
            f"{val_supported.mean():.1%} of val flows sit in a group with "
            f">={MIN_TRAIN_FLOWS} train flows")

val_windows = ValHandler.enumerate_windows(with_flow=True)
val_batches = [val_windows[i:i + val_batch_size, 1:]
               for i in range(0, val_windows.shape[0], val_batch_size)]
logger.info(f"Validation: {val_windows.shape[0]} deterministic windows -> {len(val_batches)} batches")

# --- Model ------------------------------------------------------------------
# The packet encoder's weights ride inside the stage-1 state dict, so the packet AE
# checkpoint is needed only for its decoder (the byte-eval ceiling) and its params.
seq_params, stage1_ckpt = load_SeqAE_checkpoint(STAGE1_CKPT)
model = SequenceAutoencoder(seq_params)          # freeze_packet_encoder=True by default
model.load_state_dict(stage1_ckpt["model_state_dict"])
model = model.to(device)
# The normalisation stats ride in the buffers; mirror them back into seq_params so the
# next asdict(config) in save_checkpoint carries them rather than None.
model.set_target_stats(model.target_mean.clone(), model.target_std.clone())

assert seq_params.SeqEncParams.packets_per_sequence == PACKETS_PER_SEQUENCE, (
    f"stage 1 was trained with P={seq_params.SeqEncParams.packets_per_sequence}, "
    f"this script is configured for {PACKETS_PER_SEQUENCE}")
assert seq_params.SeqEncParams.packets_per_sequence == TrainHandler.seq_len + 1

packet_ae_params, packet_ckpt = load_AE_Checkpoint(PACKET_AE_CKPT)
packet_ae = PacketAutoencoder(packet_ae_params)
packet_ae.load_state_dict(packet_ckpt["model_state_dict"])
packet_decoder = packet_ae.decoder.to(device).eval()
for p in packet_decoder.parameters():
    p.requires_grad = False

trainable = [p for p in model.parameters() if p.requires_grad]
logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable)} "
            f"(packet encoder stays frozen)")

# --- Byte-level reconstruction eval -----------------------------------------
byte_eval = build_byte_eval_set(
    handler=ValHandler, flow_offsets=val_flow_offsets, windows=val_windows[:, 1:],
    meta=val_meta, split_file=VAL_SPLIT_FILE, packet_ae_params=packet_ae_params,
    packet_encoder=model.encoder.packet_encoder, packet_decoder=packet_decoder,
    num_windows=BYTE_EVAL_WINDOWS, chunk=BYTE_EVAL_CHUNK, device=device, logger=logger)
byte_ceiling = byte_eval["ceiling"]
pad_id = byte_eval["pad_id"]

# --- Structure eval subset --------------------------------------------------
# The same windows every epoch, spread over the whole split (windows come in flow_key
# order, so the first N would all sit in one corner of it).
struct_pick = np.linspace(0, val_windows.shape[0] - 1,
                          min(STRUCT_EVAL_WINDOWS, val_windows.shape[0])).astype(np.int64)
struct_windows = val_windows[struct_pick]
struct_flows = struct_windows[:, 0]
struct_labels = torch.from_numpy(val_groups[struct_flows]).to(device)
struct_sup = val_supported[struct_flows]
struct_seq_lens = struct_windows[:, 2]
struct_batches = [struct_windows[i:i + val_batch_size, 1:]
                  for i in range(0, struct_windows.shape[0], val_batch_size)]
logger.info(f"Structure eval: {struct_windows.shape[0]} windows, "
            f"{len(set(struct_labels.tolist()))} groups present, "
            f"{struct_sup.mean():.1%} in a supported group")

optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)

steps_per_epoch = math.ceil(len(TrainHandler.starts) / batch_size)
if MAX_STEPS_PER_EPOCH is not None:
    steps_per_epoch = min(steps_per_epoch, MAX_STEPS_PER_EPOCH)
total_steps = max(1, Epochs * steps_per_epoch)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, cosine_warmup_lambda(WARMUP_STEPS, total_steps))

start_epoch = 0
best_val = float("inf")
global_step = 0

if RESUME_FROM is not None:
    _resumed_params, resume_ckpt = load_SeqAE_checkpoint(RESUME_FROM, device=device)
    model.load_state_dict(resume_ckpt["model_state_dict"])
    model.set_target_stats(model.target_mean.clone(), model.target_std.clone())
    optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in resume_ckpt:
        scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
    start_epoch = resume_ckpt["epoch"] + 1
    best_val = resume_ckpt.get("best_val", float("inf"))
    global_step = resume_ckpt.get("global_step", start_epoch * steps_per_epoch)
    logger.info(f"Resumed from {RESUME_FROM}: epoch {start_epoch}, best_val {best_val:.6f}")

# --- Metric logging ---------------------------------------------------------
metric_fields = ["epoch", "step", "lr",
                 "train_recon", "train_length", "train_struct", "train_total",
                 "train_positives", "train_groups_per_batch",
                 "val_recon", "val_length", "val_total", "val_length_acc",
                 "val_byte_acc", "val_byte_acc_nonpad",
                 "ceiling_byte_acc", "ceiling_byte_acc_nonpad",
                 "baseline_global", "baseline_per_position",
                 "struct_eff_rank",
                 "sil", "knn", "ratio",
                 "sil_sup", "knn_sup", "ratio_sup"]
metric_fields += [f"sil_{name}" for _lo, _hi, name in SEQ_LEN_BINS]
metrics = MetricsCsv(f"{output_dir}/metrics.csv", metric_fields, logger)


def autocast_ctx():
    if AMP_DTYPE is None:
        return torch.autocast("cuda", enabled=False)
    return torch.autocast("cuda", dtype=AMP_DTYPE)


def silhouette(u: np.ndarray, labels: np.ndarray) -> float:
    """Cosine silhouette, or nan where it is undefined.

    sklearn raises when there are fewer than 2 labels or as many labels as samples,
    both of which happen naturally inside a narrow seq_len bin.
    """
    if labels.shape[0] < 3:
        return float("nan")
    n_labels = len(np.unique(labels))
    if n_labels < 2 or n_labels >= labels.shape[0]:
        return float("nan")
    return float(silhouette_score(u, labels, metric="cosine"))


@torch.no_grad()
def structure_metrics() -> dict:
    """Geometry of the fixed structure-eval subset, in the space the AD suite reads.

    Deliberately measured on `z` itself rather than on any projection: the AD suite,
    embedding_viz and every detector consume `z`, so structure that does not live
    there does not count.

    Everything is reported twice -- pooled, and over the supported groups alone. On
    this split those differ enormously: 65% of val windows belong to one endpoint pair
    whose training presence is four flows of a different protocol, so the pooled
    numbers are mostly a statement about a group the loss never shaped.
    """
    Z = []
    for windows in struct_batches:
        latents, seq_lens = ValHandler.latent_batch_from_windows(windows)
        Z.append(model.encode(seq_lens.to(device), latents=latents.to(device)).float())
    Z = torch.cat(Z)
    u = F.normalize(Z, dim=-1)

    out = {"struct_eff_rank": effective_rank(Z),
           "knn": knn_group_agreement(u, struct_labels, KNN_K),
           "ratio": group_distance_stats(u, struct_labels)["ratio"]}

    sup = torch.from_numpy(struct_sup).to(device)
    if bool(sup.any()):
        u_sup, labels_sup = u[sup], struct_labels[sup]
        out["knn_sup"] = knn_group_agreement(u_sup, labels_sup, KNN_K)
        out["ratio_sup"] = group_distance_stats(u_sup, labels_sup)["ratio"]
    else:
        out["knn_sup"] = out["ratio_sup"] = float("nan")

    u_np = u.cpu().numpy()
    labels_np = struct_labels.cpu().numpy()
    out["sil"] = silhouette(u_np, labels_np)
    out["sil_sup"] = silhouette(u_np[struct_sup], labels_np[struct_sup])

    # Flow length is the known confound -- normal traffic has a median of 64 packets
    # while most attack sets are single-packet, so a pooled silhouette can improve
    # purely because groups differ in length. Repeating it inside a length bin is
    # where that is controlled rather than flagged.
    for lo, hi, name in SEQ_LEN_BINS:
        in_bin = (struct_seq_lens >= lo) & (struct_seq_lens <= hi)
        out[f"sil_{name}"] = silhouette(u_np[in_bin], labels_np[in_bin])
    return out


@torch.no_grad()
def evaluate() -> dict:
    """Deterministic pass over the val windows, plus the structure metrics."""
    model.eval()
    recon_sum = length_sum = total_sum = 0.0
    length_correct = length_total = 0
    base_global = base_position = 0.0

    for windows in val_batches:
        latents, seq_lens = ValHandler.latent_batch_from_windows(windows)
        latents = latents.to(device)
        seq_lens = seq_lens.to(device)

        with autocast_ctx():
            pred, tgt, z, len_logits, mask = model(seq_lens, latents=latents)
        pred, tgt = pred.float(), tgt.float()
        len_logits = len_logits.float() if len_logits is not None else None
        losses = model.loss(pred, tgt, mask, len_logits, seq_lens)

        recon_sum += losses["recon"].item()
        total_sum += losses["total"].item()
        if "length" in losses:
            length_sum += losses["length"].item()
            length_correct += (len_logits.argmax(dim=-1) == seq_lens).sum().item()
            length_total += seq_lens.numel()

        baselines = baseline_mses(tgt, mask)
        base_global += baselines["global"]
        base_position += baselines["per_position"]

    # Byte accuracy runs in fp32 on the fixed subset, matching how the packet AE was
    # scored -- comparing it against byte_ceiling is the whole point.
    byte_acc = byte_level_reconstruction(model, packet_decoder, byte_eval["seq_lens"],
                                         byte_eval["tokens"], latents=byte_eval["latents"],
                                         pad_token_id=pad_id, chunk=BYTE_EVAL_CHUNK)
    structure = structure_metrics()

    n_batches = max(1, len(val_batches))
    model.train()
    return {
        "val_recon": recon_sum / n_batches,
        "val_length": length_sum / n_batches,
        "val_total": total_sum / n_batches,
        "val_length_acc": length_correct / length_total if length_total else float("nan"),
        "val_byte_acc": byte_acc["all"],
        "val_byte_acc_nonpad": byte_acc["nonpad"],
        "ceiling_byte_acc": byte_ceiling["all"],
        "ceiling_byte_acc_nonpad": byte_ceiling["nonpad"],
        "baseline_global": base_global / n_batches,
        "baseline_per_position": base_position / n_batches,
        **structure,
    }


def log_eval(tag: str, m: dict):
    logger.info(f"===== {tag} =====")
    logger.info(f"Val Total Loss: {m['val_total']:.6f} "
                f"Val Reconstruction Loss: {m['val_recon']:.6f} "
                f"Val Length Accuracy: {m['val_length_acc']:.4f}")
    logger.info(f"Val Byte Accuracy: all {m['val_byte_acc']:.4f} "
                f"non-pad {m['val_byte_acc_nonpad']:.4f} "
                f"(packet-AE ceiling {byte_ceiling['all']:.4f} / "
                f"{byte_ceiling['nonpad']:.4f})")
    logger.info(f"Structure pooled:    silhouette {m['sil']:.4f} "
                f"knn@{KNN_K} {m['knn']:.4f} intra/inter {m['ratio']:.4f}")
    logger.info(f"Structure supported: silhouette {m['sil_sup']:.4f} "
                f"knn@{KNN_K} {m['knn_sup']:.4f} intra/inter {m['ratio_sup']:.4f}")
    logger.info("Silhouette by seq_len bin: " + " ".join(
        f"{name} {m[f'sil_{name}']:.4f}" for _lo, _hi, name in SEQ_LEN_BINS))
    logger.info(f"Effective rank of z: {m['struct_eff_rank']:.1f} "
                f"(of {seq_params.SeqEncParams.SeqEncoderDim} dims)")


# Stage 1's own geometry, before a single structure gradient. This is the row every
# later epoch is read against.
model.train()
if start_epoch == 0:
    baseline_metrics = evaluate()
    log_eval("Stage-1 baseline (no structure gradient yet)", baseline_metrics)
    metrics.append({"epoch": -1, "step": 0, "lr": 0.0,
                    "train_recon": float("nan"), "train_length": float("nan"),
                    "train_struct": float("nan"), "train_total": float("nan"),
                    "train_positives": float("nan"),
                    "train_groups_per_batch": float("nan"),
                    **baseline_metrics})

for epoch in range(start_epoch, Epochs):
    batches = TrainHandler.epoch_grouped_flow_batches(train_groups, batch_size, rng,
                                                      MAX_GROUP_RUN)
    if MAX_STEPS_PER_EPOCH is not None:
        batches = batches[:MAX_STEPS_PER_EPOCH]

    # How many distinct groups the sampler actually put in a batch. If this collapses
    # towards 1 the contrastive term has no negatives, which no loss value would tell
    # you on its own.
    groups_per_batch = float(np.mean([len(np.unique(train_groups[b])) for b in batches]))

    epoch_recon = epoch_length = epoch_struct = epoch_total = epoch_positives = 0.0
    window_recon = window_struct = window_total = window_positives = 0.0

    for index, flow_ids in enumerate(batches):
        latents, seq_lens = TrainHandler.draw_latent_batch(flow_ids, rng)
        latents = latents.to(device)
        seq_lens = seq_lens.to(device)
        labels = torch.from_numpy(train_groups[flow_ids]).to(device)

        # Forward Pass
        with autocast_ctx():
            pred, tgt, z, len_logits, mask = model(seq_lens, latents=latents)
        pred, tgt = pred.float(), tgt.float()
        len_logits = len_logits.float() if len_logits is not None else None
        losses = model.loss(pred, tgt, mask, len_logits, seq_lens)

        # .float() before normalising matters: a 512-dim dot product in bf16 carries
        # about three significant digits, which is not enough at these temperatures.
        u = F.normalize(z.float(), dim=-1)
        struct_loss, struct_diag = supcon(u, labels, TAU, balance_groups=BALANCE_GROUPS)
        total = losses["total"] + LAMBDA_STRUCT * struct_loss

        # Backward Pass
        total.backward()
        torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        recon = losses["recon"].item()
        length = losses["length"].item() if "length" in losses else 0.0
        struct = struct_loss.item()
        total_loss = total.item()
        positives = struct_diag["positives_per_anchor"]
        epoch_recon += recon; epoch_length += length; epoch_total += total_loss
        epoch_struct += struct; epoch_positives += positives
        window_recon += recon; window_total += total_loss
        window_struct += struct; window_positives += positives

        if (index + 1) % log_every_n_batches == 0:
            n = log_every_n_batches
            logger.info(f"Epoch {epoch+1}/{Epochs} Batch {index+1}/{len(batches)} "
                        f"lr {scheduler.get_last_lr()[0]:.2e}")
            logger.info(f"Total {window_total / n:.6f} "
                        f"Recon {window_recon / n:.6f} "
                        f"SupCon {window_struct / n:.6f} "
                        f"(positives/anchor {window_positives / n:.1f}, "
                        f"{struct_diag['groups_in_batch']:.0f} groups in batch)")
            window_recon = window_struct = window_total = window_positives = 0.0

    n_batches = max(1, len(batches))
    train_metrics = {
        "train_recon": epoch_recon / n_batches,
        "train_length": epoch_length / n_batches,
        "train_struct": epoch_struct / n_batches,
        "train_total": epoch_total / n_batches,
        "train_positives": epoch_positives / n_batches,
        "train_groups_per_batch": groups_per_batch,
    }

    val_metrics = evaluate()
    log_eval(f"Validation Results (Epoch {epoch+1})", val_metrics)
    logger.info(f"Baseline MSEs: global {val_metrics['baseline_global']:.6f} "
                f"per_position {val_metrics['baseline_per_position']:.6f}")

    metrics.append({"epoch": epoch, "step": global_step,
                    "lr": scheduler.get_last_lr()[0], **train_metrics, **val_metrics})

    extra = {"scheduler_state_dict": scheduler.state_dict(),
             "best_val": best_val, "global_step": global_step}

    epoch_path = f"{output_dir}/SequenceLevelAutoEncoder_{RUN_NAME}_E{epoch}.ckpt"
    save_checkpoint(model=model, optimizer=optimizer, epoch=epoch,
                    loss=train_metrics["train_total"], config=seq_params,
                    path=epoch_path, extra=extra)
    logger.info(f"Saved SequenceAutoencoder to {epoch_path}")

    # Selected on reconstruction alone, not on the structure term. The structuring is
    # the intervention; letting it also pick the checkpoint would make "structure
    # improved" true by construction.
    if val_metrics["val_total"] < best_val:
        best_val = val_metrics["val_total"]
        extra["best_val"] = best_val
        best_path = f"{output_dir}/SequenceLevelAutoEncoder_{RUN_NAME}_best.ckpt"
        save_checkpoint(model=model, optimizer=optimizer, epoch=epoch,
                        loss=train_metrics["train_total"], config=seq_params,
                        path=best_path, extra=extra)
        logger.info(f"New best val total loss {best_val:.6f}, saved to {best_path}")

plot_curves(
    metrics, f"{output_dir}/curves.png", "epoch",
    panels=[
        {"title": "Reconstruction vs. learn-nothing baselines",
         "ylabel": "masked MSE (normalised)",
         "series": [("train_recon", "train recon"), ("val_recon", "val recon")],
         "hlines": [("baseline_global", "baseline global", "--"),
                    ("baseline_per_position", "baseline per-position", ":")]},
        {"title": "Byte reconstruction / length accuracy", "ylabel": "accuracy",
         "ylim": (0, 1),
         "series": [("val_byte_acc", "val byte acc (all)"),
                    ("val_byte_acc_nonpad", "val byte acc (non-pad)"),
                    ("val_length_acc", "val length acc")],
         "hlines": [("ceiling_byte_acc", "packet-AE ceiling (all)", "--"),
                    ("ceiling_byte_acc_nonpad", "packet-AE ceiling (non-pad)", ":")]},
        {"title": "Embedding structure (val)", "ylabel": "score",
         "series": [("sil", "silhouette"), ("sil_sup", "silhouette (supported)"),
                    ("knn", f"knn@{KNN_K}"), ("knn_sup", f"knn@{KNN_K} (supported)"),
                    ("ratio", "intra/inter")]},
        {"title": "Contrastive loss", "ylabel": "supervised contrastive loss",
         "series": [("train_struct", "supcon")]},
        # Its own panel: the rank runs to hundreds while the loss sits near single
        # digits, so sharing an axis would flatten the loss into the baseline.
        {"title": "Collapse monitor", "ylabel": "effective rank of z",
         "series": [("struct_eff_rank", "effective rank of z")]},
    ],
    logger=logger)

logger.info(f"Done. Export embeddings with SequenceEmbeddingAD.py by setting its "
            f"RUN_NAME to {RUN_NAME!r}, then compare against {STAGE1_RUN} in "
            f"EmbeddingADSuite.py with CALIB_LABEL = EVAL_NEG_LABEL = 'test' -- `val` "
            f"was already spent on the stage-1 model.")
