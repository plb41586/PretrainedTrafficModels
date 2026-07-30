"""
Sequence-level autoencoder pretraining.

A whole flow is compressed into one CLS vector by Sequence_Encoder and decoded
back into all P packet latents by SequenceDecoder. Reconstruction is supervised
in normalised packet-latent space; the questions the run answers are whether that
single bottleneck vector beats the "learn nothing" baselines in baseline_mses
(predict the global mean latent / the per-position mean latent), and whether the
reconstructed latents are tight enough for the frozen packet decoder to turn back
into the right bytes (byte_level_reconstruction, scored against the packet AE's
own accuracy on the same packets).

The packet encoder is frozen, so this trains off a precomputed latent cache --
build it first with:
    python -m RawByteTrafficModelling.PreTraining.CachePacketLatents

Then run from the repo root:
    python -m RawByteTrafficModelling.PreTraining.SequenceLevelAutoEncoder
"""
from RawByteTrafficModelling.ModelComponents.ModelDefinitions import (
    SeqEncoderParams,
    SeqAutoEncoderParams,
    Sequence_Encoder,
    SequenceAutoencoder,
    PacketAutoencoder,
    load_AE_Checkpoint,
    load_SeqAE_checkpoint,
    save_checkpoint,
    baseline_mses,
    build_padding_mask,
    byte_level_reconstruction,
    decoder_byte_accuracy,
)
from RawByteTrafficModelling.ModelComponents.DataUtils import (
    CachedLatentSequenceHandler,
    ID_Encoder,
    PreTrainingDatasetHandler,
    load_latent_cache,
)
from RawByteTrafficModelling.ModelComponents.BackBones import MambaBackboneParams
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
import numpy as np
import torch
import logging
import math
import csv
import os

RUN_NAME = "SeqAE_EdgeIIoT_Mamba"
output_dir = f"RawByteTrafficModelling/PreTraining/TrainingOutputs/{RUN_NAME}"
os.makedirs(output_dir, exist_ok=True)   # existing scripts assume this exists; it often doesn't

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{output_dir}/SequenceLevelAutoEncoder.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

### Set Training Parameters
PACKET_AE_CKPT = "RawByteTrafficModelling/PreTraining/TrainingOutputs/EdgeIIoT_AutoEncoder_FlowSplit/PacketLevelAutoEncoder_EdgeIIoT_E1.ckpt"
TRAIN_LATENT_CACHE = "data_artefacts/IIoTset-Ferrag/flow_split/latents_EdgeIIoT_E1/train"
TEST_LATENT_CACHE = "data_artefacts/IIoTset-Ferrag/flow_split/latents_EdgeIIoT_E1/test"
# Byte-level eval needs the raw packets, which the latent cache does not hold. This
# must be the parquet TEST_LATENT_CACHE was built from (asserted against its meta.json).
VAL_SPLIT_FILE = "data_artefacts/IIoTset-Ferrag/flow_split/test.parquet"

PACKETS_PER_SEQUENCE = 65      # P, including the slot the seq-CLS overwrites
SEQ_ENCODER_DIM = 384
SEQ_DECODER_DIM = 384
SEQ_BACKBONE = "Mamba"         # "Mamba" | "Transformer" (swap MambaBackboneParams too)

Epochs = 20
batch_size = 256
val_batch_size = 256
learning_rate = 3e-4
weight_decay = 1e-2
WARMUP_STEPS = 500
GRAD_CLIP = 1.0
AMP_DTYPE = torch.bfloat16     # None disables autocast
length_loss_weight = 0.1

log_every_n_batches = 50
BYTE_EVAL_WINDOWS = 256         # fixed subset of val windows scored byte-by-byte each epoch
BYTE_EVAL_CHUNK = 64            # packets per frozen-decoder forward; bounds the logits tensor
MAX_STEPS_PER_EPOCH = None     # set to a small int for a wiring smoke test
RESUME_FROM = None
SEED = 42

torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)

device = torch.device("cuda")
assert device == torch.device("cuda")

### Load the cached packet latents
# Both caches must come from the exact weights in PACKET_AE_CKPT -- load_latent_cache
# compares a sha256 of the checkpoint, so retraining the packet model (even back to
# the same filename) fails loudly here instead of training on stale latents.
train_latents, train_flow_offsets, train_meta = load_latent_cache(TRAIN_LATENT_CACHE, PACKET_AE_CKPT)
val_latents, val_flow_offsets, val_meta = load_latent_cache(TEST_LATENT_CACHE, PACKET_AE_CKPT)
logger.info(f"Train cache: {train_meta['num_rows']} packets in {train_meta['num_flows']} flows")
logger.info(f"Val cache:   {val_meta['num_rows']} packets in {val_meta['num_flows']} flows")

TrainHandler = CachedLatentSequenceHandler(train_latents, train_flow_offsets, PACKETS_PER_SEQUENCE)
ValHandler = CachedLatentSequenceHandler(val_latents, val_flow_offsets, PACKETS_PER_SEQUENCE)

val_windows = ValHandler.enumerate_windows()
val_batches = [val_windows[i:i + val_batch_size]
               for i in range(0, val_windows.shape[0], val_batch_size)]
logger.info(f"Validation: {val_windows.shape[0]} deterministic windows -> {len(val_batches)} batches")

# --- Model Config ---
packet_ae_params, packet_ckpt = load_AE_Checkpoint(PACKET_AE_CKPT)
packet_ae = PacketAutoencoder(packet_ae_params)
packet_ae.load_state_dict(packet_ckpt["model_state_dict"])

seq_enc_params = SeqEncoderParams(
    EncoderParams=packet_ae_params.ENC_Params,
    SeqEncoderDim=SEQ_ENCODER_DIM,
    packets_per_sequence=PACKETS_PER_SEQUENCE,
    SeqBackboneType=SEQ_BACKBONE,
    SeqBackboneParams=MambaBackboneParams(dim=SEQ_ENCODER_DIM),
)

ae_params = SeqAutoEncoderParams(
    SeqEncParams=seq_enc_params,
    SeqDecoderDim=SEQ_DECODER_DIM,
    SeqDecBackboneType=SEQ_BACKBONE,
    SeqDecBackbone=MambaBackboneParams(dim=SEQ_DECODER_DIM),
    length_loss_weight=length_loss_weight,
)

# Sequence_Encoder scatters the seq-CLS at index seq_len, so P must leave room for it.
assert seq_enc_params.packets_per_sequence == TrainHandler.seq_len + 1

sequence_encoder = Sequence_Encoder(
    seq_enc_params,
    packet_encoder=packet_ae.encoder,
    freeze_packet_encoder=True,
)

model = SequenceAutoencoder(ae_params, encoder=sequence_encoder).to(device)

trainable = [p for p in model.parameters() if p.requires_grad]
logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable)}")

# --- Byte-level reconstruction eval ----------------------------------------
# Masked MSE in normalised latent space has no interpretable scale. Byte accuracy
# through the frozen packet decoder does, and it answers the question that matters:
# are the reconstructed latents tight enough for the decoder to recover the packet?
#
# The latent cache holds no bytes, so this needs the split parquet plus the
# cache-row -> parquet-row map. build_flow_index reproduces exactly the order
# CachePacketLatents wrote; the flow_key assert pins that down and the re-encode
# check below catches any residual misalignment, which would otherwise show up as
# a mysteriously terrible metric rather than an error.
assert val_meta["split_file"] == VAL_SPLIT_FILE, (
    f"byte eval would read {VAL_SPLIT_FILE} but {TEST_LATENT_CACHE} was built "
    f"from {val_meta['split_file']}")

pad_id = packet_ae_params.ENC_Params.SpecialTokens["<pad>"]
packet_id_len = packet_ae_params.ENC_Params.packet_id_len
packet_decoder = packet_ae.decoder.to(device).eval()
for p in packet_decoder.parameters():
    p.requires_grad = False

logger.info(f"Building byte-eval subset from {VAL_SPLIT_FILE} "
            f"({BYTE_EVAL_WINDOWS} windows; sorts the split, takes a moment)")
val_split = pl.read_parquet(VAL_SPLIT_FILE)
byte_encoder = ID_Encoder(SpecialIDs=packet_ae_params.ENC_Params.SpecialTokens,
                          CLS_Placement="EOS")   # must match CachePacketLatents
byte_handler = PreTrainingDatasetHandler(val_split, PACKETS_PER_SEQUENCE - 1, byte_encoder)
val_flow_index = byte_handler.build_flow_index()
assert val_flow_index["flow_key"].to_list() == val_flow_offsets["flow_key"].to_list(), \
    "flow index order does not reproduce the cache's order -- byte eval would be misaligned"

# Spread the subset over the whole val set: windows come in flow_key order, so the
# first N would all sit in one corner of the split.
byte_pick = np.linspace(0, val_windows.shape[0] - 1, BYTE_EVAL_WINDOWS).astype(np.int64)
byte_windows = val_windows[byte_pick]
byte_latents, byte_seq_lens = ValHandler.latent_batch_from_windows(byte_windows)

byte_tokens = torch.full((byte_windows.shape[0], PACKETS_PER_SEQUENCE, packet_id_len),
                         pad_id, dtype=torch.int16)   # ids max out at 261, int16 is plenty
val_row_idx = val_flow_index["row_idx"]
for w, (start, length) in enumerate(byte_windows):
    flow = int(np.searchsorted(ValHandler.starts, start, side="right") - 1)
    offset = int(start - ValHandler.starts[flow])
    rows = val_row_idx[flow].to_numpy()[offset:offset + int(length)]
    window_bytes, _ = byte_handler.get_pretraining_data(rows)
    ids = byte_encoder.construct_input_ids(window_bytes)
    byte_tokens[w, :int(length)] = torch.tensor(np.asarray(ids), dtype=torch.int16)

byte_valid = build_padding_mask(byte_seq_lens, PACKETS_PER_SEQUENCE).reshape(-1)
byte_flat_latents = byte_latents.reshape(-1, byte_latents.shape[-1])[byte_valid]
byte_flat_tokens = byte_tokens.reshape(-1, packet_id_len)[byte_valid]

# Alignment: the tokens just gathered must re-encode to the cached latents.
with torch.no_grad():
    probe = min(256, byte_flat_tokens.shape[0])
    probe_live = model.encoder.packet_encoder(
        byte_flat_tokens[:probe].long().to(device)).float().cpu()
probe_delta = (probe_live - byte_flat_latents[:probe]).abs().max().item()
assert probe_delta < 1e-2, (
    f"byte-eval tokens do not match the cached latents (max|live-cached| {probe_delta:.2e}) "
    f"-- the cache row -> parquet row mapping is off")
logger.info(f"Byte-eval alignment OK over {probe} packets: "
            f"max|live-cached| {probe_delta:.2e}")

byte_latents = byte_latents.to(device)
byte_tokens = byte_tokens.to(device)
byte_seq_lens = byte_seq_lens.to(device)

# Ceiling: the packet AE's own byte accuracy on these packets, decoding from the
# *true* latents. The sequence-AE number only means something against this.
byte_ceiling = decoder_byte_accuracy(packet_decoder,
                                     byte_flat_latents.to(device),
                                     byte_flat_tokens.to(device),
                                     pad_token_id=pad_id, chunk=BYTE_EVAL_CHUNK)
logger.info(f"Byte-eval subset: {byte_windows.shape[0]} windows, "
            f"{byte_flat_tokens.shape[0]} real packets")
logger.info(f"Packet-AE ceiling (decoding true latents): "
            f"all {byte_ceiling['all']:.4f} non-pad {byte_ceiling['nonpad']:.4f}")
del byte_flat_latents, byte_flat_tokens

optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)

steps_per_epoch = math.ceil(len(TrainHandler.starts) / batch_size)
if MAX_STEPS_PER_EPOCH is not None:
    steps_per_epoch = min(steps_per_epoch, MAX_STEPS_PER_EPOCH)
total_steps = max(1, Epochs * steps_per_epoch)


def lr_lambda(step: int) -> float:
    """Linear warmup then cosine decay to zero."""
    if step < WARMUP_STEPS:
        return (step + 1) / max(1, WARMUP_STEPS)
    progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

start_epoch = 0
best_val = float("inf")
global_step = 0

if RESUME_FROM is not None:
    _resumed_params, resume_ckpt = load_SeqAE_checkpoint(RESUME_FROM, device=device)
    model.load_state_dict(resume_ckpt["model_state_dict"])
    # The stats ride in the buffers; mirror them back into ae_params so the next
    # asdict(config) in save_checkpoint carries them rather than None.
    model.set_target_stats(model.target_mean.clone(), model.target_std.clone())
    optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in resume_ckpt:
        scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
    start_epoch = resume_ckpt["epoch"] + 1
    best_val = resume_ckpt.get("best_val", float("inf"))
    global_step = resume_ckpt.get("global_step", start_epoch * steps_per_epoch)
    logger.info(f"Resumed from {RESUME_FROM}: epoch {start_epoch}, best_val {best_val:.6f}")
else:
    # --- Latent normalisation stats -------------------------------------------
    # Reconstruction loss is measured in normalised space, so these need to come
    # from a large sample, not one batch. The cache makes that free.
    stat_rows = min(200_000, train_latents.shape[0])
    stat_idx = torch.from_numpy(rng.choice(train_latents.shape[0], stat_rows, replace=False))
    stat_sample = train_latents[stat_idx].float()
    model.set_target_stats(stat_sample.mean(dim=0), stat_sample.std(dim=0))
    logger.info(f"Target stats over {stat_rows} cached packet latents: "
                f"mean|.| {stat_sample.mean(dim=0).abs().mean():.4f} "
                f"std {stat_sample.std(dim=0).mean():.4f}")

    untrained_path = f"{output_dir}/SequenceLevelAutoEncoder_{RUN_NAME}_untrained.ckpt"
    save_checkpoint(model=model, optimizer=optimizer, epoch=-1, loss=0,
                    config=ae_params, path=untrained_path)
    logger.info(f"Saved SequenceAutoencoder to {untrained_path}")

# --- Metric logging --------------------------------------------------------
metrics_path = f"{output_dir}/metrics.csv"
metric_fields = ["epoch", "step", "lr",
                 "train_recon", "train_length", "train_total",
                 "val_recon", "val_length", "val_total", "val_length_acc",
                 "val_byte_acc", "val_byte_acc_nonpad",
                 "ceiling_byte_acc", "ceiling_byte_acc_nonpad",
                 "baseline_global", "baseline_per_position"]
# A metrics.csv from an earlier run with different columns would be appended to with
# rows that no longer line up with its header, so retire it instead of corrupting it.
if os.path.exists(metrics_path):
    with open(metrics_path, newline="") as f:
        stale_header = next(csv.reader(f), [])
    if stale_header != metric_fields:
        backup_path = f"{metrics_path}.{len(stale_header)}col.bak"
        os.rename(metrics_path, backup_path)
        logger.warning(f"metrics.csv had stale columns, moved to {backup_path}")
if not os.path.exists(metrics_path):
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(metric_fields)


def autocast_ctx():
    if AMP_DTYPE is None:
        return torch.autocast("cuda", enabled=False)
    return torch.autocast("cuda", dtype=AMP_DTYPE)


@torch.no_grad()
def evaluate():
    """Deterministic pass over the val windows. Returns a metrics dict."""
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

    # Byte accuracy runs in fp32 on the fixed subset, matching how the packet AE
    # was scored -- comparing it against byte_ceiling is the whole point.
    byte_acc = byte_level_reconstruction(model, packet_decoder, byte_seq_lens,
                                         byte_tokens, latents=byte_latents,
                                         pad_token_id=pad_id, chunk=BYTE_EVAL_CHUNK)

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
    }


model.train()

for epoch in range(start_epoch, Epochs):
    batches = TrainHandler.epoch_flow_batches(batch_size, rng)
    if MAX_STEPS_PER_EPOCH is not None:
        batches = batches[:MAX_STEPS_PER_EPOCH]

    epoch_recon = epoch_length = epoch_total = 0.0
    window_recon = window_length = window_total = 0.0

    for index, flow_ids in enumerate(batches):
        latents, seq_lens = TrainHandler.draw_latent_batch(flow_ids, rng)
        latents = latents.to(device)
        seq_lens = seq_lens.to(device)

        # Forward Pass
        with autocast_ctx():
            pred, tgt, z, len_logits, mask = model(seq_lens, latents=latents)
        pred, tgt = pred.float(), tgt.float()
        len_logits = len_logits.float() if len_logits is not None else None
        losses = model.loss(pred, tgt, mask, len_logits, seq_lens)

        # Backward Pass
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        recon = losses["recon"].item()
        length = losses["length"].item() if "length" in losses else 0.0
        total = losses["total"].item()
        epoch_recon += recon; epoch_length += length; epoch_total += total
        window_recon += recon; window_length += length; window_total += total

        if (index + 1) % log_every_n_batches == 0:
            n = log_every_n_batches
            logger.info(f"Epoch {epoch+1}/{Epochs} Batch {index+1}/{len(batches)} "
                        f"lr {scheduler.get_last_lr()[0]:.2e}")
            logger.info(f"Total Loss: {window_total / n:.6f} "
                        f"Reconstruction Loss: {window_recon / n:.6f} "
                        f"Length Loss: {window_length / n:.6f}")
            window_recon = window_length = window_total = 0.0

    n_batches = max(1, len(batches))
    train_metrics = {
        "train_recon": epoch_recon / n_batches,
        "train_length": epoch_length / n_batches,
        "train_total": epoch_total / n_batches,
    }

    val_metrics = evaluate()
    logger.info(f"===== Validation Results (Epoch {epoch+1}) =====")
    logger.info(f"Val Total Loss: {val_metrics['val_total']:.6f} "
                f"Val Reconstruction Loss: {val_metrics['val_recon']:.6f}")
    logger.info(f"Val Length Loss: {val_metrics['val_length']:.6f} "
                f"Val Length Accuracy: {val_metrics['val_length_acc']:.4f}")
    logger.info(f"Val Byte Accuracy: all {val_metrics['val_byte_acc']:.4f} "
                f"non-pad {val_metrics['val_byte_acc_nonpad']:.4f} "
                f"(packet-AE ceiling {byte_ceiling['all']:.4f} / "
                f"{byte_ceiling['nonpad']:.4f})")
    logger.info(f"Baseline MSEs: global {val_metrics['baseline_global']:.6f} "
                f"per_position {val_metrics['baseline_per_position']:.6f}")

    row = {"epoch": epoch, "step": global_step, "lr": scheduler.get_last_lr()[0],
           **train_metrics, **val_metrics}
    with open(metrics_path, "a", newline="") as f:
        csv.DictWriter(f, metric_fields).writerow(row)

    extra = {"scheduler_state_dict": scheduler.state_dict(),
             "best_val": best_val, "global_step": global_step}

    epoch_path = f"{output_dir}/SequenceLevelAutoEncoder_{RUN_NAME}_E{epoch}.ckpt"
    save_checkpoint(model=model, optimizer=optimizer, epoch=epoch,
                    loss=train_metrics["train_total"], config=ae_params,
                    path=epoch_path, extra=extra)
    logger.info(f"Saved SequenceAutoencoder to {epoch_path}")

    if val_metrics["val_total"] < best_val:
        best_val = val_metrics["val_total"]
        extra["best_val"] = best_val
        best_path = f"{output_dir}/SequenceLevelAutoEncoder_{RUN_NAME}_best.ckpt"
        save_checkpoint(model=model, optimizer=optimizer, epoch=epoch,
                        loss=train_metrics["train_total"], config=ae_params,
                        path=best_path, extra=extra)
        logger.info(f"New best val total loss {best_val:.6f}, saved to {best_path}")

# --- Curves ----------------------------------------------------------------
# Read back from the CSV rather than `history` so a resumed run still plots the
# epochs that ran before the interruption.
with open(metrics_path, newline="") as f:
    history = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]

if history:
    epochs_axis = [r["epoch"] + 1 for r in history]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs_axis, [r["train_recon"] for r in history], label="train recon")
    axes[0].plot(epochs_axis, [r["val_recon"] for r in history], label="val recon")
    axes[0].axhline(history[-1]["baseline_global"], ls="--", c="grey", label="baseline global")
    axes[0].axhline(history[-1]["baseline_per_position"], ls=":", c="black", label="baseline per-position")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("masked MSE (normalised)")
    axes[0].set_title("Reconstruction vs. learn-nothing baselines"); axes[0].legend()

    axes[1].plot(epochs_axis, [r["val_byte_acc"] for r in history], label="val byte acc (all)")
    axes[1].plot(epochs_axis, [r["val_byte_acc_nonpad"] for r in history], label="val byte acc (non-pad)")
    axes[1].plot(epochs_axis, [r["val_length_acc"] for r in history], label="val length acc")
    axes[1].axhline(history[-1]["ceiling_byte_acc"], ls="--", c="grey", label="packet-AE ceiling (all)")
    axes[1].axhline(history[-1]["ceiling_byte_acc_nonpad"], ls=":", c="black", label="packet-AE ceiling (non-pad)")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0, 1); axes[1].set_title("Byte reconstruction / length accuracy"); axes[1].legend()

    axes[2].plot(epochs_axis, [r["lr"] for r in history])
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("lr"); axes[2].set_title("Learning rate")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/curves.png", dpi=200)
    logger.info(f"Saved curves to {output_dir}/curves.png")
