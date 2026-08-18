"""
Packet-level autoencoder pretraining.

One packet's 1520 token IDs are compressed to a single latent by Packet_Encoder
(DynamicCLS pooling) and decoded back autoregressively, teacher-forced, by
AutoregressiveDecoder. The encoder this produces is the packet "embedding layer"
the sequence level freezes, so its byte-reconstruction quality is the ceiling for
everything downstream.

Run from the repo root:
    python -m RawByteTrafficModelling.PreTraining.PacketLevelAutoEncoder

Reads the flow-grouped split: whole flows live in exactly one split, so the
sequence level can reuse the same partition without a flow straddling two of
them. val.parquet is deliberately untouched -- it is the final held-out set.
"""
from RawByteTrafficModelling.ModelComponents.ModelDefinitions import (
    AutoEncoderParams,
    Packet_Encoder,
    PacketAutoencoder,
    load_AE_Checkpoint,
    save_checkpoint,
)
from RawByteTrafficModelling.ModelComponents.DataUtils import PreTrainingDatasetHandler
from RawByteTrafficModelling.ModelComponents.BackBones import MambaBackboneParams
from RawByteTrafficModelling.PreTraining.RunConfig import (
    DATASETS,
    SPECIAL_IDS,
    MetricsCsv,
    cosine_warmup_lambda,
    fixed_eval_batches,
    make_id_encoder,
    packet_encoder_params,
    plot_curves,
    resolve_device,
    setup_run,
)
import polars as pl
import numpy as np
import torch
import torch.nn.functional as F
import math

### Set Training Parameters
RUN_NAME = "PacketAE_IIoTset_d128"
DATASET = DATASETS["IIoTset-Ferrag"]
DEVICE_INDEX = 0               # AE on cuda:0, the MLM run takes cuda:1 concurrently

ENCODER_DIM = 128              # was 64; the old run's non-pad accuracy plateaued at ~0.554
ENCODER_LAYERS = 2             # unchanged, so width is the only variable vs. that run
DECODER_LAYERS = 2

# Measured ~2.0 s/step at bs=512 -> ~7.5 h per 13.3k-step epoch on one A5000.
# The cosine schedule spans the whole run, so changing this changes the LR curve
# rather than just truncating it.
Epochs = 2
batch_size = 512               # fixed. First thing to halve if the logits tensor OOMs.
learning_rate = 1e-3
weight_decay = 1e-2
WARMUP_STEPS = 1000
GRAD_CLIP = 1.0
AMP_DTYPE = torch.bfloat16     # None disables autocast

# Most of the 1520 positions are <pad>, so an unweighted objective is dominated by
# a token the model masters in the first hundred steps -- the likeliest single cause
# of the old plateau. Set to 1.0 to reproduce the previous objective exactly.
# The unweighted CE is logged alongside regardless, so the numbers stay comparable
# with _deprecated/EdgeIIoT_AutoEncoder_FlowSplit (test CE 0.13794, non-pad 0.554).
PAD_LOSS_WEIGHT = 0.1

# One train epoch is ~6.8M packets (~13.3k batches at bs=512), so per-batch logging
# and a full pass over the 1.46M-packet test split are both far too expensive.
# Log running averages on an interval, and evaluate on a fixed random subset of
# test drawn once so the number is comparable across evals.
log_every_n_batches = 200
eval_every_n_batches = 2000
eval_batch_size = 256
EVAL_BATCHES = 40
SEED = 42

# Set to an epoch/latest checkpoint to continue a killed run. Epochs are multi-hour;
# the previous run died in epoch 3 with nothing to resume from.
RESUME_FROM = None
MAX_STEPS_PER_EPOCH = None     # cap on train batches per epoch. None = a real, full epoch.

# Smoke run: same code path, both parquets, forward/backward, mid-epoch eval,
# epoch-end eval, per-epoch and best checkpoint -- in a couple of minutes rather
# than a 13.3k-batch epoch. Writes to its own output_dir so a smoke checkpoint can
# never be mistaken for a trained one. Set False for the real run.
SMOKE = False
if SMOKE:
    RUN_NAME = f"{RUN_NAME}_smoke"
    # Two short epochs rather than one: this is what caught the best-checkpoint bug,
    # since a single-epoch smoke never compares one epoch's test CE against another's.
    Epochs = 2
    MAX_STEPS_PER_EPOCH = 30
    log_every_n_batches = 10
    eval_every_n_batches = 20
    EVAL_BATCHES = 4
    WARMUP_STEPS = 5

output_dir, logger = setup_run(RUN_NAME, "PacketLevelAutoEncoder.log")
logger.info(f"Run {RUN_NAME} (SMOKE={SMOKE}) -> {output_dir}")

torch.manual_seed(SEED)
device = resolve_device(DEVICE_INDEX)
logger.info(f"Device: {device} ({torch.cuda.get_device_name(device)})")

### Load data
data = pl.read_parquet(DATASET.train)
test_data = pl.read_parquet(DATASET.test)
logger.info(data.head())
logger.info(f"train: {data.height} packets from {DATASET.train}")
logger.info(f"test:  {test_data.height} packets from {DATASET.test}")

ID_Encoder = make_id_encoder()
DataHandler = PreTrainingDatasetHandler(data, 1, ID_Encoder)
TestDataHandler = PreTrainingDatasetHandler(test_data, 1, ID_Encoder)

# --- Model Config ---
encoder_params = packet_encoder_params(dim=ENCODER_DIM, num_layers=ENCODER_LAYERS)
autoencoderparams = AutoEncoderParams(
    ENC_Params=encoder_params,
    DecBackboneType="Mamba",
    DecBackbone=MambaBackboneParams(dim=ENCODER_DIM, num_layers=DECODER_LAYERS),
    bos_token_id=SPECIAL_IDS["<BOS>"],
)

PacketEncoder = Packet_Encoder(params=autoencoderparams.ENC_Params)
AutoEncoder = PacketAutoencoder(params=autoencoderparams, encoder=PacketEncoder).to(device)
logger.info(f"Parameters: {sum(p.numel() for p in AutoEncoder.parameters())}")

optimizer = torch.optim.AdamW(AutoEncoder.parameters(), lr=learning_rate,
                              weight_decay=weight_decay)

steps_per_epoch = math.ceil(data.height / batch_size)
if MAX_STEPS_PER_EPOCH is not None:
    steps_per_epoch = min(steps_per_epoch, MAX_STEPS_PER_EPOCH)
total_steps = max(1, Epochs * steps_per_epoch)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, cosine_warmup_lambda(WARMUP_STEPS, total_steps))

pad_id = SPECIAL_IDS["<pad>"]
# Downweights the <pad> class only; index-selected per target below rather than
# handed to F.cross_entropy, so one log_softmax yields both loss variants.
class_weight = torch.ones(encoder_params.vocab_size, device=device)
class_weight[pad_id] = PAD_LOSS_WEIGHT

start_epoch = 0
best_test_ce = float("inf")
global_step = 0

if RESUME_FROM is not None:
    _resumed_params, resume_ckpt = load_AE_Checkpoint(RESUME_FROM, device=device)
    AutoEncoder.load_state_dict(resume_ckpt["model_state_dict"])
    optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in resume_ckpt:
        scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
    start_epoch = resume_ckpt["epoch"] + 1
    best_test_ce = resume_ckpt.get("best_test_ce", float("inf"))
    global_step = resume_ckpt.get("global_step", start_epoch * steps_per_epoch)
    logger.info(f"Resumed from {RESUME_FROM}: epoch {start_epoch}, "
                f"step {global_step}, best test CE {best_test_ce:.6f}")
else:
    untrained_path = f"{output_dir}/PacketLevelAutoEncoder_{RUN_NAME}_untrained.ckpt"
    save_checkpoint(model=AutoEncoder, optimizer=optimizer, epoch=-1, loss=0,
                    config=autoencoderparams, path=untrained_path)
    logger.info(f"Saved PacketAutoencoder to {untrained_path}")

eval_batches = fixed_eval_batches(test_data.height, EVAL_BATCHES, eval_batch_size, SEED)
logger.info(f"Held-out eval: {len(eval_batches)} batches x {eval_batch_size} packets from test")

# --- Metric logging --------------------------------------------------------
metric_fields = ["epoch", "step", "lr",
                 "train_loss", "train_ce", "train_acc", "train_acc_nonpad",
                 "test_loss", "test_ce", "test_acc", "test_acc_nonpad"]
metrics = MetricsCsv(f"{output_dir}/metrics.csv", metric_fields, logger)


def autocast_ctx():
    if AMP_DTYPE is None:
        return torch.autocast("cuda", enabled=False)
    return torch.autocast("cuda", dtype=AMP_DTYPE)


def encode_batch(handler: PreTrainingDatasetHandler, rows: np.ndarray) -> torch.Tensor:
    bytes_, _proto = handler.get_pretraining_data(rows)
    input_ids = handler.InputIDEncoder.construct_input_ids(bytes_)
    return torch.tensor(input_ids, dtype=torch.long).to(device)


def reconstruction_losses(logits: torch.Tensor, input_ids: torch.Tensor) -> tuple:
    """(weighted loss to optimise, unweighted CE, all-position acc, non-pad acc).

    One log_softmax via reduction="none" gives both loss variants; computing them
    as two cross_entropy calls would double the largest tensor in the step.
    """
    flat_logits = logits.reshape(-1, logits.shape[-1]).float()
    flat_targets = input_ids.reshape(-1)
    per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")

    unweighted_ce = per_token.mean()
    w = class_weight[flat_targets]
    weighted_loss = (per_token * w).sum() / w.sum()

    correct = flat_logits.argmax(dim=-1) == flat_targets
    non_pad = flat_targets != pad_id
    acc = correct.float().mean()
    acc_nonpad = (correct & non_pad).sum() / non_pad.sum().clamp(min=1)
    return weighted_loss, unweighted_ce, acc, acc_nonpad


@torch.no_grad()
def evaluate() -> dict:
    """
    Reconstruction loss and accuracy on the fixed test subset.

    Accuracy is reported twice: over all 1520 positions and over non-pad positions
    only. Most packets are far shorter than 1520, so the all-position number is
    dominated by predicting <pad> and reads ~0.97 long before the model
    reconstructs the real bytes -- non-pad is the metric that means something.
    """
    AutoEncoder.eval()
    totals = {"test_loss": 0.0, "test_ce": 0.0, "test_acc": 0.0, "test_acc_nonpad": 0.0}
    for rows in eval_batches:
        input_ids = encode_batch(TestDataHandler, rows)
        with autocast_ctx():
            logits, _latent = AutoEncoder(input_ids)
        loss, ce, acc, acc_nonpad = reconstruction_losses(logits, input_ids)
        totals["test_loss"] += loss.item()
        totals["test_ce"] += ce.item()
        totals["test_acc"] += acc.item()
        totals["test_acc_nonpad"] += acc_nonpad.item()

    AutoEncoder.train()
    n = len(eval_batches)
    return {k: v / n for k, v in totals.items()}


def write_checkpoint(path: str, epoch: int, loss: float, extra: dict = None):
    payload = {"scheduler_state_dict": scheduler.state_dict(),
               "best_test_ce": best_test_ce, "global_step": global_step}
    if extra:
        payload.update(extra)
    save_checkpoint(model=AutoEncoder, optimizer=optimizer, epoch=epoch, loss=loss,
                    config=autoencoderparams, path=path, extra=payload)


AutoEncoder.train()

for epoch in range(start_epoch, Epochs):
    batches = DataHandler.sample_epoch_packet_indices(batch_size)
    if MAX_STEPS_PER_EPOCH is not None:
        batches = batches[:MAX_STEPS_PER_EPOCH]

    epoch_sums = np.zeros(4)     # loss, ce, acc, acc_nonpad
    window_sums = np.zeros(4)
    window_n = 0

    for index, rows in enumerate(batches):
        input_ids = encode_batch(DataHandler, rows)

        # Forward Pass
        with autocast_ctx():
            logits, _latent = AutoEncoder(input_ids)
        loss, ce, acc, acc_nonpad = reconstruction_losses(logits, input_ids)

        # Backward Pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(AutoEncoder.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        step_metrics = np.array([loss.item(), ce.item(), acc.item(), acc_nonpad.item()])
        epoch_sums += step_metrics
        window_sums += step_metrics
        window_n += 1

        if (index + 1) % log_every_n_batches == 0 or index == len(batches) - 1:
            w = window_sums / window_n
            logger.info(f"Epoch {epoch+1}/{Epochs} Batch {index+1}/{len(batches)} "
                        f"lr {scheduler.get_last_lr()[0]:.2e}")
            logger.info(f"Loss: {w[0]:.6f} CE: {w[1]:.6f} "
                        f"Accuracy: {w[2]:.4f} Accuracy(non-pad): {w[3]:.4f}")
            window_sums = np.zeros(4)
            window_n = 0

        if (index + 1) % eval_every_n_batches == 0:
            test_metrics = evaluate()
            logger.info(f"[test] Epoch {epoch+1}/{Epochs} Batch {index+1}/{len(batches)} "
                        f"Loss: {test_metrics['test_loss']:.6f} "
                        f"CE: {test_metrics['test_ce']:.6f} "
                        f"Accuracy: {test_metrics['test_acc']:.4f} "
                        f"Accuracy(non-pad): {test_metrics['test_acc_nonpad']:.4f}")
            # Cheap insurance: an interrupted multi-hour epoch resumes from here.
            # Stamped with epoch-1 because this epoch is unfinished and RESUME_FROM
            # starts at ckpt["epoch"] + 1 -- so resuming replays this epoch.
            latest_path = f"{output_dir}/PacketLevelAutoEncoder_{RUN_NAME}_latest.ckpt"
            write_checkpoint(latest_path, epoch - 1, float(epoch_sums[0] / (index + 1)))

    n = max(1, len(batches))
    e = epoch_sums / n
    train_metrics = {"train_loss": e[0], "train_ce": e[1],
                     "train_acc": e[2], "train_acc_nonpad": e[3]}

    test_metrics = evaluate()
    logger.info(f"===== Test Results (Epoch {epoch+1}) =====")
    logger.info(f"Test Loss: {test_metrics['test_loss']:.6f} "
                f"Test CE: {test_metrics['test_ce']:.6f}")
    logger.info(f"Test Accuracy: {test_metrics['test_acc']:.4f} "
                f"Test Accuracy(non-pad): {test_metrics['test_acc_nonpad']:.4f}")

    metrics.append({"epoch": epoch, "step": global_step,
                    "lr": scheduler.get_last_lr()[0], **train_metrics, **test_metrics})

    # Update the best *before* writing any checkpoint: write_checkpoint embeds
    # best_test_ce, and a resume from a checkpoint carrying a stale inf would
    # forget the best-so-far and overwrite _best.ckpt with a worse model.
    # Selected on the unweighted CE so the criterion is independent of PAD_LOSS_WEIGHT.
    is_best = test_metrics["test_ce"] < best_test_ce
    if is_best:
        best_test_ce = test_metrics["test_ce"]

    epoch_path = f"{output_dir}/PacketLevelAutoEncoder_{RUN_NAME}_E{epoch}.ckpt"
    write_checkpoint(epoch_path, epoch, train_metrics["train_loss"], extra=test_metrics)
    logger.info(f"Saved PacketAutoencoder to {epoch_path}")

    if is_best:
        best_path = f"{output_dir}/PacketLevelAutoEncoder_{RUN_NAME}_best.ckpt"
        write_checkpoint(best_path, epoch, train_metrics["train_loss"], extra=test_metrics)
        logger.info(f"New best test CE {best_test_ce:.6f}, saved to {best_path}")

plot_curves(
    metrics, f"{output_dir}/curves.png", x_field="epoch",
    panels=[
        {"title": "Reconstruction loss", "ylabel": "cross-entropy",
         "series": [("train_ce", "train CE"), ("test_ce", "test CE"),
                    ("train_loss", "train loss (pad-weighted)"),
                    ("test_loss", "test loss (pad-weighted)")]},
        {"title": "Byte accuracy", "ylabel": "accuracy", "ylim": (0, 1),
         "series": [("train_acc_nonpad", "train (non-pad)"),
                    ("test_acc_nonpad", "test (non-pad)"),
                    ("train_acc", "train (all)"), ("test_acc", "test (all)")]},
        {"title": "Learning rate", "ylabel": "lr", "series": [("lr", "lr")]},
    ],
    logger=logger,
)
