"""
Packet-level masked language model pretraining.

A fraction of each packet's byte tokens is masked and reconstructed from context,
with an auxiliary protocol-hierarchy classification loss on the pooled CLS vector.
This is the alternative to PacketLevelAutoEncoder for producing a packet encoder;
both are trained on the same flow-grouped split at the same width so the two
backbones are comparable.

Run from the repo root:
    python -m RawByteTrafficModelling.PreTraining.PacketLevelMLM

Note Packet_MLM builds its own embedding + backbone rather than wrapping
Packet_Encoder, so its weights are not directly loadable into the sequence level;
the encoder spec is shared (RunConfig.packet_encoder_params) so the shapes match.
"""
from RawByteTrafficModelling.ModelComponents.ModelDefinitions import (
    MLM_Params,
    Packet_MLM,
    load_MLM_checkpoint,
    save_checkpoint,
)
from RawByteTrafficModelling.ModelComponents.DataUtils import PreTrainingDatasetHandler
from RawByteTrafficModelling.PreTraining.RunConfig import (
    DATASETS,
    PACKET_ID_LEN,
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
# keras_hub drags in TensorFlow, which by default preallocates nearly all memory on
# *every* visible GPU -- including cuda:0, where the concurrent PacketLevelAutoEncoder
# run lives. MaskedLMMaskGenerator is integer bookkeeping over a numpy array, so pin
# TF to the CPU before it initialises. Must happen before the keras_hub import.
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
from keras_hub.layers import MaskedLMMaskGenerator
import polars as pl
import numpy as np
import torch
import torch.nn.functional as F
import math

### Set Training Parameters
RUN_NAME = "PacketMLM_IIoTset_d128"
DATASET = DATASETS["IIoTset-Ferrag"]
DEVICE_INDEX = 1               # MLM on cuda:1, alongside the AE run on cuda:0

ENCODER_DIM = 128              # matches PacketLevelAutoEncoder
ENCODER_LAYERS = 2

# Measured ~2.4 s/step at bs=512 -> ~9 h per 13.3k-step epoch on one A5000.
# The cosine schedule spans the whole run, so changing this changes the LR curve.
Epochs = 2
batch_size = 512
learning_rate = 1e-3
weight_decay = 1e-2
WARMUP_STEPS = 1000
GRAD_CLIP = 1.0
AMP_DTYPE = torch.bfloat16     # None disables autocast

# The proto-hierarchy head reaches ~1.00 accuracy within a few hundred steps in
# every historical run, so weighting it 4x (as before) mostly scales a solved loss.
alpha_proto = 1.0
alpha_reconstruction = 1.0

MASK_SELECTION_RATE = 0.10
MASK_TOKEN_RATE = 0.9
RANDOM_TOKEN_RATE = 0.1

log_every_n_batches = 200
eval_every_n_batches = 2000
eval_batch_size = 256
EVAL_BATCHES = 40
SEED = 42

RESUME_FROM = None
MAX_STEPS_PER_EPOCH = None     # None = a real, full epoch

SMOKE = False
if SMOKE:
    RUN_NAME = f"{RUN_NAME}_smoke"
    Epochs = 2
    MAX_STEPS_PER_EPOCH = 30
    log_every_n_batches = 10
    eval_every_n_batches = 20
    EVAL_BATCHES = 4
    WARMUP_STEPS = 5

output_dir, logger = setup_run(RUN_NAME, "PacketLevelMLM.log")
logger.info(f"Run {RUN_NAME} (SMOKE={SMOKE}) -> {output_dir}")

torch.manual_seed(SEED)
device = resolve_device(DEVICE_INDEX)
logger.info(f"Device: {device} ({torch.cuda.get_device_name(device)})")

### Load data
train_data = pl.read_parquet(DATASET.train)
test_data = pl.read_parquet(DATASET.test)
logger.info(train_data.head())
logger.info(f"train: {train_data.height} packets from {DATASET.train}")
logger.info(f"test:  {test_data.height} packets from {DATASET.test}")

ID_Encoder = make_id_encoder()
DataHandler = PreTrainingDatasetHandler(train_data, 1, ID_Encoder)
TestDataHandler = PreTrainingDatasetHandler(test_data, 1, ID_Encoder)

# --- Protocol-hierarchy labels ---------------------------------------------
# Classes come from train only, and anything unseen maps to PROTO_IGNORE so it is
# excluded from the loss rather than silently scored against class 0. (The old
# OneHotEncoder path raised on the first unseen category in test.)
PROTO_IGNORE = -100
proto_classes = sorted(train_data["proto_hierarchy"].unique().to_list())
proto_to_idx = {name: i for i, name in enumerate(proto_classes)}
logger.info(f"proto_hierarchy classes in train: {len(proto_classes)}")
unseen_in_test = set(test_data["proto_hierarchy"].unique().to_list()) - set(proto_classes)
if unseen_in_test:
    logger.warning(f"{len(unseen_in_test)} proto_hierarchy value(s) in test are absent "
                   f"from train and will be ignored in the CLS loss: {sorted(unseen_in_test)}")


def proto_labels(protos: list[str]) -> torch.Tensor:
    return torch.tensor([proto_to_idx.get(p, PROTO_IGNORE) for p in protos],
                        dtype=torch.long, device=device)


# --- Model Config ---
ENCparams = packet_encoder_params(dim=ENCODER_DIM, num_layers=ENCODER_LAYERS)
MLM_params = MLM_Params(EncoderParams=ENCparams, NumCLSclasses=len(proto_classes))

MaskedLanguageModel = Packet_MLM(MLM_params).to(device)
logger.info(f"Parameters: {sum(p.numel() for p in MaskedLanguageModel.parameters())}")

optimizer = torch.optim.AdamW(MaskedLanguageModel.parameters(), lr=learning_rate,
                              weight_decay=weight_decay)

steps_per_epoch = math.ceil(train_data.height / batch_size)
if MAX_STEPS_PER_EPOCH is not None:
    steps_per_epoch = min(steps_per_epoch, MAX_STEPS_PER_EPOCH)
total_steps = max(1, Epochs * steps_per_epoch)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, cosine_warmup_lambda(WARMUP_STEPS, total_steps))

# Structural tokens are never masking targets: masking <pad> would make the task
# trivial, and masking <CLS> would break DynamicCLSPooling's ability to find it.
unselectable_token_ids = [SPECIAL_IDS["</s>"], SPECIAL_IDS["<pad>"],
                          SPECIAL_IDS["<CLS>"], SPECIAL_IDS["<EndPointMasking>"]]

Masker = MaskedLMMaskGenerator(
    vocabulary_size=ENCparams.vocab_size,
    mask_token_id=SPECIAL_IDS["<mask>"],
    mask_selection_length=int(PACKET_ID_LEN * 0.25),
    mask_selection_rate=MASK_SELECTION_RATE,
    mask_token_rate=MASK_TOKEN_RATE,
    random_token_rate=RANDOM_TOKEN_RATE,
    unselectable_token_ids=unselectable_token_ids,
)

start_epoch = 0
best_test_ce = float("inf")
global_step = 0

if RESUME_FROM is not None:
    _resumed_params, resume_ckpt = load_MLM_checkpoint(RESUME_FROM, device=device)
    MaskedLanguageModel.load_state_dict(resume_ckpt["model_state_dict"])
    optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in resume_ckpt:
        scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
    start_epoch = resume_ckpt["epoch"] + 1
    best_test_ce = resume_ckpt.get("best_test_ce", float("inf"))
    global_step = resume_ckpt.get("global_step", start_epoch * steps_per_epoch)
    logger.info(f"Resumed from {RESUME_FROM}: epoch {start_epoch}, "
                f"step {global_step}, best test CE {best_test_ce:.6f}")
else:
    untrained_path = f"{output_dir}/PacketLevelMLM_{RUN_NAME}_untrained.ckpt"
    save_checkpoint(model=MaskedLanguageModel, optimizer=optimizer, epoch=-1, loss=0,
                    config=MLM_params, path=untrained_path)
    logger.info(f"Saved Packet_MLM to {untrained_path}")

eval_batches = fixed_eval_batches(test_data.height, EVAL_BATCHES, eval_batch_size, SEED)
logger.info(f"Held-out eval: {len(eval_batches)} batches x {eval_batch_size} packets from test")

# --- Metric logging --------------------------------------------------------
metric_fields = ["epoch", "step", "lr",
                 "train_total", "train_recon", "train_recon_acc",
                 "train_proto", "train_proto_acc",
                 "test_total", "test_recon", "test_recon_acc",
                 "test_proto", "test_proto_acc"]
metrics = MetricsCsv(f"{output_dir}/metrics.csv", metric_fields, logger)


def autocast_ctx():
    if AMP_DTYPE is None:
        return torch.autocast("cuda", enabled=False)
    return torch.autocast("cuda", dtype=AMP_DTYPE)


def masked_batch(handler: PreTrainingDatasetHandler, rows: np.ndarray) -> tuple:
    """Rows -> (masked token ids, mask positions, original ids at those positions,
    per-position weights, proto labels).

    Uses the generator's own mask_positions/mask_ids rather than re-deriving them
    from `tokens == <mask>`: with random_token_rate=0.1 a tenth of the selected
    positions carry a random token instead of <mask>, and the old comparison
    silently dropped those from the loss.
    """
    bytes_, protos = handler.get_pretraining_data(rows)
    input_ids = handler.InputIDEncoder.construct_input_ids(bytes_)
    masked = Masker(input_ids)
    return (
        torch.tensor(np.asarray(masked["token_ids"]), dtype=torch.long, device=device),
        torch.tensor(np.asarray(masked["mask_positions"]), dtype=torch.long, device=device),
        torch.tensor(np.asarray(masked["mask_ids"]), dtype=torch.long, device=device),
        torch.tensor(np.asarray(masked["mask_weights"]), dtype=torch.float32, device=device),
        proto_labels(protos),
    )


def mlm_losses(recon_logits, cls_logits, mask_positions, mask_ids, mask_weights,
               labels) -> dict:
    """Weighted masked-token CE + proto CE, with accuracies."""
    vocab = recon_logits.shape[-1]
    # Gather only the masked positions: (B, S, V) rather than (B, 1520, V).
    selected = recon_logits.gather(
        1, mask_positions.unsqueeze(-1).expand(-1, -1, vocab)).float()

    flat_weights = mask_weights.reshape(-1)
    weight_sum = flat_weights.sum().clamp(min=1.0)
    per_token = F.cross_entropy(selected.reshape(-1, vocab), mask_ids.reshape(-1),
                                reduction="none")
    recon_loss = (per_token * flat_weights).sum() / weight_sum
    recon_correct = (selected.argmax(dim=-1) == mask_ids).float().reshape(-1)
    recon_acc = (recon_correct * flat_weights).sum() / weight_sum

    cls_logits = cls_logits.float()
    labelled = labels != PROTO_IGNORE
    if labelled.any():
        proto_loss = F.cross_entropy(cls_logits, labels, ignore_index=PROTO_IGNORE)
        proto_acc = ((cls_logits.argmax(dim=-1) == labels) & labelled).sum() / labelled.sum()
    else:
        proto_loss = torch.zeros((), device=device)
        proto_acc = torch.zeros((), device=device)

    return {"total": alpha_reconstruction * recon_loss + alpha_proto * proto_loss,
            "recon": recon_loss, "recon_acc": recon_acc,
            "proto": proto_loss, "proto_acc": proto_acc}


@torch.no_grad()
def evaluate() -> dict:
    """Averaged over the whole fixed test subset -- the old version broke out after
    one batch while still dividing by the batch count, so its numbers were ~0."""
    MaskedLanguageModel.eval()
    keys = ["total", "recon", "recon_acc", "proto", "proto_acc"]
    totals = {k: 0.0 for k in keys}
    for rows in eval_batches:
        tokens, positions, ids, weights, labels = masked_batch(TestDataHandler, rows)
        with autocast_ctx():
            recon_logits, cls_logits = MaskedLanguageModel(tokens)
        losses = mlm_losses(recon_logits, cls_logits, positions, ids, weights, labels)
        for k in keys:
            totals[k] += losses[k].item()

    MaskedLanguageModel.train()
    n = len(eval_batches)
    return {f"test_{k}": v / n for k, v in totals.items()}


def write_checkpoint(path: str, epoch: int, loss: float, extra: dict = None):
    payload = {"scheduler_state_dict": scheduler.state_dict(),
               "best_test_ce": best_test_ce, "global_step": global_step}
    if extra:
        payload.update(extra)
    save_checkpoint(model=MaskedLanguageModel, optimizer=optimizer, epoch=epoch,
                    loss=loss, config=MLM_params, path=path, extra=payload)


MaskedLanguageModel.train()
metric_keys = ["total", "recon", "recon_acc", "proto", "proto_acc"]

for epoch in range(start_epoch, Epochs):
    batches = DataHandler.sample_epoch_packet_indices(batch_size)
    if MAX_STEPS_PER_EPOCH is not None:
        batches = batches[:MAX_STEPS_PER_EPOCH]

    epoch_sums = np.zeros(len(metric_keys))
    window_sums = np.zeros(len(metric_keys))
    window_n = 0

    for index, rows in enumerate(batches):
        tokens, positions, ids, weights, labels = masked_batch(DataHandler, rows)

        # Forward Pass
        with autocast_ctx():
            recon_logits, cls_logits = MaskedLanguageModel(tokens)
        losses = mlm_losses(recon_logits, cls_logits, positions, ids, weights, labels)

        # Backward Pass
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(MaskedLanguageModel.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        step_metrics = np.array([losses[k].item() for k in metric_keys])
        epoch_sums += step_metrics
        window_sums += step_metrics
        window_n += 1

        if (index + 1) % log_every_n_batches == 0 or index == len(batches) - 1:
            w = window_sums / window_n
            logger.info(f"Epoch {epoch+1}/{Epochs} Batch {index+1}/{len(batches)} "
                        f"lr {scheduler.get_last_lr()[0]:.2e}")
            logger.info(f"Total Loss: {w[0]:.6f}")
            logger.info(f"Reconstruction Loss: {w[1]:.6f} "
                        f"Reconstruction Accuracy: {w[2]:.4f}")
            logger.info(f"ProtoHierarchy Loss: {w[3]:.6f} "
                        f"ProtoHierarchy Accuracy: {w[4]:.4f}")
            window_sums = np.zeros(len(metric_keys))
            window_n = 0

        if (index + 1) % eval_every_n_batches == 0:
            test_metrics = evaluate()
            logger.info(f"[test] Epoch {epoch+1}/{Epochs} Batch {index+1}/{len(batches)} "
                        f"Total: {test_metrics['test_total']:.6f} "
                        f"Recon: {test_metrics['test_recon']:.6f} "
                        f"Recon Acc: {test_metrics['test_recon_acc']:.4f} "
                        f"Proto Acc: {test_metrics['test_proto_acc']:.4f}")
            # Stamped with epoch-1: this epoch is unfinished and RESUME_FROM starts
            # at ckpt["epoch"] + 1, so resuming replays it.
            latest_path = f"{output_dir}/PacketLevelMLM_{RUN_NAME}_latest.ckpt"
            write_checkpoint(latest_path, epoch - 1, float(epoch_sums[0] / (index + 1)))

    n = max(1, len(batches))
    e = epoch_sums / n
    train_metrics = {"train_total": e[0], "train_recon": e[1], "train_recon_acc": e[2],
                     "train_proto": e[3], "train_proto_acc": e[4]}

    test_metrics = evaluate()
    logger.info(f"===== Test Results (Epoch {epoch+1}) =====")
    logger.info(f"Test Total Loss: {test_metrics['test_total']:.6f}")
    logger.info(f"Test Reconstruction Loss: {test_metrics['test_recon']:.6f} "
                f"Test Reconstruction Accuracy: {test_metrics['test_recon_acc']:.4f}")
    logger.info(f"Test ProtoHierarchy Loss: {test_metrics['test_proto']:.6f} "
                f"Test ProtoHierarchy Accuracy: {test_metrics['test_proto_acc']:.4f}")

    metrics.append({"epoch": epoch, "step": global_step,
                    "lr": scheduler.get_last_lr()[0], **train_metrics, **test_metrics})

    # Update the best *before* writing any checkpoint: write_checkpoint embeds
    # best_test_ce, and a resume from a checkpoint carrying a stale inf would
    # forget the best-so-far and overwrite _best.ckpt with a worse model.
    # Masked-token CE, not the total: the proto term is a solved auxiliary task and
    # should not decide which encoder is kept.
    is_best = test_metrics["test_recon"] < best_test_ce
    if is_best:
        best_test_ce = test_metrics["test_recon"]

    epoch_path = f"{output_dir}/PacketLevelMLM_{RUN_NAME}_E{epoch}.ckpt"
    write_checkpoint(epoch_path, epoch, train_metrics["train_total"], extra=test_metrics)
    logger.info(f"Saved Packet_MLM to {epoch_path}")

    if is_best:
        best_path = f"{output_dir}/PacketLevelMLM_{RUN_NAME}_best.ckpt"
        write_checkpoint(best_path, epoch, train_metrics["train_total"], extra=test_metrics)
        logger.info(f"New best test reconstruction CE {best_test_ce:.6f}, "
                    f"saved to {best_path}")

plot_curves(
    metrics, f"{output_dir}/curves.png", x_field="epoch",
    panels=[
        {"title": "Losses", "ylabel": "cross-entropy",
         "series": [("train_recon", "train masked-token"), ("test_recon", "test masked-token"),
                    ("train_proto", "train proto"), ("test_proto", "test proto"),
                    ("train_total", "train total")]},
        {"title": "Accuracy", "ylabel": "accuracy", "ylim": (0, 1),
         "series": [("train_recon_acc", "train masked-token"),
                    ("test_recon_acc", "test masked-token"),
                    ("train_proto_acc", "train proto"), ("test_proto_acc", "test proto")]},
        {"title": "Learning rate", "ylabel": "lr", "series": [("lr", "lr")]},
    ],
    logger=logger,
)
