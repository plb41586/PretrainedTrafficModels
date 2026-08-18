"""
Encode every packet of a split once with a frozen packet encoder and cache the
latents to disk.

Sequence-level training freezes the packet encoder, so re-running it every epoch
(batch_size x 65 forwards of 1520 tokens per step, through a python-loop CLS
pooling) recomputes a constant. This script pays that cost once per (split,
packet checkpoint) pair and writes, per split:

    <cache_dir>/meta.json           run provenance + row count + shard size
    <cache_dir>/flow_offsets.parquet  flow_key -> (start, length) into the latents
    <cache_dir>/shard_XXXX.npy      float16 (rows, D) latents, in cache row order

Rows are written in flow-grouped, timestamp-sorted order, so every flow is a
contiguous slice and CachedLatentSequenceHandler can batch by gather.

Sharding makes the run resumable: a shard already on disk with the right shape
is skipped, so an interrupted pass over train.parquet does not start from zero.

A cache is only valid for the exact weights it was built from -- meta.json stores
a sha256 of the checkpoint and load_latent_cache refuses a mismatch. Re-extracting
a capture also changes its flow_key strings, so a split and its caches must be
regenerated together.

Run from the repo root:
    python -m RawByteTrafficModelling.PreTraining.CachePacketLatents
"""
from RawByteTrafficModelling.ModelComponents.ModelDefinitions import (
    PacketAutoencoder,
    load_AE_Checkpoint,
)
from RawByteTrafficModelling.ModelComponents.DataUtils import (
    PreTrainingDatasetHandler,
    checkpoint_fingerprint,
)
from RawByteTrafficModelling.PreTraining.RunConfig import (
    DATASETS,
    make_id_encoder,
    resolve_device,
)
import polars as pl
import numpy as np
import torch
import logging
import json
import os

### Set Cache Parameters
DATASET = DATASETS["IIoTset-Ferrag"]
PACKET_AE_CKPT = ("RawByteTrafficModelling/PreTraining/TrainingOutputs/PacketAE_IIoTset_d128/"
                  "PacketLevelAutoEncoder_PacketAE_IIoTset_d128_best.ckpt")
# Names the cache directory: flow_split/latents_<TAG>/<split>. Change it whenever
# PACKET_AE_CKPT changes, so two generations of cache never share a path.
CACHE_TAG = "PacketAE_d128_best"
# The two splits the sequence level trains on. val is deliberately absent -- it is
# the final held-out set, and nothing that trains should have a cache for it.
SPLITS = ["train", "test"]
DEVICE_INDEX = 0

ENCODE_BATCH = 512
SHARD_ROWS = 1_000_000
# fp32 keeps the cache bit-comparable with a live encoder forward, which is what
# the equivalence check in CheckLatentCache relies on. bfloat16 roughly halves the
# wall time and breaks that check.
ENCODE_DTYPE = torch.float32

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

device = resolve_device(DEVICE_INDEX)
logger.info(f"Device: {device} ({torch.cuda.get_device_name(device)})")

# --- Load the frozen packet encoder ---------------------------------------
packet_ae_params, ckpt = load_AE_Checkpoint(PACKET_AE_CKPT)
packet_ae = PacketAutoencoder(packet_ae_params)
packet_ae.load_state_dict(ckpt["model_state_dict"])
packet_encoder = packet_ae.encoder.to(device).eval()
for p in packet_encoder.parameters():
    p.requires_grad = False

latent_dim = packet_ae_params.ENC_Params.EncoderDim
ckpt_sha = checkpoint_fingerprint(PACKET_AE_CKPT)
logger.info(f"Packet encoder from {PACKET_AE_CKPT} (epoch {ckpt.get('epoch')}, "
            f"latent_dim {latent_dim}, sha256 {ckpt_sha[:12]})")

ID_Encoder = make_id_encoder()


@torch.no_grad()
def encode_rows(handler: PreTrainingDatasetHandler, rows: np.ndarray) -> np.ndarray:
    """Original-parquet row indices -> (len(rows), D) float16 latents."""
    out = np.empty((rows.shape[0], latent_dim), dtype=np.float16)
    for i in range(0, rows.shape[0], ENCODE_BATCH):
        chunk = rows[i:i + ENCODE_BATCH]
        bytes_, _proto = handler.get_pretraining_data(chunk)
        input_ids = handler.InputIDEncoder.construct_input_ids(bytes_)
        input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
        if ENCODE_DTYPE == torch.float32:
            latents = packet_encoder(input_ids)
        else:
            with torch.autocast("cuda", dtype=ENCODE_DTYPE):
                latents = packet_encoder(input_ids)
        out[i:i + chunk.shape[0]] = latents.float().cpu().numpy().astype(np.float16)
    return out


def cache_split(split: str):
    split_file = getattr(DATASET, split)
    cache_dir = DATASET.latent_cache(CACHE_TAG, split)
    os.makedirs(cache_dir, exist_ok=True)
    logger.info(f"===== {split}: {split_file} -> {cache_dir}")

    data = pl.read_parquet(split_file)
    logger.info(f"{data.height} packets")
    DataHandler = PreTrainingDatasetHandler(data, 1, ID_Encoder)

    # --- Flow index -> cache row order ----------------------------------------
    # Every flow becomes a contiguous slice of the latent array, so the training
    # handler only ever needs (start, length) per flow.
    logger.info("Building flow index (sorts the whole split, this takes a while)")
    flow_index = DataHandler.build_flow_index()
    row_order = np.concatenate([r.to_numpy() for r in flow_index["row_idx"]]).astype(np.int64)
    lengths = flow_index["row_idx"].list.len().to_numpy().astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)

    assert row_order.shape[0] == data.height, "flow index lost rows"
    assert np.unique(row_order).shape[0] == data.height, "flow index duplicated rows"

    flow_offsets = pl.DataFrame({
        "flow_key": flow_index["flow_key"],
        "start": starts,
        "length": lengths,
    })
    flow_offsets.write_parquet(f"{cache_dir}/flow_offsets.parquet")
    logger.info(f"Wrote flow_offsets.parquet: {flow_offsets.height} flows, "
                f"{data.height} packets")

    num_rows = data.height
    num_shards = (num_rows + SHARD_ROWS - 1) // SHARD_ROWS
    logger.info(f"Encoding {num_rows} packets -> {num_shards} shard(s) "
                f"of {latent_dim}-dim latents")

    for shard in range(num_shards):
        shard_path = f"{cache_dir}/shard_{shard:04d}.npy"
        shard_start = shard * SHARD_ROWS
        shard_rows = row_order[shard_start:shard_start + SHARD_ROWS]

        if os.path.exists(shard_path):
            existing = np.load(shard_path, mmap_mode="r")
            if existing.shape == (shard_rows.shape[0], latent_dim):
                logger.info(f"Shard {shard}/{num_shards} already cached, skipping")
                continue
            logger.info(f"Shard {shard}/{num_shards} has wrong shape "
                        f"{existing.shape}, re-encoding")

        logger.info(f"Encoding shard {shard}/{num_shards} ({shard_rows.shape[0]} packets)")
        np.save(shard_path, encode_rows(DataHandler, shard_rows))
        logger.info(f"Saved {shard_path}")

    with open(f"{cache_dir}/meta.json", "w") as f:
        json.dump({
            "split_file": split_file,
            "packet_ae_ckpt": PACKET_AE_CKPT,
            # Ties the cache to these exact weights; load_latent_cache refuses a
            # mismatch, so retraining the packet model can't silently reuse stale
            # latents.
            "packet_ae_sha256": ckpt_sha,
            "latent_dim": latent_dim,
            "num_rows": int(num_rows),
            "num_flows": int(flow_offsets.height),
            "shard_rows": SHARD_ROWS,
            "num_shards": num_shards,
            "dtype": "float16",
        }, f, indent=2)
    logger.info(f"Wrote {cache_dir}/meta.json -- {split} cache complete")


for split in SPLITS:
    cache_split(split)
logger.info(f"All caches complete: {SPLITS}")
