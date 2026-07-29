"""
Encode every packet of a split once with a frozen packet encoder and cache the
latents to disk.

Sequence-level training freezes the packet encoder, so re-running it every epoch
(batch_size x 65 forwards of 1520 tokens per step, through a python-loop CLS
pooling) recomputes a constant. This script pays that cost once per (split,
packet checkpoint) pair and writes:

    <cache_dir>/meta.json           run provenance + row count + shard size
    <cache_dir>/flow_offsets.parquet  flow_key -> (start, length) into the latents
    <cache_dir>/shard_XXXX.npy      float16 (rows, D) latents, in cache row order

Rows are written in flow-grouped, timestamp-sorted order, so every flow is a
contiguous slice and CachedLatentSequenceHandler can batch by gather.

Sharding makes the run resumable: a shard already on disk with the right shape
is skipped, so an interrupted pass over train.parquet does not start from zero.

Run from the repo root:
    python -m RawByteTrafficModelling.PreTraining.CachePacketLatents
"""
from RawByteTrafficModelling.ModelComponents.ModelDefinitions import (
    PacketAutoencoder,
    load_AE_Checkpoint,
)
from RawByteTrafficModelling.ModelComponents.DataUtils import (
    ID_Encoder,
    PreTrainingDatasetHandler,
    checkpoint_fingerprint,
)
import polars as pl
import numpy as np
import torch
import logging
import json
import os

### Set Cache Parameters
SPLIT_FILE = "data_artefacts/IIoTset-Ferrag/split/val.parquet"
PACKET_AE_CKPT = "RawByteTrafficModelling/PreTraining/TrainingOutputs/EdgeIIoT_AutoEncoder/PacketLevelAutoEncoder_EdgeIIoT_E0.ckpt"
CACHE_DIR = "data_artefacts/IIoTset-Ferrag/split/latents_EdgeIIoT_E0/val"

ENCODE_BATCH = 512
SHARD_ROWS = 1_000_000
# fp32 keeps the cache bit-comparable with a live encoder forward, which is what
# the equivalence check relies on. bfloat16 roughly halves the wall time.
ENCODE_DTYPE = torch.float32

SpecialIDs = {"<pad>": 256, "</s>": 257, "<CLS>": 258, "<mask>": 259, "<EndPointMasking>": 260, "<BOS>": 261}

os.makedirs(CACHE_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{CACHE_DIR}/CachePacketLatents.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

### Load data
data = pl.read_parquet(SPLIT_FILE)
logger.info(data.head())
logger.info(data.shape)

ID_Encoder = ID_Encoder(SpecialIDs=SpecialIDs, CLS_Placement="EOS")
DataHandler = PreTrainingDatasetHandler(data, 1, ID_Encoder)

device = torch.device("cuda")

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
flow_offsets.write_parquet(f"{CACHE_DIR}/flow_offsets.parquet")
logger.info(f"Wrote flow_offsets.parquet: {flow_offsets.height} flows, {data.height} packets")

# --- Load the frozen packet encoder ---------------------------------------
packet_ae_params, ckpt = load_AE_Checkpoint(PACKET_AE_CKPT)
packet_ae = PacketAutoencoder(packet_ae_params)
packet_ae.load_state_dict(ckpt["model_state_dict"])
packet_encoder = packet_ae.encoder.to(device).eval()
for p in packet_encoder.parameters():
    p.requires_grad = False

latent_dim = packet_ae_params.ENC_Params.EncoderDim
num_rows = data.height
num_shards = (num_rows + SHARD_ROWS - 1) // SHARD_ROWS
logger.info(f"Encoding {num_rows} packets -> {num_shards} shard(s) of {latent_dim}-dim latents")


@torch.no_grad()
def encode_rows(rows: np.ndarray) -> np.ndarray:
    """Original-parquet row indices -> (len(rows), D) float16 latents."""
    out = np.empty((rows.shape[0], latent_dim), dtype=np.float16)
    for i in range(0, rows.shape[0], ENCODE_BATCH):
        chunk = rows[i:i + ENCODE_BATCH]
        bytes_, _proto = DataHandler.get_pretraining_data(chunk)
        input_ids = DataHandler.InputIDEncoder.construct_input_ids(bytes_)
        input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
        if ENCODE_DTYPE == torch.float32:
            latents = packet_encoder(input_ids)
        else:
            with torch.autocast("cuda", dtype=ENCODE_DTYPE):
                latents = packet_encoder(input_ids)
        out[i:i + chunk.shape[0]] = latents.float().cpu().numpy().astype(np.float16)
    return out


for shard in range(num_shards):
    shard_path = f"{CACHE_DIR}/shard_{shard:04d}.npy"
    shard_start = shard * SHARD_ROWS
    shard_rows = row_order[shard_start:shard_start + SHARD_ROWS]

    if os.path.exists(shard_path):
        existing = np.load(shard_path, mmap_mode="r")
        if existing.shape == (shard_rows.shape[0], latent_dim):
            logger.info(f"Shard {shard}/{num_shards} already cached, skipping")
            continue
        logger.info(f"Shard {shard}/{num_shards} has wrong shape {existing.shape}, re-encoding")

    logger.info(f"Encoding shard {shard}/{num_shards} ({shard_rows.shape[0]} packets)")
    np.save(shard_path, encode_rows(shard_rows))
    logger.info(f"Saved {shard_path}")

with open(f"{CACHE_DIR}/meta.json", "w") as f:
    json.dump({
        "split_file": SPLIT_FILE,
        "packet_ae_ckpt": PACKET_AE_CKPT,
        # Ties the cache to these exact weights; load_latent_cache refuses a
        # mismatch, so retraining the packet model can't silently reuse stale latents.
        "packet_ae_sha256": checkpoint_fingerprint(PACKET_AE_CKPT),
        "latent_dim": latent_dim,
        "num_rows": int(num_rows),
        "num_flows": int(flow_offsets.height),
        "shard_rows": SHARD_ROWS,
        "num_shards": num_shards,
        "dtype": "float16",
    }, f, indent=2)
logger.info(f"Wrote {CACHE_DIR}/meta.json -- cache complete")
