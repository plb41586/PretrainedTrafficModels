"""
Export sequence-level (flow) embeddings for UMAP inspection.

Runs the trained Sequence_Encoder over deterministic packet windows and dumps the
flow bottleneck vector z, one .npy per set, into a single directory. The filename
stem is the legend label, which is the contract Embedding_Viz.ipynb / embedding_viz.py
already expect (they glob *.npy and label by stem).

The point of the exercise: keep `train` as a dense background in the UMAP plot and
blend `test` / `val` / the per-attack sets in and out, to see whether attack flows sit
further from the training manifold than held-out normal flows do.

Two source paths, because only some splits have a packet-latent cache:
  - train / test  -> cached latents (free, no packet forward)
  - val + attacks -> raw tokens, encoded through the frozen packet encoder that rides
                     inside the sequence-AE checkpoint
Both feed the same Sequence_Encoder. The token path only encodes *real* packets and
leaves padding slots as zeros, which is exactly what the cache holds -- CROSS_CHECK
below verifies the two paths agree before any of the attack sets are written.

Run from the repo root:
    python -m RawByteTrafficModelling.AnomalyDetection.SequenceEmbeddingAD
"""
from RawByteTrafficModelling.ModelComponents.ModelDefinitions import (
    SequenceAutoencoder,
    load_SeqAE_checkpoint,
)
from RawByteTrafficModelling.ModelComponents.DataUtils import (
    CachedLatentSequenceHandler,
    ID_Encoder,
    PreTrainingDatasetHandler,
    load_latent_cache,
)
from RawByteTrafficModelling.PreTraining.RunConfig import DATASETS, resolve_device
import polars as pl
import numpy as np
import torch
import logging
import os

### Set Export Parameters
# The sequence-AE run to export. s512 was chosen from the width sweep: it reaches 96.7%
# of the packet-decoder byte-accuracy ceiling at 45% of s768's parameters.
RUN_NAME = "SeqAE_IIoTset_d128_Mamba_s512"
SEQ_AE_CKPT = (f"RawByteTrafficModelling/PreTraining/TrainingOutputs/{RUN_NAME}/"
               f"SequenceLevelAutoEncoder_{RUN_NAME}_best.ckpt")
output_dir = f"RawByteTrafficModelling/AnomalyDetection/Outputs/Embeddings/SequenceEmbeddings_{RUN_NAME}"

DATASET = DATASETS["IIoTset-Ferrag"]
CACHE_TAG = "PacketAE_d128_best"       # must match CachePacketLatents

# Sets that already have a packet-latent cache from CachePacketLatents.
CACHED_SETS = [
    ("train", DATASET.latent_cache(CACHE_TAG, "train")),
    ("test",  DATASET.latent_cache(CACHE_TAG, "test")),
]
# Sets that have no cache and must be encoded from bytes.
TOKEN_SETS = [
    ("val", DATASET.val),
]
# One .npy per attacks/*.parquet, stem = attack name. Attack labelling in this dataset
# is by filename -- no parquet carries an AttackLabel column (see TODO.md).
ATTACK_DIR = DATASET.attacks

PACKETS_PER_SEQUENCE = 65      # P, including the slot the seq-CLS overwrites
MAX_WINDOWS_PER_SET = 20_000   # evenly spaced subset; the viz subsamples again anyway
WINDOW_BATCH = 64              # windows per Sequence_Encoder forward
PACKET_BATCH = 512             # packets per packet-encoder forward (matches CachePacketLatents)

# Sanity check the token path against the cached path on `test`, which has both.
# 0 disables. A mismatch here means the flow index no longer reproduces the cache's
# row order, which would silently corrupt every token-path set.
CROSS_CHECK_WINDOWS = 32
CROSS_CHECK_SPLIT = DATASET.test
CROSS_CHECK_TOL = 1e-2

# Wiring test: tiny caps, two attack files, and a separate output dir so a smoke run
# never leaves truncated .npy files sitting where the real ones belong.
SMOKE = False

if SMOKE:
    MAX_WINDOWS_PER_SET = 200
    output_dir = f"{output_dir}_smoke"

os.makedirs(output_dir, exist_ok=True)   # logging.FileHandler below needs it to exist

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{output_dir}/SequenceEmbeddingAD.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

device = resolve_device(0)

### Load the trained sequence autoencoder
# The packet encoder's weights ride inside this state dict (encoder.packet_encoder.*),
# so the packet-AE checkpoint is not needed separately.
seq_params, seq_ckpt = load_SeqAE_checkpoint(SEQ_AE_CKPT)
model = SequenceAutoencoder(seq_params)
model.load_state_dict(seq_ckpt["model_state_dict"])
model = model.to(device).eval()

assert seq_params.SeqEncParams.packets_per_sequence == PACKETS_PER_SEQUENCE, (
    f"checkpoint was trained with P={seq_params.SeqEncParams.packets_per_sequence}, "
    f"this script is configured for {PACKETS_PER_SEQUENCE}")

SEQ_LEN = PACKETS_PER_SEQUENCE - 1                      # max real packets per window
LATENT_DIM = seq_params.SeqEncParams.EncoderParams.EncoderDim
packet_encoder = model.encoder.packet_encoder
byte_encoder = ID_Encoder(SpecialIDs=seq_params.SeqEncParams.EncoderParams.SpecialTokens,
                          CLS_Placement="EOS")          # must match CachePacketLatents
logger.info(f"Loaded {SEQ_AE_CKPT} (epoch {seq_ckpt.get('epoch')}), "
            f"packet latent dim {LATENT_DIM}, seq dim {seq_params.SeqEncParams.SeqEncoderDim}")


def enumerate_windows(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """(flow, start, length) rows -- same chopping as CachedLatentSequenceHandler.

    Non-overlapping windows of SEQ_LEN packets per flow with the remainder dropped;
    a flow shorter than SEQ_LEN yields one short window. The flow index rides along
    so each window can be traced back to its flow_key.
    """
    windows = []
    for flow, (start, length) in enumerate(zip(starts, lengths)):
        num_sequences = length // SEQ_LEN
        if num_sequences == 0:
            windows.append((flow, start, length))
        else:
            for i in range(num_sequences):
                windows.append((flow, start + i * SEQ_LEN, SEQ_LEN))
    return np.array(windows, dtype=np.int64)


def pick_windows(windows: np.ndarray) -> np.ndarray:
    """Evenly spaced deterministic subset, so the subset spans the whole split."""
    if windows.shape[0] <= MAX_WINDOWS_PER_SET:
        return windows
    idx = np.linspace(0, windows.shape[0] - 1, MAX_WINDOWS_PER_SET).astype(np.int64)
    return windows[idx]


@torch.no_grad()
def encode_windows(latents: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
    """(B, P, D) packet latents -> (B, D_seq) flow vectors."""
    return model.encode(seq_lens.to(device), latents=latents.to(device)).float().cpu()


def flow_offsets_from_index(flow_index: pl.DataFrame):
    """Cache-equivalent (row_order, starts, lengths) for a parquet split.

    Identical construction to CachePacketLatents, so a window's (start, length) means
    the same thing on the token path as it does inside a latent cache.
    """
    row_order = np.concatenate([r.to_numpy() for r in flow_index["row_idx"]]).astype(np.int64)
    lengths = flow_index["row_idx"].list.len().to_numpy().astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    return row_order, starts, lengths


@torch.no_grad()
def token_latent_batch(handler: PreTrainingDatasetHandler, row_order: np.ndarray,
                       windows: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode the real packets of each window into (B, P, D), padding slots left zero.

    Zeros are the right padding value: DynamicCLSPooling returns a zero vector for an
    all-<pad> packet, which is what the cached path stores too.
    """
    B = windows.shape[0]
    takes = windows[:, 2]
    src = np.concatenate([row_order[s:s + t] for _, s, t in windows])
    dest_b = np.repeat(np.arange(B), takes)
    dest_p = np.concatenate([np.arange(t) for t in takes])

    out = torch.zeros(B, PACKETS_PER_SEQUENCE, LATENT_DIM)
    for i in range(0, src.shape[0], PACKET_BATCH):
        rows = src[i:i + PACKET_BATCH]
        window_bytes, _proto = handler.get_pretraining_data(rows)
        input_ids = torch.tensor(handler.InputIDEncoder.construct_input_ids(window_bytes),
                                 dtype=torch.long).to(device)
        out[torch.from_numpy(dest_b[i:i + rows.shape[0]]).long(),
            torch.from_numpy(dest_p[i:i + rows.shape[0]]).long()] = \
            packet_encoder(input_ids).float().cpu()
    return out, torch.from_numpy(takes.copy()).long()


def log_length_profile(label: str, windows: np.ndarray):
    """Flow-length profile per set -- a 'cluster' can just be 'these are 1-packet flows'."""
    lens = windows[:, 2]
    logger.info(f"  {label}: {windows.shape[0]} windows, seq_len mean {lens.mean():.1f} "
                f"median {int(np.median(lens))} min {lens.min()} max {lens.max()}, "
                f"{(lens == 1).sum()} single-packet")


def embed_cached(label: str, cache_dir: str):
    """Sequence embeddings for a split that already has a packet-latent cache."""
    latents, flow_offsets, _meta = load_latent_cache(cache_dir, meta_only_ckpt(cache_dir))
    handler = CachedLatentSequenceHandler(latents, flow_offsets, PACKETS_PER_SEQUENCE)
    windows = pick_windows(enumerate_windows(handler.starts, handler.lengths))
    log_length_profile(label, windows)

    flow_keys = handler.flow_keys
    Z = []
    for i in range(0, windows.shape[0], WINDOW_BATCH):
        chunk = windows[i:i + WINDOW_BATCH]
        lat, seq_lens = handler.latent_batch_from_windows(chunk[:, 1:3])
        Z.append(encode_windows(lat, seq_lens))
    return torch.cat(Z).numpy(), [flow_keys[f] for f in windows[:, 0]], windows[:, 2]


def embed_tokens(label: str, split_file: str):
    """Sequence embeddings for a split with no cache, straight from the bytes."""
    data = pl.read_parquet(split_file)
    handler = PreTrainingDatasetHandler(data, SEQ_LEN, byte_encoder)
    flow_index = handler.build_flow_index()
    row_order, starts, lengths = flow_offsets_from_index(flow_index)
    windows = pick_windows(enumerate_windows(starts, lengths))
    log_length_profile(label, windows)

    flow_keys = flow_index["flow_key"].to_list()
    Z = []
    for i in range(0, windows.shape[0], WINDOW_BATCH):
        chunk = windows[i:i + WINDOW_BATCH]
        lat, seq_lens = token_latent_batch(handler, row_order, chunk)
        Z.append(encode_windows(lat, seq_lens))
    return torch.cat(Z).numpy(), [flow_keys[f] for f in windows[:, 0]], windows[:, 2]


def meta_only_ckpt(cache_dir: str):
    """The packet checkpoint a cache claims it was built from, for the staleness check."""
    import json
    with open(os.path.join(cache_dir, "meta.json")) as f:
        return json.load(f)["packet_ae_ckpt"]


# --- Cross-path check ------------------------------------------------------
# `test` has both a cache and a parquet, so the two paths must agree there. This is the
# only check that catches a flow-index / row-order drift between them.
if CROSS_CHECK_WINDOWS:
    check_cache = dict(CACHED_SETS)["test"]
    cc_latents, cc_offsets, _cc_meta = load_latent_cache(check_cache, meta_only_ckpt(check_cache))
    cc_handler = CachedLatentSequenceHandler(cc_latents, cc_offsets, PACKETS_PER_SEQUENCE)
    cc_windows = enumerate_windows(cc_handler.starts, cc_handler.lengths)[:CROSS_CHECK_WINDOWS]

    cc_lat, cc_lens = cc_handler.latent_batch_from_windows(cc_windows[:, 1:3])
    z_cached = encode_windows(cc_lat, cc_lens)

    cc_data = pl.read_parquet(CROSS_CHECK_SPLIT)
    cc_token_handler = PreTrainingDatasetHandler(cc_data, SEQ_LEN, byte_encoder)
    cc_row_order, cc_starts, cc_lengths = flow_offsets_from_index(cc_token_handler.build_flow_index())
    assert (cc_starts.shape == cc_handler.starts.shape
            and (cc_starts == cc_handler.starts).all()
            and (cc_lengths == cc_handler.lengths).all()), \
        (f"flow index ({cc_starts.shape[0]} flows) does not reproduce the cache's flow offsets "
         f"({cc_handler.starts.shape[0]} flows) -- the token path would be misaligned")
    cc_lat_live, cc_lens_live = token_latent_batch(cc_token_handler, cc_row_order, cc_windows)
    z_tokens = encode_windows(cc_lat_live, cc_lens_live)

    cc_delta = (z_cached - z_tokens).abs().max().item()
    assert cc_delta < CROSS_CHECK_TOL, (
        f"cached and token paths disagree by {cc_delta:.2e} on {CROSS_CHECK_WINDOWS} test "
        f"windows -- the row-order mapping is wrong, attack embeddings would be garbage")
    logger.info(f"Cross-path check OK on {CROSS_CHECK_WINDOWS} test windows: "
                f"max|cached-live| {cc_delta:.2e}")
    del cc_latents, cc_data

# --- Export ----------------------------------------------------------------
attack_files = sorted(f for f in os.listdir(ATTACK_DIR) if f.endswith(".parquet"))
if SMOKE:
    attack_files = attack_files[:2]
logger.info(f"Sets: {[n for n, _ in CACHED_SETS + TOKEN_SETS]} + {len(attack_files)} attacks")

embeddings = {}      # label -> (N, D_seq) float32
metadata = {}        # label -> (set_name, flow_keys, seq_lens)

for label, cache_dir in CACHED_SETS:
    logger.info(f"Embedding {label} from cache {cache_dir}")
    Z, flow_keys, seq_lens = embed_cached(label, cache_dir)
    embeddings[label] = Z
    metadata[label] = (label, flow_keys, seq_lens)

for label, split_file in TOKEN_SETS:
    logger.info(f"Embedding {label} from {split_file}")
    Z, flow_keys, seq_lens = embed_tokens(label, split_file)
    embeddings[label] = Z
    metadata[label] = (label, flow_keys, seq_lens)

for attack_file in attack_files:
    label = attack_file.removesuffix(".parquet")
    logger.info(f"Embedding attack {label}")
    Z, flow_keys, seq_lens = embed_tokens(label, f"{ATTACK_DIR}/{attack_file}")
    embeddings[label] = Z
    metadata[label] = ("attack", flow_keys, seq_lens)

# One .npy per label; the stem is what the viz uses as the legend entry.
for label, Z in embeddings.items():
    np.save(f"{output_dir}/{label}.npy", Z.astype(np.float32))
    logger.info(f"Saved {output_dir}/{label}.npy {Z.shape}")

# Row i of metadata.parquet describes row i of the viz's stacked X, so this has to be
# in the same order the viz stacks the files: sorted(glob("*.npy")).
meta_rows = []
for label in sorted(embeddings, key=lambda name: f"{name}.npy"):
    set_name, flow_keys, seq_lens = metadata[label]
    meta_rows.append(pl.DataFrame({
        "set": [set_name] * len(flow_keys),
        "label": [label] * len(flow_keys),
        "flow_key": flow_keys,
        "seq_len": seq_lens.astype(np.int32),
    }))
pl.concat(meta_rows).write_parquet(f"{output_dir}/metadata.parquet")

total = sum(Z.shape[0] for Z in embeddings.values())
logger.info(f"Wrote {len(embeddings)} sets, {total} windows total, "
            f"metadata at {output_dir}/metadata.parquet")
logger.info("Point Embedding_Viz.ipynb at "
            f"Outputs/Embeddings/SequenceEmbeddings_{RUN_NAME} "
            f"and pass n_max large enough that train stays dense "
            f"(load_data splits n_max evenly across {len(embeddings)} categories)")
