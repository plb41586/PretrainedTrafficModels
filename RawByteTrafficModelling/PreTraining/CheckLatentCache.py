"""
Read-only verification of one split's latent cache (set SPLIT below). Writes nothing.

1. Equivalence: for a few flows, re-encode their packets through the live packet
   encoder (token path) and compare against the cached rows.
2. Handler shapes: draw_latent_batch / enumerate_windows produce the right
   shapes, real packets land contiguously in order, pad slots are exactly zero,
   and no window crosses a flow boundary.
3. Flow length distribution, since it decides how padded the sequences are.
"""
from RawByteTrafficModelling.ModelComponents.ModelDefinitions import (
    PacketAutoencoder, load_AE_Checkpoint,
)
from RawByteTrafficModelling.ModelComponents.DataUtils import (
    PreTrainingDatasetHandler, CachedLatentSequenceHandler, load_latent_cache,
)
from RawByteTrafficModelling.PreTraining.RunConfig import (
    DATASETS, make_id_encoder, resolve_device,
)
import polars as pl
import numpy as np
import torch

# Must match CachePacketLatents' constants, or load_latent_cache's sha256 check fails.
DATASET = DATASETS["IIoTset-Ferrag"]
CACHE_TAG = "PacketAE_d128_best"
SPLIT = "train"                # "test" is the same checks over 4.7x fewer rows
PACKET_AE_CKPT = ("RawByteTrafficModelling/PreTraining/TrainingOutputs/PacketAE_IIoTset_d128/"
                  "PacketLevelAutoEncoder_PacketAE_IIoTset_d128_best.ckpt")
SPLIT_FILE = getattr(DATASET, SPLIT)
CACHE_DIR = DATASET.latent_cache(CACHE_TAG, SPLIT)
PACKETS_PER_SEQUENCE = 65
DEVICE_INDEX = 0

device = resolve_device(DEVICE_INDEX)
latents, flow_offsets, meta = load_latent_cache(CACHE_DIR, PACKET_AE_CKPT)
print(f"cache: {latents.shape} {latents.dtype}, {flow_offsets.height} flows")

data = pl.read_parquet(SPLIT_FILE)
enc = make_id_encoder()        # must match the encoder CachePacketLatents used
handler = PreTrainingDatasetHandler(data, PACKETS_PER_SEQUENCE - 1, enc)
flow_index = handler.build_flow_index()

# The cache's flow order must reproduce exactly, or every offset is wrong.
assert flow_index["flow_key"].to_list() == flow_offsets["flow_key"].to_list(), \
    "flow index order does not reproduce the cache's order"
print("flow order reproduces")

# --- 1. Equivalence: cached rows vs a live encoder forward ------------------
packet_ae_params, ckpt = load_AE_Checkpoint(PACKET_AE_CKPT)
packet_ae = PacketAutoencoder(packet_ae_params)
packet_ae.load_state_dict(ckpt["model_state_dict"])
packet_encoder = packet_ae.encoder.to(device).eval()

rng = np.random.default_rng(0)
probe_flows = rng.choice(flow_offsets.height, 5, replace=False)
worst = 0.0
for f in probe_flows:
    rows = flow_index["row_idx"][int(f)].to_numpy()[:32]
    start = int(flow_offsets["start"][int(f)])

    bytes_, _ = handler.get_pretraining_data(rows)
    input_ids = torch.tensor(enc.construct_input_ids(bytes_), dtype=torch.long).to(device)
    with torch.no_grad():
        live = packet_encoder(input_ids).cpu().float()

    cached = latents[start:start + len(rows)].float()
    delta = (live - cached).abs().max().item()
    worst = max(worst, delta)
    print(f"  flow {f}: {len(rows)} packets, max|live-cached| = {delta:.2e}")
assert worst < 1e-2, f"cached latents diverge from a live forward: {worst}"
print(f"equivalence OK (worst {worst:.2e}, fp16 storage step is ~{2**-10:.1e} relative)")

# --- 2. Handler shapes / padding -------------------------------------------
h = CachedLatentSequenceHandler(latents, flow_offsets, PACKETS_PER_SEQUENCE)
batches = h.epoch_flow_batches(64, np.random.default_rng(1))
assert sum(len(b) for b in batches) == flow_offsets.height, "epoch does not cover every flow exactly once"
assert len(np.unique(np.concatenate(batches))) == flow_offsets.height
print(f"epoch_flow_batches: {len(batches)} batches covering all {flow_offsets.height} flows once")

flow_ids = batches[0]
lat, seq_lens = h.draw_latent_batch(flow_ids, np.random.default_rng(2))
assert lat.shape == (len(flow_ids), PACKETS_PER_SEQUENCE, meta["latent_dim"]), lat.shape
assert seq_lens.min() >= 1 and seq_lens.max() <= PACKETS_PER_SEQUENCE - 1, \
    f"seq_lens out of range: {seq_lens.min()}..{seq_lens.max()}"
for i, n in enumerate(seq_lens.tolist()):
    assert lat[i, n:].abs().max() == 0, f"row {i} has non-zero padding past seq_len={n}"
print(f"draw_latent_batch: {tuple(lat.shape)}, seq_lens {seq_lens.min()}..{seq_lens.max()}, padding all zero")

# Real packets must be the flow's own rows, contiguous and in order.
for i, f in enumerate(flow_ids[:5]):
    n = int(seq_lens[i])
    s, L = int(h.starts[f]), int(h.lengths[f])
    row = lat[i, :n]
    hits = [o for o in range(L - n + 1) if torch.equal(latents[s + o:s + o + n].float(), row)]
    assert hits, f"batch row {i} is not a contiguous in-order window of flow {f}"
print("windows are contiguous, in order, and inside their own flow")

# --- 3. Deterministic val windows ------------------------------------------
w = h.enumerate_windows()
ends = w[:, 0] + w[:, 1]
flow_end = h.starts + h.lengths
owner = np.searchsorted(h.starts, w[:, 0], side="right") - 1
assert (ends <= flow_end[owner]).all(), "a window crosses a flow boundary"
assert (w[:, 1] >= 1).all() and (w[:, 1] <= PACKETS_PER_SEQUENCE - 1).all()
print(f"enumerate_windows: {w.shape[0]} windows, lengths {w[:,1].min()}..{w[:,1].max()}, none cross a flow")

vlat, vlens = h.latent_batch_from_windows(w[:256])
assert vlat.shape == (256, PACKETS_PER_SEQUENCE, meta["latent_dim"])
print(f"latent_batch_from_windows: {tuple(vlat.shape)}")

# --- 4. How padded will training actually be? ------------------------------
L = h.lengths
print(f"\nflow lengths: mean {L.mean():.1f} median {np.median(L):.0f} "
      f"p90 {np.percentile(L,90):.0f} max {L.max()}")
print(f"flows >= seq_len({h.seq_len}): {(L >= h.seq_len).sum()} / {len(L)} "
      f"({100*(L >= h.seq_len).mean():.1f}%)")
print(f"mean fill of a val window: {w[:,1].mean() / h.seq_len:.1%}")
print("\nALL CHECKS PASSED")
