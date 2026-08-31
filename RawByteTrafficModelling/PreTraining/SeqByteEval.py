"""
Byte-level evaluation set for a sequence-level model, built off a latent cache.

Masked MSE in normalised latent space has no interpretable scale. Byte accuracy
through the frozen packet decoder does, and it answers the question that matters: are
the reconstructed latents tight enough for the decoder to recover the packet?

The latent cache holds no bytes, so this needs the split parquet plus the cache-row ->
parquet-row map. `PreTrainingDatasetHandler.build_flow_index` reproduces exactly the
order `CachePacketLatents` wrote; the flow_key assert pins that down and the re-encode
probe catches any residual misalignment, which would otherwise show up as a
mysteriously terrible metric rather than an error.

This is a lift of the block SequenceLevelAutoEncoder.py works out inline (see the
comments around its byte-eval section). It lives here because the stage-2 structure
fine-tune needs the identical set to keep its numbers comparable, and a second copy of
three alignment asserts is a copy that eventually drifts.
"""
from RawByteTrafficModelling.ModelComponents.ModelDefinitions import (
    build_padding_mask,
    decoder_byte_accuracy,
)
from RawByteTrafficModelling.ModelComponents.DataUtils import (
    CachedLatentSequenceHandler,
    ID_Encoder,
    PreTrainingDatasetHandler,
)
import polars as pl
import numpy as np
import torch


@torch.no_grad()
def build_byte_eval_set(handler: CachedLatentSequenceHandler,
                        flow_offsets: pl.DataFrame,
                        windows: np.ndarray,
                        meta: dict,
                        split_file: str,
                        packet_ae_params,
                        packet_encoder: torch.nn.Module,
                        packet_decoder: torch.nn.Module,
                        num_windows: int,
                        chunk: int,
                        device: torch.device,
                        logger=None,
                        probe_tol: float = 1e-2) -> dict:
    """
    Gather a fixed subset of eval windows as raw tokens, and score the ceiling.

    Args:
        handler:      the cache handler the windows index into.
        flow_offsets: that cache's flow_offsets frame (for the flow_key order assert).
        windows:      (W, 2) (start, length) rows, e.g. handler.enumerate_windows().
        meta:         the cache's meta.json, checked against split_file.
        split_file:   the parquet the cache was built from -- the bytes come from here.
        packet_ae_params: AutoEncoderParams of the packet model (specials, id length).
        packet_encoder:   the frozen packet encoder, for the alignment probe.
        packet_decoder:   the frozen packet decoder, for the ceiling.
        num_windows:  size of the subset.
        chunk:        packets per decoder forward, bounding the logits tensor.
        device:       where the returned tensors live.

    Returns:
        dict with `tokens` (W, P, L) int16, `latents` (W, P, D), `seq_lens` (W,),
        `ceiling` ({"all", "nonpad"}) and `pad_id`.
    """
    assert meta["split_file"] == split_file, (
        f"byte eval would read {split_file} but the cache was built from "
        f"{meta['split_file']}")

    pad_id = packet_ae_params.ENC_Params.SpecialTokens["<pad>"]
    packet_id_len = packet_ae_params.ENC_Params.packet_id_len
    num_packets = handler.num_packets

    if logger:
        logger.info(f"Building byte-eval subset from {split_file} "
                    f"({num_windows} windows; sorts the split, takes a moment)")
    split = pl.read_parquet(split_file)
    byte_encoder = ID_Encoder(SpecialIDs=packet_ae_params.ENC_Params.SpecialTokens,
                              CLS_Placement="EOS")   # must match CachePacketLatents
    byte_handler = PreTrainingDatasetHandler(split, num_packets - 1, byte_encoder)
    flow_index = byte_handler.build_flow_index()
    assert flow_index["flow_key"].to_list() == flow_offsets["flow_key"].to_list(), \
        "flow index order does not reproduce the cache's order -- byte eval would be misaligned"

    # Spread the subset over the whole split: windows come in flow_key order, so the
    # first N would all sit in one corner of it.
    pick = np.linspace(0, windows.shape[0] - 1, num_windows).astype(np.int64)
    byte_windows = windows[pick]
    latents, seq_lens = handler.latent_batch_from_windows(byte_windows)

    tokens = torch.full((byte_windows.shape[0], num_packets, packet_id_len),
                        pad_id, dtype=torch.int16)   # ids max out at 261, int16 is plenty
    row_idx = flow_index["row_idx"]
    for w, (start, length) in enumerate(byte_windows):
        flow = int(np.searchsorted(handler.starts, start, side="right") - 1)
        offset = int(start - handler.starts[flow])
        rows = row_idx[flow].to_numpy()[offset:offset + int(length)]
        window_bytes, _ = byte_handler.get_pretraining_data(rows)
        ids = byte_encoder.construct_input_ids(window_bytes)
        tokens[w, :int(length)] = torch.tensor(np.asarray(ids), dtype=torch.int16)

    valid = build_padding_mask(seq_lens, num_packets).reshape(-1)
    flat_latents = latents.reshape(-1, latents.shape[-1])[valid]
    flat_tokens = tokens.reshape(-1, packet_id_len)[valid]

    # Alignment: the tokens just gathered must re-encode to the cached latents.
    probe = min(256, flat_tokens.shape[0])
    probe_live = packet_encoder(flat_tokens[:probe].long().to(device)).float().cpu()
    probe_delta = (probe_live - flat_latents[:probe]).abs().max().item()
    assert probe_delta < probe_tol, (
        f"byte-eval tokens do not match the cached latents (max|live-cached| "
        f"{probe_delta:.2e}) -- the cache row -> parquet row mapping is off")
    if logger:
        logger.info(f"Byte-eval alignment OK over {probe} packets: "
                    f"max|live-cached| {probe_delta:.2e}")

    # Ceiling: the packet AE's own byte accuracy on these packets, decoding from the
    # *true* latents. A sequence-AE number only means something against this.
    ceiling = decoder_byte_accuracy(packet_decoder,
                                    flat_latents.to(device),
                                    flat_tokens.to(device),
                                    pad_token_id=pad_id, chunk=chunk)
    if logger:
        logger.info(f"Byte-eval subset: {byte_windows.shape[0]} windows, "
                    f"{flat_tokens.shape[0]} real packets")
        logger.info(f"Packet-AE ceiling (decoding true latents): "
                    f"all {ceiling['all']:.4f} non-pad {ceiling['nonpad']:.4f}")

    return {"tokens": tokens.to(device),
            "latents": latents.to(device),
            "seq_lens": seq_lens.to(device),
            "ceiling": ceiling,
            "pad_id": pad_id}
