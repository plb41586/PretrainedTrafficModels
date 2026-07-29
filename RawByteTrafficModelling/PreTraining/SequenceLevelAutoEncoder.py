from RawByteTrafficModelling.ModelComponents.DataUtils import PreTrainingDatasetHandler, ID_Encoder
from RawByteTrafficModelling.ModelComponents.ModelDefinitions import (
    SeqEncoderParams,
    SeqAutoEncoderParams,
    Sequence_Encoder,
    SequenceAutoencoder,
    PacketAutoencoder,
    load_AE_Checkpoint,
    compute_target_stats,
    retrieval_accuracy,
    baseline_mses,
)
from RawByteTrafficModelling.ModelComponents.BackBones import MambaBackboneParams
import polars as pl
import torch

device = torch.device("cuda")

# NOTE: this is a minimal smoke test to verify the SequenceAutoencoder wiring in
# ModelDefinitions.py with a real backward pass. The data loading below reuses
# PreTrainingDatasetHandler.draw_sequence_batch as-is and is not meant to be
# efficient or correct for full training -- it gets redone once the model side
# is confirmed to work.

pretrained_encoder_ckpt = "RawByteTrafficModelling/PreTraining/TrainingOutputs/EdgeIIoT_AutoEncoder/PacketLevelAutoEncoder_EdgeIIoT_E0.ckpt"

SpecialIDs = {"<pad>": 256, "</s>": 257, "<CLS>": 258, "<mask>": 259, "<EndPointMasking>": 260, "<BOS>": 261}

data = pl.read_parquet("data_artefacts/IIoTset-Ferrag/split/train.parquet")

# CLS (Classify) token replaces Start Of Sequence token
id_encoder = ID_Encoder(SpecialIDs=SpecialIDs, CLS_Placement="EOS")

# pad_sequence_IDs always pads a flow out to seq_len + 1 packets -- the extra
# slot is where Sequence_Encoder writes the seq-level CLS.
PACKETS_PER_SEQUENCE = 65
DataHandler = PreTrainingDatasetHandler(data, seq_len=PACKETS_PER_SEQUENCE - 1, encoder=id_encoder)

# --- Load the pretrained packet-level autoencoder and reuse its encoder -----
packet_ae_params, ckpt = load_AE_Checkpoint(pretrained_encoder_ckpt)
packet_ae = PacketAutoencoder(packet_ae_params)
packet_ae.load_state_dict(ckpt["model_state_dict"])

seq_enc_params = SeqEncoderParams(
    EncoderParams=packet_ae_params.ENC_Params,
    SeqEncoderDim=384,
    packets_per_sequence=PACKETS_PER_SEQUENCE,
    SeqBackboneType="Mamba",
    SeqBackboneParams=MambaBackboneParams(dim=384),
)

ae_params = SeqAutoEncoderParams(
    SeqEncParams=seq_enc_params,
    SeqDecoderDim=384,
    SeqDecBackboneType="Mamba",
    SeqDecBackbone=MambaBackboneParams(dim=384),
)

sequence_encoder = Sequence_Encoder(
    seq_enc_params,
    packet_encoder=packet_ae.encoder,
    freeze_packet_encoder=True,
)

model = SequenceAutoencoder(ae_params, encoder=sequence_encoder).to(device)

# --- Grab one batch, use it to set the latent-normalisation stats ----------
BATCH_SIZE = 8
tokens, labels, seq_lens = DataHandler.draw_sequence_batch(BATCH_SIZE)
tokens = tokens.long().to(device)
seq_lens = seq_lens.to(device)

with torch.no_grad():
    latents = model.encoder.encode_packets(tokens)
model.set_target_stats(*compute_target_stats(latents, seq_lens))

# --- One training step (forward + backward), to sanity-check the model -----
optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)

pred, tgt, z, len_logits, mask = model(seq_lens, tokens=tokens)
losses = model.loss(pred, tgt, mask, len_logits, seq_lens)
losses["total"].backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)

print(f"total_loss={losses['total'].item():.4f} recon_loss={losses['recon'].item():.4f}")

# --- Eval on the same batch, just to see the numbers are sane --------------
with torch.no_grad():
    pred, tgt, z, len_logits, mask = model(seq_lens, tokens=tokens)
    acc, n = retrieval_accuracy(pred, tgt, mask)
    print(f"retrieval_accuracy={acc:.4f} (n={n}) baseline_mses={baseline_mses(tgt, mask)}")
