from RawByteTrafficModelling.ModelComponents.ModelDefinitions import  ModelParams, Packet_MLM, Packet_Encoder, DynamicCLSPooling, TransformerBackbone, MambaBackbone, AutoregressiveDecoder, PacketAutoencoder
from RawByteTrafficModelling.ModelComponents.DataUtils import ID_Encoder, PreTrainingDatasetHandler
import polars as pl
import torch
from keras_hub.layers import MaskedLMMaskGenerator
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import torch.nn as nn
import logging
import torch.nn.functional as F
import os

output_dir = "RawByteTrafficModelling/AnomalyDetection/Outputs"
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{output_dir}/PacketLevelAutoEncoder_EdgeIIoT.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Model Config ---
vocab_size = 262
emb_dim = 32
seq_lvl_dim = 32
bytes_per_packet = 1520       # token length per packet
packets_per_sequence = 64     # max packets per sequence
num_classes = 14
batch_size = 1024

data = pl.read_parquet("/home/plb41586/workspace/data_artefacts/IIoTset-Ferrag/NormalMerged.parquet")
logger.info(data.head())

# CLS (Classify) token replaces Start Of Sequence  token
ID_Encoder = ID_Encoder(SpecialIDs = {"<pad>": 256, "</s>": 257, "<CLS>": 258, "<mask>": 259, "<EndPointMasking>": 260, "<BOS>": 261}, CLS_Placement="EOS")
DataHandler = PreTrainingDatasetHandler(data, 1, ID_Encoder)

# Init Label ProtoHierarchy Encoder
ProtoHierarchyEncoder = OneHotEncoder(sparse_output=False, dtype=np.float32)
ProtoHierarchyEncodings = ProtoHierarchyEncoder.fit_transform(DataHandler.data["proto_hierarchy"].unique().to_numpy().reshape(-1, 1))

device = torch.device("cuda")
assert device == torch.device("cuda")


# Backbone = TransformerBackbone(d_model=emb_dim, nhead=4, num_layers=2, max_len=1520).to(device)
Backbone = MambaBackbone(d_model=emb_dim, num_layers=2, d_state=16, d_conv=4, expand=2).to(device)

MaskedLanguageModel = Packet_MLM(vocab_size=vocab_size, 
                                embedding_dim=emb_dim, 
                                num_CLS_classes=ProtoHierarchyEncodings.shape[1],
                                CLS_Pooling = DynamicCLSPooling(DataHandler.InputIDEncoder.SpecialIDs["<CLS>"]),
                                Backbone=Backbone,
                                device=device)


PacketEncoder = Packet_Encoder(vocab_size=vocab_size,
                                embedding_dim=emb_dim,
                                input_len=bytes_per_packet,
                                latent_dim=emb_dim,
                                device=device,
                                embedding=MaskedLanguageModel.embedding,
                                BackBone=MaskedLanguageModel.Backbone,
                                Pooling=MaskedLanguageModel.CLS_Pooling).to(device)

DecoderBackbone = MambaBackbone(d_model=emb_dim, num_layers=2, d_state=16, d_conv=4, expand=2).to(device)
Decoder = AutoregressiveDecoder(vocab_size=vocab_size,
                                embedding_dim=emb_dim,
                                max_len=bytes_per_packet,
                                Backbone=DecoderBackbone,
                                bos_token_id=DataHandler.InputIDEncoder.SpecialIDs["<BOS>"],
                                device=device)

AutoEncoder = PacketAutoencoder(vocab_size=vocab_size,
                                embedding_dim=emb_dim,
                                max_len=bytes_per_packet,
                                decoder_backbone=DecoderBackbone,
                                bos_token_id=DataHandler.InputIDEncoder.SpecialIDs["<BOS>"],
                                device=device,
                                encoder=PacketEncoder,
)

AutoEncoder.load_state_dict(torch.load(f"/home/plb41586/workspace/RawByteTrafficModelling/PreTraining/TrainingOutputs/PacketLevelAutoEncoder_EdgeIIoT.pth"))

loss_fct = nn.CrossEntropyLoss()

losses = []

with torch.no_grad():
    batches = DataHandler.sample_epoch_packet_indices(batch_size)
    for index, batch in enumerate(batches):
        bytes, proto_hierarchy = DataHandler.get_pretraining_data(batch)
        input_ids = DataHandler.InputIDEncoder.construct_input_ids(bytes)
        input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
        #Perform Forward Pass
        logits, latent = AutoEncoder(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), input_ids.reshape(-1))

        predictions = torch.argmax(logits, dim=-1)
        reconstruction_accuracy = (predictions == input_ids).float().mean().item()

        logger.info(f"Pretraining Batch {index}/{len(batches)}")
        logger.info(f"Total Loss: {loss.item()}")
        logger.info(f"Reconstruction Loss: {loss} Reconstruction Accuracy: {reconstruction_accuracy}")
        losses.append(loss.item())


losses = np.array(losses)
np.save(f"{output_dir}/AutoEncoder_AnomalyDetection_NormalLosses.npy", losses)

attack_dir = 'data_artefacts/IIoTset-Ferrag/attacks'
attack_files = os.listdir(attack_dir)
for file in attack_files:
    if not file.endswith('.parquet'): continue
    path = f"{attack_dir}/{file}"
    attack_data = pl.read_parquet(path)
    logger.info(data.head())
    AttackHandler = PreTrainingDatasetHandler(attack_data, 1, ID_Encoder)
    attack_losses = []
    with torch.no_grad():
        batches = AttackHandler.sample_epoch_packet_indices(batch_size)
        for index, batch in enumerate(batches):
            bytes, proto_hierarchy = AttackHandler.get_pretraining_data(batch)
            input_ids = AttackHandler.InputIDEncoder.construct_input_ids(bytes)
            input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
            #Perform Forward Pass
            logits, latent = AutoEncoder(input_ids)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), input_ids.reshape(-1))

            predictions = torch.argmax(logits, dim=-1)
            reconstruction_accuracy = (predictions == input_ids).float().mean().item()

            logger.info(f"Pretraining Batch {index}/{len(batches)}")
            logger.info(f"Total Loss: {loss.item()}")
            logger.info(f"Reconstruction Loss: {loss} Reconstruction Accuracy: {reconstruction_accuracy}")
            attack_losses.append(loss.item())

    attack_losses = np.array(attack_losses)
    attack_name = file.removesuffix(".parquet")
    np.save(f"{output_dir}/AutoEncoder_AnomalyDetection_{attack_name}_Losses.npy", attack_losses)
print("Done")