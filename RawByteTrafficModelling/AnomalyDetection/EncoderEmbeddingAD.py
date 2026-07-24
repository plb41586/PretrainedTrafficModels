from RawByteTrafficModelling.ModelComponents.ModelDefinitions import  MLM_Params, Packet_MLM, Packet_Encoder, DynamicCLSPooling, TransformerBackbone, MambaBackbone, AutoregressiveDecoder, PacketAutoencoder, load_AE_Checkpoint, load_MLM_checkpoint
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

output_dir = "RawByteTrafficModelling/AnomalyDetection/Outputs/PacketEmbeddingsMLM"
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


device = torch.device("cuda:0")

AEpath = '/home/plb41586/workspace/RawByteTrafficModelling/PreTraining/TrainingOutputs/test/PacketLevelAutoEncoder_EdgeIIoT.ckpt'
MLM_Path = '/home/plb41586/workspace/RawByteTrafficModelling/PreTraining/TrainingOutputs/test/PacketLevelMLM_EdgeIIoT.pth'

# AEparams, ckpt = load_AE_Checkpoint(AEpath)
MLMparams, ckpt = load_MLM_checkpoint(MLM_Path)

# AE_model = PacketAutoencoder(AEparams)
# AE_model.load_state_dict(ckpt['model_state_dict'])
MLM_model = Packet_MLM(MLMparams)
MLM_model.load_state_dict(ckpt['model_state_dict'])

loss_fct = nn.CrossEntropyLoss()

losses = []

# --- Model Config ---
# vocab_size = AEparams.ENC_Params.vocab_size
vocab_size = MLMparams.EncoderParams.vocab_size
batch_size = 1024


# CLS (Classify) token replaces Start Of Sequence  token
ID_Encoder = ID_Encoder(SpecialIDs = {"<pad>": 256, "</s>": 257, "<CLS>": 258, "<mask>": 259, "<EndPointMasking>": 260, "<BOS>": 261}, CLS_Placement="EOS")

# packet_encoder_model = AE_model.encoder
packet_encoder_model = Packet_Encoder(embedding=MLM_model.embedding, BackBone=MLM_model.Backbone, params=MLMparams.EncoderParams)
packet_encoder_model = packet_encoder_model.to(device)

attack_dir = 'data_artefacts/IIoTset-Ferrag/attacks'
attack_files = os.listdir(attack_dir)
normal_file = "/home/plb41586/workspace/data_artefacts/IIoTset-Ferrag/NormalMerged.parquet"

files = []
for file in attack_files:
    files.append(os.path.join(attack_dir, file))
files.append(normal_file)

for file in files:
    if not file.endswith('.parquet'): continue
    data = pl.read_parquet(file)
    logger.info(data.head())
    DataHandler = PreTrainingDatasetHandler(data, 1, ID_Encoder)
    embeddings = []
    with torch.no_grad():
        batches = DataHandler.sample_epoch_packet_indices(batch_size)
        for index, batch in enumerate(batches):
            bytes, proto_hierarchy = DataHandler.get_pretraining_data(batch)
            input_ids = DataHandler.InputIDEncoder.construct_input_ids(bytes)
            input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
            #Perform Forward Pass
            latent = packet_encoder_model(input_ids)

            logger.info(f"Pretraining Batch {index}/{len(batches)}")
            embeddings.append(latent)
            if index == 300:
                break
        embeddings = torch.cat(embeddings, dim=0)
        embeddings = embeddings.cpu().numpy()
        new_filename = os.path.splitext(os.path.basename(file))[0] + ".npy"
        output_file = os.path.join(output_dir, new_filename)
        np.save(output_file, embeddings)

print("Done")