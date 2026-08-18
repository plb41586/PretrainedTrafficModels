from RawByteTrafficModelling.ModelComponents.ModelDefinitions import  MLM_Params, Packet_MLM, Packet_Encoder, DynamicCLSPooling, TransformerBackbone, MambaBackbone, AutoregressiveDecoder, PacketAutoencoder, load_AE_Checkpoint, load_MLM_checkpoint
from RawByteTrafficModelling.ModelComponents.DataUtils import ID_Encoder, PreTrainingDatasetHandler
from RawByteTrafficModelling.PreTraining.RunConfig import DATASETS, make_id_encoder, resolve_device
import polars as pl
import torch
from keras_hub.layers import MaskedLMMaskGenerator
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import torch.nn as nn
import logging
import torch.nn.functional as F
import os

DATASET = DATASETS["IIoTset-Ferrag"]
output_dir = ("RawByteTrafficModelling/AnomalyDetection/Outputs/Embeddings/"
              "PacketEmbeddings_PacketAE_IIoTset_d128")
os.makedirs(output_dir, exist_ok=True)   # the FileHandler below cannot create it
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


device = resolve_device(0)

AEpath = ('RawByteTrafficModelling/PreTraining/TrainingOutputs/PacketAE_IIoTset_d128/'
          'PacketLevelAutoEncoder_PacketAE_IIoTset_d128_best.ckpt')
AEparams, ckpt = load_AE_Checkpoint(AEpath)
AE_model = PacketAutoencoder(AEparams)
AE_model.load_state_dict(ckpt['model_state_dict'])
vocab_size = AEparams.ENC_Params.vocab_size
packet_encoder_model = AE_model.encoder
packet_encoder_model.to(device)

# Swap the AE encoder above for the MLM's by uncommenting this block:
# MLM_Path = 'RawByteTrafficModelling/PreTraining/TrainingOutputs/PacketMLM_IIoTset_d128/PacketLevelMLM_PacketMLM_IIoTset_d128_best.ckpt'
# MLMparams, ckpt = load_MLM_checkpoint(MLM_Path)
# MLM_model = Packet_MLM(MLMparams)
# MLM_model.load_state_dict(ckpt['model_state_dict'])
# vocab_size = MLMparams.EncoderParams.vocab_size
# packet_encoder_model = Packet_Encoder(embedding=MLM_model.embedding, BackBone=MLM_model.Backbone, params=MLMparams.EncoderParams)
# packet_encoder_model = packet_encoder_model.to(device)

loss_fct = nn.CrossEntropyLoss()
losses = []
batch_size = 1024

# CLS (Classify) token replaces Start Of Sequence  token
ID_Encoder = make_id_encoder()


attack_dir = DATASET.attacks
attack_files = os.listdir(attack_dir)
# val: the final held-out split, so the "normal" embeddings are of packets the
# packet encoder never trained on.
normal_file = DATASET.val

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