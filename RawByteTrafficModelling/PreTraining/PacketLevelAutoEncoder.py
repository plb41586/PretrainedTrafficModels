from RawByteTrafficModelling.ModelComponents.ModelDefinitions import EncoderParams, Packet_Encoder, PacketAutoencoder, AutoEncoderParams, save_checkpoint
from RawByteTrafficModelling.ModelComponents.DataUtils import ID_Encoder, PreTrainingDatasetHandler
from RawByteTrafficModelling.ModelComponents.BackBones import TransformerBackbone, TransformerBackboneParams, MambaBackbone, MambaBackboneParams
import polars as pl
import torch
from keras_hub.layers import MaskedLMMaskGenerator
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import torch.nn as nn
import logging
import torch.nn.functional as F
from RawByteTrafficModelling.ModelComponents.ModelDefinitions import load_MLM_checkpoint

# MLMparams, ckpt = load_MLM_checkpoint("/home/plb41586/workspace/RawByteTrafficModelling/PreTraining/TrainingOutputs/EdgeIIoT_64/PacketLevelMLM_EdgeIIoT_E2.pth")

# MaskedLanguageModel = Packet_MLM(MLMparams) 

# MaskedLanguageModel.load_state_dict(ckpt['model_state_dict'])

# bos_token_id = MLMparams.EncoderParams.SpecialTokens['<BOS>']

SpecialIDs = {"<pad>": 256, "</s>": 257, "<CLS>": 258, "<mask>": 259, "<EndPointMasking>": 260, "<BOS>": 261}

encoder_params = EncoderParams(
    vocab_size=262,
    EncoderDim=64,
    packet_id_len=1520,
    pooling_type="DynamicCLS",
    BackboneType="Mamba",
    BackboneParams=MambaBackboneParams(dim=64),
    CLS_ID=SpecialIDs["<CLS>"],
    SpecialTokens=SpecialIDs
    )

autoencoderparams = AutoEncoderParams(ENC_Params=encoder_params,
                                      DecBackboneType="Mamba",
                                      DecBackbone=MambaBackboneParams(dim=encoder_params.EncoderDim),
                                      bos_token_id=encoder_params.SpecialTokens['<BOS>'])

output_dir = "RawByteTrafficModelling/PreTraining/TrainingOutputs/test"
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

### Set Training Parameters
Epochs = 3
learning_rate = 8e-4
batch_size = 128

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
# Backbone = MambaBackbone(d_model=emb_dim, num_layers=2, d_state=16, d_conv=4, expand=2).to(device)

PacketEncoder = Packet_Encoder(params=autoencoderparams.ENC_Params)

AutoEncoder = PacketAutoencoder(params = autoencoderparams,
                                encoder=PacketEncoder
).to(device)


loss_fct = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(AutoEncoder.parameters(), lr=learning_rate, weight_decay=1e-2)

AutoEncoderModel_path = f"{output_dir}/PacketLevelAutoEncoder_EdgeIIoT_untrained.ckpt"
save_checkpoint(model=AutoEncoder,
                optimizer=optimizer,
                epoch=-1,
                loss=0,
                config=autoencoderparams,
                path=AutoEncoderModel_path)
logger.info(f"Saved MaskedLanguageModel to {AutoEncoderModel_path}")

for epoch in range(Epochs):
    batches = DataHandler.sample_epoch_packet_indices(batch_size)
    for index, batch in enumerate(batches):
        bytes, proto_hierarchy = DataHandler.get_pretraining_data(batch)
        input_ids = DataHandler.InputIDEncoder.construct_input_ids(bytes)
        input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
        #Perform Forward Pass
        logits, latent = AutoEncoder(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, autoencoderparams.ENC_Params.vocab_size), input_ids.reshape(-1))

        predictions = torch.argmax(logits, dim=-1)
        reconstruction_accuracy = (predictions == input_ids).float().mean().item()

        # Backward Pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        logger.info(f"Epoch {epoch+1}/{Epochs}")
        logger.info(f"Pretraining Batch {index}/{len(batches)}")
        logger.info(f"Total Loss: {loss.item()}")
        logger.info(f"Reconstruction Loss: {loss} Reconstruction Accuracy: {reconstruction_accuracy}")
        if reconstruction_accuracy > 0.45 and batch_size < 1024:
            logger.info("Increasing Batch Size")
            batch_size = 1024
            learning_rate = learning_rate * 5
            optimizer.param_groups[0]['lr'] = learning_rate
            
    AutoEncoderModel_path = f"{output_dir}/PacketLevelAutoEncoder_EdgeIIoT_E{epoch}.ckpt"
    save_checkpoint(model=AutoEncoder,
                    optimizer=optimizer,
                    epoch=epoch,
                    loss=loss,
                    config=autoencoderparams,
                    path=AutoEncoderModel_path)
    logger.info(f"Saved MaskedLanguageModel to {AutoEncoderModel_path}")