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
from RawByteTrafficModelling.ModelComponents.ModelDefinitions import load_checkpoint

config, ckpt = load_checkpoint("/home/plb41586/workspace/RawByteTrafficModelling/PreTraining/TrainingOutputs/test/PacketLevelMLM_EdgeIIoT.pth")

MaskedLanguageModel = Packet_MLM(config) 

MaskedLanguageModel.load_state_dict(ckpt['model_state_dict'])

output_dir = "RawByteTrafficModelling/PreTraining/TrainingOutputs"
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
Epochs = 1

# # --- Model Config ---
# vocab_size = 262
# emb_dim = 32
# seq_lvl_dim = 32
# bytes_per_packet = 1520       # token length per packet
# packets_per_sequence = 64     # max packets per sequence
# num_classes = 14
# batch_size = 1024

# learning_rate = 8e-4

# alpha_proto = 4
# alpha_reconstruction = 1

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


loss_fct = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(AutoEncoder.parameters(), lr=learning_rate, weight_decay=1e-2)

unselectable_token_ids = [DataHandler.InputIDEncoder.SpecialIDs["</s>"], 
                        DataHandler.InputIDEncoder.SpecialIDs["<pad>"],
                        DataHandler.InputIDEncoder.SpecialIDs["<CLS>"],
                        DataHandler.InputIDEncoder.SpecialIDs["<EndPointMasking>"]]


Masker = MaskedLMMaskGenerator( vocabulary_size = ModelParams.vocab_size, 
                                mask_token_id = DataHandler.InputIDEncoder.SpecialIDs["<mask>"], 
                                mask_selection_length=ModelParams.packet_id_len*0.25, 
                                mask_selection_rate=0.10,
                                mask_token_rate=0.9,
                                random_token_rate=0.1,
                                unselectable_token_ids=unselectable_token_ids)

for i in range(Epochs):
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

        # Backward Pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        logger.info(f"Epoch {i+1}/{Epochs}")
        logger.info(f"Pretraining Batch {index}/{len(batches)}")
        logger.info(f"Total Loss: {loss.item()}")
        logger.info(f"Reconstruction Loss: {loss} Reconstruction Accuracy: {reconstruction_accuracy}")

        # if reconstruction_accuracy > 0.45 and CLS_accuracy > 0.80 and batch_size < 2048:
        #     logger.info("Increasing Batch Size")
        #     batch_size = 2048
        #     learning_rate = learning_rate * 10
        #     optimizer.param_groups[0]['lr'] = learning_rate
        #     break
            
AutoEncoderModel_path = f"{output_dir}/PacketLevelAutoEncoder_EdgeIIoT.pth"
torch.save(AutoEncoder.state_dict(), AutoEncoderModel_path)
logger.info(f"Saved MaskedLanguageModel to {AutoEncoderModel_path}")