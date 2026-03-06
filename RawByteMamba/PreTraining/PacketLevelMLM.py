from RawByteMamba.SequenceClassifierComponents.ModelDefinitions import  ModelParams, Packet_MLM, Packet_Encoder, DynamicCLSPooling, TransformerBackbone, MambaBackbone
from RawByteMamba.SequenceClassifierComponents.DataUtils import ID_Encoder, PreTrainingDatasetHandler
import polars as pl
import torch
from keras_hub.layers import MaskedLMMaskGenerator
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import torch.nn as nn
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('RawByteMamba/PreTraining/PacketLevelMLM.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

### Set Training Parameters
Epochs = 10

# --- Model Config ---
vocab_size = 261
emb_dim = 32
seq_lvl_dim = 32
bytes_per_packet = 1520       # token length per packet
packets_per_sequence = 64     # max packets per sequence
num_classes = 14
batch_size = 256

alpha_proto = 4
alpha_reconstruction = 1

data = pl.read_parquet("/home/plb41586/workspace/data_artefacts/CICAPT_Phase1.parquet")
logger.info(data.head())

# CLS (Classify) token replaces Start Of Sequence  token
ID_Encoder = ID_Encoder(SpecialIDs = {"<pad>": 256, "</s>": 257, "<CLS>": 258, "<mask>": 259, "<EndPointMasking>": 260}, CLS_Placement="EOS")
DataHandler = PreTrainingDatasetHandler(data, 1, ID_Encoder)

# Init Label ProtoHierarchy Encoder
ProtoHierarchyEncoder = OneHotEncoder(sparse_output=False, dtype=np.float32)
ProtoHierarchyEncodings = ProtoHierarchyEncoder.fit_transform(DataHandler.data["proto_hierarchy"].unique().to_numpy().reshape(-1, 1))


device = torch.device("cuda")
assert device == torch.device("cuda")

# TransformerBackbone = TransformerBackbone(d_model=emb_dim, nhead=4, num_layers=2, max_len=1520).to(device)
MambaBackbone = MambaBackbone(d_model=emb_dim, num_layers=2, d_state=16, d_conv=4, expand=2).to(device)


MaskedLanguageModel = Packet_MLM(vocab_size=vocab_size, 
                                embedding_dim=emb_dim, 
                                num_CLS_classes=ProtoHierarchyEncodings.shape[1],
                                CLS_Pooling = DynamicCLSPooling(DataHandler.InputIDEncoder.SpecialIDs["<CLS>"]),
                                Backbone=MambaBackbone,
                                device=device)


loss_fct = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(MaskedLanguageModel.parameters(), lr=8e-4, weight_decay=1e-2)

unselectable_token_ids = [DataHandler.InputIDEncoder.SpecialIDs["</s>"], 
                        DataHandler.InputIDEncoder.SpecialIDs["<pad>"],
                        DataHandler.InputIDEncoder.SpecialIDs["<CLS>"],
                        DataHandler.InputIDEncoder.SpecialIDs["<EndPointMasking>"]]


Masker = MaskedLMMaskGenerator( vocabulary_size = ModelParams.vocab_size, 
                                mask_token_id = DataHandler.InputIDEncoder.SpecialIDs["<mask>"], 
                                mask_selection_length=ModelParams.packet_id_len*0.25, 
                                mask_selection_rate=0.25,
                                mask_token_rate=0.9,
                                random_token_rate=0.1,
                                unselectable_token_ids=unselectable_token_ids)

for i in range(Epochs):
    batches = DataHandler.sample_epoch_packet_indices(batch_size)

    for index, batch in enumerate(batches):
        bytes, proto_hierarchy = DataHandler.get_pretraining_data(batch)
        input_ids = DataHandler.InputIDEncoder.construct_input_ids(bytes)
        masked_ids = Masker(input_ids)
        masked_ids = np.array([masked_ids["token_ids"]]).squeeze()
        masked_ids = torch.tensor(masked_ids, dtype=torch.long).to(device)
        #Perform Forward Pass
        reconstruction_logits, CLS_logits = MaskedLanguageModel(masked_ids)

        # Encode proto_hierarchy and calculate proto_hierarchy loss
        proto_label_encodings = ProtoHierarchyEncoder.transform(np.array(proto_hierarchy).reshape(-1, 1))
        proto_label_encodings = torch.tensor(proto_label_encodings, dtype=torch.float32).to(device)
        proto_hierarchyloss = loss_fct(CLS_logits, torch.argmax(proto_label_encodings, dim=-1))

        # Get mask token indices and calculate reconstruction loss
        mask_token_id = DataHandler.InputIDEncoder.SpecialIDs["<mask>"]
        mask_token_indices = (masked_ids == mask_token_id)
        masked_logits = reconstruction_logits[mask_token_indices].to(device)
        input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
        masked_labels = input_ids[mask_token_indices]
        reconstruction_loss = loss_fct(masked_logits, masked_labels)

        predictions = torch.argmax(masked_logits, dim=-1)
        reconstruction_accuracy = (predictions == masked_labels).float().mean().item()

        CLS_predictions = torch.argmax(CLS_logits, dim=-1)
        CLS_accuracy = (CLS_predictions == torch.argmax(proto_label_encodings, dim=-1)).float().mean().item()

        # Calculate total loss
        loss = alpha_proto*proto_hierarchyloss + alpha_reconstruction*reconstruction_loss

        # Backward Pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        logger.info(f"Epoch {i+1}/{Epochs}")
        logger.info(f"Pretraining Batch {index}/{len(batches)}")
        logger.info(f"Total Loss: {loss.item()}")
        logger.info(f"Reconstruction Loss: {reconstruction_loss} Reconstruction Accuracy: {reconstruction_accuracy}")
        logger.info(f"ProtoHierarchy Loss: {proto_hierarchyloss} ProtoHierarchy Accuracy: {CLS_accuracy}")