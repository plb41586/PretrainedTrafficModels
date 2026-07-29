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
import os
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

RUN_NAME = "EdgeIIoT_AutoEncoder_FlowSplit"

### Set Training Parameters
Epochs = 3
learning_rate = 8e-4
batch_size = 128
# Cap on train batches per epoch. None = a real, full epoch.
MAX_STEPS_PER_EPOCH = None

# Flow-grouped split: whole flows live in exactly one split, so the sequence-level
# task can reuse the same partition without a flow ever straddling two of them.
# val.parquet is deliberately untouched here -- it is held back for the
# sequence-level autoencoder, so packet-level training never sees it.
SPLIT_DIR = "data_artefacts/IIoTset-Ferrag/flow_split"
TRAIN_FILE = f"{SPLIT_DIR}/train.parquet"
TEST_FILE = f"{SPLIT_DIR}/test.parquet"

# One train epoch is ~6.8M packets (~53k batches at bs=128), so per-batch logging
# and a full pass over the 1.46M-packet test split are both far too expensive.
# Log running averages on an interval, and evaluate on a fixed random subset of
# test that is drawn once so the number is comparable across evals.
log_every_n_batches = 200
eval_every_n_batches = 5000
eval_batch_size = 256
EVAL_BATCHES = 40
SEED = 42

# Smoke run: same code path, both parquets, forward/backward, mid-epoch eval,
# epoch-end eval, per-epoch and best checkpoint -- in a couple of minutes rather
# than a 53k-batch epoch. Writes to its own output_dir so a smoke checkpoint can
# never be mistaken for a trained one. Set False for the real run.
SMOKE = False
if SMOKE:
    RUN_NAME = f"{RUN_NAME}_smoke"
    Epochs = 1
    MAX_STEPS_PER_EPOCH = 30
    log_every_n_batches = 10
    eval_every_n_batches = 20
    EVAL_BATCHES = 4

output_dir = f"RawByteTrafficModelling/PreTraining/TrainingOutputs/{RUN_NAME}"
os.makedirs(output_dir, exist_ok=True)
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
logger.info(f"Run {RUN_NAME} (SMOKE={SMOKE}) -> {output_dir}")

data = pl.read_parquet(TRAIN_FILE)
test_data = pl.read_parquet(TEST_FILE)
logger.info(data.head())
logger.info(f"train: {data.height} packets from {TRAIN_FILE}")
logger.info(f"test:  {test_data.height} packets from {TEST_FILE}")

# CLS (Classify) token replaces Start Of Sequence  token
ID_Encoder = ID_Encoder(SpecialIDs = {"<pad>": 256, "</s>": 257, "<CLS>": 258, "<mask>": 259, "<EndPointMasking>": 260, "<BOS>": 261}, CLS_Placement="EOS")
DataHandler = PreTrainingDatasetHandler(data, 1, ID_Encoder)
TestDataHandler = PreTrainingDatasetHandler(test_data, 1, ID_Encoder)

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

# Drawn once, so every eval scores the same packets and the curve is a curve
# rather than a fresh sample each time.
eval_rng = np.random.default_rng(SEED)
eval_row_indices = eval_rng.choice(test_data.height,
                                   size=min(EVAL_BATCHES * eval_batch_size, test_data.height),
                                   replace=False)
eval_batches = [eval_row_indices[i:i + eval_batch_size]
                for i in range(0, eval_row_indices.shape[0], eval_batch_size)]
logger.info(f"Held-out eval: {len(eval_batches)} batches x {eval_batch_size} packets from test")

pad_id = SpecialIDs["<pad>"]


@torch.no_grad()
def evaluate():
    """
    Reconstruction loss and accuracy on the fixed test subset.

    Reports accuracy twice: over all 1520 positions (comparable with the training
    metric below) and over non-pad positions only. Most packets are far shorter
    than 1520, so the all-position number is dominated by predicting <pad> and
    reads high long before the model reconstructs any real bytes.
    """
    AutoEncoder.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_acc_nonpad = 0.0
    for batch in eval_batches:
        bytes, proto_hierarchy = TestDataHandler.get_pretraining_data(batch)
        input_ids = TestDataHandler.InputIDEncoder.construct_input_ids(bytes)
        input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)

        logits, latent = AutoEncoder(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, autoencoderparams.ENC_Params.vocab_size), input_ids.reshape(-1))

        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions == input_ids)
        non_pad = (input_ids != pad_id)

        total_loss += loss.item()
        total_acc += correct.float().mean().item()
        total_acc_nonpad += (correct & non_pad).sum().item() / max(non_pad.sum().item(), 1)

    AutoEncoder.train()
    n = len(eval_batches)
    return total_loss / n, total_acc / n, total_acc_nonpad / n


best_test_loss = float("inf")

for epoch in range(Epochs):
    batches = DataHandler.sample_epoch_packet_indices(batch_size)
    if MAX_STEPS_PER_EPOCH is not None:
        batches = batches[:MAX_STEPS_PER_EPOCH]
    running_loss = 0.0
    running_acc = 0.0
    running_n = 0
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

        running_loss += loss.item()
        running_acc += reconstruction_accuracy
        running_n += 1

        if (index + 1) % log_every_n_batches == 0 or index == len(batches) - 1:
            logger.info(f"Epoch {epoch+1}/{Epochs} Batch {index+1}/{len(batches)} "
                        f"Reconstruction Loss: {running_loss / running_n} "
                        f"Reconstruction Accuracy: {running_acc / running_n}")
            running_loss = 0.0
            running_acc = 0.0
            running_n = 0

        if (index + 1) % eval_every_n_batches == 0:
            test_loss, test_acc, test_acc_nonpad = evaluate()
            logger.info(f"[test] Epoch {epoch+1}/{Epochs} Batch {index+1}/{len(batches)} "
                        f"Loss: {test_loss} Accuracy: {test_acc} Accuracy(non-pad): {test_acc_nonpad}")

        if reconstruction_accuracy > 0.45 and batch_size < 1024:
            logger.info("Increasing Batch Size")
            batch_size = 1024
            learning_rate = learning_rate * 5
            optimizer.param_groups[0]['lr'] = learning_rate

    test_loss, test_acc, test_acc_nonpad = evaluate()
    logger.info(f"[test] Epoch {epoch+1}/{Epochs} end "
                f"Loss: {test_loss} Accuracy: {test_acc} Accuracy(non-pad): {test_acc_nonpad}")

    AutoEncoderModel_path = f"{output_dir}/PacketLevelAutoEncoder_EdgeIIoT_E{epoch}.ckpt"
    save_checkpoint(model=AutoEncoder,
                    optimizer=optimizer,
                    epoch=epoch,
                    loss=loss,
                    config=autoencoderparams,
                    path=AutoEncoderModel_path)
    logger.info(f"Saved MaskedLanguageModel to {AutoEncoderModel_path}")

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_path = f"{output_dir}/PacketLevelAutoEncoder_EdgeIIoT_best.ckpt"
        save_checkpoint(model=AutoEncoder,
                        optimizer=optimizer,
                        epoch=epoch,
                        loss=loss,
                        config=autoencoderparams,
                        path=best_path,
                        extra={"test_loss": test_loss,
                               "test_acc": test_acc,
                               "test_acc_nonpad": test_acc_nonpad})
        logger.info(f"New best test loss {test_loss}, saved to {best_path}")
