import torch
import torch.nn as nn
from mamba_ssm import Mamba
from dataclasses import dataclass

@dataclass
class ModelParams:
    """
    MLM Model Parameters
    """
    vocab_size: int = 260
    dim: int = 64
    packet_id_len: int = 1520
    pooling_type: str = "DynamicCLS"
    """
    PacketLevel Classifier Parameters
    """
    latent_dim: int = 64,
    latent_len: int = 1,
    """
    Sequence Classifier
    """
    PacketSequenceLength: int =16
    SeqClassifierDim: int = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvSequenceReducer(nn.Module):
    '''
    Convolutional Sequence Reducer

    This module reduces the sequence length of an input tensor using a depthwise 1D convolution. 
    It processes input sequences along the temporal dimension by applying a convolution with a 
    kernel size and stride of 4, effectively downsampling the sequence length by a factor of 4 
    while preserving the number of features (channels). 

    Args:
        input_dim (int): The number of input features (channels).

    Shape:
        - Input: (batch_size, seq_len, input_dim)
        - Output: (batch_size, seq_len // 4, input_dim)

    Example:
        >>> reducer = ConvSequenceReducer(input_dim=128)
        >>> x = torch.randn(32, 100, 128)  # (batch_size, seq_len, input_dim)
        >>> out = reducer(x)
        >>> print(out.shape)  # (32, 25, 128)
    '''
    def __init__(self, input_dim):
        super(ConvSequenceReducer, self).__init__()
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=4, stride=4, groups=input_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change to (batch, channels, seq_len) for Conv1d
        x = self.conv(x)         # Apply convolution
        x = x.permute(0, 2, 1)   # Change back to (batch, seq_len, channels)
        return x

class ConvSequenceUpsampler(nn.Module):
    '''
    Convolutional Sequence Upsampler

    This module increases the sequence length of an input tensor using a depthwise 1D transposed 
    convolution. It expands the temporal dimension by a factor of 4 while preserving the number 
    of features (channels). The transposed convolution is applied independently to each channel.

    Args:
        input_dim (int): The number of input features (channels).

    Shape:
        - Input: (batch_size, seq_len, input_dim)
        - Output: (batch_size, seq_len * 4, input_dim)

    Example:
        >>> upsampler = ConvSequenceUpsampler(input_dim=128)
        >>> x = torch.randn(32, 25, 128)  # (batch_size, seq_len, input_dim)
        >>> out = upsampler(x)
        >>> print(out.shape)  # (32, 100, 128)
    '''

    def __init__(self, input_dim):
        super(ConvSequenceUpsampler, self).__init__()
        self.deconv = nn.ConvTranspose1d(in_channels=input_dim, 
                                         out_channels=input_dim, 
                                         kernel_size=4, 
                                         stride=4, 
                                         groups=input_dim)  # Keeps channels independent

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change to (batch, channels, seq_len)
        x = self.deconv(x)      # Apply transposed convolution
        x = x.permute(0, 2, 1)  # Change back to (batch, seq_len, channels)
        return x

class CLSPooling(nn.Module):
    '''
    CLS Pooling

    This module extracts the hidden state corresponding to the CLS token from a sequence of 
    hidden states. The CLS token is the first token in the sequence, which is typically used 
    as an aggregate representation of the entire sequence in BERT-like models.

    Shape:
        - Input: (batch_size, seq_len, hidden_dim)
        - Output: (batch_size, hidden_dim)

    Example:
        >>> pooling = CLSPooling()
        >>> x = torch.randn(32, 100, 768)  # (batch_size, seq_len, hidden_dim)
        >>> out = pooling(x)
        >>> print(out.shape)  # (32, 768)
    '''

    def __init__(self):
        super(CLSPooling, self).__init__()

    def forward(self, x, _):
        return x[:, 0, :]  # Extract CLS token

class CLSPoolingEOS(nn.Module):
    '''
    CLS Pooling

    This module extracts the hidden state corresponding to the CLS token from a sequence of 
    hidden states. The CLS token is the last token in the sequence, which is typically used 
    as an aggregate representation of the entire sequence in BERT-like models.

    Shape:
        - Input: (batch_size, seq_len, hidden_dim)
        - Output: (batch_size, hidden_dim)

    Example:
        >>> pooling = CLSPooling()
        >>> x = torch.randn(32, 100, 768)  # (batch_size, seq_len, hidden_dim)
        >>> out = pooling(x)
        >>> print(out.shape)  # (32, 768)
    '''

    def __init__(self):
        super(CLSPooling, self).__init__()

    def forward(self, x, _):
        return x[:, -1, :]  # Extract CLS token

class DynamicCLSPooling(nn.Module):
    """
    Dynamic CLS Pooling

    This module extracts the hidden state corresponding to the CLS token from a sequence of 
    hidden states. Instead of assuming a fixed position for the CLS token, it dynamically finds 
    its location based on the provided token IDs.

    Shape:
        - Input:
            - hidden_states: (batch_size, seq_len, hidden_dim)
            - input_ids: (batch_size, seq_len) - Tokenized input sequences
        - Output: (batch_size, hidden_dim)

    Example:
        >>> pooling = DynamicCLSPooling(cls_token_id=101)  # Example CLS token ID
        >>> x = torch.randn(32, 100, 768)  # (batch_size, seq_len, hidden_dim)
        >>> input_ids = torch.randint(0, 30522, (32, 100))  # Random token IDs
        >>> out = pooling(x, input_ids)
        >>> print(out.shape)  # (32, 768)
    """

    def __init__(self, cls_token_id: int):
        super(DynamicCLSPooling, self).__init__()
        self.cls_token_id = cls_token_id

    def forward(self, hidden_states, input_ids):
        """
        Extracts CLS token embeddings dynamically.

        Args:
            hidden_states (torch.Tensor): (batch_size, seq_len, hidden_dim) hidden states.
            input_ids (torch.Tensor): (batch_size, seq_len) token IDs to locate CLS token.

        Returns:
            torch.Tensor: (batch_size, hidden_dim) Extracted CLS token embeddings.
        """
        # Find the index of CLS token for each sequence in the batch
        cls_positions = (input_ids == self.cls_token_id).nonzero(as_tuple=True)

        # Initialize a tensor to store CLS embeddings
        batch_size, hidden_dim = hidden_states.shape[0], hidden_states.shape[2]
        cls_embeddings = torch.zeros(batch_size, hidden_dim, device=hidden_states.device)

        # Iterate over each sequence in the batch
        for batch_idx in range(batch_size):
            indices = cls_positions[1][cls_positions[0] == batch_idx]  # Get CLS indices
            if len(indices) > 0:
                cls_embeddings[batch_idx] = hidden_states[batch_idx, indices[0], :]

        return cls_embeddings


class Packet_MLM(nn.Module):
    def __init__(   self, 
                    vocab_size: int, 
                    embedding_dim: int,  
                    num_CLS_classes: int,
                    CLS_Pooling: nn.Module, 
                    device: torch.device):
        super().__init__()
        self.device = device
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim).to(device)
        
        self.BackBone = Mamba(
                d_model=embedding_dim,
                d_state=16,
                d_conv=4,
                expand=2
            ).to(device)
        
        
        self.reconstruction_output = nn.Linear(embedding_dim, vocab_size, bias=False).to(device)
        self.CLS_Pooling = CLS_Pooling.to(device)
        self.CLS_output = nn.Linear(embedding_dim, num_CLS_classes, bias=False).to(device)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model. Returns the raw logits.
        
        Args:
            tokens (torch.Tensor): The input tokens.
        
        Returns:
            torch.Tensor: The raw logits.
        """
        h = self.embedding(tokens)
        h = self.BackBone(h)
        reconstruction_output = self.reconstruction_output(h)
        CLS = self.CLS_Pooling(h, tokens)
        CLS_output = self.CLS_output(CLS)
        return reconstruction_output, CLS_output   

class Packet_Classifier(nn.Module):
    def __init__(   self, 
                    vocab_size: int, 
                    embedding_dim: int,
                    input_len: int,
                    latent_dim: int,
                    latent_len: int,  
                    num_classes: int,
                    device: torch.device,
                    embedding: nn.Module = None,
                    BackBone: nn.Module = None,
                    Pooling: nn.Module = None):
        super().__init__()
        self.device = device
        
        if embedding is None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim).to(device)
        else:
            self.embedding = embedding.to(device)
        
        if BackBone is None:
            self.BackBone = Mamba(
                    d_model=embedding_dim,
                    d_state=16,
                    d_conv=4,
                    expand=2
                ).to(device)
        else:
            self.BackBone = BackBone.to(device)

        if Pooling is None:
            assert embedding_dim == latent_dim, "Embedding and latent dimensions must match for default pooling."
            assert latent_len == input_len, "Input and latent lengths must match for default pooling."
            self.Pooler = nn.Identity()
        else:
            self.Pooler = Pooling.to(device)
        
        self.output = nn.Linear(latent_dim*latent_len, num_classes, bias=False).to(device)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model. Returns the raw logits.
        
        Args:
            tokens (torch.Tensor): The input tokens.
        
        Returns:
            torch.Tensor: The raw logits.
        """
        h = self.embedding(tokens)
        h = self.BackBone(h)
        h = self.Pooler(h, tokens)
        h = h.view(tokens.shape[0], -1)
        output = self.output(h)
        return output

class Packet_Encoder(nn.Module):
    def __init__(   self, 
                    vocab_size: int, 
                    embedding_dim: int,
                    input_len: int,
                    latent_dim: int,
                    latent_len: int,  
                    num_classes: int,
                    device: torch.device,
                    embedding: nn.Module = None,
                    BackBone: nn.Module = None,
                    Pooling: nn.Module = None):
        super().__init__()
        self.device = device
        
        if embedding is None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim).to(device)
        else:
            self.embedding = embedding.to(device)
        
        if BackBone is None:
            self.BackBone = Mamba(
                    d_model=embedding_dim,
                    d_state=16,
                    d_conv=4,
                    expand=2
                ).to(device)
        else:
            self.BackBone = BackBone.to(device)

        if Pooling is None:
            assert embedding_dim == latent_dim, "Embedding and latent dimensions must match for default pooling."
            assert latent_len == input_len, "Input and latent lengths must match for default pooling."
            self.Pooler = nn.Identity()
        else:
            self.Pooler = Pooling.to(device)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model. Returns the raw logits.
        
        Args:
            tokens (torch.Tensor): The input tokens.
        
        Returns:
            torch.Tensor: The compressed latent representation.
        """
        h = self.embedding(tokens)
        h = self.BackBone(h)
        h = self.Pooler(h, tokens)
        return h

class SequenceClassifier(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 embedding_dim: int, 
                 seq_lvl_dim: int, 
                 latent_len: int, 
                 sequence_length: int,
                 num_classes: int, 
                 device: torch.device,
                 PacketEncoder: nn.Module):
        super().__init__()
        self.device = device

        self.encoder = PacketEncoder.to(device)

        if seq_lvl_dim != embedding_dim:
            self.seq_embedding = nn.Linear(embedding_dim, seq_lvl_dim).to(device)
        else:
            self.seq_embedding = nn.Identity().to(device)

        self.SeqBackBone = Mamba(
                d_model=seq_lvl_dim,
                d_state=16,
                d_conv=4,
                expand=2
            ).to(device)

        self.SeqCLSembedding = nn.Embedding(1, embedding_dim).to(device)

        self.embedding_dim = embedding_dim
        self.seq_lvl_dim = seq_lvl_dim
        self.latent_len = latent_len + 1 # account for CLS token 
        self.vocab_size = vocab_size
        self.output = nn.Linear(seq_lvl_dim, num_classes).to(device)

    def forward(self, tokens: [list[torch.Tensor]], seq_lens) -> torch.Tensor:
        """
        Forward pass of the model. Returns the raw logits.
        
        Args:
            tokens (torch.Tensor): The input tokens.
            seq_lens (torch.Tensor): The sequence length of the input tokens.
        
        Returns:
            torch.Tensor: The raw logits.
        """
        batch_size = tokens.shape[0]
        latent_logits = []
        seqCLS = self.SeqCLSembedding(torch.LongTensor([0]).to(self.device))
        for batch_index in range(batch_size):
            batch = tokens[batch_index]
            encoded = self.encoder(batch)
            # Add the CLS token to the seq_len position
            seq_len = seq_lens[batch_index]
            encoded[seq_len] = seqCLS
            latent_logits.append(encoded)
        latent_logits = torch.stack(latent_logits)


        h = self.seq_embedding(latent_logits)
        h = self.SeqBackBone(h)
        h = h.contiguous()

        ##Perform pooling
        # Reshape index to (batch_size, 1, 1) and expand to match data's shape
        seq_lens = seq_lens.view(batch_size, 1, 1)
        seq_lens = seq_lens.expand(-1, -1, self.seq_lvl_dim).long().to(self.device)
        # Use torch.gather to retrieve values
        h = torch.gather(h, 1, seq_lens)

        h = h.view(batch_size, -1)
        output = self.output(h)
        return output

    def forward_ff(self, tokens: torch.Tensor, seq_lens) -> torch.Tensor:
        """
        Attempt to speed up the forward pass by removing the for loop
        Forward pass of the model.
        Returns the raw logits.
        
        Args:
            tokens (torch.Tensor): The input tokens.
            seq_lens (torch.Tensor): The sequence length of the input tokens.
        
        Returns:
            torch.Tensor: The raw logits.
        """
        batch_size = tokens.shape[0]
        tokens = tokens.reshape(batch_size*self.latent_len, -1)
        latent_logits = self.encoder(tokens)
        latent_logits = latent_logits.reshape(batch_size, self.latent_len, -1)
        # Add the CLS token to the seq_len position
        seq_lens = seq_lens.view(batch_size, 1, 1)
        seq_len_inserts = seq_lens.expand(-1, -1, self.embedding_dim).long().to(self.device)
        seqCLS = self.SeqCLSembedding(torch.LongTensor([0]).to(self.device))
        seqCLS = seqCLS.expand(batch_size, -1, -1)

        latent_logits = latent_logits.scatter(1, seq_len_inserts, seqCLS)
        

        h = self.seq_embedding(latent_logits)

        h = self.SeqBackBone(h)
        h = h.contiguous()

        ##Perform pooling
        # Reshape index to (batch_size, 1, 1) and expand to match data's shape
        seq_len_retrievals = seq_lens.expand(-1, -1, self.seq_lvl_dim).long().to(self.device)
        # Use torch.gather to retrieve values
        h = torch.gather(h, 1, seq_len_retrievals)

        h = h.view(batch_size, -1)
        output = self.output(h)
        return output

# class ClassifierModel(nn.Module):
#     def __init__(self, Backbone, Embedding, EmbeddingDim, latent_dim, NumClasses, latent_len, device: torch.device):
#         super().__init__()
#         self.embedding = Embedding.to(device)
#         self.backbone = Backbone.to(device)
#         self.output = nn.Linear(latent_dim*latent_len, NumClasses, bias=False).to(device)

#     def forward(self, tokens:torch.Tensor):
#         """
#         Forward pass of the model. Returns the raw classifier logits.
#         Args:
#             tokens (torch.Tensor): The input tokens.
#         Returns:
#             torch.Tensor: The raw logits.
#         """
#         h = self.embedding(tokens)
#         h = self.backbone(h)
#         h = h.contiguous()
#         h = h.view(-1, h.size(1) * h.size(2))
#         output = self.output(h)
#         return output

#     def predict(self, tokens:torch.Tensor):
#         """
#         Predict the attack label class probabilities.
#         Args:
#             tokens (torch.Tensor): The input tokens.
#         Returns:
#             torch.Tensor: The class probabilities.
#         """
#         logits = self(tokens)
#         return torch.softmax(logits, dim=-1)



class HierarchicalModel(nn.Module):
    def __init__(self, n, emb_dim, latent_len, num_classes, embedding, encoder):
        super(HierarchicalModel, self).__init__()
        self.emb_dim = emb_dim
        self.latent_len = latent_len
        self.embedding = embedding
        self.encoder = encoder
        self.seq2seq =  Mamba(
                d_model=emb_dim,
                d_state=16,
                d_conv=4,
                expand=2
            ).to(device)
        self.classifier = nn.Linear(n*latent_len*emb_dim, num_classes)

    def forward(self, input):
        batch_size = input.shape[0]
        latent_logits = []
        for batch_index in range(batch_size):
            batch = input[batch_index]
            embedded = self.embedding(batch)
            latent_logits.append(self.encoder(embedded))
        latent_logits = torch.stack(latent_logits)
        latent_logits = latent_logits.reshape(batch_size, -1, self.emb_dim) # -1 is n*latent_len
        latent_logits = self.seq2seq(latent_logits)
        latent_logits = latent_logits.view(batch_size, -1)
        return self.classifier(latent_logits)  # Classification output


if __name__ == "__main__":
    # --- Config ---
    vocab_size = 260
    emb_dim = 64
    seq_lvl_dim = 64
    packet_len = 1520       # token length per packet
    latent_len = 1           # encoder output length per packet (after pooling)
    seq_len = 16             # max packets per sequence
    num_classes = 5
    batch_size = 4

    print(f"Device: {device}")
    print("Building Packet_Encoder...")

    # Build the packet-level encoder with CLS pooling (reduces each packet to 1 token)
    encoder = Packet_Encoder(
        vocab_size=vocab_size,
        embedding_dim=emb_dim,
        input_len=packet_len,
        latent_dim=emb_dim,
        latent_len=latent_len,
        num_classes=num_classes,
        device=device,
        Pooling=CLSPooling(),
    )

    print("Building SequenceClassifier...")
    model = SequenceClassifier(
        vocab_size=vocab_size,
        embedding_dim=emb_dim,
        seq_lvl_dim=seq_lvl_dim,
        latent_len=latent_len,
        sequence_length=seq_len,
        num_classes=num_classes,
        device=device,
        PacketEncoder=encoder,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # --- Generate random data ---
    # Each batch item: (seq_len+1, packet_len) — +1 for the CLS slot
    # seq_lens: actual number of packets per sample (CLS goes at this index)
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len + 1, packet_len)).to(device)
    seq_lens = torch.randint(1, seq_len, (batch_size,))  # random lengths in [1, seq_len)

    print(f"\nInput tokens shape:  {tokens.shape}")
    print(f"Sequence lengths:    {seq_lens.tolist()}")

    # --- Forward pass (loop version) ---
    print("\n--- forward (loop) ---")
    model.eval()
    with torch.no_grad():
        logits = model(tokens, seq_lens)
    print(f"Output logits shape: {logits.shape}")  # expect (batch_size, num_classes)
    print(f"Logits:\n{logits}")

    # --- Forward pass (fast/flat version) ---
    print("\n--- forward_ff (batched) ---")
    with torch.no_grad():
        logits_ff = model.forward_ff(tokens, seq_lens)
    print(f"Output logits shape: {logits_ff.shape}")
    print(f"Logits:\n{logits_ff}")

    # --- Quick sanity checks ---
    assert logits.shape == (batch_size, num_classes), f"Unexpected shape: {logits.shape}"
    assert logits_ff.shape == (batch_size, num_classes), f"Unexpected shape: {logits_ff.shape}"
    print("\nAll checks passed!")