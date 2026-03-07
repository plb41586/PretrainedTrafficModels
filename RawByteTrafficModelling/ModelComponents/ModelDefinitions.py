import torch
import torch.nn as nn
from mamba_ssm import Mamba
from dataclasses import dataclass
from RawByteTrafficModelling.ModelComponents.DataUtils import ID_Encoder
import math

@dataclass
class ModelParams:
    """
    MLM Model Parameters
    """
    vocab_size: int = 262
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

class MambaBackbone(nn.Module):
    """Stacked Mamba (S4/SSM) backbone for sequence modeling.

    Applies multiple Mamba layers with residual connections and layer
    normalization.

    Args:
        d_model:    Dimensionality of the input and output embeddings.
        num_layers: Number of stacked Mamba layers.
        d_state:    State dimensionality of the structured state space model.
        d_conv:     Kernel size of the local convolution in each Mamba block.
        expand:     Expansion factor for the inner projection dimension.
        dropout:    Dropout probability applied before and after the layer stack.
    """

    def __init__(self, d_model: int, num_layers: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the input through all Mamba layers.

        Args:
            x: Input tensor of shape ``(batch, seq_len, d_model)``.

        Returns:
            Output tensor of the same shape after applying all Mamba
            layers with residual connections, layer normalization, and
            dropout.
        """
        h = self.dropout(x)
        for mamba, norm in zip(self.layers, self.norms):
            h = norm(mamba(h) + h)
        h = self.dropout(h)
        return h


class TransformerBackbone(nn.Module):
    """Standard Transformer encoder backbone with sinusoidal positional encoding.

    Wraps ``nn.TransformerEncoder`` with fixed sinusoidal position embeddings
    and configurable depth/dropout.

    Args:
        d_model:    Dimensionality of the input and output embeddings.
        nhead:      Number of attention heads per layer.
        num_layers: Number of stacked TransformerEncoderLayers.
        max_len:    Maximum supported sequence length for positional encoding.
        dropout:    Dropout probability applied to positional embeddings and
                    within each transformer layer.
    """

    def __init__(self, d_model: int, nhead: int, num_layers: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self._init_sinusoidal_encoding(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def _init_sinusoidal_encoding(self, max_len: int, d_model: int) -> None:
        """Generate and register fixed sinusoidal positional encodings.

        Args:
            max_len: Maximum sequence length.
            d_model: Embedding dimensionality.
        """
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pos_encoding', pe.unsqueeze(0))

    @staticmethod
    def generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate an upper-triangular causal attention mask.

        Args:
            seq_len: Length of the sequence.
            device:  Device to create the mask on.

        Returns:
            A ``(seq_len, seq_len)`` mask with ``-inf`` above the diagonal
            and ``0`` on and below, suitable for ``nn.TransformerEncoder``.
        """
        return torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, causal: bool = False) -> torch.Tensor:
        """Run the input through positional encoding and all transformer layers.

        Args:
            x:      Input tensor of shape ``(batch, seq_len, d_model)``.
            mask:   Optional attention mask passed to the transformer encoder.
                    Takes precedence over ``causal`` if provided.
            causal: If ``True`` and ``mask`` is ``None``, automatically
                    generates a causal (upper-triangular) attention mask.

        Returns:
            Output tensor of the same shape.
        """
        h = self.dropout(x + self.pos_encoding[:, :x.size(1), :])
        if mask is None and causal:
            mask = self.generate_causal_mask(x.size(1), x.device)
        h = self.transformer_encoder(h, mask=mask)
        h = self.dropout(h)
        return h


class Packet_MLM(nn.Module):
    """Dual-head model for Masked Language Modeling and sequence classification.

    Combines a token embedding layer, a swappable sequence backbone (e.g.
    ``TransformerBackbone`` or ``MambaBackbone``), and two output heads:
    one for per-token reconstruction (MLM) and one for sequence-level
    classification (CLS).

    Args:
        vocab_size:       Size of the token vocabulary.
        embedding_dim:    Dimensionality of token embeddings (must match
                          the backbone's ``d_model``).
        num_CLS_classes:  Number of target classes for the classification head.
        CLS_Pooling:      Module that reduces the backbone's per-token output
                          to a single sequence-level vector. Receives
                          ``(hidden_states, tokens)`` and returns a tensor
                          of shape ``(batch, embedding_dim)``.
        Backbone:         Sequence modeling backbone (any ``nn.Module`` mapping
                          ``(batch, seq_len, embedding_dim)`` →
                          ``(batch, seq_len, embedding_dim)``).
        device:           Device to place all sub-modules on.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, num_CLS_classes: int,
                 CLS_Pooling: nn.Module, Backbone: nn.Module, device: torch.device):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.Backbone = Backbone
        self.reconstruction_output = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.CLS_Pooling = CLS_Pooling
        self.CLS_output = nn.Linear(embedding_dim, num_CLS_classes, bias=False)
        self.to(device)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass producing both reconstruction and classification logits.

        Args:
            tokens: Integer token indices of shape ``(batch, seq_len)``.

        Returns:
            A tuple of:
                - **reconstruction_output**: Per-token logits over the
                  vocabulary, shape ``(batch, seq_len, vocab_size)``.
                - **CLS_output**: Classification logits, shape
                  ``(batch, num_CLS_classes)``.
        """
        h = self.embedding(tokens)
        h = self.Backbone(h)
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


class AutoregressiveDecoder(nn.Module):
    """Autoregressive decoder that reconstructs a token sequence from a CLS embedding.

    The CLS embedding is prepended as the first token in the decoder input,
    acting as a bottleneck representation. During training, teacher forcing
    is used by shifting the ground truth tokens and prepending the CLS
    embedding. During inference, tokens are generated one at a time.

    Args:
        vocab_size:     Size of the token vocabulary.
        embedding_dim:  Dimensionality of token embeddings (must match the
                        CLS embedding and backbone ``d_model``).
        max_len:        Maximum sequence length for generation.
        Backbone:       Sequence modeling backbone (any ``nn.Module`` mapping
                        ``(batch, seq_len, embedding_dim)`` →
                        ``(batch, seq_len, embedding_dim)``). Should support
                        causal masking if using a Transformer backbone.
        bos_token_id:   Beginning-of-sequence token id used to seed
                        autoregressive inference.
        device:         Device to place all sub-modules on.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, max_len: int,
                 Backbone: nn.Module, bos_token_id: int, device: torch.device):
        super().__init__()
        self.max_len = max_len
        self.bos_token_id = bos_token_id
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.cls_proj = nn.Linear(embedding_dim, embedding_dim)
        self.Backbone = Backbone
        self.output_head = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.to(device)

    def _build_decoder_input(self, cls_embedding: torch.Tensor, tokens: torch.Tensor = None) -> torch.Tensor:
        """Prepend the projected CLS embedding to the token embeddings.

        Args:
            cls_embedding: CLS vector of shape ``(batch, embedding_dim)``.
            tokens:        Optional token indices of shape ``(batch, seq_len)``.
                           If ``None``, only the CLS embedding is returned.

        Returns:
            Decoder input of shape ``(batch, 1 + seq_len, embedding_dim)``
            or ``(batch, 1, embedding_dim)`` if tokens are not provided.
        """
        cls_token = self.cls_proj(cls_embedding).unsqueeze(1)  # (batch, 1, embedding_dim)
        if tokens is None:
            return cls_token
        token_embeds = self.embedding(tokens)  # (batch, seq_len, embedding_dim)
        return torch.cat([cls_token, token_embeds], dim=1)

    def forward(self, cls_embedding: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
        """Training forward pass with teacher forcing.

        The input to the decoder is ``[CLS_proj | target_tokens[:-1]]``,
        i.e. the CLS embedding followed by all target tokens except the
        last. The model predicts each next token, producing logits aligned
        with ``target_tokens``.

        Args:
            cls_embedding:  CLS vector of shape ``(batch, embedding_dim)``.
            target_tokens:  Ground truth token indices of shape
                            ``(batch, seq_len)`` used for teacher forcing.

        Returns:
            Logits of shape ``(batch, seq_len, vocab_size)`` aligned with
            ``target_tokens`` (i.e. logits[i] predicts target_tokens[i]).
        """
        # Shift right: drop last target token so output length matches target
        shifted = target_tokens[:, :-1]  # (batch, seq_len - 1)
        h = self._build_decoder_input(cls_embedding, shifted)  # (batch, 1 + seq_len - 1, dim)
        h = self.Backbone(h)
        logits = self.output_head(h)  # (batch, seq_len, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, cls_embedding: torch.Tensor, max_len: int = None,
                 temperature: float = 1.0, eos_token_id: int = None) -> torch.Tensor:
        """Autoregressively generate a token sequence from a CLS embedding.

        Args:
            cls_embedding:  CLS vector of shape ``(batch, embedding_dim)``.
            max_len:        Maximum number of tokens to generate. Defaults
                            to ``self.max_len``.
            temperature:    Sampling temperature. Values < 1 sharpen the
                            distribution, > 1 flatten it. Use 0 for greedy.
            eos_token_id:   Optional end-of-sequence token id. Generation
                            stops early for a sample once this token is
                            produced.

        Returns:
            Generated token indices of shape ``(batch, generated_len)``.
        """
        max_len = max_len or self.max_len
        batch_size = cls_embedding.size(0)
        device = cls_embedding.device

        generated = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            h = self._build_decoder_input(cls_embedding, generated)
            h = self.Backbone(h)
            next_logits = h[:, -1, :]  # (batch, embedding_dim)
            next_logits = self.output_head(next_logits)  # (batch, vocab_size)

            if temperature == 0:
                next_token = next_logits.argmax(dim=-1)
            else:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

            # Don't update already finished sequences
            next_token = next_token.masked_fill(finished, self.bos_token_id)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

            if eos_token_id is not None:
                finished = finished | (next_token == eos_token_id)
                if finished.all():
                    break

        return generated


class SequenceClassifier(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 embedding_dim: int, 
                 seq_lvl_dim: int, 
                 packets_per_sequence: int, 
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
        self.latent_len = packets_per_sequence # account for CLS token 
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

if __name__ == "__main__":
    import numpy as np
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CLS (Classify) token placed ad End Of Sequence
    TokenIDEncoder = ID_Encoder(SpecialIDs = {"<pad>": 256, "</s>": 257, "<CLS>": 258, "<mask>": 259}, CLS_Placement="EOS")

    # --- Config ---
    vocab_size = 260
    emb_dim = 32
    seq_lvl_dim = 32
    bytes_per_packet = 1520       # token length per packet
    packets_per_sequence = 64             # max packets per sequence
    num_classes = 14
    batch_size = 32

    print(f"Device: {device}")
    print("Building Packet_Encoder...")

    # Build the packet-level encoder with CLS pooling (reduces each packet to 1 token)
    encoder = Packet_Encoder(
        vocab_size=vocab_size,
        embedding_dim=emb_dim,
        input_len=bytes_per_packet,
        latent_dim=emb_dim,
        latent_len=1,
        num_classes=num_classes,
        device=device,
        Pooling=DynamicCLSPooling(258),
    )

    print("Building SequenceClassifier...")
    model = SequenceClassifier(
        vocab_size=vocab_size,
        embedding_dim=emb_dim,
        seq_lvl_dim=seq_lvl_dim,
        packets_per_sequence=packets_per_sequence,
        num_classes=num_classes,
        device=device,
        PacketEncoder=encoder,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # --- Generate random data ---
    data_samples = []
    passes = 1
    for i in range(passes):
        input_data = torch.randint(0, vocab_size-1, (batch_size, packets_per_sequence, bytes_per_packet)).to(device)
        data_samples.append(input_data)
    PackSeqLens = torch.randint(1, packets_per_sequence, (batch_size,))

    PackSeqLens = PackSeqLens.to(device)
    # print(f"\nInput tokens shape:  {tokens.shape}")
    # print(f"Sequence lengths:    {seq_lens.tolist()}")

    # --- Forward pass (loop version) ---
    print("\n--- forward (loop) ---")
    model.eval().to(device)
    with torch.no_grad():
        if device == "cuda":
            torch.cuda.synchronize()
        
        for i in range(passes):
            logits = model.forward(data_samples[i], PackSeqLens)
            logits_ff = model.forward_ff(data_samples[i], PackSeqLens)

    print(f"Output logits shape: {logits.shape}")  # expect (batch_size, num_classes)
    print(f"Logits:\n{logits}")

    print(f"Output logits shape: {logits_ff.shape}")
    print(f"Logits:\n{logits_ff}")

    # --- Quick sanity checks ---
    assert logits.shape == (batch_size, num_classes), f"Unexpected shape: {logits.shape}"
    assert logits_ff.shape == (batch_size, num_classes), f"Unexpected shape: {logits_ff.shape}"
    diff = (logits-logits_ff).sum().cpu().numpy()
    assert diff == np.array(0., dtype="float32"), "Logits dont match up!"

    print("\nAll checks passed!")