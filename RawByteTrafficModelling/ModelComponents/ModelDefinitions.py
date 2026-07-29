import torch
import torch.nn as nn
from mamba_ssm import Mamba
from dataclasses import dataclass, asdict, fields
from RawByteTrafficModelling.ModelComponents.DataUtils import ID_Encoder
import math
from RawByteTrafficModelling.ModelComponents.BackBones import BackboneParams, TransformerBackboneParams, MambaBackboneParams, MambaBackbone, TransformerBackbone

# Backbone Factory
BACKBONES = {"Transformer": (TransformerBackboneParams, TransformerBackbone),
             "Mamba": (MambaBackboneParams, MambaBackbone)}

def build_backbone(kind, params):
    try: _, cls = BACKBONES[kind]
    except KeyError: raise ValueError(f"Unsupported backbone: {kind}")
    return cls(params)


def unpack_backbone_params(type: str, config: dict):
    if type == "Mamba":
        BackboneParams = MambaBackboneParams(**config)
    elif type == "Transformer":
        BackboneParams = TransformerBackboneParams(**config)
    else: raise Exception("Backbone Type is not supported")
    return BackboneParams

@dataclass
class EncoderParams:
    vocab_size: int
    EncoderDim: int
    packet_id_len: int
    pooling_type: str
    BackboneType: str
    BackboneParams: BackboneParams
    CLS_ID: int
    SpecialTokens: dict

def unpack_encoder_params(config: dict):
    params = EncoderParams(**config)
    params.BackboneParams = unpack_backbone_params(params.BackboneType,
                                                   params.BackboneParams)
    return params


@dataclass
class MLM_Params:
    EncoderParams: EncoderParams
    NumCLSclasses: int = None

def load_MLM_checkpoint(path: str, device='cpu'):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = MLM_Params(**ckpt['config'])
    config.EncoderParams = unpack_encoder_params(config.EncoderParams)
    return config, ckpt


@dataclass
class PacketClassifierParams:
    EncoderParams: EncoderParams
    NumAttackClasses: int = None

@dataclass
class AutoEncoderParams:
    ENC_Params: EncoderParams
    DecBackboneType: str
    DecBackbone: BackboneParams
    bos_token_id: int

def load_AE_Checkpoint(path: str, device='cpu'):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = AutoEncoderParams(**ckpt['config'])
    config.ENC_Params = unpack_encoder_params(config.ENC_Params)
    config.DecBackbone = unpack_backbone_params(config.DecBackboneType,
                                                config.DecBackbone)
    return config, ckpt

def save_checkpoint(model, optimizer, epoch, loss, config, path: str, extra: dict = None):
    """``extra`` is merged into the saved dict -- scheduler state, best val loss, etc."""
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': asdict(config),
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)

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

    def __init__(self, params: MLM_Params, Backbone = None):
        super().__init__()
        ENCparams = params.EncoderParams
        self.embedding = nn.Embedding(ENCparams.vocab_size, ENCparams.EncoderDim)
        if Backbone == None:
            self.Backbone = build_backbone(ENCparams.BackboneType, ENCparams.BackboneParams)
        else:
            self.Backbone = Backbone
        self.reconstruction_output = nn.Linear(ENCparams.EncoderDim, ENCparams.vocab_size, bias=False)
        self.CLS_Pooling = DynamicCLSPooling(ENCparams.CLS_ID)
        self.CLS_output = nn.Linear(ENCparams.EncoderDim, params.NumCLSclasses, bias=False)

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
                    params: PacketClassifierParams,
                    encoder: nn.Module = None):
        super().__init__()
        ENCparams = params.EncoderParams
        
        if encoder is None:
            self.encoder = Packet_Encoder(ENCparams)
        else:
            self.encoder = encoder
        
        self.output = nn.Linear(ENCparams.EncoderDim, params.NumAttackClasses, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model. Returns the raw logits.
        
        Args:
            tokens (torch.Tensor): The input tokens.
        
        Returns:
            torch.Tensor: The raw logits.
        """
        h = self.encoder(tokens)
        h = h.view(tokens.shape[0], -1)
        output = self.output(h)
        return output

class Packet_Encoder(nn.Module):
    def __init__(   self, 
                    params: EncoderParams,
                    embedding: nn.Module = None,
                    BackBone: nn.Module = None,
                    Pooling: nn.Module = None):
        super().__init__()
        
        if embedding is None:
            self.embedding = nn.Embedding(params.vocab_size, params.EncoderDim)
        else:
            self.embedding = embedding
        
        if BackBone == None:
            self.Backbone = build_backbone(params.BackboneType, params.BackboneParams)
        else:
            self.Backbone = BackBone

        if Pooling is None:
            self.Pooling = DynamicCLSPooling(params.CLS_ID)
        else:
            self.Pooling = Pooling

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model. Returns the raw logits.
        
        Args:
            tokens (torch.Tensor): The input tokens.
        
        Returns:
            torch.Tensor: The compressed latent representation.
        """
        h = self.embedding(tokens)
        h = self.Backbone(h)
        h = self.Pooling(h, tokens)
        return h
        
class AutoregressiveDecoder(nn.Module):
    """Autoregressive decoder that reconstructs a token sequence from a CLS embedding.

    The projected CLS embedding acts as the first position in the decoder
    input (the bottleneck representation). During training, teacher forcing
    prepends the CLS embedding to the shifted target tokens. During
    inference, the CLS embedding alone seeds generation and tokens are
    produced one at a time.

    Args (packed in AutoEncoderParams):
        vocab_size:     Size of the token vocabulary.
        embedding_dim:  Dimensionality of token embeddings (must match the
                        CLS embedding and backbone ``d_model``).
        max_len:        Maximum sequence length for generation.
        DecBackbone:    Backbone params for the decoder. Should support
                        causal masking if using a Transformer backbone.
        pad_token_id:   Pad token id, used to fill finished sequences during
                        generation.
    """

    def __init__(self, params: AutoEncoderParams, Backbone=None):
        super().__init__()
        embedding_dim = params.ENC_Params.EncoderDim
        vocab_size = params.ENC_Params.vocab_size
        self.max_len = params.ENC_Params.packet_id_len
        self.pad_token_id = params.ENC_Params.SpecialTokens["<pad>"]
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.cls_proj = nn.Linear(embedding_dim, embedding_dim)
        if Backbone is None:
            self.Backbone = build_backbone(params.DecBackboneType, params.DecBackbone)
        else:
            self.Backbone = Backbone
        self.output_head = nn.Linear(embedding_dim, vocab_size, bias=False)

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
        if tokens is None or tokens.size(1) == 0:
            return cls_token
        token_embeds = self.embedding(tokens)  # (batch, seq_len, embedding_dim)
        return torch.cat([cls_token, token_embeds], dim=1)

    def forward(self, cls_embedding: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
        """Training forward pass with teacher forcing.

        The decoder input is ``[CLS_proj | target_tokens[:-1]]`` — the CLS
        embedding followed by all target tokens except the last — so the
        output at position i predicts ``target_tokens[i]``.

        Args:
            cls_embedding:  CLS vector of shape ``(batch, embedding_dim)``.
            target_tokens:  Ground truth token indices of shape
                            ``(batch, seq_len)``.

        Returns:
            Logits of shape ``(batch, seq_len, vocab_size)`` aligned with
            ``target_tokens``.
        """
        shifted = target_tokens[:, :-1]  # (batch, seq_len - 1)
        h = self._build_decoder_input(cls_embedding, shifted)  # (batch, seq_len, dim)
        h = self.Backbone(h)
        logits = self.output_head(h)  # (batch, seq_len, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, cls_embedding: torch.Tensor, max_len: int = None,
                 temperature: float = 1.0, eos_token_id: int = None) -> torch.Tensor:
        """Autoregressively generate a token sequence from a CLS embedding.

        Generation starts from the CLS embedding alone (matching training,
        where position 0 is the CLS token and predicts the first real token).

        Args:
            cls_embedding:  CLS vector of shape ``(batch, embedding_dim)``.
            max_len:        Maximum number of tokens to generate. Defaults
                            to ``self.max_len``.
            temperature:    Sampling temperature. Values < 1 sharpen, > 1
                            flatten. Use 0 for greedy.
            eos_token_id:   Optional end-of-sequence token id. Generation
                            stops early once every sample has emitted it.

        Returns:
            Generated token indices of shape ``(batch, generated_len)``.
        """
        max_len = max_len or self.max_len
        batch_size = cls_embedding.size(0)
        device = cls_embedding.device

        generated = torch.empty((batch_size, 0), dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            h = self._build_decoder_input(cls_embedding, generated)
            h = self.Backbone(h)
            next_logits = self.output_head(h[:, -1, :])  # (batch, vocab_size)

            if temperature == 0:
                next_token = next_logits.argmax(dim=-1)
            else:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

            # Keep finished sequences padded
            next_token = next_token.masked_fill(finished, self.pad_token_id)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

            if eos_token_id is not None:
                finished = finished | (next_token == eos_token_id)
                if finished.all():
                    break

        return generated

class PacketAutoencoder(nn.Module):
    """End-to-end autoencoder that encodes a token sequence into a latent
    representation via a ``Packet_Encoder`` and reconstructs it
    autoregressively via an ``AutoregressiveDecoder``.

    Args:
        vocab_size:     Size of the token vocabulary.
        embedding_dim:  Dimensionality of token embeddings.
        max_len:        Maximum sequence length for reconstruction.
        decoder_backbone: Backbone module for the decoder (any ``nn.Module``
                        mapping ``(batch, seq_len, embedding_dim)`` →
                        ``(batch, seq_len, embedding_dim)``).
        bos_token_id:   Beginning-of-sequence token id used to seed
                        autoregressive generation.
        device:         Device to place all sub-modules on.
        encoder:        Optional pre-built ``Packet_Encoder``. If ``None``,
                        a default encoder is constructed from the remaining
                        keyword arguments.
        encoder_kwargs: Additional keyword arguments forwarded to
                        ``Packet_Encoder`` when ``encoder`` is ``None``
                        (e.g. ``BackBone``, ``Pooling``, ``latent_dim``,
                        ``latent_len``, ``input_len``).
    """

    def __init__(self, 
                params: AutoEncoderParams,
                decoder: nn.Module = None,
                encoder: nn.Module = None):
        super().__init__()

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = Packet_Encoder(params.ENC_Params)

        if decoder is not None:
            self.decoder = decoder
        else:
            self.decoder = AutoregressiveDecoder(params)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Training forward pass with teacher forcing.

        Encodes the input tokens into a latent representation, then
        autoregressively reconstructs them using teacher forcing.

        Args:
            tokens: Input token indices of shape ``(batch, seq_len)``.

        Returns:
            A tuple of:
                - **logits**: Reconstruction logits of shape
                  ``(batch, seq_len, vocab_size)`` aligned with ``tokens``.
                - **latent**: The encoder's latent output, useful for
                  auxiliary losses (e.g. regularization, contrastive).
        """
        latent = self.encoder(tokens)
        logits = self.decoder(latent, tokens)
        return logits, latent

    @torch.no_grad()
    def reconstruct(self, tokens: torch.Tensor, max_len: int = None,
                    temperature: float = 1.0, eos_token_id: int = None) -> torch.Tensor:
        """Encode and autoregressively reconstruct a token sequence.

        Args:
            tokens:        Input token indices of shape ``(batch, seq_len)``.
            max_len:       Maximum generation length. Defaults to the
                           decoder's ``max_len``.
            temperature:   Sampling temperature (0 for greedy).
            eos_token_id:  Optional early stopping token.

        Returns:
            Reconstructed token indices of shape ``(batch, generated_len)``.
        """
        latent = self.encoder(tokens)
        return self.decoder.generate(latent, max_len=max_len,
                                     temperature=temperature,
                                     eos_token_id=eos_token_id)

class SequenceClassifier(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 embedding_dim: int, 
                 seq_lvl_dim: int, 
                 packets_per_sequence: int, 
                 num_classes: int, 
                 PacketEncoder: nn.Module):
        super().__init__()

        self.encoder = PacketEncoder

        if seq_lvl_dim != embedding_dim:
            self.seq_embedding = nn.Linear(embedding_dim, seq_lvl_dim)
        else:
            self.seq_embedding = nn.Identity()

        self.SeqBackBone = Mamba(
                d_model=seq_lvl_dim,
                d_state=16,
                d_conv=4,
                expand=2
            )

        self.SeqCLSembedding = nn.Embedding(1, embedding_dim)

        self.embedding_dim = embedding_dim
        self.seq_lvl_dim = seq_lvl_dim
        self.latent_len = packets_per_sequence # account for CLS token 
        self.vocab_size = vocab_size
        self.output = nn.Linear(seq_lvl_dim, num_classes)

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
        seqCLS = self.SeqCLSembedding(torch.zeros(1, dtype=torch.long, device=tokens.device))
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
        seq_lens = seq_lens.expand(-1, -1, self.seq_lvl_dim).long()
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
        seq_len_inserts = seq_lens.expand(-1, -1, self.embedding_dim).long()
        seqCLS = self.SeqCLSembedding(torch.zeros(1, dtype=torch.long, device=tokens.device))
        seqCLS = seqCLS.expand(batch_size, -1, -1)

        latent_logits = latent_logits.scatter(1, seq_len_inserts, seqCLS)
        

        h = self.seq_embedding(latent_logits)

        h = self.SeqBackBone(h)
        h = h.contiguous()

        ##Perform pooling
        # Reshape index to (batch_size, 1, 1) and expand to match data's shape
        seq_len_retrievals = seq_lens.expand(-1, -1, self.seq_lvl_dim).long()
        # Use torch.gather to retrieve values
        h = torch.gather(h, 1, seq_len_retrievals)

        h = h.view(batch_size, -1)
        output = self.output(h)
        return output

# ---------------------------------------------------------------------------
# Sequence-level params
# ---------------------------------------------------------------------------

@dataclass
class SeqEncoderParams:
    EncoderParams: EncoderParams        # packet-level encoder
    SeqEncoderDim: int                  # D_seq
    packets_per_sequence: int           # P (includes the slot used by seq-CLS)
    SeqBackboneType: str
    SeqBackboneParams: BackboneParams   # d_model must equal SeqEncoderDim


def unpack_seq_encoder_params(config: dict):
    params = SeqEncoderParams(**config)
    params.EncoderParams = unpack_encoder_params(params.EncoderParams)
    params.SeqBackboneParams = unpack_backbone_params(params.SeqBackboneType,
                                                      params.SeqBackboneParams)
    return params


@dataclass
class SeqAutoEncoderParams:
    SeqEncParams: SeqEncoderParams
    SeqDecoderDim: int                  # D_dec
    SeqDecBackboneType: str
    SeqDecBackbone: BackboneParams      # d_model must equal SeqDecoderDim
    target_mean: list = None            # (D,) packet-latent normalisation stats
    target_std: list = None
    length_head: bool = True
    length_loss_weight: float = 0.1


def unpack_seq_ae_params(config: dict):
    params = SeqAutoEncoderParams(**config)
    params.SeqEncParams = unpack_seq_encoder_params(params.SeqEncParams)
    params.SeqDecBackbone = unpack_backbone_params(params.SeqDecBackboneType,
                                                   params.SeqDecBackbone)
    return params


def load_SeqAE_checkpoint(path: str, device='cpu'):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = unpack_seq_ae_params(ckpt['config'])
    return config, ckpt

def build_padding_mask(seq_lens: torch.Tensor, num_packets: int) -> torch.Tensor:
    """(B,) sequence lengths -> (B, P) bool mask, True where a real packet sits."""
    ar = torch.arange(num_packets, device=seq_lens.device).unsqueeze(0)
    return ar < seq_lens.view(-1, 1)


class Sequence_Encoder(nn.Module):
    """Encodes a batch of packet sequences into a single flow-level vector.

    Mirrors ``Packet_Encoder`` one level up: the packet encoder plays the role
    of the embedding table, a sequence backbone contextualises the per-packet
    latents, and a CLS position is pooled out.

    The sequence CLS embedding is written at index ``seq_len`` (i.e. directly
    after the last real packet), matching ``SequenceClassifier`` and keeping
    the model valid under a causal backbone.

    Accepts either raw ``tokens`` of shape ``(B, P, L)`` or pre-computed
    packet ``latents`` of shape ``(B, P, D)``. The latter path is what you
    want when the packet encoder is frozen and its outputs are cached.
    """

    def __init__(self, params: SeqEncoderParams,
                 packet_encoder: nn.Module = None,
                 SeqBackbone: nn.Module = None,
                 freeze_packet_encoder: bool = True):
        super().__init__()
        self.params = params
        D = params.EncoderParams.EncoderDim
        D_seq = params.SeqEncoderDim
        self.packet_latent_dim = D
        self.seq_lvl_dim = D_seq
        self.num_packets = params.packets_per_sequence

        if packet_encoder is None:
            self.packet_encoder = Packet_Encoder(params.EncoderParams)
        else:
            self.packet_encoder = packet_encoder

        self.frozen_packet_encoder = freeze_packet_encoder
        if freeze_packet_encoder:
            self.freeze_packet_encoder()

        self.seq_embedding = nn.Linear(D, D_seq) if D != D_seq else nn.Identity()

        if SeqBackbone is None:
            self.SeqBackbone = build_backbone(params.SeqBackboneType,
                                              params.SeqBackboneParams)
        else:
            self.SeqBackbone = SeqBackbone

        # CLS lives in sequence space, so it is not distorted by seq_embedding
        self.SeqCLSembedding = nn.Parameter(torch.zeros(1, 1, D_seq))
        nn.init.normal_(self.SeqCLSembedding, std=0.02)

    def freeze_packet_encoder(self):
        self.frozen_packet_encoder = True
        for p in self.packet_encoder.parameters():
            p.requires_grad = False
        self.packet_encoder.eval()

    def unfreeze_packet_encoder(self):
        self.frozen_packet_encoder = False
        for p in self.packet_encoder.parameters():
            p.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen_packet_encoder:
            self.packet_encoder.eval()   # keep frozen encoder out of train mode
        return self

    def encode_packets(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, P, L) tokens -> (B, P, D) packet latents, without the python loop."""
        B, P, L = tokens.shape
        flat = tokens.reshape(B * P, L)
        if self.frozen_packet_encoder:
            with torch.no_grad():
                latents = self.packet_encoder(flat)
        else:
            latents = self.packet_encoder(flat)
        return latents.reshape(B, P, -1)

    def forward(self, seq_lens: torch.Tensor,
                tokens: torch.Tensor = None,
                latents: torch.Tensor = None):
        """
        Args:
            seq_lens: (B,) number of real packets per sequence.
            tokens:   (B, P, L) raw byte tokens, or
            latents:  (B, P, D) pre-computed packet latents.

        Returns:
            z:       (B, D_seq) flow-level bottleneck vector.
            latents: (B, P, D)  packet latents (the AE reconstruction targets).
        """
        if latents is None:
            if tokens is None:
                raise ValueError("Provide either tokens or latents")
            latents = self.encode_packets(tokens)

        B, P, _ = latents.shape
        seq_lens = seq_lens.view(B).long()

        h = self.seq_embedding(latents)                       # (B, P, D_seq)

        cls_idx = seq_lens.view(B, 1, 1).expand(-1, -1, self.seq_lvl_dim)
        h = h.scatter(1, cls_idx, self.SeqCLSembedding.expand(B, -1, -1))

        h = self.SeqBackbone(h)
        h = h.contiguous()

        z = torch.gather(h, 1, cls_idx).squeeze(1)            # (B, D_seq)
        return z, latents

class SequenceDecoder(nn.Module):
    """Reconstructs P packet latents from a single flow vector, in parallel.

    Input to the backbone is ``[cls_proj(z) | query_0 ... query_{P-1}]``.
    Under a causal backbone each query still sees the CLS at position 0 plus
    the queries preceding it; since the queries carry no content, no
    information leaks and no autoregressive drift accumulates.
    """

    def __init__(self, params: SeqAutoEncoderParams, Backbone: nn.Module = None):
        super().__init__()
        D_seq = params.SeqEncParams.SeqEncoderDim
        D_dec = params.SeqDecoderDim
        D = params.SeqEncParams.EncoderParams.EncoderDim
        self.num_packets = params.SeqEncParams.packets_per_sequence

        self.cls_proj = nn.Linear(D_seq, D_dec)
        self.pos_queries = nn.Parameter(torch.zeros(1, self.num_packets, D_dec))
        nn.init.normal_(self.pos_queries, std=0.02)

        if Backbone is None:
            self.Backbone = build_backbone(params.SeqDecBackboneType,
                                           params.SeqDecBackbone)
        else:
            self.Backbone = Backbone

        self.output_head = nn.Linear(D_dec, D)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """(B, D_seq) -> (B, P, D) predicted packet latents (normalised space)."""
        B = z.size(0)
        cls_token = self.cls_proj(z).unsqueeze(1)             # (B, 1, D_dec)
        queries = self.pos_queries.expand(B, -1, -1)          # (B, P, D_dec)
        h = torch.cat([cls_token, queries], dim=1)            # (B, P+1, D_dec)
        h = self.Backbone(h)
        return self.output_head(h[:, 1:, :])                  # drop CLS position

class SequenceAutoencoder(nn.Module):
    """Option A: hard bottleneck at the sequence CLS.

    A whole flow is compressed into one vector and decoded back into all P
    packet latents. Reconstruction is supervised in *normalised* packet-latent
    space; an auxiliary length head makes the bottleneck carry sequence length
    so that standalone reconstruction at inference is well defined.
    """

    def __init__(self, params: SeqAutoEncoderParams,
                 encoder: nn.Module = None,
                 decoder: nn.Module = None,
                 freeze_packet_encoder: bool = True):
        super().__init__()
        self.params = params
        D = params.SeqEncParams.EncoderParams.EncoderDim
        D_seq = params.SeqEncParams.SeqEncoderDim
        P = params.SeqEncParams.packets_per_sequence
        self.num_packets = P

        self.encoder = encoder if encoder is not None else Sequence_Encoder(
            params.SeqEncParams, freeze_packet_encoder=freeze_packet_encoder)
        self.decoder = decoder if decoder is not None else SequenceDecoder(params)

        self.length_output = nn.Linear(D_seq, P + 1) if params.length_head else None
        self.length_loss_weight = params.length_loss_weight

        mean = torch.zeros(D) if params.target_mean is None \
            else torch.tensor(params.target_mean, dtype=torch.float)
        std = torch.ones(D) if params.target_std is None \
            else torch.tensor(params.target_std, dtype=torch.float)
        self.register_buffer("target_mean", mean)
        self.register_buffer("target_std", std)

    # -- target normalisation ------------------------------------------------

    def set_target_stats(self, mean: torch.Tensor, std: torch.Tensor, eps=1e-5):
        self.target_mean.copy_(mean.to(self.target_mean.device))
        self.target_std.copy_(std.clamp_min(eps).to(self.target_std.device))
        self.params.target_mean = self.target_mean.tolist()
        self.params.target_std = self.target_std.tolist()

    def normalize(self, latents):
        return (latents - self.target_mean) / self.target_std

    def denormalize(self, latents):
        return latents * self.target_std + self.target_mean

    # -- forward / loss ------------------------------------------------------

    def forward(self, seq_lens, tokens=None, latents=None):
        """Returns (pred_norm, target_norm, z, length_logits, mask).

        ``pred_norm`` and ``target_norm`` are both in normalised latent space,
        shape (B, P, D). ``mask`` is (B, P) and True at real packets.
        """
        z, latents = self.encoder(seq_lens, tokens=tokens, latents=latents)
        pred_norm = self.decoder(z)
        target_norm = self.normalize(latents)
        mask = build_padding_mask(seq_lens.view(-1), self.num_packets)
        length_logits = self.length_output(z) if self.length_output is not None else None
        return pred_norm, target_norm, z, length_logits, mask

    def loss(self, pred_norm, target_norm, mask, length_logits=None, seq_lens=None):
        """Padding-masked MSE in normalised space, plus optional length CE."""
        m = mask.unsqueeze(-1).float()
        se = ((pred_norm - target_norm) ** 2) * m
        recon = se.sum() / (m.sum().clamp_min(1.0) * pred_norm.size(-1))

        out = {"recon": recon, "total": recon}
        if length_logits is not None and seq_lens is not None:
            len_loss = nn.functional.cross_entropy(length_logits, seq_lens.view(-1).long())
            out["length"] = len_loss
            out["total"] = recon + self.length_loss_weight * len_loss
        return out

    # -- inference -----------------------------------------------------------

    @torch.no_grad()
    def encode(self, seq_lens, tokens=None, latents=None):
        z, _ = self.encoder(seq_lens, tokens=tokens, latents=latents)
        return z

    @torch.no_grad()
    def reconstruct_latents(self, z):
        """(B, D_seq) -> (B, P, D) in original (un-normalised) latent space."""
        return self.denormalize(self.decoder(z))

    @torch.no_grad()
    def predicted_lengths(self, z):
        if self.length_output is None:
            return None
        return self.length_output(z).argmax(dim=-1)

@torch.no_grad()
def flatten_valid(pred_norm, target_norm, mask):
    m = mask.reshape(-1)
    return pred_norm.reshape(-1, pred_norm.size(-1))[m], \
           target_norm.reshape(-1, target_norm.size(-1))[m]


@torch.no_grad()
def retrieval_accuracy(pred_norm, target_norm, mask, chunk: int = 1024):
    """Fraction of predicted latents whose nearest true latent is the correct one.

    Computed over all valid packets in the batch, so the difficulty scales with
    batch size -- report the batch size alongside the number.
    """
    p, t = flatten_valid(pred_norm, target_norm, mask)
    n = p.size(0)
    if n < 2:
        return float('nan'), n
    correct = 0
    for i in range(0, n, chunk):
        d = torch.cdist(p[i:i + chunk], t)                     # (chunk, N)
        nn_idx = d.argmin(dim=1)
        tgt = torch.arange(i, min(i + chunk, n), device=p.device)
        correct += (nn_idx == tgt).sum().item()
    return correct / n, n


@torch.no_grad()
def baseline_mses(target_norm, mask):
    """Two 'learn nothing' baselines the model must beat.

    global:       predict the mean latent everywhere
    per_position: predict the mean latent for each packet index
    """
    m = mask.unsqueeze(-1).float()
    denom = (m.sum() * target_norm.size(-1)).clamp_min(1.0)

    glob = (target_norm * m).sum(dim=(0, 1)) / m.sum(dim=(0, 1)).clamp_min(1.0)
    glob_mse = (((target_norm - glob) ** 2) * m).sum() / denom

    pos_denom = m.sum(dim=0).clamp_min(1.0)                    # (P, 1)
    pos = (target_norm * m).sum(dim=0) / pos_denom             # (P, D)
    pos_mse = (((target_norm - pos.unsqueeze(0)) ** 2) * m).sum() / denom

    return {"global": glob_mse.item(), "per_position": pos_mse.item()}


@torch.no_grad()
def byte_level_reconstruction(seq_ae, packet_decoder, seq_lens, tokens,
                              latents=None, pad_token_id=None, temperature=0.0):
    """Eval-only: pipe reconstructed latents through the frozen packet decoder.

    Returns byte-level accuracy against the original packets, directly
    comparable to the packet-level AE numbers. No gradients flow here.
    """
    z, _ = seq_ae.encoder(seq_lens, tokens=tokens, latents=latents)
    lat_hat = seq_ae.reconstruct_latents(z)                    # (B, P, D)
    B, P, D = lat_hat.shape
    gen = packet_decoder.generate(lat_hat.reshape(B * P, D),
                                  temperature=temperature)     # (B*P, L')
    ref = tokens.reshape(B * P, -1)
    L = min(gen.size(1), ref.size(1))
    gen, ref = gen[:, :L], ref[:, :L]

    pkt_mask = build_padding_mask(seq_lens.view(-1), P).reshape(-1, 1)
    byte_mask = pkt_mask.expand(-1, L)
    if pad_token_id is not None:
        byte_mask = byte_mask & (ref != pad_token_id)
    return ((gen == ref) & byte_mask).sum().item() / byte_mask.sum().clamp_min(1).item()

@torch.no_grad()
def precompute_latents(packet_encoder, dataloader, device='cuda'):
    """Cache frozen packet latents so the AE trains on (B, P, D) floats.

    Expects the loader to yield (tokens (B,P,L), seq_lens (B,)).
    Returns (latents (N,P,D), seq_lens (N,)) on CPU.
    """
    packet_encoder.eval().to(device)
    all_lat, all_lens = [], []
    for tokens, seq_lens in dataloader:
        B, P, L = tokens.shape
        lat = packet_encoder(tokens.to(device).reshape(B * P, L))
        all_lat.append(lat.reshape(B, P, -1).cpu())
        all_lens.append(seq_lens.cpu())
    return torch.cat(all_lat), torch.cat(all_lens)


def compute_target_stats(latents, seq_lens, eps=1e-5):
    """Per-dimension mean/std over *valid* packets only."""
    mask = build_padding_mask(seq_lens.view(-1), latents.size(1))
    valid = latents[mask]                                      # (N_valid, D)
    return valid.mean(dim=0), valid.std(dim=0).clamp_min(eps)