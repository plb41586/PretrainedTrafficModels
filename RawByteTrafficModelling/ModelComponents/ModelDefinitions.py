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

def save_checkpoint(model, optimizer, epoch, loss, config, path: str):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': asdict(config),
    }, path)

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

# class AutoregressiveDecoder(nn.Module):
#     """Autoregressive decoder that reconstructs a token sequence from a CLS embedding.

#     The CLS embedding is prepended as the first token in the decoder input,
#     acting as a bottleneck representation. During training, teacher forcing
#     is used by shifting the ground truth tokens and prepending the CLS
#     embedding. During inference, tokens are generated one at a time.

#     Args(Packed in AutoEncoderParams):
#         vocab_size:     Size of the token vocabulary.
#         embedding_dim:  Dimensionality of token embeddings (must match the
#                         CLS embedding and backbone ``d_model``).
#         max_len:        Maximum sequence length for generation.
#         Backbone:       Sequence modeling backbone (any ``nn.Module`` mapping
#                         ``(batch, seq_len, embedding_dim)`` →
#                         ``(batch, seq_len, embedding_dim)``). Should support
#                         causal masking if using a Transformer backbone.
#         bos_token_id:   Beginning-of-sequence token id used to seed
#                         autoregressive inference.
#     """

#     def __init__(self, params: AutoEncoderParams, Backbone=None):
#         super().__init__()
#         embedding_dim = params.ENC_Params.EncoderDim
#         vocab_size = params.ENC_Params.vocab_size
#         self.max_len = params.ENC_Params.packet_id_len
#         self.bos_token_id = params.ENC_Params.SpecialTokens["<BOS>"]
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.cls_proj = nn.Linear(embedding_dim, embedding_dim)
#         if Backbone == None:
#             self.Backbone = build_backbone(params.DecBackboneType, params.DecBackbone)
#         else:
#             self.Backbone = Backbone
#         self.output_head = nn.Linear(embedding_dim, vocab_size, bias=False)

#     def _build_decoder_input(self, cls_embedding: torch.Tensor, tokens: torch.Tensor = None) -> torch.Tensor:
#         """Prepend the projected CLS embedding to the token embeddings.

#         Args:
#             cls_embedding: CLS vector of shape ``(batch, embedding_dim)``.
#             tokens:        Optional token indices of shape ``(batch, seq_len)``.
#                            If ``None``, only the CLS embedding is returned.

#         Returns:
#             Decoder input of shape ``(batch, 1 + seq_len, embedding_dim)``
#             or ``(batch, 1, embedding_dim)`` if tokens are not provided.
#         """
#         cls_token = self.cls_proj(cls_embedding).unsqueeze(1)  # (batch, 1, embedding_dim)
#         if tokens is None:
#             return cls_token
#         token_embeds = self.embedding(tokens)  # (batch, seq_len, embedding_dim)
#         return torch.cat([cls_token, token_embeds], dim=1)

#     def forward(self, cls_embedding: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
#         """Training forward pass with teacher forcing.

#         The input to the decoder is ``[CLS_proj | target_tokens[:-1]]``,
#         i.e. the CLS embedding followed by all target tokens except the
#         last. The model predicts each next token, producing logits aligned
#         with ``target_tokens``.

#         Args:
#             cls_embedding:  CLS vector of shape ``(batch, embedding_dim)``.
#             target_tokens:  Ground truth token indices of shape
#                             ``(batch, seq_len)`` used for teacher forcing.

#         Returns:
#             Logits of shape ``(batch, seq_len, vocab_size)`` aligned with
#             ``target_tokens`` (i.e. logits[i] predicts target_tokens[i]).
#         """
#         # Shift right: drop last target token so output length matches target
#         shifted = target_tokens[:, :-1]  # (batch, seq_len - 1)
#         h = self._build_decoder_input(cls_embedding, shifted)  # (batch, 1 + seq_len - 1, dim)
#         h = self.Backbone(h)
#         logits = self.output_head(h)  # (batch, seq_len, vocab_size)
#         return logits

#     @torch.no_grad()
#     def generate(self, cls_embedding: torch.Tensor, max_len: int = None,
#                  temperature: float = 1.0, eos_token_id: int = None) -> torch.Tensor:
#         """Autoregressively generate a token sequence from a CLS embedding.

#         Args:
#             cls_embedding:  CLS vector of shape ``(batch, embedding_dim)``.
#             max_len:        Maximum number of tokens to generate. Defaults
#                             to ``self.max_len``.
#             temperature:    Sampling temperature. Values < 1 sharpen the
#                             distribution, > 1 flatten it. Use 0 for greedy.
#             eos_token_id:   Optional end-of-sequence token id. Generation
#                             stops early for a sample once this token is
#                             produced.

#         Returns:
#             Generated token indices of shape ``(batch, generated_len)``.
#         """
#         max_len = max_len or self.max_len
#         batch_size = cls_embedding.size(0)
#         device = cls_embedding.device

#         generated = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=device)
#         finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

#         for _ in range(max_len - 1):
#             h = self._build_decoder_input(cls_embedding, generated)
#             h = self.Backbone(h)
#             next_logits = h[:, -1, :]  # (batch, embedding_dim)
#             next_logits = self.output_head(next_logits)  # (batch, vocab_size)

#             if temperature == 0:
#                 next_token = next_logits.argmax(dim=-1)
#             else:
#                 probs = torch.softmax(next_logits / temperature, dim=-1)
#                 next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

#             # Don't update already finished sequences
#             next_token = next_token.masked_fill(finished, self.bos_token_id)
#             generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

#             if eos_token_id is not None:
#                 finished = finished | (next_token == eos_token_id)
#                 if finished.all():
#                     break

#         return generated

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
        seqCLS = self.SeqCLSembedding(torch.LongTensor([0]))
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
        seqCLS = self.SeqCLSembedding(torch.LongTensor([0]))
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

# if __name__ == "__main__":
#     import numpy as np
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # CLS (Classify) token placed ad End Of Sequence
#     TokenIDEncoder = ID_Encoder(SpecialIDs = {"<pad>": 256, "</s>": 257, "<CLS>": 258, "<mask>": 259}, CLS_Placement="EOS")

#     # --- Config ---
#     backboneparams = MambaBackboneParams(
#         dim=32
#     )
#     params = MLM_Params(
#         vocab_size=262,
#         EncoderDim=32,
#         packet_id_len=1520,
#         pooling_type="DynamicCLS",
#         EncoderBackboneType="Mamba",
#         EncoderBackboneParams=MambaBackboneParams(dim=32),
#         CLS_ID=TokenIDEncoder.SpecialIDs["<CLS>"],
#         NumCLSclasses=14,
#         NumAttackClasses=14,
#         PacketSequenceLength=16,
#         SeqClassifierDim=64)
    
#     MLM = Packet_MLM(params)

#     vocab_size = 260
#     emb_dim = 32
#     seq_lvl_dim = 32
#     bytes_per_packet = 1520       # token length per packet
#     packets_per_sequence = 64             # max packets per sequence
#     num_classes = 14
#     batch_size = 32

#     print(f"Device: {device}")
#     print("Building Packet_Encoder...")

#     # Build the packet-level encoder with CLS pooling (reduces each packet to 1 token)
#     encoder = Packet_Encoder(
#         vocab_size=vocab_size,
#         embedding_dim=emb_dim,
#         input_len=bytes_per_packet,
#         latent_dim=emb_dim,
#         latent_len=1,
#         num_classes=num_classes,
#         device=device,
#         Pooling=DynamicCLSPooling(258),
#     )

#     print("Building SequenceClassifier...")
#     model = SequenceClassifier(
#         vocab_size=vocab_size,
#         embedding_dim=emb_dim,
#         seq_lvl_dim=seq_lvl_dim,
#         packets_per_sequence=packets_per_sequence,
#         num_classes=num_classes,
#         device=device,
#         PacketEncoder=encoder,
#     )

#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"Total parameters: {total_params:,}")

#     # --- Generate random data ---
#     data_samples = []
#     passes = 1
#     for i in range(passes):
#         input_data = torch.randint(0, vocab_size-1, (batch_size, packets_per_sequence, bytes_per_packet)).to(device)
#         data_samples.append(input_data)
#     PackSeqLens = torch.randint(1, packets_per_sequence, (batch_size,))

#     PackSeqLens = PackSeqLens.to(device)
#     # print(f"\nInput tokens shape:  {tokens.shape}")
#     # print(f"Sequence lengths:    {seq_lens.tolist()}")

#     # --- Forward pass (loop version) ---
#     print("\n--- forward (loop) ---")
#     model.eval().to(device)
#     with torch.no_grad():
#         if device == "cuda":
#             torch.cuda.synchronize()
        
#         for i in range(passes):
#             logits = model.forward(data_samples[i], PackSeqLens)
#             logits_ff = model.forward_ff(data_samples[i], PackSeqLens)

#     print(f"Output logits shape: {logits.shape}")  # expect (batch_size, num_classes)
#     print(f"Logits:\n{logits}")

#     print(f"Output logits shape: {logits_ff.shape}")
#     print(f"Logits:\n{logits_ff}")

#     # --- Quick sanity checks ---
#     assert logits.shape == (batch_size, num_classes), f"Unexpected shape: {logits.shape}"
#     assert logits_ff.shape == (batch_size, num_classes), f"Unexpected shape: {logits_ff.shape}"
#     diff = (logits-logits_ff).sum().cpu().numpy()
#     assert diff == np.array(0., dtype="float32"), "Logits dont match up!"

#     print("\nAll checks passed!")