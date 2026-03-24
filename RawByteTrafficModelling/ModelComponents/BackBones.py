import torch
import torch.nn as nn
from mamba_ssm import Mamba
from dataclasses import dataclass

@dataclass
class BackboneParams:
    dim: int
    num_layers: int = 2
    dropout: float = 0.1
@dataclass
class TransformerBackboneParams(BackboneParams):
    num_layers: int = 2
    nhead: int = 4
    max_len: int = 1520

@dataclass
class MambaBackboneParams(BackboneParams):
    num_layers: int = 2
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2

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

    def __init__(self, params: MambaBackboneParams):
        super().__init__()
        self.dropout = nn.Dropout(p=params.dropout)
        self.layers = nn.ModuleList([
            Mamba(
                d_model=params.dim,
                d_state=params.d_state,
                d_conv=params.d_conv,
                expand=params.expand,
            )
            for _ in range(params.num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(params.dim)
            for _ in range(params.num_layers)
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

    def __init__(self, params: TransformerBackboneParams):
        super().__init__()
        self._init_sinusoidal_encoding(params.max_len, params.dim)
        self.dropout = nn.Dropout(p=params.dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=params.dim,
            nhead=params.nhead,
            dropout=params.dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=params.num_layers,
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