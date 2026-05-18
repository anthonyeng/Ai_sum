"""
Temporal Transformer for video segment importance scoring.

A lightweight Transformer encoder that processes a sequence of video segment
features and predicts per-segment importance scores. Uses:
  - Positional encoding (sinusoidal) for temporal order
  - Multi-head self-attention to capture long-range segment dependencies
  - Feed-forward layers for importance regression

Architecture:
    Input (batch, seq_len, input_dim=512)
    → Linear projection to d_model
    → Positional encoding
    → N × TransformerEncoderLayer (self-attention + FFN)
    → Linear head → sigmoid
    → Output (batch, seq_len) importance scores in [0, 1]

Compared to BiLSTM:
  - Captures global context (attention over all segments) vs local (LSTM hidden state)
  - Parallelizable during training (no sequential dependency)
  - Better at long videos where LSTM forgets early segments

Usage:
    from src.models.temporal_transformer import TemporalTransformer
    model = TemporalTransformer(input_dim=512)
    scores = model(features, mask)  # (batch, seq_len)
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 2000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalTransformer(nn.Module):
    """
    Transformer encoder for segment importance scoring.

    Args:
        input_dim: dimension of input features (512 for ResNet18)
        d_model: internal transformer dimension
        nhead: number of attention heads
        num_layers: number of transformer encoder layers
        dim_feedforward: FFN hidden dimension
        dropout: dropout rate
        max_seq_len: maximum video length in segments
    """

    def __init__(
        self,
        input_dim: int = 512,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 2000,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model

        # Project input features to transformer dimension
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Importance scoring head
        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim) segment features
            mask: (batch, seq_len) boolean mask (True = valid, False = padding)

        Returns:
            (batch, seq_len) importance scores in [0, 1]
        """
        # Project to d_model
        x = self.input_projection(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Create attention mask for padding (True = ignore in PyTorch convention)
        src_key_padding_mask = None
        if mask is not None:
            src_key_padding_mask = ~mask  # PyTorch expects True = pad

        # Transformer encoding
        x = self.transformer_encoder(
            x, src_key_padding_mask=src_key_padding_mask
        )

        # Score each segment
        scores = self.score_head(x).squeeze(-1)  # (batch, seq_len)

        return scores

    def get_attention_weights(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> list[torch.Tensor]:
        """
        Extract attention weights from each layer for visualization.

        Returns:
            list of (batch, nhead, seq_len, seq_len) attention weight tensors
        """
        x = self.input_projection(x)
        x = self.pos_encoder(x)

        src_key_padding_mask = ~mask if mask is not None else None

        attention_weights = []
        for layer in self.transformer_encoder.layers:
            # Use the self-attention sublayer directly
            x_norm = layer.norm1(x)
            attn_output, attn_weight = layer.self_attn(
                x_norm, x_norm, x_norm,
                key_padding_mask=src_key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )
            attention_weights.append(attn_weight.detach())

            # Complete the forward pass through this layer
            x = x + layer.dropout1(attn_output)
            x = x + layer._ff_block(layer.norm2(x))

        return attention_weights
