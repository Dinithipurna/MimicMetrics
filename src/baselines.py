"""
Baseline models for EMI Challenge.
- ViT + 3-layer GRU (reported score: 0.09)
- Wav2Vec2 + Linear (reported score: 0.24)
"""

import torch
import torch.nn as nn


class VitGRUBaseline(nn.Module):
    """Vision Transformer features → 3-layer GRU → emotion prediction."""
    
    def __init__(self, input_dim=384, hidden_dim=256, num_layers=3, output_dim=6, dropout=0.2):
        """
        Args:
            input_dim: ViT embedding dimension (384)
            hidden_dim: GRU hidden dimension
            num_layers: Number of GRU layers
            output_dim: Number of emotion labels (6)
            dropout: Dropout rate
        """
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, 384) ViT embeddings
        Returns:
            (batch_size, 6) emotion scores
        """
        # GRU forward pass
        gru_out, _ = self.gru(x)  # (batch, seq_len, hidden_dim)
        
        # Global average pooling over sequence dimension
        pooled = gru_out.mean(dim=1)  # (batch, hidden_dim)
        
        # Linear projection to emotion space
        logits = self.fc(pooled)  # (batch, 6)
        
        return logits


class Wav2Vec2LinearBaseline(nn.Module):
    """Wav2Vec2 features → Linear layer → emotion prediction."""
    
    def __init__(self, input_dim=768, output_dim=6):
        """
        Args:
            input_dim: Wav2Vec2 embedding dimension (768)
            output_dim: Number of emotion labels (6)
        """
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, 768) Wav2Vec2 embeddings
        Returns:
            (batch_size, 6) emotion scores
        """
        # Global average pooling over sequence dimension
        pooled = x.mean(dim=1)  # (batch, 768)
        
        # Linear projection to emotion space
        logits = self.fc(pooled)  # (batch, 6)
        
        return logits
