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


class Wav2Vec2LSTMBaseline(nn.Module):
    """Wav2Vec2 features → LSTM → Linear layer → emotion prediction."""

    def __init__(self, input_dim=768, hidden_dim=256, output_dim=6, num_layers=2, bidirectional=True, dropout=0.3):
        """
        Args:
            input_dim: Wav2Vec2 embedding dimension (768)
            hidden_dim: Hidden dimension of the LSTM
            output_dim: Number of emotion labels (6)
            num_layers: Number of LSTM layers
            bidirectional: Whether the LSTM is bidirectional
            dropout: Dropout rate
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        """
        Args:
            x: (batch_size, seq_len, 768) Wav2Vec2 embeddings
            lengths: (batch_size,) Actual lengths of each sequence in the batch
        Returns:
            (batch_size, 6) emotion scores
        """
        # Pack the padded sequence
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Pass through LSTM
        packed_out, _ = self.lstm(packed_x)

        # Unpack the sequence
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        # Apply global average pooling over the sequence dimension
        pooled = lstm_out.mean(dim=1)  # (batch_size, hidden_dim * num_directions)

        # Apply dropout and pass through the fully connected layer
        logits = self.fc(self.dropout(pooled))  # (batch_size, 6)

        return logits


class MultimodalConcatBaseline(nn.Module):
    """Concatenate ViT + Wav2Vec2 features → MLP → emotion prediction."""
    
    def __init__(self, vit_dim=384, wav2vec2_dim=768, hidden_dim=512, output_dim=6, dropout=0.3):
        """
        Args:
            vit_dim: ViT embedding dimension (384)
            wav2vec2_dim: Wav2Vec2 embedding dimension (768)
            hidden_dim: Hidden dimension for fusion MLP
            output_dim: Number of emotion labels (6)
            dropout: Dropout rate
        """
        super().__init__()
        concat_dim = vit_dim + wav2vec2_dim  # 384 + 768 = 1152
        
        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, vit_emb, wav2vec2_emb):
        """
        Args:
            vit_emb: (batch_size, seq_len_v, 384) ViT embeddings
            wav2vec2_emb: (batch_size, seq_len_a, 768) Wav2Vec2 embeddings
        Returns:
            (batch_size, 6) emotion scores
        """
        # Pool both modalities
        vit_pooled = vit_emb.mean(dim=1)  # (batch, 384)
        wav2vec2_pooled = wav2vec2_emb.mean(dim=1)  # (batch, 768)
        
        # Concatenate
        fused = torch.cat([vit_pooled, wav2vec2_pooled], dim=1)  # (batch, 1152)
        
        # MLP fusion
        logits = self.fusion(fused)  # (batch, 6)
        
        return logits


class MultimodalCrossAttentionBaseline(nn.Module):
    """Cross-attention fusion between ViT and Wav2Vec2 → emotion prediction."""
    
    def __init__(self, vit_dim=384, wav2vec2_dim=768, hidden_dim=256, 
                 num_heads=4, output_dim=6, dropout=0.3):
        """
        Args:
            vit_dim: ViT embedding dimension (384)
            wav2vec2_dim: Wav2Vec2 embedding dimension (768)
            hidden_dim: Hidden dimension for projection
            num_heads: Number of attention heads
            output_dim: Number of emotion labels (6)
            dropout: Dropout rate
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Project both modalities to same dimension
        self.vit_proj = nn.Linear(vit_dim, hidden_dim)
        self.wav2vec2_proj = nn.Linear(wav2vec2_dim, hidden_dim)
        
        # Cross-attention: video attends to audio
        self.cross_attn_v2a = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Cross-attention: audio attends to video
        self.cross_attn_a2v = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Fusion head
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, vit_emb, wav2vec2_emb):
        """
        Args:
            vit_emb: (batch_size, seq_len_v, 384) ViT embeddings
            wav2vec2_emb: (batch_size, seq_len_a, 768) Wav2Vec2 embeddings
        Returns:
            (batch_size, 6) emotion scores
        """
        # Project to common dimension
        vit_proj = self.vit_proj(vit_emb)  # (batch, seq_v, hidden)
        wav2vec2_proj = self.wav2vec2_proj(wav2vec2_emb)  # (batch, seq_a, hidden)
        
        # Cross-attention: video attends to audio
        v_attended, _ = self.cross_attn_v2a(
            query=vit_proj, 
            key=wav2vec2_proj, 
            value=wav2vec2_proj
        )  # (batch, seq_v, hidden)
        
        # Cross-attention: audio attends to video
        a_attended, _ = self.cross_attn_a2v(
            query=wav2vec2_proj,
            key=vit_proj,
            value=vit_proj
        )  # (batch, seq_a, hidden)
        
        # Pool attended representations
        v_pooled = v_attended.mean(dim=1)  # (batch, hidden)
        a_pooled = a_attended.mean(dim=1)  # (batch, hidden)
        
        # Concatenate and fuse
        fused = torch.cat([v_pooled, a_pooled], dim=1)  # (batch, hidden*2)
        logits = self.fusion(fused)  # (batch, 6)
        
        return logits
