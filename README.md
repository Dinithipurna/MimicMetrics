# Staged Multimodal Fusion Training

A flexible framework for training multimodal emotion recognition models using a staged training pipeline. This implementation focuses on **Emotional Mimicry Intensity (EMI)** recognition by combining text, audio, and visual modalities through a principled four-stage training procedure.

## Key Features

- **Four-Stage Training Pipeline**:
  - Stage 1: TextMLPEncoder (trained on text embeddings)
  - Stage 2: AudioAttentionEncoder (trained on audio embeddings)  
  - Stage 3: VisionAttentionEncoder (trained on visual embeddings)
  - Stage 5: FusionRegressor (trained on frozen encoder outputs)
  
- **Optional Motion Stage**: Add per-frame motion sequences as an additional modality
- **Early Stopping & Learning Rate Scheduling**: Per-stage adaptive training
- **Modality Dropout**: Optional probabilistic modality dropping for robustness
- **Flexible Loss Functions**: CCC (Concordance Correlation Coefficient) or combined CCC+MSE
- **Checkpoint Management**: Automatic best-model saving with metadata

## Architecture Overview

```
Text Embeddings    Audio Embeddings    Visual Embeddings    [Motion Sequences]
        ↓                  ↓                    ↓                    ↓
    [MLP]          [Attention Pool]     [Attention Pool]    [Attention Pool]
        ↓                  ↓                    ↓                    ↓
   Text Encoder    Audio Encoder      Vision Encoder      Motion Encoder
        ↓                  ↓                    ↓                    ↓
         └──────────────────┴────────────────┴──────────────────────┘
                                ↓
                         Concatenate Features
                                ↓
                        Fusion Regressor (MLP)
                                ↓
                        Emotion Predictions (6D)
```

### Model Components

**TextMLPEncoder**: Two-layer MLP with LayerNorm and GELU activation
- Input: [B, text_dim] → Output: [B, text_hidden]

**AudioAttentionEncoder**: Attention-based pooling followed by MLP
- Input: [B, T_a, audio_dim] → Attention Pool → Output: [B, audio_hidden]

**VisionAttentionEncoder**: Attention-based pooling followed by MLP
- Input: [B, T_v, vision_dim] → Attention Pool → Output: [B, vision_hidden]

**MotionAttentionEncoder** (optional): Per-frame motion pooling
- Input: [B, T_m, motion_dim] → Attention Pool → Output: [B, motion_hidden]

**FusionRegressor**: Final prediction head
- Input: [B, text_hidden + audio_hidden + vision_hidden + motion_hidden] → Output: [B, 6]

## Installation

### Requirements
- Python 3.8+
- CUDA-capable GPU (recommended) or CPU

### Setup

```bash
# Clone or navigate to the repository
cd MimicMetrics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data Format

### Embedding Files Structure

Store pre-computed embeddings as pickle files:

```
data/
├── embeddings/
│   ├── face/
│   │   ├── 00001.pkl      # Shape: (T_v, 768) or (T_v, 384)
│   │   ├── 00002.pkl
│   │   └── ...
│   ├── audio/
│   │   ├── 00001.pkl      # Shape: (T_a, 1024)
│   │   ├── 00002.pkl
│   │   └── ...
│   └── text/
│       ├── 00001.pkl      # Shape: (768,) or (D_t,)
│       ├── 00002.pkl
│       └── ...
├── motion/
│   ├── 00001.pkl          # Shape: (T-1, motion_feat_dim)
│   ├── 00002.pkl
│   └── ...
└── splits/
    ├── train_split.csv    # Columns: Filename, Admiration, Amusement, ...
    └── val_split.csv
```

### CSV Format

Training and validation split CSVs must contain:
- `Filename` column: video ID/filename
- Label columns (emotion scores): `Admiration`, `Amusement`, `Determination`, `Empathic Pain`, `Excitement`, `Joy`

Example:
```csv
Filename,Admiration,Amusement,Determination,Empathic Pain,Excitement,Joy
00001,1.2,0.8,0.5,0.3,1.0,0.9
00002,0.5,1.5,0.7,0.4,0.8,1.1
...
```

## Quick Start

### Basic Usage (Text + Audio + Vision)

```bash
python src/staged_fusion_training.py \
  --face_dir ./data/embeddings/face \
  --audio_dir ./data/embeddings/audio \
  --text_dir ./data/embeddings/text \
  --train_csv ./data/splits/train_split.csv \
  --valid_csv ./data/splits/val_split.csv \
  --save_dir ./results/checkpoints \
  --log_dir ./logs \
  --batch_size 16 \
  --device cuda:0
```

### With Motion Features

```bash
python src/staged_fusion_training.py \
  --face_dir ./data/embeddings/face \
  --audio_dir ./data/embeddings/audio \
  --text_dir ./data/embeddings/text \
  --train_csv ./data/splits/train_split.csv \
  --valid_csv ./data/splits/val_split.csv \
  --use_motion_seq \
  --motion_seq_dir ./data/motion \
  --motion_hidden_dim 128 \
  --motion_epochs 100 \
  --save_dir ./results/checkpoints \
  --log_dir ./logs \
  --device cuda:0
```

## Citation

**If you use this framework in your research, please cite:**

> Dinithi Dissanayake, Shaveen Silva, Ovindu Atukorala, Prasanth Sasikumar, and
> Suranga Nanayakkara. 2026. *Two-Stage Multimodal Framework for Emotion Mimicry
> Intensity Prediction.* In *Proceedings of the IEEE/CVF Conference on Computer
> Vision and Pattern Recognition Workshops (CVPR Workshops).*


