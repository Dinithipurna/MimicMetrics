# MimicMetrics: Two-Stage Multimodal Framework for Emotion Mimicry Intensity Prediction

This repository contains the implementation of **MimicMetrics**, our submission to the **Hume-ABAW10 Emotional Mimicry Intensity (EMI) Challenge** at the CVPR ABAW Workshop.

Our framework predicts continuous emotion mimicry intensity across six affective dimensions: **Admiration, Amusement, Determination, Empathic Pain, Excitement, and Joy**. The approach uses a staged multimodal training pipeline that combines **text, audio, vision, and optional motion features** through a lightweight fusion regressor.

Our team placed **3rd in the EMI Challenge**, achieving an average Pearson correlation of **0.57** on the official test set.

---

## Poster

![MimicMetrics Poster](assets/poster.pdf)

> Replace `assets/poster.png` with the actual path to your poster image or PDF preview.

---

## Links

- **Paper:** https://arxiv.org/pdf/2605.21869  
- **Code:** https://github.com/Dinithipurna/MimicMetrics  

---

## Overview

Emotion mimicry intensity prediction is a challenging multimodal affective computing task. Unlike categorical emotion recognition, EMI prediction requires estimating continuous intensity values for multiple affective dimensions from in-the-wild video clips.

The dataset contains strong temporal variability, sparse and imbalanced labels, and partial missingness in the text modality. To address these challenges, MimicMetrics uses a **two-stage multimodal learning strategy**:

1. Train modality-specific encoders independently.
2. Remove the unimodal prediction heads and fuse the learned representations using a lightweight MLP regressor.

This design allows each modality to first learn a stable standalone representation before cross-modal fusion.

---

## Key Features

- **Staged multimodal training pipeline**
  - Stage 1: Text encoder
  - Stage 2: Audio encoder
  - Stage 3: Vision encoder
  - Optional stage: Motion encoder
  - Final stage: Multimodal fusion regressor

- **Multimodal input support**
  - Text embeddings
  - Audio embeddings
  - Visual face embeddings
  - Optional OpenFace AU + head-pose motion sequences

- **Attention-based temporal pooling**
  - Used for variable-length audio, vision, and motion sequences

- **Missing text fallback**
  - Samples with missing text embeddings are handled using zero-vector fallback

- **Modality dropout**
  - Randomly drops modality embeddings during fusion training to improve robustness

- **CCC-oriented objective**
  - Supports CCC loss and combined CCC + MSE loss for continuous intensity prediction

- **Checkpointing and metadata logging**
  - Saves best-performing models and training metadata for each stage

---

## Architecture

```text
Text Embedding      Audio Sequence      Visual Sequence      [Motion Sequence]
     |                    |                    |                     |
 Text MLP          Attention Pool       Attention Pool        Attention Pool
     |                    |                    |                     |
 Text Encoder      Audio Encoder        Vision Encoder        Motion Encoder
     |                    |                    |                     |
     +--------------------+--------------------+---------------------+
                              |
                     Concatenate Features
                              |
                     Fusion Regressor (MLP)
                              |
              6D Emotion Intensity Prediction
```

---

## Modalities and Feature Dimensions

| Modality | Feature type | Input dimension | Encoder output |
|---|---|---:|---:|
| Text | GTE text embedding | 768 | 384 |
| Audio | Custom wav2vec2 embedding | 1024 | 384 |
| Vision | DINOv2 face embedding | 768 | 384 |
| Motion | OpenFace AU + head-pose sequence | 23 | 128 |
| Output | EMI emotion dimensions | — | 6 |

The optional motion branch uses a compact **23D OpenFace descriptor** consisting of:

- **17 facial AU intensity features**
- **6 head-pose features**

Motion did not provide a major standalone advantage, but it served as a behaviorally meaningful complementary cue and remains an interesting direction for future work.

---

## Model Components

### TextMLPEncoder

Processes sentence-level text embeddings using a lightweight MLP.

```text
Input:  [B, 768]
Output: [B, 384]
```

### AudioAttentionEncoder

Processes variable-length audio embeddings using masked attention pooling followed by an MLP projection.

```text
Input:  [B, T_audio, 1024]
Output: [B, 384]
```

### VisionAttentionEncoder

Processes variable-length DINOv2 face embeddings using masked attention pooling.

```text
Input:  [B, T_vision, 768]
Output: [B, 384]
```

### MotionAttentionEncoder

Processes OpenFace AU + head-pose sequences using masked attention pooling.

```text
Input:  [B, T_motion, 23]
Output: [B, 128]
```

### FusionRegressor

Concatenates learned modality representations and predicts the six EMI dimensions.

Without motion:

```text
Input: 384 + 384 + 384 = 1152
Output: 6
```

With motion:

```text
Input: 384 + 384 + 384 + 128 = 1280
Output: 6
```

---

## Data Format

### Expected Directory Structure

Store pre-extracted embeddings as pickle files:

```text
data/
├── embeddings/
│   ├── face/
│   │   ├── 00001.pkl      # Shape: (T_v, 768)
│   │   ├── 00002.pkl
│   │   └── ...
│   ├── audio/
│   │   ├── 00001.pkl      # Shape: (T_a, 1024)
│   │   ├── 00002.pkl
│   │   └── ...
│   └── text/
│       ├── 00001.pkl      # Shape: (768,)
│       ├── 00002.pkl
│       └── ...
├── motion/
│   ├── 00001.pkl          # Shape: (T_m, 23)
│   ├── 00002.pkl
│   └── ...
└── splits/
    ├── train_split.csv
    └── val_split.csv
```

### CSV Format

Training and validation CSV files should contain:

- `Filename`
- `Admiration`
- `Amusement`
- `Determination`
- `Empathic Pain`
- `Excitement`
- `Joy`

Example:

```csv
Filename,Admiration,Amusement,Determination,Empathic Pain,Excitement,Joy
00001,0.12,0.08,0.05,0.03,0.10,0.09
00002,0.05,0.15,0.07,0.04,0.08,0.11
```

---

## Installation

```bash
git clone https://github.com/Dinithipurna/MimicMetrics.git
cd MimicMetrics

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

For Windows:

```bash
venv\Scripts\activate
```

---

## Quick Start

### Train Text + Audio + Vision Fusion

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

### Train with Optional Motion Branch

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
  --batch_size 16 \
  --device cuda:0
```

---

## Training Objective

The model supports CCC loss and combined CCC + MSE loss.

The default combined objective is:

```text
L = α L_CCC + (1 - α) L_MSE
```

where `α = 0.7`.

CCC encourages predictions to follow the same intensity pattern as the ground truth while staying close to the correct value range.

---

## Results Summary

Across our submitted systems, the strongest validation performance was achieved by the text-audio-vision-motion fusion model under the expanded 4:1 split:

```text
Validation Pearson: 0.4722
```

The strongest official test performance was achieved by the text-audio-vision fusion model under the expanded 4:1 split:

```text
Test Pearson: 0.570
Challenge ranking: 3rd place
```

Overall, text and audio provided the strongest standalone signals, while video and motion acted as more selective complementary cues, especially for dimensions such as Joy, Amusement, and Excitement.

---

## Citation

If you use this framework or build on our work, please cite:

```bibtex
@inproceedings{dissanayake2026mimicmetrics,
  title={Two-Stage Multimodal Framework for Emotion Mimicry Intensity Prediction},
  author={Dissanayake, Dinithi and Silva, Shaveen and Atukorala, Ovindu and Sasikumar, Prasanth and Nanayakkara, Suranga},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  year={2026}
}
```

---

## Acknowledgements

This work was developed as part of the Hume-ABAW10 Emotional Mimicry Intensity Challenge. We thank the challenge organizers and the ABAW workshop for providing the benchmark and evaluation platform.
