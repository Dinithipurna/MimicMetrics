# EMI Challenge — Emotional Mimicry Intensity

Beat the state-of-the-art (SoTA) for Emotional Mimicry Intensity recognition.

## Project Structure

```
EMI_Challenge/
├── data/
│   ├── raw/           # Raw dataset files
│   ├── processed/     # Preprocessed data
│   └── splits/        # Train/val/test splits
├── models/            # Model architectures and checkpoints
├── notebooks/         # EDA and analysis notebooks
├── src/               # Training and evaluation code
├── results/
│   ├── checkpoints/   # Model weights
│   ├── logs/          # Training logs
│   └── predictions/   # Model predictions
├── requirements.txt   # Dependencies
└── config.yaml        # Configuration
```

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare data**:
   Place raw data in `data/raw/`

3. **Extract pre-computed embeddings** (optional, for baseline):
   ```bash
   # ViT embeddings
   unzip /data/EMI/vit.zip -d results/checkpoints/baseline/
   
   # Wav2Vec2 embeddings
   unzip /data/EMI/wav2vec2.zip -d results/checkpoints/baseline/
   ```

4. **Train baseline models**:
   ```bash
   # Train both baselines (ViT+GRU and Wav2Vec2+Linear)
   python run_baselines.py

   # Or train individually
   python src/train_baselines.py --model vit
   python src/train_baselines.py --model wav2vec2
   ```

5. **Evaluate baseline models**:
   ```bash
   python src/inference.py --model both --split valid
   ```

6. **Train custom model (optional)**:
   ```bash
   python src/train.py --config config.yaml
   ```

7. **Evaluate custom model (optional)**:
   ```bash
   python src/evaluate.py --checkpoint results/checkpoints/best.pt
   ```

## Baselines

- **ViT baseline**: precomputed ViT embeddings → 3-layer GRU → 6 emotion scores
- **Wav2Vec2 baseline**: precomputed Wav2Vec2 embeddings → linear head → 6 emotion scores
- **Training loss**: MSE (`nn.MSELoss`)
- **Saved checkpoints**: `results/checkpoints/trained/best_vit.pt`, `results/checkpoints/trained/best_wav2vec2.pt`
- **Predictions output**: `results/predictions/{split}/`

For full baseline details, see `BASELINES.md`.

## SoTA Benchmark

- **Current Best**: [Add SoTA metric here]
- **Target**: Beat by X%

## Experiments

- [ ] Baseline model
- [ ] Feature engineering
- [ ] Ensemble methods
- [ ] Fine-tuning
