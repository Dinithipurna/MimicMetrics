# EMI Challenge Baselines

Quick guide to training and evaluating the baseline models.

## Baselines Overview

1. **ViT + 3-layer GRU** (Reported Score: **MSE 0.09**)
   - Input: Vision Transformer embeddings (38 frames × 384 dims)
   - Architecture: 3-layer GRU with mean pooling + linear projection
   - Output: 6 emotion intensity scores

2. **Wav2Vec2 + Linear** (Reported Score: **MSE 0.24**)
   - Input: Wav2Vec2 audio embeddings (294 frames × 768 dims)
   - Architecture: Mean pooling + single linear layer
   - Output: 6 emotion intensity scores

## Installation

```bash
cd /home/dinithi/Documents/EMI_Challenge
pip install -r requirements.txt
```

Ensure `torch`, `torchvision`, `librosa`, `scipy`, `scikit-learn`, and `tqdm` are installed.

## Training

### Train ViT Baseline
```bash
cd /home/dinithi/Documents/EMI_Challenge
python src/train_baselines.py --model vit --epochs 50 --batch_size 32 --lr 1e-3
```

### Train Wav2Vec2 Baseline
```bash
python src/train_baselines.py --model wav2vec2 --epochs 50 --batch_size 32 --lr 1e-3
```

### Train Both
```bash
python src/train_baselines.py --model both --epochs 50 --batch_size 32 --lr 1e-3
```

### Training Options

- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 1e-3)
- `--device`: Device to use - 'cpu' or 'cuda' (default: cuda)

### Expected Training Output

```
ViT Baseline Training:
- Epoch 1/50
  Train Loss: 0.2345
  Val MSE:    0.1523
  Val MAE:    0.2891
  Val Pearson: 0.4562
...
Best Val MSE: 0.0892 (Epoch 23)
```

## Evaluation

### Evaluate on Validation Set
```bash
python src/inference.py --model both --split valid
```

### Evaluate on Train Set
```bash
python src/inference.py --model vit --split train
```

### Inference Options

- `--model`: Model to evaluate - 'vit', 'wav2vec2', or 'both' (default: both)
- `--split`: Data split - 'train', 'valid', or 'test' (default: valid)
- `--device`: Device to use - 'cpu' or 'cuda' (default: cuda)

## Output Files

Training outputs:
- Checkpoints: `results/checkpoints/trained/best_vit.pt`, `best_wav2vec2.pt`
- Results: `results/checkpoints/trained/training_results.json`

Inference outputs:
- Predictions: `results/predictions/{split}/{model_name}_predictions.csv`

## Data Structure

The baseline training expects embeddings in:
```
results/checkpoints/baseline/
├── vit/
│   ├── 00000.pkl (38, 384) numpy array
│   ├── 00001.pkl
│   └── ...
└── wav2vec2/
    ├── 00000.pkl (294, 768) numpy array
    ├── 00001.pkl
    └── ...
```

And split files in:
```
data/splits/
├── train_split.csv
└── valid_split.csv
```

CSV format:
```
Filename,Admiration,Amusement,Determination,Empathic Pain,Excitement,Joy
00000,0.333,0.333,0.0,0.0,0.333,0.0
00001,0.0,0.0,0.0,0.0,0.5,0.0
...
```

## Performance Benchmarks

Current baseline results (from organizers):

| Model | Dataset | MSE  | MAE  | Pearson |
|-------|---------|------|------|---------|
| ViT + GRU | Valid | 0.09 | ?    | ?       |
| Wav2Vec2 + Linear | Valid | 0.24 | ?    | ?       |

Your results may vary based on hyperparameters and implementation details.

## Optimization Tips

- **For faster training**: Reduce `--batch_size` to 16 or use `--device cpu` for debugging
- **For better scores**: Try different learning rates (1e-4, 1e-3, 1e-2)
- **For longer training**: Increase `--epochs` to 100+
- **GPU memory**: If OOM, reduce batch size

## Common Issues

### "embedding files not found"
Check that ViT and Wav2Vec2 embeddings are extracted:
```bash
ls /home/dinithi/Documents/EMI_Challenge/results/checkpoints/baseline/vit/ | wc -l
ls /home/dinithi/Documents/EMI_Challenge/results/checkpoints/baseline/wav2vec2/ | wc -l
```

Should show ~8000+ files for each.

### "CSV files not found"
Verify splits exist:
```bash
ls /home/dinithi/Documents/EMI_Challenge/data/splits/
```

### CUDA errors
Try using CPU:
```bash
python src/train_baselines.py --model vit --device cpu
```

## References

- Vision Transformer (ViT): [Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929)
- Wav2Vec2: [Baevski et al., 2020](https://arxiv.org/abs/2006.11477)
- EMI Challenge: [Challenge Details](link_to_challenge)
