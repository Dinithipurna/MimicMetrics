"""
Training script for multimodal fusion baselines.
Usage:
    python train_multimodal.py --model concat       # Train concat fusion
    python train_multimodal.py --model attention    # Train cross-attention fusion
    python train_multimodal.py --model both         # Train both
"""

import argparse
import os
import json
import gc
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tqdm

from src.baselines import MultimodalConcatBaseline, MultimodalCrossAttentionBaseline
from src.multimodal_dataloader import create_multimodal_dataloaders


class MultimodalTrainer:
    """Trainer for multimodal fusion models."""
    
    def __init__(self, model, device, learning_rate=1e-3):
        self.model = model.to(device)
        self.device = device
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
    
    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for vit_emb, wav2vec2_emb, labels in tqdm.tqdm(dataloader, desc="Training"):
            vit_emb = vit_emb.to(self.device)
            wav2vec2_emb = wav2vec2_emb.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            logits = self.model(vit_emb, wav2vec2_emb)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    @torch.no_grad()
    def evaluate(self, dataloader):
        """Evaluate on validation set."""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        total_loss = 0
        
        for vit_emb, wav2vec2_emb, labels in tqdm.tqdm(dataloader, desc="Evaluating"):
            vit_emb = vit_emb.to(self.device)
            wav2vec2_emb = wav2vec2_emb.to(self.device)
            labels = labels.to(self.device)
            
            logits = self.model(vit_emb, wav2vec2_emb)
            loss = self.criterion(logits, labels)
            
            total_loss += loss.item()
            all_preds.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        mse = mean_squared_error(all_labels, all_preds)
        mae = mean_absolute_error(all_labels, all_preds)
        
        flat_preds = all_preds.flatten()
        flat_labels = all_labels.flatten()
        pearson_corr, _ = pearsonr(flat_labels, flat_preds)
        
        return {
            'loss': total_loss / len(dataloader),
            'mse': mse,
            'mae': mae,
            'pearson': pearson_corr
        }
    
    def save_checkpoint(self, path):
        """Save model checkpoint."""
        torch.save(self.model.state_dict(), path)
        print(f"Checkpoint saved to {path}")


def train_multimodal(model_name, dataloaders, device, num_epochs=50, learning_rate=1e-3, save_dir=None):
    """
    Train a multimodal fusion model.
    
    Args:
        model_name: 'concat' or 'attention'
        dataloaders: Dict of dataloaders
        device: torch device
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        save_dir: Directory to save checkpoints
    """
    
    # Create model
    if model_name == 'concat':
        model = MultimodalConcatBaseline(
            vit_dim=384, wav2vec2_dim=768, 
            hidden_dim=512, output_dim=6, dropout=0.3
        )
    elif model_name == 'attention':
        model = MultimodalCrossAttentionBaseline(
            vit_dim=384, wav2vec2_dim=768,
            hidden_dim=256, num_heads=4, output_dim=6, dropout=0.3
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    trainer = MultimodalTrainer(model, device, learning_rate=learning_rate)
    
    best_mse = float('inf')
    best_epoch = 0
    history = {'train_loss': [], 'val_metrics': []}
    
    print(f"\n{'='*60}")
    print(f"Training Multimodal {model_name.upper()} Fusion")
    print(f"{'='*60}\n")
    
    for epoch in range(num_epochs):
        train_loss = trainer.train_epoch(dataloaders['train'])
        val_metrics = trainer.evaluate(dataloaders['valid'])
        
        history['train_loss'].append(train_loss)
        history['val_metrics'].append(val_metrics)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val MSE:    {val_metrics['mse']:.4f}")
        print(f"  Val MAE:    {val_metrics['mae']:.4f}")
        print(f"  Val Pearson: {val_metrics['pearson']:.4f}")
        
        # Save best checkpoint
        if val_metrics['mse'] < best_mse:
            best_mse = val_metrics['mse']
            best_epoch = epoch
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                checkpoint_path = os.path.join(save_dir, f"best_multimodal_{model_name}.pt")
                trainer.save_checkpoint(checkpoint_path)
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best Epoch: {best_epoch+1}, Best Val MSE: {best_mse:.4f}")
    print(f"{'='*60}\n")
    
    return trainer, history


def main():
    parser = argparse.ArgumentParser(description='Train multimodal fusion baselines')
    parser.add_argument('--model', choices=['concat', 'attention', 'both'], default='concat',
                        help='Which fusion to train')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--attention_batch_size', type=int, default=8,
                        help='Batch size for cross-attention model (lower to avoid OOM)')
    parser.add_argument('--attention_eval_batch_size', type=int, default=1,
                        help='Validation batch size for cross-attention model')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', choices=['cpu', 'cuda:0', 'cuda:1'], default='cuda:1', help='Device')
    args = parser.parse_args()
    
    # Paths
    base_dir = Path('/home/dinithi/Documents/EMI_Challenge')
    vit_dir = base_dir / 'results/checkpoints/baseline/vit'
    wav2vec2_dir = base_dir / 'results/checkpoints/baseline/wav2vec2'
    train_csv = base_dir / 'data/splits/train_split.csv'
    valid_csv = base_dir / 'data/splits/valid_split.csv'
    checkpoint_dir = base_dir / 'results/checkpoints/trained'
    
    # Verify paths
    assert vit_dir.exists(), f"ViT embeddings not found at {vit_dir}"
    assert wav2vec2_dir.exists(), f"Wav2Vec2 embeddings not found at {wav2vec2_dir}"
    assert train_csv.exists(), f"Train split CSV not found at {train_csv}"
    assert valid_csv.exists(), f"Valid split CSV not found at {valid_csv}"
    
    # Create dataloaders
    print("Loading multimodal data...")
    dataloaders = create_multimodal_dataloaders(
        str(vit_dir), str(wav2vec2_dir),
        str(train_csv), str(valid_csv),
        batch_size=args.batch_size, num_workers=4
    )

    attention_dataloaders = create_multimodal_dataloaders(
        str(vit_dir), str(wav2vec2_dir),
        str(train_csv), str(valid_csv),
        batch_size=args.attention_batch_size,
        valid_batch_size=args.attention_eval_batch_size,
        num_workers=4
    )
    
    # Device
    if args.device == 'cuda:0' and torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif args.device == 'cuda:1' and torch.cuda.is_available():
        device = torch.device('cuda:1')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}\n")
    
    # Train models
    results = {}
    
    if args.model in ['concat', 'both']:
        trainer_concat, history_concat = train_multimodal(
            'concat', dataloaders, device,
            num_epochs=args.epochs, learning_rate=args.lr,
            save_dir=str(checkpoint_dir)
        )
        results['concat'] = history_concat

        if str(device).startswith('cuda'):
            del trainer_concat
            gc.collect()
            torch.cuda.empty_cache()
    
    if args.model in ['attention', 'both']:
        trainer_attention, history_attention = train_multimodal(
            'attention', attention_dataloaders, device,
            num_epochs=args.epochs, learning_rate=args.lr,
            save_dir=str(checkpoint_dir)
        )
        results['attention'] = history_attention
    
    # Save results
    if results:
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            return obj
        
        results_json = convert_for_json(results)
        results_path = checkpoint_dir / 'multimodal_training_results.json'
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        print(f"Results saved to {results_path}")


if __name__ == '__main__':
    main()
