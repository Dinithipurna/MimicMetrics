"""
Training script for EMI baselines.
Usage:
    python train_baselines.py --model vit     # Train ViT+GRU baseline
    python train_baselines.py --model wav2vec2 # Train Wav2Vec2+Linear baseline
    python train_baselines.py --model both    # Train both
"""

import argparse
import os
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tqdm
import logging
import datetime

# Configure logging to save logs in separate files for each training run
def setup_logging(training_name):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path(f"logs/{training_name}_{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "training.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode='w')
        ]
    )
    return log_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log", mode='w')
    ]
)

from src.baselines import VitGRUBaseline, Wav2Vec2LinearBaseline, Wav2Vec2LSTMBaseline
from src.dataloaders import create_dataloaders


class BaselineTrainer:
    """Trainer for baseline models."""
    
    def __init__(self, model, device, learning_rate=1e-3):
        self.model = model.to(device)
        self.device = device
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
    
    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0

        for embeddings, labels in tqdm.tqdm(dataloader, desc="Training"):
            embeddings = embeddings.to(self.device)
            labels = labels.to(self.device)

            # Calculate sequence lengths for the batch and move to CPU as int64
            lengths = torch.tensor([e.size(0) for e in embeddings], dtype=torch.int64, device="cpu")

            # Forward pass
            logits = self.model(embeddings, lengths)
            loss = self.criterion(logits, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)
    
    @torch.no_grad()
    def evaluate(self, dataloader):
        """Evaluate on validation set. Returns MSE, MAE, Pearson correlation."""
        self.model.eval()

        all_preds = []
        all_labels = []
        total_loss = 0

        for embeddings, labels in tqdm.tqdm(dataloader, desc="Evaluating"):
            embeddings = embeddings.to(self.device)
            labels = labels.to(self.device)

            # Calculate sequence lengths for the batch and move to CPU as int64
            lengths = torch.tensor([e.size(0) for e in embeddings], dtype=torch.int64, device="cpu")

            logits = self.model(embeddings, lengths)
            loss = self.criterion(logits, labels)

            total_loss += loss.item()
            all_preds.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        mse = mean_squared_error(all_labels, all_preds)
        mae = mean_absolute_error(all_labels, all_preds)

        # Compute Pearson correlation for each dimension and average
        pearson_corrs = []
        for i in range(all_labels.shape[1]):
            corr, _ = pearsonr(all_labels[:, i], all_preds[:, i])
            pearson_corrs.append(corr)
        avg_pearson_corr = np.mean(pearson_corrs)

        return {
            'loss': total_loss / len(dataloader),
            'mse': mse,
            'mae': mae,
            'pearson': avg_pearson_corr
        }
    
    def save_checkpoint(self, path):
        """Save model checkpoint."""
        torch.save(self.model.state_dict(), path)
        print(f"Checkpoint saved to {path}")


def train_baseline(model_name, dataloaders, device, num_epochs=50, learning_rate=1e-3, save_dir=None):
    """
    Train a baseline model.
    
    Args:
        model_name: 'vit' or 'wav2vec2'
        dataloaders: Dict of dataloaders from create_dataloaders()
        device: torch device
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        save_dir: Directory to save checkpoints (optional)
    """
    
    # Create model
    if model_name == 'vit':
        model = VitGRUBaseline(input_dim=384, hidden_dim=256, num_layers=3, output_dim=6)
        train_loader = dataloaders['vit_train']
        valid_loader = dataloaders['vit_valid']
    elif model_name == 'wav2vec2':
        model = Wav2Vec2LinearBaseline(input_dim=768, output_dim=6)
        train_loader = dataloaders['wav2vec2_train']
        valid_loader = dataloaders['wav2vec2_valid']
    elif model_name == 'wav2vec2_lstm':
        model = Wav2Vec2LSTMBaseline(input_dim=768, output_dim=6)
        train_loader = dataloaders['wav2vec2_train']
        valid_loader = dataloaders['wav2vec2_valid']
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    trainer = BaselineTrainer(model, device, learning_rate=learning_rate)
    
    best_mse = float('inf')
    best_epoch = 0
    history = {'train_loss': [], 'val_metrics': []}
    best_pearson = -float('inf')  # Track best Pearson correlation
    
    logging.info(f"{'='*60}")
    logging.info(f"Training {model_name.upper()} Baseline")
    logging.info(f"{'='*60}\n")
    
    for epoch in range(num_epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_metrics = trainer.evaluate(valid_loader)
        
        history['train_loss'].append(train_loss)
        history['val_metrics'].append(val_metrics)
        
        logging.info(f"Epoch {epoch+1}/{num_epochs}")
        logging.info(f"  Train Loss: {train_loss:.4f}")
        logging.info(f"  Val MSE:    {val_metrics['mse']:.4f}")
        logging.info(f"  Val MAE:    {val_metrics['mae']:.4f}")
        logging.info(f"  Val Pearson: {val_metrics['pearson']:.4f}")

        # Save best checkpoint based on MSE
        if val_metrics['mse'] < best_mse:
            best_mse = val_metrics['mse']
            best_epoch = epoch
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                checkpoint_path = os.path.join(save_dir, f"best_{model_name}.pt")
                trainer.save_checkpoint(checkpoint_path)
                logging.info(f"New best model saved at epoch {epoch+1} with MSE: {best_mse:.4f}")

        # Save best checkpoint based on Pearson correlation
        if val_metrics['pearson'] > best_pearson:
            best_pearson = val_metrics['pearson']
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                checkpoint_path = os.path.join(save_dir, f"best_pearson_{model_name}.pt")
                trainer.save_checkpoint(checkpoint_path)
                logging.info(f"New best model saved at epoch {epoch+1} with Pearson Correlation: {best_pearson:.4f}")

    logging.info(f"{'='*60}")
    logging.info(f"Training Complete!")
    logging.info(f"Best Epoch: {best_epoch+1}, Best Val MSE: {best_mse:.4f}")
    logging.info(f"Best Pearson Correlation: {best_pearson:.4f}")
    logging.info(f"{'='*60}\n")
    
    return trainer, history


def main():
    parser = argparse.ArgumentParser(description='Train EMI baselines')
    parser.add_argument('--model', choices=['vit', 'wav2vec2', 'wav2vec2_lstm', 'both'], default='both',
                        help='Which model to train')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda', help='Device')
    parser.add_argument('--vit_dir', type=str, default="/home/dinithi/Documents/EMI_Challenge/results/checkpoints/baseline/vit", help='Path to ViT embeddings directory')
    parser.add_argument('--wav2vec2_dir', type=str, default="/home/dinithi/Documents/EMI_Challenge/results/checkpoints/baseline/wav2vec2", help='Path to Wav2Vec2 embeddings directory')
    parser.add_argument('--train_csv', type=str, default="/home/dinithi/Documents/EMI_Challenge/data/splits/train_split.csv", help='Path to training split CSV')
    parser.add_argument('--valid_csv', type=str, default="/home/dinithi/Documents/EMI_Challenge/data/splits/valid_split.csv", help='Path to validation split CSV')
    args = parser.parse_args()

    # Parse arguments
    model_name = args.model
    if model_name == 'vit':
        model = VitGRUBaseline(input_dim=384, hidden_dim=256, num_layers=3, output_dim=6)
        training_name = 'vit_gru_baseline'
    elif model_name == 'wav2vec2':
        model = Wav2Vec2LinearBaseline(input_dim=768, output_dim=6)
        training_name = 'wav2vec2_linear_baseline'
    elif model_name == 'wav2vec2_lstm':
        model = Wav2Vec2LSTMBaseline(input_dim=768, output_dim=6)
        training_name = 'wav2vec2_lstm_baseline'
    else:
        raise ValueError("Invalid model choice. Use 'vit', 'wav2vec2', or 'wav2vec2_lstm'.")

    # Setup logging
    log_file = setup_logging(training_name)
    print(f"Logging to {log_file}")

    # Paths
    save_dir = Path("results") / training_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Train the model
    trainer, history = train_baseline(
        model_name=model_name,
        dataloaders=create_dataloaders(
            vit_dir=args.vit_dir,
            wav2vec2_dir=args.wav2vec2_dir,
            train_csv=args.train_csv,
            valid_csv=args.valid_csv,
            batch_size=args.batch_size
        ),
        device=torch.device(args.device),
        num_epochs=args.epochs,
        learning_rate=args.lr,
        save_dir=save_dir
    )

    # Log results instead of dumping to JSON
    results_log_file = save_dir / f"results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    with open(results_log_file, 'w') as f:
        f.write("Training Results:\n")
        for epoch, metrics in enumerate(history['val_metrics']):
            f.write(f"Epoch {epoch+1}:\n")
            f.write(f"  Train Loss: {history['train_loss'][epoch]:.4f}\n")
            f.write(f"  Val MSE: {metrics['mse']:.4f}\n")
            f.write(f"  Val MAE: {metrics['mae']:.4f}\n")
            f.write(f"  Val Pearson: {metrics['pearson']:.4f}\n\n")

    logging.info(f"Results saved to {results_log_file}")

if __name__ == "__main__":
    main()
