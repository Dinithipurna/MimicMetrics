"""
Training script for EMI predictor using unimodal text embeddings.
Usage:
    python train_text_baseline.py --epochs 50 --batch_size 32 --lr 1e-3 --device cuda
"""

import argparse
import os
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import tqdm
import logging
import datetime
import pandas as pd
import pickle

# Configure logging with start time in the log file name
start_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_file_name = f"{start_time}_textbaseline_log.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file_name, mode='w')
    ]
)

class TextBaselineModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, bidirectional=True):
        super(TextBaselineModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)

    def forward(self, x):
        # Ensure input has 3 dimensions (batch_size, seq_len, input_dim)
        if x.dim() == 2:  # If input is (batch_size, input_dim)
            x = x.unsqueeze(1)  # Add a sequence length dimension

        # Pass through LSTM
        x, _ = self.lstm(x)  # x is now (batch_size, seq_len, hidden_dim * num_directions)

        # Take the last hidden state from the sequence
        x = x[:, -1, :]  # (batch_size, hidden_dim * num_directions)

        # Pass through the fully connected layer
        x = self.fc(x)  # (batch_size, output_dim)
        return x

# Normalize embeddings and labels
def create_dataloader(embeddings_dir, split_csv, batch_size):
    split_data = pd.read_csv(split_csv)
    ids = split_data['Filename'].values
    labels = split_data.iloc[:, 1:].values  # Assuming labels are in columns after 'id'

    embeddings = []
    skipped_files = []  # List to log skipped files
    valid_shape = None  # To store the shape of valid embeddings

    for id_ in ids:
        id_str = str(id_).zfill(5)  # Ensure leading zeros
        embedding_path = os.path.join(embeddings_dir, f"{id_str}.pkl")
        try:
            with open(embedding_path, 'rb') as f:
                embedding = pickle.load(f)
                if valid_shape is None:
                    valid_shape = embedding.shape  # Set the valid shape from the first embedding
                if embedding.shape != valid_shape:
                    logging.warning(f"Shape mismatch for {id_str}: {embedding.shape}. Skipping.")
                    skipped_files.append(id_str)
                    continue
                embeddings.append(embedding)
        except FileNotFoundError:
            logging.warning(f"File not found: {embedding_path}. Using default embedding.")
            skipped_files.append(id_str)
            embeddings.append(np.zeros(valid_shape) if valid_shape else np.zeros((1,)))  # Use default embedding

    # Log all skipped files at once
    if skipped_files:
        logging.info(f"Total skipped files: {len(skipped_files)}. Skipped IDs: {skipped_files}")

    embeddings = np.array(embeddings)
    # Normalize embeddings and labels
    embeddings = (embeddings - np.mean(embeddings, axis=0)) / (np.std(embeddings, axis=0) + 1e-8)
    labels = (labels - np.mean(labels, axis=0)) / (np.std(labels, axis=0) + 1e-8)

    dataset = torch.utils.data.TensorDataset(torch.tensor(embeddings, dtype=torch.float32),
                                             torch.tensor(labels, dtype=torch.float32))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

class TextBaselineTrainer:
    def __init__(self, model, device, learning_rate=1e-3, weight_decay=1e-5):
        self.model = model.to(device)
        self.device = device
        self.optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        for embeddings, labels in tqdm.tqdm(dataloader, desc="Training"):
            embeddings = embeddings.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(embeddings)
            loss = self.criterion(outputs, labels)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
        return total_loss / len(dataloader)

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        for embeddings, labels in tqdm.tqdm(dataloader, desc="Evaluating"):
            embeddings = embeddings.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(embeddings)
            loss = self.criterion(outputs, labels)
            total_loss += loss.item()

            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        mse = mean_squared_error(all_labels, all_preds)
        mae = mean_absolute_error(all_labels, all_preds)
        pearson_corrs = [pearsonr(all_labels[:, i], all_preds[:, i])[0] for i in range(all_labels.shape[1])]
        avg_pearson = np.mean(pearson_corrs)

        # Log warning if Pearson correlation is negative
        if avg_pearson < 0:
            logging.warning(f"Negative Pearson correlation detected: {avg_pearson:.4f}")

        return {
            'loss': total_loss / len(dataloader),
            'mse': mse,
            'mae': mae,
            'pearson': avg_pearson
        }

    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)
        logging.info(f"Checkpoint saved to {path}")

def main():
    parser = argparse.ArgumentParser(description='Train EMI predictor using text embeddings')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda', help='Device')
    args = parser.parse_args()

    # Paths
    embeddings_dir = Path('results/checkpoints/baseline/text_embeddings')
    train_csv = Path('data/splits/train_split.csv')
    valid_csv = Path('data/splits/valid_split.csv')
    save_dir = Path('results/checkpoints/baseline/text')
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create dataloaders
    train_loader = create_dataloader(embeddings_dir, train_csv, args.batch_size)
    valid_loader = create_dataloader(embeddings_dir, valid_csv, args.batch_size)

    # Model and trainer
    input_dim = 768  # Adjust based on your text embeddings
    hidden_dim = 256
    output_dim = 6
    model = TextBaselineModel(input_dim, hidden_dim, output_dim)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    trainer = TextBaselineTrainer(model, device, learning_rate=args.lr)

    # Training loop
    best_pearson = -float('inf')
    for epoch in range(args.epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_metrics = trainer.evaluate(valid_loader)

        logging.info(f"Epoch {epoch+1}/{args.epochs}")
        logging.info(f"  Train Loss: {train_loss:.4f}")
        logging.info(f"  Val MSE: {val_metrics['mse']:.4f}")
        logging.info(f"  Val MAE: {val_metrics['mae']:.4f}")
        logging.info(f"  Val Pearson: {val_metrics['pearson']:.4f}")

        if val_metrics['pearson'] > best_pearson:
            best_pearson = val_metrics['pearson']
            checkpoint_path = save_dir / 'best_text_model.pt'
            trainer.save_checkpoint(checkpoint_path)

    logging.info(f"Training complete. Best Pearson: {best_pearson:.4f}")

if __name__ == '__main__':
    main()