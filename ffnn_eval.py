#!/usr/bin/env python
"""
Test Feed Forward Neural Networks (square and triangular) for wind farm revenue prediction.
Accepts architecture parameters from command line.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import argparse
import hashlib
import logging
import json
from scipy.stats import ks_2samp
import time
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FFNN(nn.Module):
    def __init__(self, input_size, layer_sizes, dropout_rate=0.1):
        """
        Initialize FFNN with configurable architecture.
        Args:
            input_size: Number of input features
            layer_sizes: List of hidden layer sizes (excluding input/output)
            dropout_rate: Dropout probability
        """
        super().__init__()
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for size in layer_sizes:
            layers.extend([
                nn.Linear(prev_size, size),
                nn.ReLU(),
                nn.BatchNorm1d(size),
                nn.Dropout(dropout_rate)
            ])
            prev_size = size
        
        # Output layer (single value prediction)
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze()

def load_data(data_dir, revenue_scenario):
    """Load and prepare data for training."""
    # Load datasets
    train_data = np.load(data_dir / 'train.npz')
    valid_data = np.load(data_dir / 'valid.npz')
    test_uniform = np.load(data_dir / 'test_uniform.npz')
    test_exp1 = np.load(data_dir / 'test_exp1.npz')
    test_exp2 = np.load(data_dir / 'test_exp2.npz')
    
    # Extract features and targets
    data = {}
    for name, dataset in [
        ('train', train_data),
        ('valid', valid_data),
        ('test_uniform', test_uniform),
        ('test_exp1', test_exp1),
        ('test_exp2', test_exp2)
    ]:
        X = dataset['X']
        y = dataset[f'y_revenue_{revenue_scenario}']
        y_raw = dataset[f'raw_revenue_{revenue_scenario}']
        
        # Convert to PyTorch tensors
        data[name] = {
            'X': torch.FloatTensor(X),
            'y': torch.FloatTensor(y),
            'y_raw': torch.FloatTensor(y_raw)
        }
    
    return data

def create_dataloaders(data, batch_size):
    """Create DataLoaders for training."""
    # Get number of CPU cores and limit workers accordingly
    num_workers = min(os.cpu_count() or 1, 2)  # Use at most 2 workers
    logger.info(f"Using {num_workers} DataLoader workers")
    
    loaders = {}
    for name, tensors in data.items():
        dataset = TensorDataset(tensors['X'], tensors['y'])
        shuffle = name == 'train'
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
    return loaders

def evaluate_model(model, dataloader, data_tensors, device):
    """Evaluate model using same metrics as XGBoost comparison."""
    model.eval()
    
    # Get predictions
    y_pred = []
    with torch.no_grad():
        for X, _ in dataloader:
            X = X.to(device)
            pred = model(X).cpu().numpy()
            y_pred.append(pred)
    
    y_pred = np.concatenate(y_pred)
    y_true = data_tensors['y'].numpy()  # Using normalized values
    
    # Calculate metrics using normalized values
    ks_stat, _ = ks_2samp(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mean_diff = np.abs(np.mean(y_true) - np.mean(y_pred))
    
    return {
        'ks_statistic': float(ks_stat),
        'rmse': float(rmse),
        'mean_difference': float(mean_diff)
    }

def train_model(
    model, train_loader, valid_loader, valid_tensors,
    device, learning_rate, max_epochs, patience,
    max_valid_loss=1e6  # Add maximum validation loss threshold
):
    """Train model with early stopping."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    best_valid_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    final_epoch = 0
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        train_losses = []
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        valid_losses = []
        
        with torch.no_grad():
            for X, y in valid_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                loss = criterion(pred, y)
                valid_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        valid_loss = np.mean(valid_losses)
        
        # Check for divergence
        if valid_loss > max_valid_loss:
            logger.warning(f"Validation loss {valid_loss:.2f} exceeded threshold {max_valid_loss:.2f}. Stopping training.")
            break
        
        # Early stopping check
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Log progress
        if (epoch + 1) % 10 == 0:
            metrics = evaluate_model(model, valid_loader, valid_tensors, device)
            logger.info(
                f"Epoch {epoch+1}: train_loss={train_loss:.4f}, "
                f"valid_loss={valid_loss:.4f}, valid_ks={metrics['ks_statistic']:.4f}"
            )
        
        final_epoch = epoch + 1
        
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model, final_epoch

def get_config_hash(args):
    """Create a unique hash for this configuration"""
    config_dict = {
        'scenario_idx': args.scenario_idx,
        'architecture': args.architecture,
        'hidden_width': args.hidden_width,
        'num_layers': args.num_layers,
        'learning_rate': args.learning_rate,
        'dropout_rate': args.dropout_rate
    }
    # Convert to string and hash
    return hashlib.md5(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()

def results_exist(output_dir, config_hash):
    """Check if results already exist for this configuration"""
    if not output_dir.exists():
        return False
        
    # Look for any results file that contains this config
    for results_file in output_dir.glob('results_*.json'):
        try:
            with open(results_file) as f:
                results = json.load(f)
                if results.get('config_hash') == config_hash:
                    logger.info(f"Results already exist: {results_file}")
                    return True
        except Exception as e:
            logger.warning(f"Error reading {results_file}: {e}")
            continue
    return False

def main():
    parser = argparse.ArgumentParser()
    
    # Data and architecture parameters
    parser.add_argument('--data_dir', type=str, default='preprocessed_data',
                       help='Directory containing preprocessed data')
    parser.add_argument('--scenario_idx', type=int, required=True,
                       help='Index of revenue scenario to process')
    parser.add_argument('--architecture', type=str, choices=['square', 'triangle'],
                       required=True, help='Network architecture type')
    parser.add_argument('--hidden_width', type=int, required=True,
                       help='Width of hidden layers (for square) or first hidden layer (for triangle)')
    parser.add_argument('--num_layers', type=int, required=True,
                       help='Number of hidden layers')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--max_epochs', type=int, default=10000)
    parser.add_argument('--patience', type=int, default=1000)
    parser.add_argument('--max_valid_loss', type=float, default=1e6,
                       help='Maximum validation loss before stopping training')
    
    args = parser.parse_args()

    # Define revenue scenarios
    scenarios = [
        '1yr_0.1', '1yr_0.2', '1yr_0.3',
        '10yr_0.1', '10yr_0.2', '10yr_0.3'
    ]
    scenario = scenarios[args.scenario_idx]
    
    # Create config hash and results dictionary early
    config_hash = get_config_hash(args)
    results = {
        'config': vars(args),
        'config_hash': config_hash,
        'layer_sizes': None,
        'final_epochs': None,
        'metrics': {}
    }
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    data_dir = Path(args.data_dir)
    
    # Check if results exist BEFORE loading data
    output_dir = Path(f'ffnn_results3_{scenario}_{args.architecture}')
    if results_exist(output_dir, config_hash):
        logger.info("Results already exist for this configuration. Skipping.")
        return
    
    # Load data
    logger.info(f"Loading data for scenario: {scenario}")
    data = load_data(data_dir, scenario)
    
    # Create dataloaders
    dataloaders = create_dataloaders(data, args.batch_size)
    
    # Determine layer sizes based on architecture
    input_size = data['train']['X'].shape[1]
    if args.architecture == 'square':
        layer_sizes = [args.hidden_width] * args.num_layers
    else:  # triangle
        decay = (args.hidden_width - 32) / (args.num_layers - 1)
        layer_sizes = [
            int(args.hidden_width - i * decay)
            for i in range(args.num_layers)
        ]
    
    # Update results with layer sizes
    results['layer_sizes'] = layer_sizes
    
    # Create and train model
    model = FFNN(
        input_size=input_size,
        layer_sizes=layer_sizes,
        dropout_rate=args.dropout_rate
    )
    logger.info(f"Created {args.architecture} FFNN with layers: {[input_size] + layer_sizes + [1]}")
    
    try:
        model, final_epochs = train_model(
            model=model,
            train_loader=dataloaders['train'],
            valid_loader=dataloaders['valid'],
            valid_tensors=data['valid'],
            device=device,
            learning_rate=args.learning_rate,
            max_epochs=args.max_epochs,
            patience=args.patience,
            max_valid_loss=args.max_valid_loss
        )
        
        # Update results with final epochs
        results['final_epochs'] = final_epochs
        
        # Evaluate on all datasets
        for name, loader in dataloaders.items():
            metrics = evaluate_model(model, loader, data[name], device)
            results['metrics'][name] = metrics
            logger.info(f"\n{name} metrics:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
        
        # Save results and model
        try:
            output_dir.mkdir(exist_ok=True, parents=True)
            
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            results_file = output_dir / f'results_{timestamp}.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            model_file = output_dir / f'model_{timestamp}.pt'
            torch.save(model.state_dict(), model_file)
            
            logger.info(f"\nResults saved to: {results_file}")
            logger.info(f"Model saved to: {model_file}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise
            
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

if __name__ == '__main__':
    main()
