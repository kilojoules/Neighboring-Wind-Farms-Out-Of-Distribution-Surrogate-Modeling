#!/usr/bin/env python
"""
Test Linear Models for wind farm revenue prediction.
Follows similar structure to XGBoost and FFNN tests.
"""

import numpy as np
from pathlib import Path
import json
import logging
import argparse
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from scipy.stats import ks_2samp
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(data_dir, target_scenario):
    """Load preprocessed data for given revenue scenario."""
    train_data = np.load(data_dir / 'train.npz')
    valid_data = np.load(data_dir / 'valid.npz')
    test_uniform = np.load(data_dir / 'test_uniform.npz')
    test_exp1 = np.load(data_dir / 'test_exp1.npz')
    test_exp2 = np.load(data_dir / 'test_exp2.npz')
    
    logger.info("Data files loaded successfully")
    
    # Get X features and target variables
    data = {}
    for name, dataset in [
        ('train', train_data),
        ('valid', valid_data),
        ('test_uniform', test_uniform),
        ('test_exp1', test_exp1),
        ('test_exp2', test_exp2)
    ]:
        data[name] = {
            'X': dataset['X'],
            'y': dataset[f'y_revenue_{target_scenario}'].ravel(),
            'y_raw': dataset[f'raw_revenue_{target_scenario}'].ravel()
        }
        logger.info(f"{name} shapes - X: {data[name]['X'].shape}, y: {data[name]['y'].shape}")
    
    return data

def evaluate_model(model, X, y_true):
    """Evaluate model using multiple metrics."""
    y_pred = model.predict(X)
    
    ks_stat, _ = ks_2samp(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mean_diff = np.abs(np.mean(y_true) - np.mean(y_pred))
    
    return {
        'ks_statistic': float(ks_stat),
        'rmse': float(rmse),
        'mean_difference': float(mean_diff)
    }

def run_parameter_sweep(data):
    """Run parameter sweep for different linear models."""
    X_train, y_train = data['train']['X'], data['train']['y']
    X_valid, y_valid = data['valid']['X'], data['valid']['y']
    
    # Parameter grid
    models = {
        'ridge': {
            'class': Ridge,
            'params': [
                {'alpha': alpha}
                for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]
            ]
        },
        'lasso': {
            'class': Lasso,
            'params': [
                {'alpha': alpha}
                for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]
            ]
        },
        'elastic_net': {
            'class': ElasticNet,
            'params': [
                {'alpha': alpha, 'l1_ratio': l1_ratio}
                for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]
                for l1_ratio in [0.2, 0.5, 0.8]
            ]
        }
    }
    
    results = []
    best_score = float('inf')
    best_model = None
    best_config = None
    
    total_combinations = sum(len(model['params']) for model in models.values())
    logger.info(f"Starting parameter sweep with {total_combinations} combinations")
    
    for model_name, model_config in models.items():
        model_class = model_config['class']
        
        for params in model_config['params']:
            start_time = time.time()
            
            # Train model
            model = model_class(random_state=42, **params)
            model.fit(X_train, y_train)
            
            # Evaluate on all datasets
            metrics = {}
            for dataset_name, dataset in data.items():
                metrics[dataset_name] = evaluate_model(
                    model, dataset['X'], dataset['y']
                )
            
            # Track results
            config = {
                'model_type': model_name,
                'params': params
            }
            result = {
                'config': config,
                'metrics': metrics
            }
            results.append(result)
            
            # Update best model if validation RMSE improves
            val_rmse = metrics['valid']['rmse']
            if val_rmse < best_score:
                best_score = val_rmse
                best_model = model
                best_config = config
            
            # Log progress
            elapsed = time.time() - start_time
            logger.info(
                f"Completed {model_name} with {params} "
                f"(RMSE: {val_rmse:.4f}, Time: {elapsed:.1f}s)"
            )
    
    return results, best_model, best_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='preprocessed_data',
                       help='Directory containing preprocessed data')
    parser.add_argument('--scenario_idx', type=int, required=True,
                       help='Index of revenue scenario to process')
    args = parser.parse_args()
    
    # Define revenue scenarios
    scenarios = [
        '1yr_0.1', '1yr_0.2', '1yr_0.3',
        '10yr_0.1', '10yr_0.2', '10yr_0.3'
    ]
    scenario = scenarios[args.scenario_idx]
    
    # Setup directories
    data_dir = Path(args.data_dir)
    output_dir = Path(f'linear_results_{scenario}')
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    logger.info(f"Loading data for scenario: {scenario}")
    data = load_data(data_dir, scenario)
    
    # Run parameter sweep
    logger.info("Starting parameter sweep")
    results, best_model, best_config = run_parameter_sweep(data)
    
    # Compile final results
    final_results = {
        'scenario': scenario,
        'best_config': best_config,
        'results': results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save results
    results_file = output_dir / 'sweep_results.json'
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    logger.info("Parameter sweep complete!")
    logger.info(f"Best model type: {best_config['model_type']}")
    logger.info(f"Best parameters: {best_config['params']}")
    logger.info(f"Results saved to: {results_file}")

if __name__ == '__main__':
    main()
