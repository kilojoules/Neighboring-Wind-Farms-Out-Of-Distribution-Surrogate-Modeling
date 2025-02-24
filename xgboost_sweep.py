#!/usr/bin/env python
"""
XGBoost parameter sweep for wind farm revenue prediction.
Run as an array job across different revenue scenarios.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
import json
import logging
import argparse
from sklearn.metrics import mean_squared_error
from scipy.stats import ks_2samp
import pickle
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
    
    # Log available keys
    logger.info(f"Available keys in train data: {train_data.files}")
    
    # Get X features
    X_train = train_data['X']
    X_valid = valid_data['X']
    X_test_uniform = test_uniform['X']
    X_test_exp1 = test_exp1['X']
    X_test_exp2 = test_exp2['X']
    
    # Get target variable and ensure it's flattened
    y_train = train_data[f'y_revenue_{target_scenario}'].ravel()
    y_valid = valid_data[f'y_revenue_{target_scenario}'].ravel()
    y_test_uniform = test_uniform[f'y_revenue_{target_scenario}'].ravel()
    y_test_exp1 = test_exp1[f'y_revenue_{target_scenario}'].ravel()
    y_test_exp2 = test_exp2[f'y_revenue_{target_scenario}'].ravel()
    
    # Log shapes for debugging
    logger.info(f"Training shapes - X: {X_train.shape}, y: {y_train.shape}")
    logger.info(f"Validation shapes - X: {X_valid.shape}, y: {y_valid.shape}")
    logger.info(f"Test shapes - Uniform: {X_test_uniform.shape}, Exp1: {X_test_exp1.shape}, Exp2: {X_test_exp2.shape}")
    
    return {
        'train': (X_train, y_train),
        'valid': (X_valid, y_valid),
        'test_uniform': (X_test_uniform, y_test_uniform),
        'test_exp1': (X_test_exp1, y_test_exp1),
        'test_exp2': (X_test_exp2, y_test_exp2)
    }

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

def run_parameter_sweep(data, output_dir, n_jobs=10):
    """Run XGBoost parameter sweep and save results."""
    X_train, y_train = data['train']
    X_valid, y_valid = data['valid']
    
    # Parameter grid
    param_grid = {
        'learning_rate': [0.1, 0.01, 0.001],
        'max_depth': [3, 6, 9],
        'subsample': [0.8, 0.9, 1.0],
        'n_estimators': [100, 500, 1000, 2000, 5000],
        'min_child_weight': [1, 3, 5],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    results = []
    best_score = float('inf')
    best_model = None
    best_params = None
    
    # Generate all parameter combinations
    param_combinations = [
        {
            'learning_rate': lr,
            'max_depth': md,
            'subsample': ss,
            'n_estimators': ne,
            'min_child_weight': mcw,
            'colsample_bytree': cbt,
            'random_state': 42,
            'n_jobs': n_jobs
        }
        for lr in param_grid['learning_rate']
        for md in param_grid['max_depth']
        for ss in param_grid['subsample']
        for ne in param_grid['n_estimators']
        for mcw in param_grid['min_child_weight']
        for cbt in param_grid['colsample_bytree']
    ]
    
    total_combinations = len(param_combinations)
    logger.info(f"Starting parameter sweep with {total_combinations} combinations")
    
    for i, params in enumerate(param_combinations, 1):
        start_time = time.time()
        
        # Train model
        model = xgb.XGBRegressor(
            **params,
            callbacks=[
                xgb.callback.EarlyStopping(
                    rounds=50,
                    save_best=True
                )
            ]
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False
        )
        
        # Evaluate on all datasets
        metrics = {}
        for dataset_name, (X, y) in data.items():
            metrics[dataset_name] = evaluate_model(model, X, y)
        
        # Track results
        result = {
            'params': params,
            'metrics': metrics,
            'best_iteration': model.best_iteration if hasattr(model, 'best_iteration') else None
        }
        results.append(result)
        
        # Update best model if validation KS statistic improves
        val_ks = metrics['valid']['ks_statistic']
        if val_ks < best_score:
            best_score = val_ks
            best_model = model
            best_params = params
        
        # Log progress
        elapsed = time.time() - start_time
        logger.info(f"Completed {i}/{total_combinations} "
                   f"(KS: {val_ks:.4f}, Time: {elapsed:.1f}s)")
    
    return results, best_model, best_params

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario_idx', type=int, required=True,
                       help='Index of revenue scenario to process')
    args = parser.parse_args()
    
    # Verify data directory exists
    data_dir = Path('preprocessed_data')
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Define revenue scenarios
    scenarios = [
        '1yr_0.1', '1yr_0.2', '1yr_0.3',
        '10yr_0.1', '10yr_0.2', '10yr_0.3'
    ]
    
    # Get scenario for this job
    scenario = scenarios[args.scenario_idx]
    
    # Setup directories
    data_dir = Path('preprocessed_data')
    output_dir = Path(f'xgboost_results_{scenario}')
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    logger.info(f"Loading data for scenario: {scenario}")
    data = load_data(data_dir, scenario)
    
    # Run parameter sweep
    logger.info("Starting parameter sweep")
    results, best_model, best_params = run_parameter_sweep(
        data, output_dir, n_jobs=10
    )
    
    # Save results
    logger.info("Saving results")
    results_file = output_dir / 'sweep_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save best model
    model_file = output_dir / 'best_model.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(best_model, f)
    
    logger.info("Parameter sweep complete!")
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best validation KS: {best_score}")

if __name__ == '__main__':
    main()
