#!/usr/bin/env python
"""
Preprocess wind farm data with proper handling of multiple farm features.
Each sample has 9 farms, each with 3 features:
- rated_power
- construction_day
- ss_seed (categorical)
"""

import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import json
import logging
from datetime import datetime
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WindFarmPreprocessor:
    def __init__(self, discount_rates=[0.1, 0.2, 0.3]):
        self.discount_rates = discount_rates
        self.feature_names = [
            'rated_power',
            'construction_day',
            'ss_seed'  # categorical
        ]
        # One scaler per numerical feature type
        self.power_scaler = StandardScaler()
        self.day_scaler = StandardScaler()
        self.revenue_scalers = {}
        
    def calculate_revenue(self, power_data, time_dim='time'):
        """Calculate discounted revenue for different periods."""
        revenues = {}
        
        year1_days = 365
        total_days = power_data[time_dim].size
        time_years = np.arange(total_days) / 365.0
        
        for rate in self.discount_rates:
            discount_factors = 1 / (1 + rate) ** time_years
            
            # First year revenue
            year1_revenue = (power_data.isel({time_dim: slice(0, year1_days)}) * 
                           discount_factors[:year1_days]).sum(dim=time_dim)
            
            # Full lifetime revenue
            full_revenue = (power_data * discount_factors).sum(dim=time_dim)
            
            revenues[f'1yr_{rate}'] = year1_revenue
            revenues[f'10yr_{rate}'] = full_revenue
            
        return revenues
    
    def extract_features(self, dataset, fit_scalers=False):
        """
        Extract and normalize features while maintaining farm structure.
        Returns array of shape (n_samples, n_farms * n_features).
        """
        n_samples = dataset.dims['sample']
        n_farms = dataset.dims['farm']
        
        # Initialize output array
        features = np.zeros((n_samples, n_farms * len(self.feature_names)))
        
        # Process each feature
        for farm_idx in range(n_farms):
            base_idx = farm_idx * len(self.feature_names)
            
            # Rated power (normalize)
            power_values = dataset.rated_power.isel(farm=farm_idx).values
            if fit_scalers:
                power_norm = self.power_scaler.fit_transform(power_values.reshape(-1, 1)).ravel()
            else:
                power_norm = self.power_scaler.transform(power_values.reshape(-1, 1)).ravel()
            features[:, base_idx] = power_norm
            
            # Construction day (normalize)
            day_values = dataset.construction_day.isel(farm=farm_idx).values
            if fit_scalers:
                day_norm = self.day_scaler.fit_transform(day_values.reshape(-1, 1)).ravel()
            else:
                day_norm = self.day_scaler.transform(day_values.reshape(-1, 1)).ravel()
            features[:, base_idx + 1] = day_norm
            
            # ss_seed (categorical, keep as is)
            seed_values = dataset.ss_seed.isel(farm=farm_idx).values
            features[:, base_idx + 2] = seed_values
        
        return features
    
    def process_dataset(self, dataset, fit_scalers=False):
        """Process complete dataset into features and revenues."""
        # Extract features
        X = self.extract_features(dataset, fit_scalers=fit_scalers)
        
        # Calculate revenues
        revenues = self.calculate_revenue(dataset.Power)
        
        # Package results
        processed = {
            'X': X,  # Shape: (n_samples, n_farms * n_features)
            'raw_power': dataset.Power.values
        }
        
        # Add revenue targets - both raw and normalized
        for rev_key, rev_values in revenues.items():
            raw_values = rev_values.values
            
            # Store raw values
            processed[f'raw_revenue_{rev_key}'] = raw_values
            
            # Create/use scaler for this revenue type
            if fit_scalers:
                scaler = StandardScaler()
                normalized_values = scaler.fit_transform(
                    raw_values.reshape(-1, 1)
                ).ravel()
                self.revenue_scalers[rev_key] = scaler
            else:
                scaler = self.revenue_scalers[rev_key]
                normalized_values = scaler.transform(
                    raw_values.reshape(-1, 1)
                ).ravel()
            
            processed[f'y_revenue_{rev_key}'] = normalized_values
            
        return processed
    
    def save_scalers(self, output_dir):
        """Save all scalers for future use."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save feature scalers
        joblib.dump(self.power_scaler, output_dir / 'power_scaler.joblib')
        joblib.dump(self.day_scaler, output_dir / 'day_scaler.joblib')
        
        # Save revenue scalers
        for rev_key, scaler in self.revenue_scalers.items():
            joblib.dump(scaler, output_dir / f'revenue_scaler_{rev_key}.joblib')
    
    @classmethod
    def load_scalers(cls, input_dir):
        """Load saved scalers and create preprocessor instance."""
        input_dir = Path(input_dir)
        preprocessor = cls()
        
        # Load feature scalers
        preprocessor.power_scaler = joblib.load(input_dir / 'power_scaler.joblib')
        preprocessor.day_scaler = joblib.load(input_dir / 'day_scaler.joblib')
        
        # Load revenue scalers
        for scaler_path in input_dir.glob('revenue_scaler_*.joblib'):
            rev_key = scaler_path.stem.replace('revenue_scaler_', '')
            preprocessor.revenue_scalers[rev_key] = joblib.load(scaler_path)
            
        return preprocessor

def save_metadata(output_dir, preprocessor):
    """Save preprocessing metadata."""
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'feature_names': preprocessor.feature_names,
        'n_farms': 9,
        'features_per_farm': len(preprocessor.feature_names),
        'total_features': 9 * len(preprocessor.feature_names),
        'discount_rates': preprocessor.discount_rates,
        'train_samples': 5000,
        'valid_samples': 2000,
        'revenue_periods': ['1yr', '10yr'],
        'distributions': ['uniform', 'exp1', 'exp2'],
        'normalized_features': ['rated_power', 'construction_day'],
        'categorical_features': ['ss_seed'],
        'scaler_type': 'StandardScaler'
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

def verify_shapes(data, dataset_name):
    """Verify that all arrays have consistent first dimension (n_samples)."""
    n_samples = None
    inconsistencies = []
    
    for key, array in data.items():
        if n_samples is None:
            n_samples = array.shape[0]
        elif array.shape[0] != n_samples:
            inconsistencies.append(
                f"{key}: expected {n_samples} samples, got {array.shape[0]}"
            )
    
    if inconsistencies:
        logger.error(f"\nShape inconsistencies in {dataset_name}:")
        for msg in inconsistencies:
            logger.error(f"  {msg}")
        raise ValueError(f"Inconsistent shapes in {dataset_name}")
    else:
        logger.info(f"\n{dataset_name} shapes verified: {n_samples} samples")
        for key, array in data.items():
            logger.info(f"  {key}: {array.shape}")

def main():
    # Set up directories
    output_dir = Path('preprocessed_data')
    output_dir.mkdir(exist_ok=True)
    scaler_dir = output_dir / 'scalers'
    
    # Load datasets
    logger.info("Loading raw datasets...")
    uniform = xr.load_dataset("uniform.nc")
    exp1 = xr.load_dataset("exponential_1yr.nc")
    exp2 = xr.load_dataset("exponential_2yr.nc")
    
    # Initialize preprocessor
    preprocessor = WindFarmPreprocessor()
    
    # Process training data (first 5000 uniform samples)
    logger.info("Processing training data...")
    train_data = uniform.isel(sample=slice(0, 5000))
    train_processed = preprocessor.process_dataset(train_data, fit_scalers=True)
    verify_shapes(train_processed, "Training data")
    np.savez(output_dir / 'train.npz', **train_processed)
    
    # Save scalers after fitting on training data
    preprocessor.save_scalers(scaler_dir)
    
    # Process validation data (next 2000 uniform samples)
    logger.info("Processing validation data...")
    valid_data = uniform.isel(sample=slice(5000, 7000))
    valid_processed = preprocessor.process_dataset(valid_data)
    verify_shapes(valid_processed, "Validation data")
    np.savez(output_dir / 'valid.npz', **valid_processed)
    
    # Process test data
    logger.info("Processing test data...")
    
    # Remaining uniform samples
    test_uniform = uniform.isel(sample=slice(7000, None))
    test_uniform_processed = preprocessor.process_dataset(test_uniform)
    verify_shapes(test_uniform_processed, "Test uniform data")
    np.savez(output_dir / 'test_uniform.npz', **test_uniform_processed)
    
    # Exponential distributions
    test_exp1_processed = preprocessor.process_dataset(exp1)
    verify_shapes(test_exp1_processed, "Test exp1 data")
    np.savez(output_dir / 'test_exp1.npz', **test_exp1_processed)
    
    test_exp2_processed = preprocessor.process_dataset(exp2)
    verify_shapes(test_exp2_processed, "Test exp2 data")
    np.savez(output_dir / 'test_exp2.npz', **test_exp2_processed)
    
    # Save metadata
    save_metadata(output_dir, preprocessor)
    
    logger.info("Preprocessing complete!")
    logger.info(f"Data saved to: {output_dir}")
    logger.info(f"Scalers saved to: {scaler_dir}")

if __name__ == "__main__":
    main()
