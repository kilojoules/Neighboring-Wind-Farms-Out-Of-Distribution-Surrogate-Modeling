#!/usr/bin/env python3

import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import sys 
import matplotlib.pyplot as plt 
from optuna.visualization import matplotlib as optuna_plot

# --- Configuration ---
BASE_PROJECT_DIR = "/work/users/juqu/Neighboring-Wind-Farms-Out-Of-Distribution-Surrogate-Modeling" 
UNIFORM_DATA_FILE = os.path.join(BASE_PROJECT_DIR, 'uniform_results/processed_xy_data/uniform_xy_data.csv')

N_FARMS = 9
INPUT_FEATURES_PER_FARM_BASENAMES = ['rated_power', 'construction_day', 'ss_seed']
FEATURE_COLUMNS = []
for i in range(N_FARMS):
    for basename in INPUT_FEATURES_PER_FARM_BASENAMES:
        FEATURE_COLUMNS.append(f"{basename}_{i}")

TARGET_COLUMN = 'mean_power_all_time'

N_OPTIMIZATION_SAMPLES_DEFAULT = 80 
N_HOLDOUT_TEST_SAMPLES_DEFAULT = 20 

N_OPTUNA_TRIALS = 50 
N_CV_FOLDS = 4      
EARLY_STOPPING_ROUNDS_FIXED = 10 # Fixed value for early stopping

XGB_FIXED_PARAMS = {
    'objective': 'reg:squarederror',
    'random_state': 42,
    'n_jobs': -1 
}

# --- Helper Functions ---

def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"ERROR: Data file not found at {file_path}")
        return None
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {file_path}, shape: {df.shape}")
        return df
    except Exception as e:
        print(f"ERROR: Could not load data from {file_path}. Error: {e}")
        return None

# --- Optuna Objective Function ---

def objective(trial, X, y):
    optuna_suggested_params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000, step=50),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.1),
        'gamma': trial.suggest_float('gamma', 0, 0.5, step=0.1),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True)
    }
    
    # Combine Optuna's suggestions with fixed parameters and early stopping for constructor
    model_params_for_constructor = {}
    model_params_for_constructor.update(XGB_FIXED_PARAMS) # Start with fixed ones
    model_params_for_constructor.update(optuna_suggested_params) # Add Optuna's suggestions
    model_params_for_constructor['early_stopping_rounds'] = EARLY_STOPPING_ROUNDS_FIXED # Add early stopping

    kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=trial.number) 
    cv_rmse_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBRegressor(**model_params_for_constructor)
        
        # eval_set is now crucial because early_stopping_rounds is in the constructor
        model.fit(X_train_fold, y_train_fold,
                  eval_set=[(X_val_fold, y_val_fold)],
                  verbose=False) 
        
        preds = model.predict(X_val_fold)
        rmse = np.sqrt(mean_squared_error(y_val_fold, preds))
        cv_rmse_scores.append(rmse)

    return np.mean(cv_rmse_scores)

# --- Main Execution ---

if __name__ == '__main__':
    print("--- Loading Data ---")
    uniform_df = load_data(UNIFORM_DATA_FILE)

    if uniform_df is None or uniform_df.empty:
        print("Exiting due to data loading issues.")
        sys.exit(1)
    
    num_total_samples = len(uniform_df)
    n_optimization_samples = N_OPTIMIZATION_SAMPLES_DEFAULT
    n_holdout_test_samples = N_HOLDOUT_TEST_SAMPLES_DEFAULT

    if num_total_samples != (N_OPTIMIZATION_SAMPLES_DEFAULT + N_HOLDOUT_TEST_SAMPLES_DEFAULT) or \
       (N_OPTIMIZATION_SAMPLES_DEFAULT + N_HOLDOUT_TEST_SAMPLES_DEFAULT == 0 and num_total_samples > 0) : # check if defaults were 0 but data exists
        print(f"Note: Total samples ({num_total_samples}) requires adjusting default split "
              f"({N_OPTIMIZATION_SAMPLES_DEFAULT}+{N_HOLDOUT_TEST_SAMPLES_DEFAULT}). Adjusting split.")
        if num_total_samples < 20: 
            print(f"ERROR: Dataset too small ({num_total_samples} samples) for opt/test split. Need at least ~20. Exiting.")
            sys.exit(1)
        
        n_holdout_test_samples = max(10, int(0.2 * num_total_samples))
        temp_opt_samples = num_total_samples - n_holdout_test_samples
        
        if temp_opt_samples < N_CV_FOLDS: 
            # Reduce holdout to make opt set large enough for N_CV_FOLDS, if possible
            n_holdout_test_samples = num_total_samples - N_CV_FOLDS 
            if n_holdout_test_samples < 0: n_holdout_test_samples = 0 
        
        n_optimization_samples = num_total_samples - n_holdout_test_samples

        if n_optimization_samples < N_CV_FOLDS :
             print(f"ERROR: Optimization set ({n_optimization_samples}) too small for {N_CV_FOLDS} folds after reserving test set. "
                   "Reduce N_CV_FOLDS, get more data, or reduce desired holdout size. Exiting.")
             sys.exit(1)
        print(f"Adjusted data split: Total={num_total_samples}, Opt set={n_optimization_samples}, Holdout test set={n_holdout_test_samples}")

    X_all = uniform_df[FEATURE_COLUMNS]
    y_all = uniform_df[TARGET_COLUMN]

    print(f"\n--- Splitting Data ---")
    if n_optimization_samples + n_holdout_test_samples > num_total_samples:
        print("ERROR: Sum of optimization and holdout samples somehow exceeds total samples. Check logic.")
        sys.exit(1)

    X_opt = X_all.iloc[:n_optimization_samples]
    y_opt = y_all.iloc[:n_optimization_samples]
    
    X_holdout_test = X_all.iloc[n_optimization_samples : n_optimization_samples + n_holdout_test_samples]
    y_holdout_test = y_all.iloc[n_optimization_samples : n_optimization_samples + n_holdout_test_samples]
    
    print(f"Optimization set size: {len(X_opt)} samples.")
    if n_holdout_test_samples > 0:
        print(f"Hold-out test set size: {len(X_holdout_test)} samples.")
        if X_holdout_test.empty and not y_holdout_test.empty:
             print("WARNING: Hold-out test set X or y is empty when it shouldn't be. Check split logic.")
    else:
        print("No hold-out test set will be used (n_holdout_test_samples is 0).")

    print(f"\n--- Starting Optuna Hyperparameter Optimization ---")
    print(f"Number of Optuna trials: {N_OPTUNA_TRIALS}")
    print(f"Number of CV folds within each trial: {N_CV_FOLDS}")
    print(f"Early stopping rounds (fixed for constructor): {EARLY_STOPPING_ROUNDS_FIXED}")
    
    study = optuna.create_study(direction='minimize')
    try:
        study.optimize(lambda trial: objective(trial, X_opt, y_opt), n_trials=N_OPTUNA_TRIALS)
    except Exception as e:
        print(f"An error occurred during Optuna optimization: {e}")
        print("Please check your data and XGBoost installation.")
        if hasattr(study, 'trials') and study.trials:
             print(f"Optuna ran for {len(study.trials)} trials before stopping.")
        sys.exit(1)

    print("\n--- Optuna Optimization Finished ---")
    if not study.trials: 
        print("No Optuna trials completed. Cannot determine best parameters or plot results.")
        sys.exit(1)
        
    print(f"Number of finished trials: {len(study.trials)}")
    best_trial = study.best_trial
    print("Best trial:")
    print(f"  Value (minimized avg CV RMSE): {best_trial.value:.4f}")
    print("  Best Parameters (tuned by Optuna): ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    print("\n--- Generating Optuna Visualization Plots ---")
    try:
        fig_history = optuna_plot.plot_optimization_history(study)
        plt.savefig("optuna_optimization_history.png")
        print("Saved optuna_optimization_history.png")
        plt.close(fig_history.figure if hasattr(fig_history, 'figure') else fig_history)

        fig_importance = optuna_plot.plot_param_importances(study)
        plt.savefig("optuna_param_importances.png")
        print("Saved optuna_param_importances.png")
        plt.close(fig_importance.figure if hasattr(fig_importance, 'figure') else fig_importance)

        key_params_for_slice_plot = ['n_estimators', 'learning_rate', 'max_depth']
        params_to_plot_slice = [p for p in key_params_for_slice_plot if p in best_trial.params]
        if params_to_plot_slice:
            fig_slice = optuna_plot.plot_slice(study, params=params_to_plot_slice)
            plt.savefig("optuna_slice_plot.png")
            print("Saved optuna_slice_plot.png")
            plt.close(fig_slice.figure if hasattr(fig_slice, 'figure') else fig_slice)
        else:
            print("Skipping slice plot as key parameters were not tuned or found in best_trial.params.")
    except Exception as e:
        print(f"Error generating Optuna plots (matplotlib might be needed, or an issue with study object): {e}")

    print("\n--- Training Final Model with Best Parameters ---")
    # Combine Optuna's best params with fixed params and early stopping for the constructor
    final_model_params = {}
    final_model_params.update(XGB_FIXED_PARAMS)       # Start with fixed
    final_model_params.update(best_trial.params)      # Add Optuna's best
    final_model_params['early_stopping_rounds'] = EARLY_STOPPING_ROUNDS_FIXED # Add early stopping for final model
    # Ensure n_estimators from Optuna is used, not overwritten if it was in XGB_FIXED_PARAMS (it isn't here)

    final_model = xgb.XGBRegressor(**final_model_params)
    
    # Train on the full optimization set. For final evaluation, we need an eval_set if early_stopping_rounds is active.
    # We'll use the holdout test set as eval_set here for training the *final* model,
    # which is common practice if you want early stopping on the final training run.
    # Alternatively, train on X_opt without early stopping for the tuned n_estimators.
    # Given we have a separate holdout, using it as eval_set for early stopping the *final training run* is fine.
    if not X_holdout_test.empty and not y_holdout_test.empty:
        print(f"Training final model on {len(X_opt)} samples, using hold-out set ({len(X_holdout_test)} samples) for early stopping monitoring.")
        final_model.fit(X_opt, y_opt, eval_set=[(X_holdout_test, y_holdout_test)], verbose=False)
    else:
        print(f"Training final model on {len(X_opt)} samples (no hold-out set for early stopping monitoring during this final fit).")
        final_model.fit(X_opt, y_opt, verbose=False) 
    print("Final model training complete.")

    if not X_holdout_test.empty and not y_holdout_test.empty:
        print("\n--- Evaluating Final Model on Hold-out Test Set ---")
        test_predictions = final_model.predict(X_holdout_test)
        test_mae = mean_absolute_error(y_holdout_test, test_predictions)
        test_rmse = np.sqrt(mean_squared_error(y_holdout_test, test_predictions))
        test_r2 = r2_score(y_holdout_test, test_predictions)
        print(f"Hold-out Test MAE: {test_mae:.4f}")
        print(f"Hold-out Test RMSE: {test_rmse:.4f}")
        print(f"Hold-out Test R2: {test_r2:.4f}")
        
        try:
            plt.figure(figsize=(10, max(8, len(FEATURE_COLUMNS) // 2))) 
            xgb.plot_importance(final_model, height=0.9)
            plt.title("Feature Importance (Final Model)")
            plt.tight_layout()
            plt.savefig("final_model_feature_importance.png")
            print("Saved final_model_feature_importance.png")
            plt.close()
        except Exception as e:
            print(f"Error plotting feature importance: {e}")
    else:
        print("\nHold-out test set is empty or was not created, skipping final evaluation and feature importance plot.")
        
    print("\n--- Script Finished ---")
