import h5py
import xarray as xr
import numpy as np
import os
from sklearn.model_selection import train_test_split

# --- Function to load data from the H5 file ---
def load_data_from_h5_by_sample(h5_filepath):
    """
    Loads data from an H5 file structured with '/sample_XXXX' groups,
    extracting only the 'Power' data for farm 0,
    into a single xarray.Dataset.
    """
    all_sample_datasets = []
    
    with h5py.File(h5_filepath, 'r') as f:
        sample_groups_names = [name for name in f.keys() if name.startswith('sample_')]
        sample_numbers = []
        for name in sample_groups_names:
            try:
                sample_numbers.append(int(name.split('_')[1]))
            except ValueError:
                continue
        sample_numbers.sort()


        for sample_num in sample_numbers:
            group_name = f'sample_{sample_num}'
            if group_name in f:
                sample_group = f[group_name]
                
                if 'Power' not in sample_group:
                    print(f"Warning: 'Power' dataset not found in group {group_name}. Skipping sample.")
                    print(f"Available keys in {group_name}: {list(sample_group.keys())}")
                    continue
                
                if 'time' not in sample_group:
                    print(f"Warning: 'time' dataset not found in group {group_name}. Skipping sample.")
                    print(f"Available keys in {group_name}: {list(sample_group.keys())}")
                    continue

                try:
                    power_data_all_farms = sample_group['Power'][:]
                    time_coords = sample_group['time'][:]

                    if power_data_all_farms.ndim >= 2 and power_data_all_farms.shape[0] > 0:
                        power_data_farm0 = power_data_all_farms[0, :]
                    else:
                        print(f"Warning: Power data for {group_name} is not 2D or has no farms. Skipping sample.")
                        print(f"Power data shape: {power_data_all_farms.shape}")
                        continue


                    power_da = xr.DataArray(
                        power_data_farm0,
                        coords={'time': time_coords},
                        dims=['time']
                    )

                    sample_ds = xr.Dataset(
                        {'Power': power_da},
                        coords={'sample': sample_num}
                    )
                    all_sample_datasets.append(sample_ds)

                except Exception as e:
                    print(f"Error processing group {group_name}: {e}")
                    print(f"Available keys in {group_name} during error: {list(sample_group.keys())}")
                    continue

            else:
                print(f"Warning: Group {group_name} not found in H5 file. This should not happen if iterating from keys.")

    if not all_sample_datasets:
        raise ValueError("No valid sample data found in the H5 file after filtering.")

    concatenated_ds = xr.concat(all_sample_datasets, dim='sample')

    return concatenated_ds

# --- Placeholder for calculate_revenue function ---
def calculate_revenue(power_data, discount_rate=0.03):
    """
    Calculates first-year and total (10-year) revenue from power time series.
    Assumes power_data is in GWh/year and is (sample, time).
    Time dimension is assumed to be daily data for 3653 days (approx 10 years).
    """
    days_per_year = 365
    num_years = 10 

    price_per_GWh = 50.0 # Example price, adjust as needed

    daily_revenue = power_data * price_per_GWh

    first_year_daily_revenue = daily_revenue[:, :days_per_year]
    first_year_revenue = first_year_daily_revenue.sum(dim='time')

    total_revenue_discounted = xr.zeros_like(first_year_revenue, dtype=float)

    for i in range(num_years):
        start_day = i * days_per_year
        end_day = (i + 1) * days_per_year
        
        if start_day >= daily_revenue.shape[1]:
            break
        
        current_year_daily_revenue = daily_revenue[:, start_day:end_day]
        current_year_total_revenue = current_year_daily_revenue.sum(dim='time')
        
        discount_factor = 1.0 / ((1.0 + discount_rate) ** i)
        total_revenue_discounted += current_year_total_revenue * discount_factor
    
    revenues_ds = xr.Dataset(
        {
            'first_year_revenue': first_year_revenue,
            'total_revenue': total_revenue_discounted
        },
        coords={'sample': power_data['sample']}
    )

    return revenues_ds


# --- Main preprocessing script logic ---
if __name__ == "__main__":
    # --- Configuration ---
    # Adjust this variable for each run to process different data sets
    # e.g., "uniform_results", "exponential_1yr_results", "exponential_2yr_results"
    RESULTS_SUBDIR = "exponential_1yr_results" # Set this for the current run

    H5_RESULTS_FILE = os.path.join(f'./{RESULTS_SUBDIR}/concatenated/', 'all_samples.h5')
    
    BASE_SAMPLE_FILENAME = RESULTS_SUBDIR.replace('_results', '_samples') 
    ORIGINAL_SAMPLES_FILE = f'./{BASE_SAMPLE_FILENAME}.nc' 

    # --- IMPORTANT: Standardized output directory for XGBoost script ---
    OUTPUT_DIR = './preprocessed_data/' 
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 1. Load Power time series from H5 file ---
    print(f"Loading Power data from {H5_RESULTS_FILE}...")
    try:
        combined_sim_results = load_data_from_h5_by_sample(H5_RESULTS_FILE)
        print("Power data loaded successfully.")
        print(f"Loaded {combined_sim_results.sizes['sample']} samples.")
        print(combined_sim_results)
    except Exception as e:
        print(f"Failed to load Power data from H5: {e}")
        exit()


    # --- 2. Load input features (X) from original sample file ---
    print(f"Loading input features from {ORIGINAL_SAMPLES_FILE}...")
    try:
        original_features_ds = xr.load_dataset(ORIGINAL_SAMPLES_FILE)
        print("Input features loaded successfully.")
    except Exception as e:
        print(f"Failed to load original features from {ORIGINAL_SAMPLES_FILE}: {e}")
        exit()

    features_to_extract = [
        'rated_power', 'rotor_diameter', 'hub_height', 'construction_day',
        'ss_seed'
    ]

    samples_in_h5 = combined_sim_results['sample'].values
    
    aligned_features_ds = original_features_ds[features_to_extract].sel(sample=samples_in_h5)
    
    if 'farm' in aligned_features_ds.dims and aligned_features_ds.sizes['farm'] > 0:
        aligned_features_ds = aligned_features_ds.isel(farm=0).drop_vars('farm')
    else:
        print("Warning: 'farm' dimension not found in original features or is empty. Proceeding without farm selection for features.")

    final_preprocessed_ds = xr.merge([combined_sim_results, aligned_features_ds])
    print("\nFinal preprocessed dataset merged:")
    print(final_preprocessed_ds)


    # --- 3. Calculate Revenue ---
    print("\nCalculating revenues...")
    revenues_ds = calculate_revenue(final_preprocessed_ds['Power'])
    print("Revenues calculated successfully.")
    print(revenues_ds)


    # --- 4. Prepare data for XGBoost (X and y) and save to .npz ---
    print("\nPreparing data for XGBoost and saving to .npz files...")

    X_data = final_preprocessed_ds[features_to_extract].to_array(dim='feature').T.to_numpy()

    y_1yr = revenues_ds['first_year_revenue'].to_numpy()
    y_total = revenues_ds['total_revenue'].to_numpy()

    # --- Split data and save to .npz based on RESULTS_SUBDIR ---
    # This section needs to be run multiple times, once for each RESULTS_SUBDIR.
    # The 'train.npz' and 'valid.npz' should ideally be generated from one main dataset
    # (e.g., exponential_1yr_results or a combined training set).
    # Then, subsequent runs for other RESULTS_SUBDIRs generate the specific test sets.

    if RESULTS_SUBDIR == "exponential_1yr_results":
        # This run can be used to generate the main train/validation/test split
        X_train_val, X_test, y_1yr_train_val, y_1yr_test, y_total_train_val, y_total_test = \
            train_test_split(X_data, y_1yr, y_total, test_size=0.2, random_state=42)

        X_train, X_val, y_1yr_train, y_1yr_val, y_total_train, y_total_val = \
            train_test_split(X_train_val, y_1yr_train_val, y_total_train_val, test_size=0.25, random_state=42)

        np.savez(os.path.join(OUTPUT_DIR, 'train.npz'), X=X_train, y_revenue_1yr_0_1=y_1yr_train, y_revenue_10yr_0_1=y_total_train)
        np.savez(os.path.join(OUTPUT_DIR, 'valid.npz'), X=X_val, y_revenue_1yr_0_1=y_1yr_val, y_revenue_10yr_0_1=y_total_val)
        
        # Save a test file specific to this result subdir
        np.savez(os.path.join(OUTPUT_DIR, 'test_exp1.npz'), X=X_test, y_revenue_1yr_0_1=y_1yr_test, y_revenue_10yr_0_1=y_total_test)

    elif RESULTS_SUBDIR == "uniform_results":
        # All data from this run becomes the 'test_uniform' set
        np.savez(os.path.join(OUTPUT_DIR, 'test_uniform.npz'), X=X_data, y_revenue_1yr_0_1=y_1yr, y_revenue_10yr_0_1=y_total)

    elif RESULTS_SUBDIR == "exponential_2yr_results":
        # All data from this run becomes the 'test_exp2' set
        np.savez(os.path.join(OUTPUT_DIR, 'test_exp2.npz'), X=X_data, y_revenue_1yr_0_1=y_1yr, y_revenue_10yr_0_1=y_total)

    else:
        print(f"Warning: No specific saving logic for RESULTS_SUBDIR: {RESULTS_SUBDIR}. Saving as 'test_all_samples.npz'.")
        np.savez(os.path.join(OUTPUT_DIR, 'test_all_samples.npz'), X=X_data, y_revenue_1yr_0_1=y_1yr, y_revenue_10yr_0_1=y_total)

    print(f"Processed data saved to {OUTPUT_DIR}")
    print("Preprocessing complete!")
