#!/usr/bin/env python3

import os
import glob
import sys
import time
from collections import defaultdict
import xarray as xr
import numpy as np
import pandas as pd

# --- Configuration ---
BASE_PROJECT_DIR = "/work/users/juqu/Neighboring-Wind-Farms-Out-Of-Distribution-Surrogate-Modeling"

# Input features (X values) configuration
INPUT_FEATURES_PER_FARM_BASENAMES = ['rated_power', 'construction_day', 'ss_seed'] 
N_FARMS = 9
INPUT_FEATURE_COLUMN_NAMES = []
for i in range(N_FARMS):
    for basename in INPUT_FEATURES_PER_FARM_BASENAMES:
        INPUT_FEATURE_COLUMN_NAMES.append(f"{basename}_{i}")


DISTRIBUTIONS_CONFIG = [
    {
        'id': 'uniform',
        'results_subdir_name': 'uniform_results', 
        'samples_nc_file': 'uniform_samples.nc', 
        'total_samples': 5000,
        'files_per_sample': 10
    },
    {
        'id': 'exp1yr',
        'results_subdir_name': 'exponential_1yr_results',
        'samples_nc_file': 'exponential_1yr_samples.nc', 
        'total_samples': 5000,
        'files_per_sample': 10
    },
    {
        'id': 'exp2yr',
        'results_subdir_name': 'exponential_2yr_results',
        'samples_nc_file': 'exponential_2yr_samples.nc', 
        'total_samples': 5000,
        'files_per_sample': 10
    }
]

# Output (Y values) configuration
POWER_VARIABLE_IN_NC = "Power" 
WT_COORDINATE_UPPER_BOUND = 65 
N_FIRST_TIMESTEPS_FOR_MEAN = 360
# --- End Configuration ---

def find_progress(current_result_dir, current_total_samples, current_files_per_sample, dist_id_for_log):
    print(f"DEBUG [{dist_id_for_log}]: Scanning directory: {current_result_dir}")
    if not os.path.isdir(current_result_dir):
        print(f"ERROR [{dist_id_for_log}]: Result directory does not exist: {current_result_dir}")
        return {"completed": [], "incomplete": [], "missing": list(range(1, current_total_samples + 1)),
                "total_completed": 0, "total_incomplete": 0, "total_missing": current_total_samples,
                "percent_complete": 0}
    all_files = glob.glob(os.path.join(current_result_dir, "res_*_*.nc"))
    sample_counts = defaultdict(int)
    for f_path in all_files:
        basename = os.path.basename(f_path)
        parts = basename.split('_')
        if len(parts) >= 3 and parts[0] == 'res' and parts[1].isdigit():
            try: sample_counts[int(parts[1])] += 1
            except ValueError: pass
    completed_ids, incomplete_info = [], []
    all_found_ids = set(sample_counts.keys())
    # expected_ids are 1-based if current_total_samples refers to 1-based sample IDs
    expected_ids = set(range(1, current_total_samples + 1)) 
    completion_threshold = current_files_per_sample - 2
    if completion_threshold < 1: completion_threshold = 1
    for sample_id in sorted(list(all_found_ids)):
        if sample_counts[sample_id] >= completion_threshold: completed_ids.append(sample_id)
        else: incomplete_info.append((sample_id, sample_counts[sample_id]))
    missing_ids = sorted(list(expected_ids - all_found_ids))
    try:
        with open(f'missing_cases_{dist_id_for_log}.dat', 'w') as outf: outf.write(' '.join(map(str, missing_ids)))
    except IOError as e: print(f"ERROR [{dist_id_for_log}]: Could not write missing_cases_{dist_id_for_log}.dat: {e}")
    return {"completed": sorted(completed_ids), "incomplete": sorted(incomplete_info), "missing": missing_ids,
            "total_completed": len(completed_ids), "total_incomplete": len(incomplete_info),
            "total_missing": len(missing_ids), "percent_complete": (len(completed_ids) / current_total_samples * 100) if current_total_samples > 0 else 0}

def print_progress_report(progress, current_total_samples, current_files_per_sample, dist_id_for_report):
    completion_threshold = current_files_per_sample - 2
    if completion_threshold < 1: completion_threshold = 1
    print("\n" + "=" * 50 + f"\nPROGRESS REPORT - {dist_id_for_report} - {time.strftime('%Y-%m-%d %H:%M:%S')}\n" + "=" * 50)
    print(f"Target Samples: {current_total_samples}\nFiles per Sample Threshold: ~{completion_threshold}")
    print("-" * 50 + f"\nCompleted & Processable: {progress['total_completed']} ({progress['percent_complete']:.1f}%)\n"
            f"Incomplete: {progress['total_incomplete']}\nMissing: {progress['total_missing']}\n" + "=" * 50)

def delete_incomplete_files(incomplete_list, current_result_dir_to_clean, dist_id_for_log):
    deleted_files_count = 0; deleted_samples_count = 0
    print(f"\n[{dist_id_for_log}] Deleting files for {len(incomplete_list)} incomplete samples in: {current_result_dir_to_clean}")
    for sample_id, file_count in incomplete_list:
        file_pattern = os.path.join(current_result_dir_to_clean, f"res_{sample_id}_*.nc")
        files_to_delete = glob.glob(file_pattern)
        if not files_to_delete: continue
        deleted_samples_count += 1
        for filepath in files_to_delete:
            try:
                os.remove(filepath)
                deleted_files_count += 1
            except OSError as e: print(f"     [{dist_id_for_log}] ERROR deleting {os.path.basename(filepath)}: {e}")
    print(f"\n[{dist_id_for_log}] Deletion Summary: Attempted for {deleted_samples_count} incomplete samples. Deleted {deleted_files_count} files.")

def calculate_io_and_power_means(completed_ids_to_process, current_result_dir_nc_files, original_samples_dataset,
                                 output_csv_file_for_dist, power_var_name_in_nc, wt_upper_bound,
                                 n_first_ts_for_mean, dist_id_for_log, original_samples_nc_filepath_for_log=""):
    print(f"\n[{dist_id_for_log}] Starting X features and Y power mean calculation for {len(completed_ids_to_process)} samples.")
    print(f"[{dist_id_for_log}] Output CSV target: {output_csv_file_for_dist}")

    if not completed_ids_to_process:
        print(f"[{dist_id_for_log}] No completed samples to process.")
        return

    results_list = []
    processed_sample_count = 0
    total_to_process = len(completed_ids_to_process)

    for i, sample_id_from_res_files in enumerate(completed_ids_to_process): # This ID is from res_K_*.nc filenames
        print(f"  [{dist_id_for_log}] Processing sample {i+1}/{total_to_process} (File ID: {sample_id_from_res_files})")
        
        # --- MODIFIED SECTION FOR X-FEATURE EXTRACTION ---
        # The ID from res_K_*.nc (sample_id_from_res_files) corresponds to the 
        # 0-indexed 'sample' coordinate that was used by run_cluster.py K 
        # when it did ds.sel(sample=K) to get its inputs.
        # Therefore, use this ID directly to select X-features from original_samples_dataset.
        # This assumes original_samples_dataset has a 0-indexed 'sample' coordinate.
        x_feature_sample_index = sample_id_from_res_files 
        # --- END MODIFIED SECTION ---

        current_sample_x_values = {}
        try:
            sample_input_features_ds = original_samples_dataset.sel(sample=x_feature_sample_index)
            for farm_idx in range(N_FARMS):
                for basename_idx, feat_basename in enumerate(INPUT_FEATURES_PER_FARM_BASENAMES):
                    col_name = INPUT_FEATURE_COLUMN_NAMES[farm_idx * len(INPUT_FEATURES_PER_FARM_BASENAMES) + basename_idx]
                    current_sample_x_values[col_name] = sample_input_features_ds[feat_basename].isel(farm=farm_idx).item()
        except KeyError: # Or other relevant xarray selection errors
            print(f"     WARNING [{dist_id_for_log}]: X-feature sample index {x_feature_sample_index} (derived from res file ID {sample_id_from_res_files}) not found in original samples file '{original_samples_nc_filepath_for_log}'. Skipping this sample.")
            continue 
        except Exception as e_xfeat:
            print(f"     ERROR [{dist_id_for_log}]: Could not extract X features for res file ID {sample_id_from_res_files} (using X-feature index {x_feature_sample_index}). Error: {e_xfeat}. Skipping.")
            continue

        sample_file_pattern = os.path.join(current_result_dir_nc_files, f"res_{sample_id_from_res_files}_*.nc")
        sample_files = sorted(glob.glob(sample_file_pattern))
        if not sample_files:
            print(f"     WARNING [{dist_id_for_log}]: Res file ID {sample_id_from_res_files}: No .nc files found for Y processing. Skipping.")
            continue

        datasets_for_sample = []
        sample_combined_ds = None
        mean_p_first_n_val = np.nan
        mean_p_all_val = np.nan

        try:
            for f_path in sample_files:
                try: datasets_for_sample.append(xr.open_dataset(f_path))
                except Exception as e_open: print(f"     ERROR [{dist_id_for_log}]: Could not open {f_path} for res file ID {sample_id_from_res_files}: {e_open}.")
            
            if not datasets_for_sample:
                print(f"     WARNING [{dist_id_for_log}]: No files loaded for Y processing for res file ID {sample_id_from_res_files}. Using NaN for Y values.")
            else:
                sample_combined_ds = xr.concat(datasets_for_sample, dim="time", join="outer", fill_value=np.nan)
                if power_var_name_in_nc not in sample_combined_ds:
                    print(f"     ERROR [{dist_id_for_log}]: Y var '{power_var_name_in_nc}' not found for res file ID {sample_id_from_res_files}. Vars: {list(sample_combined_ds.data_vars)}. Using NaN for Y values.")
                else:
                    power_data_all_turbines = sample_combined_ds[power_var_name_in_nc]
                    if 'wt' not in power_data_all_turbines.dims:
                        print(f"     ERROR [{dist_id_for_log}]: 'wt' dim not in '{power_var_name_in_nc}' for res file ID {sample_id_from_res_files}. Using NaN.")
                    else:
                        power_selected_turbines = power_data_all_turbines.sel(wt=slice(0, wt_upper_bound))
                        if power_selected_turbines.wt.size > 0:
                            total_power_ts = power_selected_turbines.sum(dim='wt', skipna=True).data
                            if total_power_ts.ndim == 1 and total_power_ts.size > 0:
                                actual_n_first = min(n_first_ts_for_mean, total_power_ts.size)
                                if actual_n_first > 0: mean_p_first_n_val = np.nanmean(total_power_ts[:actual_n_first])
                                mean_p_all_val = np.nanmean(total_power_ts)
                            else: print(f"     WARNING [{dist_id_for_log}]: Summed power for res file ID {sample_id_from_res_files} not 1D or empty. Using NaN.")
                        else: 
                            max_wt_coord = power_data_all_turbines.wt.max().item() if power_data_all_turbines.wt.size > 0 else 'N/A (no turbines)'
                            print(f"     WARNING [{dist_id_for_log}]: No turbines selected with wt <= {wt_upper_bound} for res file ID {sample_id_from_res_files}. Max wt coord found: {max_wt_coord}. Using NaN.")
        except Exception as e_ycalc:
            print(f"     ERROR [{dist_id_for_log}]: Failed during Y calculation for res file ID {sample_id_from_res_files}: {e_ycalc}. Using NaN for Y values.")
        finally:
            for ds_indiv in datasets_for_sample: ds_indiv.close()
            if sample_combined_ds: sample_combined_ds.close()

        # The sample_id in the CSV will be the ID from the res_K_*.nc filenames
        current_sample_result = {'sample_id': sample_id_from_res_files} 
        current_sample_result.update(current_sample_x_values)
        current_sample_result[f'mean_power_first_{n_first_ts_for_mean}'] = mean_p_first_n_val
        current_sample_result['mean_power_all_time'] = mean_p_all_val
        results_list.append(current_sample_result)
        processed_sample_count +=1

    if not results_list:
        print(f"\n[{dist_id_for_log}] No samples were successfully processed to include both X and Y features.")
        return

    results_df = pd.DataFrame(results_list)
    ordered_columns = ['sample_id'] + INPUT_FEATURE_COLUMN_NAMES + [f'mean_power_first_{n_first_ts_for_mean}', 'mean_power_all_time']
    
    final_columns = []
    for col in ordered_columns:
        if col in results_df.columns:
            final_columns.append(col)
        else: # Should ideally not happen if X features are always extracted or NaNs are filled
            print(f"  DEV WARNING [{dist_id_for_log}]: Expected column '{col}' not found in results_df. It will be missing in CSV.")
            
    results_df = results_df[final_columns] # Use only columns that actually exist
    results_df.sort_values(by='sample_id', inplace=True)
    try:
        os.makedirs(os.path.dirname(output_csv_file_for_dist), exist_ok=True)
        results_df.to_csv(output_csv_file_for_dist, index=False, float_format='%.6g')
        print(f"\n[{dist_id_for_log}] Calculated X & Y for {processed_sample_count} samples. Saved to: {output_csv_file_for_dist}")
    except IOError as e:
        print(f"\nERROR [{dist_id_for_log}]: Could not write CSV to {output_csv_file_for_dist}: {e}")

def main():
    print(f"--- Starting Main Processing Script (X and Y) ---")
    current_base_project_dir = BASE_PROJECT_DIR
    AUTO_DELETE_INCOMPLETE = False 
    AUTO_PROCESS_XY = True     
    SAMPLES_LIMIT_FOR_TESTING = 5000

    for dist_config in DISTRIBUTIONS_CONFIG:
        dist_id = dist_config['id']
        dist_subdir = dist_config['results_subdir_name']
        samples_nc_filename = dist_config['samples_nc_file'] 
        current_total_samples = dist_config['total_samples']
        current_files_per_sample = dist_config['files_per_sample']

        print(f"\n\n=== PROCESSING DISTRIBUTION: {dist_id.upper()} ===")
        
        current_result_dir_for_y = os.path.join(current_base_project_dir, dist_subdir, "wake")
        original_samples_nc_path = os.path.join(current_base_project_dir, samples_nc_filename) 

        current_output_dir_xy = os.path.join(current_base_project_dir, dist_subdir, "processed_xy_data")
        current_final_output_csv = os.path.join(current_output_dir_xy, f"{dist_id}_xy_data.csv") 

        if not os.path.isdir(current_result_dir_for_y):
            print(f"ERROR: Directory for Y data (res_*.nc) not found for {dist_id}: {current_result_dir_for_y}. Skipping.")
            continue
        if not os.path.exists(original_samples_nc_path):
            print(f"ERROR: File for X data ('{samples_nc_filename}') not found at '{original_samples_nc_path}' for {dist_id}. Skipping.")
            continue
        
        original_samples_ds = None 
        try:
            original_samples_ds = xr.load_dataset(original_samples_nc_path)
            # Basic check for 'sample' coordinate
            if 'sample' not in original_samples_ds.coords:
                 print(f"  ERROR [{dist_id}]: 'sample' coordinate not found in {samples_nc_filename}. Verify file structure.")
                 original_samples_ds.close() # Close if problematic
                 continue # Skip this distribution
            # You might add more checks here if needed, e.g., if sample is 0-indexed
        except Exception as e_load_x:
            print(f"ERROR [{dist_id}]: Could not load original samples file {original_samples_nc_path} for X features. Error: {e_load_x}. Skipping.")
            if original_samples_ds: original_samples_ds.close()
            continue

        progress_data = find_progress(current_result_dir_for_y, current_total_samples, current_files_per_sample, dist_id)
        print_progress_report(progress_data, current_total_samples, current_files_per_sample, dist_id)

        if AUTO_DELETE_INCOMPLETE and progress_data['incomplete']:
            delete_incomplete_files(progress_data['incomplete'], current_result_dir_for_y, dist_id)
        elif progress_data['incomplete']:
            print(f"\n[{dist_id}] Skipping deletion of incomplete files (AUTO_DELETE_INCOMPLETE is False).")

        if AUTO_PROCESS_XY and progress_data['completed']:
            completed_ids_for_this_dist = progress_data['completed']
            ids_to_process = completed_ids_for_this_dist
            
            if SAMPLES_LIMIT_FOR_TESTING > 0 and len(completed_ids_for_this_dist) > SAMPLES_LIMIT_FOR_TESTING:
                print(f"\n[{dist_id}] NOTE: Limiting processing to the first {SAMPLES_LIMIT_FOR_TESTING} completed samples for testing.")
                ids_to_process = completed_ids_for_this_dist[:SAMPLES_LIMIT_FOR_TESTING]
            elif len(completed_ids_for_this_dist) > 0:
                print(f"\n[{dist_id}] Processing all {len(completed_ids_for_this_dist)} completed samples.")
            
            if ids_to_process:
                calculate_io_and_power_means(
                    ids_to_process, current_result_dir_for_y, original_samples_ds, 
                    current_final_output_csv, POWER_VARIABLE_IN_NC, 
                    WT_COORDINATE_UPPER_BOUND, N_FIRST_TIMESTEPS_FOR_MEAN, dist_id,
                    original_samples_nc_filepath_for_log=original_samples_nc_path # Pass for logging
                )
            else:
                print(f"[{dist_id}] No completed samples available to process for {dist_id}.")
        elif AUTO_PROCESS_XY: 
            print(f"[{dist_id}] No completed samples found to process.")
        else: 
            print(f"[{dist_id}] X-Y data processing skipped (AUTO_PROCESS_XY is False).")
        
        if original_samples_ds: original_samples_ds.close() 
            
    print("\n--- Main Processing Script Finished ---")

if __name__ == "__main__":
    main()
