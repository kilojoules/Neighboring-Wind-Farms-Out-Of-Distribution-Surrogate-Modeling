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

DISTRIBUTIONS_CONFIG = [
    {
        'id': 'uniform',
        'results_subdir_name': 'uniform_results',
        'total_samples': 5000,  # Corrected from 7000
        'files_per_sample': 10
    },
    {
        'id': 'exp1yr',
        'results_subdir_name': 'exponential_1yr_results',
        'total_samples': 5000,
        'files_per_sample': 10
    },
    {
        'id': 'exp2yr',
        'results_subdir_name': 'exponential_2yr_results',
        'total_samples': 5000,
        'files_per_sample': 10
    }
]

POWER_VARIABLE_IN_NC = "Power"
WT_COORDINATE_UPPER_BOUND = 65
N_FIRST_TIMESTEPS_FOR_MEAN = 360
# --- End Configuration ---

def find_progress(current_result_dir, current_total_samples, current_files_per_sample, dist_id_for_log):
    print(f"DEBUG [{dist_id_for_log}]: Scanning directory: {current_result_dir}")
    if not os.path.isdir(current_result_dir):
        print(f"ERROR [{dist_id_for_log}]: Result directory does not exist: {current_result_dir}")
        return {
            "completed": [], "incomplete": [], "missing": list(range(1, current_total_samples + 1)),
            "total_completed": 0, "total_incomplete": 0, "total_missing": current_total_samples,
            "percent_complete": 0
        }
        
    all_files = glob.glob(os.path.join(current_result_dir, "res_*_*.nc"))
    sample_counts = defaultdict(int)
    for f in all_files:
        basename = os.path.basename(f)
        parts = basename.split('_')
        if len(parts) >= 3 and parts[0] == 'res' and parts[1].isdigit():
            try:
                sample_id = int(parts[1])
                sample_counts[sample_id] += 1
            except ValueError:
                pass 
    completed_ids = []
    incomplete_info = []
    all_found_ids = set(sample_counts.keys())
    expected_ids = set(range(1, current_total_samples + 1))
    completion_threshold = current_files_per_sample - 2
    if completion_threshold < 1: completion_threshold = 1

    for sample_id in sorted(list(all_found_ids)):
        count = sample_counts[sample_id]
        if count >= completion_threshold: 
            completed_ids.append(sample_id)
        else:
            incomplete_info.append((sample_id, count))
            
    missing_ids = sorted(list(expected_ids - all_found_ids))
    missing_cases_file = f'missing_cases_{dist_id_for_log}.dat'
    try:
        with open(missing_cases_file, 'w') as outf:
            outf.write(' '.join([str(s) for s in missing_ids]))
    except IOError as e:
        print(f"ERROR [{dist_id_for_log}]: Could not write missing cases file {missing_cases_file}: {e}")
    total_completed = len(completed_ids)
    percent_complete = (total_completed / current_total_samples) * 100 if current_total_samples > 0 else 0
    return {
        "completed": sorted(completed_ids), "incomplete": sorted(incomplete_info), "missing": missing_ids,
        "total_completed": total_completed, "total_incomplete": len(incomplete_info),
        "total_missing": len(missing_ids), "percent_complete": percent_complete
    }

def print_progress_report(progress, current_total_samples, current_files_per_sample, dist_id_for_report):
    completion_threshold = current_files_per_sample - 2
    if completion_threshold < 1: completion_threshold = 1
    print("\n" + "=" * 50)
    print(f"PROGRESS REPORT - {dist_id_for_report} - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    print(f"Target Samples: {current_total_samples}")
    print(f"Files per Sample Threshold (approx): {completion_threshold} (for considering a sample complete enough)")
    print("-" * 50)
    print(f"Completed & Processable (initially found): {progress['total_completed']} / {current_total_samples} samples ({progress['percent_complete']:.1f}%)")
    print(f"Incomplete (fewer than {completion_threshold} files): {progress['total_incomplete']} samples")
    print(f"Missing (no files found): {progress['total_missing']} samples")
    print("=" * 50)

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
            except OSError as e: print(f"    [{dist_id_for_log}] ERROR deleting {os.path.basename(filepath)}: {e}")
    print(f"\n[{dist_id_for_log}] Deletion Summary: Attempted for {deleted_samples_count} incomplete samples. Deleted {deleted_files_count} files.")

def calculate_and_save_power_means(completed_ids_to_process, current_result_dir_nc_files, output_csv_file_for_dist, 
                                   power_var_name_in_nc, wt_upper_bound, n_first_ts_for_mean, dist_id_for_log):
    # The completed_ids_to_process list is now potentially sliced to the first 100
    
    print(f"\n[{dist_id_for_log}] Starting power mean calculation for {len(completed_ids_to_process)} samples.")
    print(f"[{dist_id_for_log}] Output CSV target: {output_csv_file_for_dist}")
    print(f"[{dist_id_for_log}] Power variable: '{power_var_name_in_nc}', summing wt up to {wt_upper_bound}, first N_ts for mean: {n_first_ts_for_mean}")

    if not completed_ids_to_process:
        print(f"[{dist_id_for_log}] No completed samples to process.")
        return

    power_means_results = []
    processed_sample_count = 0
    total_to_process = len(completed_ids_to_process)

    for i, sample_id in enumerate(completed_ids_to_process):
        print(f"  [{dist_id_for_log}] Processing sample {i+1}/{total_to_process} (ID: {sample_id})")
        
        sample_file_pattern = os.path.join(current_result_dir_nc_files, f"res_{sample_id}_*.nc")
        sample_files = sorted(glob.glob(sample_file_pattern))
        if not sample_files:
             print(f"    WARNING [{dist_id_for_log}]: Sample {sample_id}: No .nc files found. Skipping.")
             continue

        datasets_for_sample = []
        sample_combined_ds = None
        try:
            for f_path in sample_files:
                try:
                    datasets_for_sample.append(xr.open_dataset(f_path))
                except Exception as e_open:
                    print(f"    ERROR [{dist_id_for_log}]: Could not open file {f_path} for sample {sample_id}. Error: {e_open}.")
            if not datasets_for_sample:
                print(f"    WARNING [{dist_id_for_log}]: No files successfully loaded for sample {sample_id}. Skipping.")
                continue
            sample_combined_ds = xr.concat(datasets_for_sample, dim="time", join="outer", fill_value=np.nan)
        except Exception as e_concat:
            print(f"    ERROR [{dist_id_for_log}]: Failed to load or concatenate files for sample {sample_id}: {e_concat}. Skipping.")
            continue
        finally:
            for ds_indiv in datasets_for_sample: ds_indiv.close()
        
        try:
            if power_var_name_in_nc not in sample_combined_ds:
                print(f"    ERROR [{dist_id_for_log}]: Power var '{power_var_name_in_nc}' not found for sample {sample_id}. Vars: {list(sample_combined_ds.data_vars)}. Skipping.")
                continue
            power_data_all_turbines = sample_combined_ds[power_var_name_in_nc]
            if 'wt' not in power_data_all_turbines.dims:
                print(f"    ERROR [{dist_id_for_log}]: 'wt' dim not in '{power_var_name_in_nc}' for sample {sample_id}. Skipping.")
                continue
            
            power_selected_turbines_per_wt = power_data_all_turbines.sel(wt=slice(0, wt_upper_bound))
            if power_selected_turbines_per_wt.wt.size == 0:
                # This might happen if wt_upper_bound is too low or wt coordinates are unexpected
                max_wt_coord = power_data_all_turbines.wt.max().item() if power_data_all_turbines.wt.size > 0 else 'N/A (no turbines)'
                print(f"    WARNING [{dist_id_for_log}]: No turbines selected with wt <= {wt_upper_bound} for sample {sample_id}. Max wt coord found: {max_wt_coord}. Skipping.")
                continue

            total_power_ts_da = power_selected_turbines_per_wt.sum(dim='wt', skipna=True)
            power_data_array = total_power_ts_da.data
            if power_data_array.ndim != 1 or power_data_array.size == 0:
                print(f"    ERROR/WARNING [{dist_id_for_log}]: Summed power for sample {sample_id} is not 1D or is empty. Shape: {power_data_array.shape}. Skipping.")
                continue

            actual_n_first = min(n_first_ts_for_mean, power_data_array.size)
            mean_p_first_n_val = np.nanmean(power_data_array[:actual_n_first]) if actual_n_first > 0 else np.nan
            mean_p_all_val = np.nanmean(power_data_array) if power_data_array.size > 0 else np.nan
            
            power_means_results.append({
                'sample_id': sample_id,
                f'mean_power_first_{n_first_ts_for_mean}': mean_p_first_n_val,
                'mean_power_all_time': mean_p_all_val
            })
            processed_sample_count +=1
        except Exception as e_calc:
            print(f"    ERROR [{dist_id_for_log}]: Failed during calculation for sample {sample_id}: {e_calc}. Skipping.")
        finally:
            if sample_combined_ds: sample_combined_ds.close()

    if not power_means_results:
        print(f"\n[{dist_id_for_log}] No samples processed for power means.")
        return

    results_df = pd.DataFrame(power_means_results)
    results_df.sort_values(by='sample_id', inplace=True)
    try:
        os.makedirs(os.path.dirname(output_csv_file_for_dist), exist_ok=True)
        results_df.to_csv(output_csv_file_for_dist, index=False, float_format='%.6g')
        print(f"\n[{dist_id_for_log}] Calculated power means for {processed_sample_count} samples. Saved to: {output_csv_file_for_dist}")
    except IOError as e:
        print(f"\nERROR [{dist_id_for_log}]: Could not write CSV to {output_csv_file_for_dist}: {e}")

def main():
    print(f"--- Starting Main Processing Script ---")
    current_base_project_dir = BASE_PROJECT_DIR

    AUTO_DELETE_INCOMPLETE = False 
    AUTO_PROCESS_MEANS = True    
    SAMPLES_LIMIT_FOR_TESTING = 100 # Limit processing to this many completed samples per distribution

    for dist_config in DISTRIBUTIONS_CONFIG:
        dist_id = dist_config['id']
        dist_subdir = dist_config['results_subdir_name']
        current_total_samples = dist_config['total_samples']
        current_files_per_sample = dist_config['files_per_sample']

        print(f"\n\n=== PROCESSING DISTRIBUTION: {dist_id.upper()} ===")
        current_result_dir = os.path.join(current_base_project_dir, dist_subdir, "wake")
        current_output_dir_power_means = os.path.join(current_base_project_dir, dist_subdir, "power_means_output")
        current_final_output_csv = os.path.join(current_output_dir_power_means, f"{dist_id}_power_means_first{N_FIRST_TIMESTEPS_FOR_MEAN}.csv")

        if not os.path.isdir(current_result_dir):
            print(f"ERROR: Result directory for .nc files not found for {dist_id}: {current_result_dir}. Skipping.")
            continue

        progress_data = find_progress(current_result_dir, current_total_samples, current_files_per_sample, dist_id)
        print_progress_report(progress_data, current_total_samples, current_files_per_sample, dist_id)

        if AUTO_DELETE_INCOMPLETE and progress_data['incomplete']:
            print(f"\n[{dist_id}] Automatically deleting files for {progress_data['total_incomplete']} incomplete samples.")
            delete_incomplete_files(progress_data['incomplete'], current_result_dir, dist_id)
        elif progress_data['incomplete']:
             print(f"\n[{dist_id}] Skipping deletion of incomplete files (AUTO_DELETE_INCOMPLETE is False).")

        if AUTO_PROCESS_MEANS and progress_data['completed']:
            completed_ids_for_this_dist = progress_data['completed']
            
            # Apply the sample limit for testing
            if SAMPLES_LIMIT_FOR_TESTING > 0 and len(completed_ids_for_this_dist) > SAMPLES_LIMIT_FOR_TESTING:
                print(f"\n[{dist_id}] NOTE: Limiting processing to the first {SAMPLES_LIMIT_FOR_TESTING} completed samples for testing (out of {len(completed_ids_for_this_dist)} found).")
                ids_to_process = completed_ids_for_this_dist[:SAMPLES_LIMIT_FOR_TESTING]
            else:
                ids_to_process = completed_ids_for_this_dist
            
            if ids_to_process:
                print(f"\n[{dist_id}] Proceeding to process {len(ids_to_process)} samples for power means.")
                calculate_and_save_power_means(
                    ids_to_process, current_result_dir, current_final_output_csv,
                    POWER_VARIABLE_IN_NC, WT_COORDINATE_UPPER_BOUND, N_FIRST_TIMESTEPS_FOR_MEAN,
                    dist_id
                )
            else:
                print(f"[{dist_id}] No completed samples available to process after applying limit (or none found).")

        elif AUTO_PROCESS_MEANS: 
            print(f"[{dist_id}] No completed samples found to process for power means.")
        else: 
             print(f"[{dist_id}] Power mean calculation skipped (AUTO_PROCESS_MEANS is False).")
             
    print("\n--- Main Processing Script Finished ---")

if __name__ == "__main__":
    main()
