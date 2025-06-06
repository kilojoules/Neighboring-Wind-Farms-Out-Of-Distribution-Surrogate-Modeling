import xarray as xr
import glob
import os
import re
from collections import defaultdict
import matplotlib.pyplot as plt

def parse_filename_details(filepath):
    """Extracts sample_id and chunk_id from a filename."""
    match = re.search(r'res_(\d+)_(\d+)\.nc$', os.path.basename(filepath))
    if match:
        return {
            'filepath': filepath,
            'sample_id': int(match.group(1)),
            'chunk_id': int(match.group(2))
        }
    return None

def load_and_process_sample_power(data_dir, sample_ids_to_load, num_turbines_to_include, rolling_window_days):
    """
    Loads NetCDF files for specified sample_ids, sums over turbines, 
    applies a rolling mean, and then concatenates the results.
    Returns a DataArray of smoothed power (sample_id, time).
    """
    file_pattern = os.path.join(data_dir, "res_*.nc")
    all_filepaths = glob.glob(file_pattern)

    if not all_filepaths:
        print(f"No files found matching pattern: {file_pattern}")
        return None

    parsed_files_info = [parse_filename_details(fp) for fp in all_filepaths if parse_filename_details(fp)]
    if not parsed_files_info:
        print(f"No files could be parsed in: {data_dir}")
        return None

    files_by_sample_id_map = defaultdict(list)
    for info in parsed_files_info:
        files_by_sample_id_map[info['sample_id']].append(info)

    processed_sample_power_list = []
    actual_turbines_summed_for_sample = {} # To store for logging/title

    for target_s_id in sample_ids_to_load:
        if target_s_id in files_by_sample_id_map:
            sample_file_chunks = sorted(files_by_sample_id_map[target_s_id], key=lambda x: x['chunk_id'])
            chunk_filepaths = [info['filepath'] for info in sample_file_chunks]

            if not chunk_filepaths:
                continue
            
            print(f"Processing sample_id: {target_s_id}...")
            # 1. Load daily data for the current sample
            current_sample_ds = xr.open_mfdataset(
                chunk_filepaths, combine='nested', concat_dim='time'
            )
            daily_power_all_wt_single_sample = current_sample_ds['Power'] # DataArray(wt, time)

            # 2. Sum over specified turbines for this sample
            daily_site_power_single_sample = None
            turbines_summed_this_sample = 0
            if 'wt' in daily_power_all_wt_single_sample.dims:
                total_turbines_in_sample = daily_power_all_wt_single_sample.sizes['wt']
                if total_turbines_in_sample > 0:
                    turbines_summed_this_sample = min(num_turbines_to_include, total_turbines_in_sample)
                    power_subset_turbines = daily_power_all_wt_single_sample.isel(wt=slice(0, turbines_summed_this_sample))
                    daily_site_power_single_sample = power_subset_turbines.sum(dim='wt', skipna=True) # DataArray(time)
                else:
                    print(f"  Warning: Sample {target_s_id} 'wt' dimension has size 0.")
                    # Create an empty/zero array for this sample's power to maintain structure if needed
                    # This depends on how you want to handle missing 'wt' data for a sample
                    daily_site_power_single_sample = xr.DataArray(
                        [0] * daily_power_all_wt_single_sample.sizes.get('time',0), 
                        dims=['time'], 
                        coords={'time': daily_power_all_wt_single_sample.coords.get('time', [])}
                    )

            elif daily_power_all_wt_single_sample.size > 0 : # No 'wt' dim, but data exists (assume it's already site power)
                print(f"  Warning: Sample {target_s_id} has no 'wt' dimension. Using raw power data.")
                daily_site_power_single_sample = daily_power_all_wt_single_sample
            
            actual_turbines_summed_for_sample[target_s_id] = turbines_summed_this_sample

            if daily_site_power_single_sample is None or daily_site_power_single_sample.size == 0:
                print(f"  No site power data for sample {target_s_id} after turbine summation.")
                continue

            # 3. Calculate rolling average for this sample
            smoothed_single_sample_power = daily_site_power_single_sample.rolling(
                time=rolling_window_days, center=True, min_periods=1
            ).mean(skipna=True) # DataArray(time)
            
            # Add sample_id coordinate and expand dimension for concatenation
            smoothed_single_sample_power = smoothed_single_sample_power.assign_coords(sample_id=target_s_id)
            smoothed_single_sample_power = smoothed_single_sample_power.expand_dims('sample_id') # Now (sample_id:1, time)
            
            processed_sample_power_list.append(smoothed_single_sample_power)
        else:
            print(f"Data for requested sample_id: {target_s_id} not found in the directory.")

    if not processed_sample_power_list:
        print("No data was processed for any of the specified sample IDs.")
        return None, {}

    # 4. Concatenate all processed (smoothed) samples
    print("Concatenating processed samples...")
    final_smoothed_power = xr.concat(processed_sample_power_list, dim='sample_id')
    
    return final_smoothed_power, actual_turbines_summed_for_sample


if __name__ == '__main__':
    data_directory = './uniform_results/wake/'
    first_sample_to_plot = 1
    number_of_samples_to_plot = 100
    target_sample_ids = range(first_sample_to_plot, first_sample_to_plot + number_of_samples_to_plot)
    num_turbines_to_include = 65
    rolling_window_days = 360

    print(f"Attempting to load and process power data for sample IDs: {list(target_sample_ids)}")
    
    # The function now returns the already smoothed data and info about turbines summed
    smoothed_site_power, turbines_info = load_and_process_sample_power(
        data_directory, 
        target_sample_ids, 
        num_turbines_to_include, 
        rolling_window_days
    )

    if smoothed_site_power is None or smoothed_site_power.sample_id.size == 0:
        print("Failed to load or process data for the specified samples. Exiting.")
    else:
        print(f"Data processed for {smoothed_site_power.sample_id.size} sample(s): {smoothed_site_power.sample_id.values}. Preparing plot...")
        
        plt.figure(figsize=(14, 7))
        
        plot_data_for_hue = smoothed_site_power.copy()
        if plot_data_for_hue.sample_id.dtype != 'object' and plot_data_for_hue.sample_id.dtype != '<U':
             plot_data_for_hue['sample_id'] = plot_data_for_hue['sample_id'].astype(str)

        plot_data_for_hue.plot.line(x='time', hue='sample_id', add_legend=False)
        
        loaded_ids_str = ', '.join(plot_data_for_hue.sample_id.values)
        # Note: turbines_info might vary per sample if some have fewer than num_turbines_to_include.
        # For simplicity, we'll just state the general aim for the title.
        # A more complex title could list per-sample turbine counts if they vary significantly.
        turbine_sum_info = f" (Sum over up to first {num_turbines_to_include} Turbines)" if num_turbines_to_include > 0 else ""
        
        plt.title(f'{rolling_window_days}-Day Rolling Avg Power: Samples {loaded_ids_str}{turbine_sum_info}')
        plt.xlabel('Time (Days)')
        plt.ylabel(f'{rolling_window_days}-Day Rolling Avg Power{turbine_sum_info}')
        plt.grid(True)
        plt.tight_layout()
        
        print("Displaying plot...")
        plt.savefig('powers')
        plt.clf()
