import xarray as xr
import glob
import os
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# --- (Assume parse_filename_details and load_and_process_sample_power functions are defined as before) ---
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
        print(f"No files found matching pattern: {file_pattern} in directory: {data_dir}")
        return None, {}

    parsed_files_info = [parse_filename_details(fp) for fp in all_filepaths if parse_filename_details(fp)]
    if not parsed_files_info:
        print(f"No files could be parsed in: {data_dir}")
        return None, {}

    files_by_sample_id_map = defaultdict(list)
    for info in parsed_files_info:
        files_by_sample_id_map[info['sample_id']].append(info)

    processed_sample_power_list = []
    actual_turbines_summed_for_sample = {}

    for target_s_id in sample_ids_to_load:
        if target_s_id in files_by_sample_id_map:
            sample_file_chunks = sorted(files_by_sample_id_map[target_s_id], key=lambda x: x['chunk_id'])
            chunk_filepaths = [info['filepath'] for info in sample_file_chunks]

            if not chunk_filepaths:
                print(f"  No chunk files found for sample_id: {target_s_id} in {data_dir} (unexpected).")
                continue
            
            print(f"Processing sample_id: {target_s_id} from {data_dir}...")
            current_sample_ds = None 
            try:
                current_sample_ds = xr.open_mfdataset(
                    chunk_filepaths, combine='nested', concat_dim='time', engine='netcdf4'
                )
            except Exception as e:
                print(f"  Error loading files for sample {target_s_id} from {data_dir}: {e}")
                if current_sample_ds: current_sample_ds.close() 
                continue
            
            if 'Power' not in current_sample_ds:
                print(f"  'Power' variable not found in dataset for sample {target_s_id} from {data_dir}.")
                if current_sample_ds: current_sample_ds.close()
                continue
                
            daily_power_all_wt_single_sample = current_sample_ds['Power']

            daily_site_power_single_sample = None
            turbines_summed_this_sample = 0
            if 'wt' in daily_power_all_wt_single_sample.dims:
                total_turbines_in_sample = daily_power_all_wt_single_sample.sizes['wt']
                if total_turbines_in_sample > 0:
                    turbines_to_sum_count = min(num_turbines_to_include, total_turbines_in_sample)
                    if num_turbines_to_include > total_turbines_in_sample:
                         print(f"  Info: Sample {target_s_id} has {total_turbines_in_sample} turbines, less than requested {num_turbines_to_include}. Using all {total_turbines_in_sample} available.")
                    
                    power_subset_turbines = daily_power_all_wt_single_sample.isel(wt=slice(0, turbines_to_sum_count))
                    daily_site_power_single_sample = power_subset_turbines.sum(dim='wt', skipna=True)
                    turbines_summed_this_sample = turbines_to_sum_count
                else:
                    print(f"  Warning: Sample {target_s_id} 'wt' dimension has size 0.")
                    daily_site_power_single_sample = xr.DataArray(
                        [0] * daily_power_all_wt_single_sample.sizes.get('time',0),
                        dims=['time'],
                        coords={'time': daily_power_all_wt_single_sample.coords.get('time', [])}
                    )
            elif daily_power_all_wt_single_sample.ndim > 0 and 'time' in daily_power_all_wt_single_sample.dims:
                print(f"  Info: Sample {target_s_id} has no 'wt' dimension. Using raw power data as site power.")
                daily_site_power_single_sample = daily_power_all_wt_single_sample
                turbines_summed_this_sample = 1 
            else:
                print(f"  Warning: Sample {target_s_id} 'Power' data is not in expected format (no 'wt' and not 1D time series). Skipping.")
                if current_sample_ds: current_sample_ds.close()
                continue
            
            actual_turbines_summed_for_sample[target_s_id] = turbines_summed_this_sample

            if daily_site_power_single_sample is None or daily_site_power_single_sample.size == 0:
                print(f"  No site power data for sample {target_s_id} after turbine summation or data was empty.")
                if current_sample_ds: current_sample_ds.close()
                continue

            if 'time' not in daily_site_power_single_sample.dims or daily_site_power_single_sample.sizes['time'] < 1:
                print(f"  Sample {target_s_id} has no valid 'time' dimension for rolling mean. Skipping.")
                if current_sample_ds: current_sample_ds.close()
                continue
                
            actual_rolling_window = min(rolling_window_days, daily_site_power_single_sample.sizes['time'])
            if actual_rolling_window < 1 : 
                 print(f"  Sample {target_s_id} has time dimension size < 1 ({daily_site_power_single_sample.sizes['time']}). Skipping rolling mean.")
                 if current_sample_ds: current_sample_ds.close()
                 continue

            smoothed_single_sample_power = daily_site_power_single_sample.rolling(
                time=actual_rolling_window, center=True, min_periods=1
            ).mean(skipna=True)
            
            smoothed_single_sample_power = smoothed_single_sample_power.assign_coords(sample_id=target_s_id)
            smoothed_single_sample_power = smoothed_single_sample_power.expand_dims('sample_id')
            
            processed_sample_power_list.append(smoothed_single_sample_power)
            if current_sample_ds: current_sample_ds.close()
        else:
            print(f"Data for requested sample_id: {target_s_id} not found in the directory {data_dir}.")

    if not processed_sample_power_list:
        print(f"No data was processed for any of the specified sample IDs in {data_dir}.")
        return None, {}

    print(f"Concatenating processed samples from {data_dir}...")
    final_smoothed_power = xr.concat(processed_sample_power_list, dim='sample_id')
    
    return final_smoothed_power, actual_turbines_summed_for_sample

if __name__ == '__main__':
    data_directory_wake = './uniform_results/wake/'
    data_directory_no_wake = './uniform_results/no_wake/'

    first_sample_to_plot = 1
    number_of_samples_to_plot = 100 
    target_sample_ids = list(range(first_sample_to_plot, first_sample_to_plot + number_of_samples_to_plot))
    
    num_turbines_to_include = 65
    rolling_window_days = 360

    print(f"Attempting to load and process power data for sample IDs: {target_sample_ids}")
    
    print(f"\n--- Loading WAKE data from: {data_directory_wake} ---")
    smoothed_power_wake, turbines_info_wake = load_and_process_sample_power(
        data_directory_wake,
        target_sample_ids,
        num_turbines_to_include,
        rolling_window_days
    )

    print(f"\n--- Loading NO_WAKE data from: {data_directory_no_wake} ---")
    smoothed_power_no_wake, turbines_info_no_wake = load_and_process_sample_power(
        data_directory_no_wake,
        target_sample_ids,
        num_turbines_to_include,
        rolling_window_days
    )

    plot_data_dict = {}
    if smoothed_power_wake is not None and smoothed_power_wake.sample_id.size > 0:
        plot_data_dict['wake'] = smoothed_power_wake
        print(f"Wake data successfully processed for {smoothed_power_wake.sample_id.size} sample(s).")
    else:
        print("No wake data was processed or available.")

    if smoothed_power_no_wake is not None and smoothed_power_no_wake.sample_id.size > 0:
        plot_data_dict['no_wake'] = smoothed_power_no_wake
        print(f"No-wake data successfully processed for {smoothed_power_no_wake.sample_id.size} sample(s).")
    else:
        print("No no-wake data was processed or available.")

    if not plot_data_dict:
        print("\nNo data available for plotting from any source. Exiting.")
    else:
        print("\nPreparing combined plot...")
        combined_ds = xr.Dataset(plot_data_dict)
        plot_data_array = combined_ds.to_array(dim='source')
        # plot_data_array dimensions are now (source, sample_id, time)

        if plot_data_array.size == 0:
            print("Combined data array is empty. Cannot plot. Exiting.")
        else:
            print(f"Original combined plot_data_array dimensions: {plot_data_array.dims}")
            
            # --- MODIFIED PLOTTING SECTION for explicit two-color plotting ---
            plt.figure(figsize=(14, 7))
            ax = plt.gca() # Get current axes

            # Define explicit colors for sources
            # Using common distinct colors; you can customize these
            colors = {'wake': 'C0', 'no_wake': 'C1'} # C0 is blue, C1 is orange by default

            legend_handles = {} # To store one line handle per source for the legend

            # Iterate through each source type ('wake', 'no_wake')
            for src_name_da in plot_data_array.source:
                src_name = str(src_name_da.values) # Extract string value e.g. 'wake'
                
                # Select data for the current source: results in (sample_id, time) DataArray
                data_for_current_source = plot_data_array.sel(source=src_name_da)
                
                # Iterate through each sample_id for this source
                for sample_id_val in data_for_current_source.sample_id.values:
                    # Select the time series for this specific sample_id and source
                    single_time_series = data_for_current_source.sel(sample_id=sample_id_val)
                    
                    # Only plot if there's actual data (all NaNs might occur if a sample is missing for a source)
                    if not single_time_series.isnull().all():
                        # Determine if we need to add a label for the legend
                        # (only for the first valid line of each source type)
                        if src_name not in legend_handles:
                            line, = ax.plot(single_time_series.time.values, 
                                            single_time_series.values, 
                                            color=colors.get(src_name, 'gray'), # Fallback to gray if src_name not in colors
                                            label=src_name)
                            legend_handles[src_name] = line
                        else:
                            ax.plot(single_time_series.time.values, 
                                    single_time_series.values, 
                                    color=colors.get(src_name, 'gray'))
            
            if legend_handles:
                ax.legend(handles=legend_handles.values(), labels=legend_handles.keys(), title="Source")
            # --- END OF MODIFIED PLOTTING SECTION ---
            
            title_sources_str = ' & '.join(plot_data_array.source.astype(str).values)
            
            unique_sample_ids = np.unique(plot_data_array.sample_id.values)
            if len(unique_sample_ids) > 0:
                if len(unique_sample_ids) > 5:
                    samples_descriptor = f"{len(unique_sample_ids)} unique samples (e.g., {unique_sample_ids[0]}...{unique_sample_ids[-1]})"
                else:
                    samples_descriptor = f"Samples {', '.join(map(str, unique_sample_ids))}"
            else:
                 samples_descriptor = " (No specific samples found in combined data)"

            turbine_sum_info = f" (Sum over up to {num_turbines_to_include} Turbines)" if num_turbines_to_include > 0 else ""
            
            plt.title(f'{rolling_window_days}-Day Rolling Avg Power from {title_sources_str} Sources\n{samples_descriptor}{turbine_sum_info}')
            plt.xlabel('Time (Days)')
            plt.ylabel(f'{rolling_window_days}-Day Rolling Avg Power')
            plt.grid(True)
            plt.tight_layout()
            
            output_filename = 'powers_wake_vs_nowake.png'
            plt.savefig(output_filename)
            print(f"\nPlot saved as {output_filename}")
            plt.clf()
            print("Script finished.")
