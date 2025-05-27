import h5py
import numpy as np
import xarray as xr
import pandas as pd
import os
import glob
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import traceback # For detailed error printing

# --- PyWake Imports ---
from py_wake.site.xrsite import XRSite
from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine
from py_wake import Nygaard_2022
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.wind_turbines import WindTurbines, WindTurbine

# --- Configuration Section (USER TO PROVIDE/CONFIRM) ---
H5_LAYOUT_FILE = "re_precomputed_layouts.h5"

# --- parse_filename_details and load_and_process_sample_power functions ---
def parse_filename_details(filepath):
    match = re.search(r'res_(\d+)_(\d+)\.nc$', os.path.basename(filepath))
    if match:
        return {'filepath': filepath, 'sample_id': int(match.group(1)), 'chunk_id': int(match.group(2))}
    return None

def load_and_process_sample_power(data_dir, sample_ids_to_load, num_turbines_to_include_in_sum, rolling_window_days):
    file_pattern = os.path.join(data_dir, "res_*.nc")
    all_filepaths = glob.glob(file_pattern)
    if not all_filepaths:
        print(f"No files found: {file_pattern} in {data_dir}")
        return None
    parsed_files_info = [parse_filename_details(fp) for fp in all_filepaths if parse_filename_details(fp)]
    if not parsed_files_info:
        print(f"No files parsed in: {data_dir}")
        return None

    files_by_sample_id_map = defaultdict(list)
    for info in parsed_files_info: files_by_sample_id_map[info['sample_id']].append(info)
    processed_sample_power_list = []

    for target_s_id in sample_ids_to_load:
        if target_s_id in files_by_sample_id_map:
            sample_file_chunks = sorted(files_by_sample_id_map[target_s_id], key=lambda x: x['chunk_id'])
            chunk_filepaths = [info['filepath'] for info in sample_file_chunks]
            if not chunk_filepaths: continue
            print(f"Processing sample_id: {target_s_id} from {data_dir}...")
            current_sample_ds = None
            try:
                current_sample_ds = xr.open_mfdataset(chunk_filepaths, combine='nested', concat_dim='time', engine='netcdf4', chunks={'time': 'auto'})
            except Exception as e:
                print(f"  Error loading files for sample {target_s_id} from {data_dir}: {e}")
                if current_sample_ds: current_sample_ds.close()
                continue
            if 'Power' not in current_sample_ds:
                print(f"  'Power' variable not found for {target_s_id} from {data_dir}.")
                if current_sample_ds: current_sample_ds.close()
                continue
            
            power_data_array = current_sample_ds['Power']
            time_coords_to_preserve = current_sample_ds.coords['time']

            daily_power_all_wt_single_sample = power_data_array
            daily_site_power_single_sample = None
            if 'wt' in daily_power_all_wt_single_sample.dims:
                total_turbines_in_file = daily_power_all_wt_single_sample.sizes['wt']
                if total_turbines_in_file > 0:
                    turbines_to_sum_count = min(num_turbines_to_include_in_sum, total_turbines_in_file)
                    if num_turbines_to_include_in_sum > total_turbines_in_file:
                        print(f"  Info: Sample {target_s_id} has {total_turbines_in_file} turbines, < requested {num_turbines_to_include_in_sum}. Summing {total_turbines_in_file}.")
                    power_subset_turbines = daily_power_all_wt_single_sample.isel(wt=slice(0, turbines_to_sum_count))
                    daily_site_power_single_sample = power_subset_turbines.sum(dim='wt', skipna=True)
                else:
                    daily_site_power_single_sample = xr.DataArray(
                        [0]*daily_power_all_wt_single_sample.sizes.get('time',0),
                        dims=['time'],
                        coords={'time': time_coords_to_preserve}
                    )
            elif daily_power_all_wt_single_sample.ndim > 0 and 'time' in daily_power_all_wt_single_sample.dims:
                daily_site_power_single_sample = daily_power_all_wt_single_sample
            else:
                if current_sample_ds: current_sample_ds.close(); continue
            
            if daily_site_power_single_sample is None or daily_site_power_single_sample.size == 0 :
                if current_sample_ds: current_sample_ds.close(); continue
            if 'time' not in daily_site_power_single_sample.dims or daily_site_power_single_sample.sizes['time'] < 1:
                if current_sample_ds: current_sample_ds.close(); continue
            
            actual_rolling_window = min(rolling_window_days, daily_site_power_single_sample.sizes['time'])
            if actual_rolling_window < 1:
                # If window is less than 1, rolling mean is not meaningful or will error.
                # Keep the original data, or decide on other handling (e.g. skip smoothing)
                smoothed_power = daily_site_power_single_sample 
                print(f"  Info: Rolling window {actual_rolling_window} is < 1 for sample {target_s_id}. Using unsmoothed data.")

            else:
                 smoothed_power = daily_site_power_single_sample.rolling(time=actual_rolling_window, center=True, min_periods=1).mean(skipna=True)

            smoothed_power = smoothed_power.assign_coords(sample_id=target_s_id).expand_dims('sample_id')
            processed_sample_power_list.append(smoothed_power)
            if current_sample_ds: current_sample_ds.close()
            
    if not processed_sample_power_list: return None
    final_smoothed_power = xr.concat(processed_sample_power_list, dim='sample_id')
    return final_smoothed_power

# --- Helper function get_layout_from_h5 ---
def get_layout_from_h5(farm_idx, type_idx, seed, h5_file_path):
    config_key = f"farm{farm_idx}_t{type_idx}_s{seed}"
    if not os.path.exists(h5_file_path):
        print(f"ERROR: Layout HDF5 file not found: {h5_file_path}")
        return None, None
    try:
        with h5py.File(h5_file_path, 'r') as f:
            if config_key in f:
                layout_dataset = f[config_key]['layout']
                if layout_dataset.shape[0] == 2 and layout_dataset.ndim == 2:
                    layout_data = layout_dataset[:]
                    x_coords = np.asarray(layout_data[0])
                    y_coords = np.asarray(layout_data[1])
                    if x_coords.ndim == 1 and y_coords.ndim == 1 and len(x_coords) == len(y_coords):
                        return x_coords, y_coords
                    else:
                        print(f"ERROR: Layout data for '{config_key}' has inconsistent x/y shapes: x_shape={x_coords.shape}, y_shape={y_coords.shape}")
                        return None, None
                else:
                    print(f"ERROR: Layout data for '{config_key}' has unexpected shape: {layout_dataset.shape}. Expected (2, N_turbines).")
                    return None, None
            else:
                print(f"ERROR: Config key '{config_key}' not found in {h5_file_path}.")
                return None, None
    except Exception as e:
        print(f"Error reading layout from HDF5 file for key '{config_key}': {e}")
        traceback.print_exc()
        return None, None

# --- Main Analysis Script ---
if __name__ == "__main__":
    # --- General Configuration ---
    data_directory_wake = './uniform_results/wake/'
    data_directory_no_wake = './uniform_results/no_wake/'
    first_sample_to_plot = 1
    number_of_samples_to_plot = 500
    target_sample_ids = list(range(first_sample_to_plot, first_sample_to_plot + number_of_samples_to_plot))
    num_turbines_to_include_for_neighbor_cases = 65
    rolling_window_days = 360

    # =================================================================================
    # --- PYWAKE BASELINE CONFIGURATION (ACTION: USER TO PROVIDE/CONFIRM THESE DETAILS) ---
    # =================================================================================
    BASELINE_FARM_IDX = 0
    BASELINE_TYPE_IDX = 5
    BASELINE_SEED = 0
    BASELINE_TURBINE_NAME = 'BaselineIsolatedTurbine'
    BASELINE_TURBINE_D = 240.0
    BASELINE_TURBINE_H = 150.0
    # Rated power in kW for GenericWindTurbine's power_norm argument
    BASELINE_TURBINE_P_REF_KW = 15000.0

    # Define target units for plotting (must match units of res_*.nc 'Power' data)
    TARGET_PLOT_UNITS_NAME = "MW" # Options: "W", "kW", "MW" - <<< USER ACTION: SET THIS!
    # Rated Power of ONE baseline turbine, in the TARGET_PLOT_UNITS_NAME defined above
    RATED_TURBINE_POWER_IN_TARGET_UNITS = 15.0 # <<< USER ACTION: SET THIS!
    
    SITE_FILE_FOR_PYWAKE = 'ref_site.nc'
    # =================================================================================

    # Initialize plot variables
    plot_data_array = None
    isolated_sim_smoothed_power = None
    time_varying_max_potential_smoothed = None # Renamed for clarity
    N_BASELINE_TURBINES = 0
    time_coords_for_sim = None
    ws_sim_input = None
    wd_sim_input = None
    max_potential_power_baseline = 0.0 # Initialize for fallback

    # 1. Load WAKE and NO_WAKE data
    print(f"\n--- Loading WAKE data from: {data_directory_wake} ---")
    smoothed_power_wake = load_and_process_sample_power(data_directory_wake, target_sample_ids, num_turbines_to_include_for_neighbor_cases, rolling_window_days)

    print(f"\n--- Loading NO_WAKE data from: {data_directory_no_wake} ---")
    smoothed_power_no_wake = load_and_process_sample_power(data_directory_no_wake, target_sample_ids, num_turbines_to_include_for_neighbor_cases, rolling_window_days)
    
    plot_data_dict = {}
    if smoothed_power_wake is not None: plot_data_dict['wake'] = smoothed_power_wake
    if smoothed_power_no_wake is not None: plot_data_dict['no_wake'] = smoothed_power_no_wake

    if plot_data_dict:
        try:
            valid_plot_data_dict = {k: v for k, v in plot_data_dict.items() if v is not None}
            if valid_plot_data_dict:
                combined_ds = xr.Dataset(valid_plot_data_dict)
                plot_data_array = combined_ds.to_array(dim='source')
                print(f"Wake/No-Wake data combined. Plot_data_array dims: {plot_data_array.dims}, coords: {plot_data_array.coords}")
                
                # Attempt to extract ws/wd for PyWake simulation IF they are part of the 'time' coordinate
                # This structure assumes 'ws' and 'wd' might be non-dimensional coordinates of 'time'
                if 'time' in plot_data_array.coords:
                    time_coord_obj = plot_data_array.coords['time']
                    if hasattr(time_coord_obj, 'coords') and \
                       'ws' in time_coord_obj.coords and \
                       'wd' in time_coord_obj.coords:
                        time_coords_for_sim = time_coord_obj # Keep the DataArray for time
                        ws_sim_input = time_coord_obj.coords['ws'].data
                        wd_sim_input = time_coord_obj.coords['wd'].data
                        print(f"Extracted ws/wd for PyWake from loaded data (length: {len(ws_sim_input)}).")
                    else:
                        # If ws/wd not in time.coords, try to get them from the dataset variables if they exist with matching 'time'
                        if 'ws' in plot_data_array.coords and 'wd' in plot_data_array.coords and \
                           plot_data_array.coords['ws'].dims == ('time',) and \
                           plot_data_array.coords['wd'].dims == ('time,') :
                           time_coords_for_sim = plot_data_array.coords['time']
                           ws_sim_input = plot_data_array.coords['ws'].data
                           wd_sim_input = plot_data_array.coords['wd'].data
                           print(f"Extracted ws/wd for PyWake from data variables (length: {len(ws_sim_input)}).")
                        else:
                            print("Warning: Could not extract ws/wd coordinates robustly from loaded plot_data_array for PyWake sim.")
                            print("Ensure 'ws' and 'wd' are coordinates of 'time' or standalone coordinates along 'time' dimension.")
            else:
                print("No valid wake/no-wake data to combine into plot_data_array.")
        except Exception as e:
            print(f"Error combining wake/no_wake data: {e}"); traceback.print_exc()
    else:
        print("No wake or no-wake data loaded successfully.")

    # 2. Prepare and Run PyWake simulation for ISOLATED BASELINE
    print("\n--- Attempting PyWake simulation for isolated baseline ---")
    baseline_x_coords, baseline_y_coords = get_layout_from_h5(BASELINE_FARM_IDX, BASELINE_TYPE_IDX, BASELINE_SEED, H5_LAYOUT_FILE)

    baseline_pywake_turbine_obj = None

    if baseline_x_coords is None or baseline_y_coords is None:
        print("Failed to load baseline layout. Skipping PyWake simulation.")
    elif not (isinstance(baseline_x_coords, np.ndarray) and isinstance(baseline_y_coords, np.ndarray)):
        print("Loaded baseline layout x or y are not numpy arrays. Skipping PyWake simulation.")
    elif len(baseline_x_coords) != len(baseline_y_coords):
        print("Loaded baseline layout x and y coordinates have different lengths. Skipping PyWake simulation.")
    else:
        N_BASELINE_TURBINES = len(baseline_x_coords)
        if N_BASELINE_TURBINES == 0:
            print("Baseline layout loaded but has no turbines. Skipping PyWake simulation.")
        else:
            print(f"Successfully loaded layout for {N_BASELINE_TURBINES} baseline turbines.")

            if ws_sim_input is not None and wd_sim_input is not None and time_coords_for_sim is not None:
                try:
                    baseline_pywake_turbine_obj = GenericWindTurbine(
                        name=BASELINE_TURBINE_NAME,
                        diameter=BASELINE_TURBINE_D,
                        hub_height=BASELINE_TURBINE_H,
                        power_norm=BASELINE_TURBINE_P_REF_KW
                    )
                    windTurbines_for_sim = WindTurbines.from_WindTurbine_lst([baseline_pywake_turbine_obj])
                    print(f"Baseline PyWake turbine model ('{BASELINE_TURBINE_NAME}') defined.")

                    print(f"Loading site for PyWake from: {SITE_FILE_FOR_PYWAKE}")
                    if not os.path.exists(SITE_FILE_FOR_PYWAKE):
                        raise FileNotFoundError(f"PyWake site file {SITE_FILE_FOR_PYWAKE} not found.")
                    site_for_pywake_sim = XRSite.load(SITE_FILE_FOR_PYWAKE)
                    
                    wf_model = Nygaard_2022(site_for_pywake_sim, windTurbines_for_sim)
                    print(f"PyWake model ({type(wf_model).__name__}) instantiated.")

                    print(f"Running PyWake simulation ({N_BASELINE_TURBINES} turbines, {len(ws_sim_input)} time steps)...")
                    # Ensure time_coords_for_sim.data is 1D array of timestamps/indices for PyWake
                    time_input_for_pywake = time_coords_for_sim.data
                    if time_input_for_pywake.ndim > 1: # Safety check
                        time_input_for_pywake = time_input_for_pywake.flatten()


                    sim_res = wf_model(
                        x=baseline_x_coords, y=baseline_y_coords,
                        wd=wd_sim_input, ws=ws_sim_input,
                        TI=0.1, # Assuming a constant TI, adjust if variable TI is available/needed
                        time=time_input_for_pywake # Pass the 1D time data
                    )
                    
                    if hasattr(sim_res, 'Power'):
                        total_farm_power_watts = sim_res.Power.sum(dim='wt') # Power is in Watts from PyWake
                        
                        conversion_factor_from_watts = 1.0
                        
                        isolated_power_plot_units = total_farm_power_watts / conversion_factor_from_watts

                        isolated_power_da = xr.DataArray(
                            isolated_power_plot_units.data, # Use .data to get NumPy array
                            coords={'time': time_coords_for_sim.data}, dims=['time'] # Match time coord
                        )
                        actual_rolling_window_sim = min(rolling_window_days, len(isolated_power_da.time))
                        if actual_rolling_window_sim >= 1:
                            isolated_sim_smoothed_power = isolated_power_da.rolling(
                                time=actual_rolling_window_sim, center=True, min_periods=1).mean(skipna=True)
                        else:
                            isolated_sim_smoothed_power = isolated_power_da
                        print("PyWake simulation for isolated baseline (total farm power) complete and processed.")

                        # Calculate time-varying max potential power
                        single_turbine_potential_power_watts = baseline_pywake_turbine_obj.power(ws=ws_sim_input) # Watts
                        single_turbine_potential_plot_units = single_turbine_potential_power_watts / conversion_factor_from_watts
                        time_varying_max_farm_potential_plot_units = single_turbine_potential_plot_units * N_BASELINE_TURBINES
                        
                        max_potential_da = xr.DataArray(
                            time_varying_max_farm_potential_plot_units, # This is a numpy array
                            coords={'time': time_coords_for_sim.data}, dims=['time']
                        )
                        if actual_rolling_window_sim >= 1:
                            time_varying_max_potential_smoothed = max_potential_da.rolling(
                                time=actual_rolling_window_sim, center=True, min_periods=1).mean(skipna=True)
                        else:
                            time_varying_max_potential_smoothed = max_potential_da
                        print("Time-varying max potential power calculated and processed.")
                        
                    else:
                        print("PyWake sim ran, but 'Power' attr not found in sim_res.")
                except Exception as e:
                    print(f"ERROR during PyWake simulation for baseline: {e}"); traceback.print_exc()
            else:
                print("Skipping PyWake simulation: ws/wd/time data not available from loaded .nc files.")
            
            if N_BASELINE_TURBINES > 0 and time_varying_max_potential_smoothed is None :
                max_potential_power_baseline = RATED_TURBINE_POWER_IN_TARGET_UNITS * N_BASELINE_TURBINES
                print(f"Calculated static Max potential power for baseline (fallback): {max_potential_power_baseline:.2f} {TARGET_PLOT_UNITS_NAME}")


    # 3. Plotting Section
    if not plot_data_dict and isolated_sim_smoothed_power is None: # Check if any data exists at all
        print("\nNo data available for plotting from any source. Exiting.")
    else:
        print("\nPreparing plot...")
        plt.figure(figsize=(15, 9))
        ax = plt.gca()
        colors = {
            'wake': 'red', 'no_wake': 'purple',
            'isolated_baseline': 'blue',
            'max_potential_time_varying': 'k',
            'max_potential_static': 'k' # Added for fallback static line
        }
        my_names = {
            'wake': 'Total Power', 'no_wake': 'External Losses',
            'isolated_baseline': 'Internal Losses',
            'max_potential_time_varying': 'No Wake',
            'max_potential_static': f"Rated Max ({N_BASELINE_TURBINES} WT)" # For fallback
        }
        
        legend_artist_map = {} # Stores artists for the legend: key -> artist

        plot_data_numpy = None
        time_values_np = None
        source_coords_np = None

        if plot_data_array is not None and plot_data_array.size > 0:
            print("Optimizing: Pre-fetching data to NumPy arrays for plotting...")
            try:
                if hasattr(plot_data_array.coords['time'], 'compute'):
                    time_values_np = plot_data_array.coords['time'].compute().data
                else:
                    time_values_np = plot_data_array.coords['time'].data
                
                if hasattr(plot_data_array, 'compute'):
                    plot_data_numpy = plot_data_array.compute().data
                else:
                    plot_data_numpy = plot_data_array.data

                if hasattr(plot_data_array.coords['source'], 'compute'):
                    source_coords_np = plot_data_array.coords['source'].compute().data
                else:
                    source_coords_np = plot_data_array.coords['source'].data
                
                print(f"Data pre-fetched. Plot data shape: {plot_data_numpy.shape if plot_data_numpy is not None else 'N/A'}, Time values length: {len(time_values_np) if time_values_np is not None else 'N/A'}")

            except Exception as e:
                print(f"ERROR pre-fetching data for plotting: {e}")
                traceback.print_exc()
                # plot_data_numpy, time_values_np, source_coords_np remain None or as set before error

            if plot_data_numpy is not None and time_values_np is not None and source_coords_np is not None:
                for i_src, src_name_val in enumerate(source_coords_np):
                    src_name = str(src_name_val) 

                    for i_sample in range(plot_data_numpy.shape[1]):
                        single_time_series_np = plot_data_numpy[i_src, i_sample, :]

                        if np.all(np.isnan(single_time_series_np)):
                            continue

                        line_color = colors.get(src_name, 'grey')
                        alpha_val = 1.0 # As per user preference
                        lw_val = 1.0    # As per user preference
                        
                        label_for_legend = None
                        if src_name not in legend_artist_map:
                            label_for_legend = my_names.get(src_name, src_name).capitalize()
                        
                        line, = ax.plot(time_values_np, single_time_series_np,
                                        color=line_color, label=label_for_legend,
                                        alpha=alpha_val, linewidth=lw_val)
                        
                        if src_name not in legend_artist_map:
                            legend_artist_map[src_name] = line
            else:
                print("Skipping plotting of wake/no-wake data due to issues in pre-fetching or missing data.")
        
        if isolated_sim_smoothed_power is not None:
            if not isolated_sim_smoothed_power.isnull().all():
                key_isolated = 'isolated_baseline'
                label_isolated = my_names.get(key_isolated, "Isolated Baseline")
                line, = ax.plot(isolated_sim_smoothed_power.time.values, isolated_sim_smoothed_power.values,
                                color=colors[key_isolated], label=label_isolated,
                                linewidth=2.0, linestyle='-')
                if key_isolated not in legend_artist_map:
                    legend_artist_map[key_isolated] = line
        
        if time_varying_max_potential_smoothed is not None:
            if not time_varying_max_potential_smoothed.isnull().all():
                key_max_power = 'max_potential_time_varying'
                label_max_power = my_names.get(key_max_power, "Max Potential Power")
                line, = ax.plot(time_varying_max_potential_smoothed.time.values,
                                time_varying_max_potential_smoothed.values,
                                color=colors[key_max_power],
                                label=label_max_power, linewidth=2.0, linestyle='-')
                if key_max_power not in legend_artist_map:
                    legend_artist_map[key_max_power] = line
        elif N_BASELINE_TURBINES > 0 and max_potential_power_baseline > 0: 
            key_max_static = 'max_potential_static'
            # Ensure my_names has this entry for consistent label lookup
            if key_max_static not in my_names:
                 my_names[key_max_static] = f"Rated Max ({N_BASELINE_TURBINES} WT)" # Default if not set earlier
            label_max_power_static = my_names.get(key_max_static)

            if key_max_static not in legend_artist_map:
                line = ax.axhline(max_potential_power_baseline, color=colors.get(key_max_static, 'k'), # Use .get for color too
                                  linestyle=':', label=label_max_power_static, linewidth=2)
                legend_artist_map[key_max_static] = line

        if legend_artist_map:
            handles_for_legend = []
            labels_for_legend = []
            # Iterate through my_names to try and preserve a somewhat logical order for the legend
            # or use legend_artist_map.keys() if order is defined by plotting sequence
            ordered_keys_for_legend = [k for k in ['wake', 'no_wake', 'isolated_baseline', 'max_potential_time_varying', 'max_potential_static'] if k in legend_artist_map]
            # Add any other keys that might have been added to legend_artist_map but are not in the predefined list
            for k in legend_artist_map.keys():
                if k not in ordered_keys_for_legend:
                    ordered_keys_for_legend.append(k)

            for key in ordered_keys_for_legend:
                if key in my_names: # Ensure the key exists in my_names for the label
                    handles_for_legend.append(legend_artist_map[key])
                    labels_for_legend.append(my_names[key].capitalize()) # Use .capitalize() for consistency
                else: # Fallback if a key made it to artist map but not my_names (should not happen with current logic)
                    handles_for_legend.append(legend_artist_map[key])
                    labels_for_legend.append(key.capitalize())


            ax.legend(handles=handles_for_legend, labels=labels_for_legend, loc='best')
        
        plt.title(f"{rolling_window_days}-Day Rolling Avg Power & PyWake Baseline Simulation", fontsize=16)
        plt.xlabel('Time (Days from start of common period)', fontsize=12)
        plt.ylabel(f'Power (W)', fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        
        output_filename = 'powers_comparison_with_pywake_sim_integrated.png'
        plt.savefig(output_filename)
        print(f"\nPlot saved as {output_filename}")
        plt.clf() 
        print("Script finished.")
