#!/usr/bin/env python3

import os
import glob
import sys
import time
from collections import defaultdict

# Configuration
RESULT_DIR = "/work/users/juqu/Neighboring-Wind-Farms-Out-Of-Distribution-Surrogate-Modeling/results/wake"
OUTPUT_DIR = "/work/users/juqu/Neighboring-Wind-Farms-Out-Of-Distribution-Surrogate-Modeling/results/concatenated"
FINAL_OUTPUT = os.path.join(OUTPUT_DIR, "all_samples.nc")
TOTAL_SAMPLES = 5000
FILES_PER_SAMPLE = 10

def check_progress():
    """Check progress of sample processing and return completed samples."""
    # Get all result files
    all_files = glob.glob(os.path.join(RESULT_DIR, "res_*_*.nc"))
    
    # Count files per sample
    sample_counts = defaultdict(int)
    for f in all_files:
        basename = os.path.basename(f)
        try:
            # Extract sample ID
            sample_id = int(basename.split('_')[1])
            sample_counts[sample_id] += 1
        except:
            continue
    
    # Track completed and incomplete samples
    completed = []
    incomplete = []
    for sample_id, count in sample_counts.items():
        if count == FILES_PER_SAMPLE:
            completed.append(sample_id)
        else:
            incomplete.append((sample_id, count))
    
    completed.sort()
    incomplete.sort()
    
    return {
        "completed": completed,
        "incomplete": incomplete,
        "total_completed": len(completed),
        "total_incomplete": len(incomplete),
        "total_missing": TOTAL_SAMPLES - len(sample_counts),
        "percent_complete": (len(completed) / TOTAL_SAMPLES) * 100
    }

def print_report(progress):
    """Print a progress report."""
    print("\n" + "=" * 50)
    print(f"PROGRESS REPORT - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    print(f"Completed: {progress['total_completed']} / {TOTAL_SAMPLES} samples ({progress['percent_complete']:.1f}%)")
    print(f"Incomplete: {progress['total_incomplete']} samples")
    print(f"Missing: {progress['total_missing']} samples")
    
    # Show some completed samples
    if progress['completed']:
        sample_str = str(progress['completed'][:10])
        if len(progress['completed']) > 10:
            sample_str = sample_str[:-1] + ", ...]"
        print(f"\nCompleted samples: {sample_str}")
    
    # Show some incomplete samples
    if progress['incomplete']:
        print("\nIncomplete samples (sample_id, files_count):")
        for i, (sample_id, count) in enumerate(progress['incomplete'][:5]):
            print(f"  Sample {sample_id}: {count}/{FILES_PER_SAMPLE} files")
        if len(progress['incomplete']) > 5:
            print(f"  ... and {len(progress['incomplete']) - 5} more")
    
    print("=" * 50)


import h5py
import sys
import os
import glob
import xarray as xr
import numpy as np

def concatenate_all_samples_hdf5(completed_samples, output_file):
    """Concatenate all completed samples into a single HDF5 file."""
    if not completed_samples:
        print("No completed samples to concatenate.")
        return False

    try:
        with h5py.File(output_file, 'w') as hf:
            for i, sample_id in enumerate(completed_samples):
                if i % 100 == 0:
                    print(f"Processing sample {i+1}/{len(completed_samples)}")

                sample_files = sorted(glob.glob(os.path.join(RESULT_DIR, f"res_{sample_id}_*.nc")))

                if len(sample_files) != FILES_PER_SAMPLE:
                    print(f"Warning: Sample {sample_id} has {len(sample_files)} files, expected {FILES_PER_SAMPLE}")
                    continue

                try:
                    sample_datasets = [xr.open_dataset(f) for f in sample_files]
                    sample_concat = xr.concat(sample_datasets, dim="time")

                    # Store data in HDF5
                    for var_name, data_array in sample_concat.data_vars.items():
                        if var_name not in hf:
                            # Create dataset if it doesn't exist
                            hf.create_dataset(var_name, shape=(len(completed_samples), len(sample_concat.time), *data_array.shape[1:]), dtype=data_array.dtype)
                        hf[var_name][i, :, ...] = data_array.values

                    # Store sample_id as metadata
                    hf.attrs[f"sample_id_{i}"] = sample_id

                    for ds in sample_datasets:
                        ds.close()

                except Exception as e:
                    print(f"Error processing sample {sample_id}: {str(e)}")
                    continue

        print(f"Successfully concatenated {len(completed_samples)} samples into {output_file}")
        return True

    except Exception as e:
        print(f"Error concatenating samples: {str(e)}")
        return False

def concatenate_all_samples(completed_samples):
    """Concatenate all completed samples into a single NetCDF file."""
    if not completed_samples:
        print("No completed samples to concatenate.")
        return False
    
    # Make sure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    try:
        import xarray as xr
        import numpy as np
        
        print(f"Concatenating {len(completed_samples)} samples into a single file...")
        
        # Process each sample and add to a list of datasets
        all_datasets = []
        
        for i, sample_id in enumerate(completed_samples):
            if i % 100 == 0:
                print(f"Processing sample {i+1}/{len(completed_samples)}")
            
            # Get all files for this sample
            sample_files = sorted(glob.glob(os.path.join(RESULT_DIR, f"res_{sample_id}_*.nc")))
            
            if len(sample_files) != FILES_PER_SAMPLE:
                print(f"Warning: Sample {sample_id} has {len(sample_files)} files, expected {FILES_PER_SAMPLE}")
                continue
            
            try:
                # Open and concatenate along time dimension
                sample_datasets = [xr.open_dataset(f) for f in sample_files]
                sample_concat = xr.concat(sample_datasets, dim="time")
                
                # Add sample_id as a coordinate
                sample_concat = sample_concat.assign_coords(sample_id=sample_id)
                sample_concat = sample_concat.expand_dims("sample_id")
                
                all_datasets.append(sample_concat)
                
                # Close all sample datasets to free memory
                for ds in sample_datasets:
                    ds.close()
                    
            except Exception as e:
                print(f"Error processing sample {sample_id}: {str(e)}")
                continue
        
        if not all_datasets:
            print("No datasets were successfully processed.")
            return False
            
        # Combine all samples along sample_id dimension
        print("Merging all samples...")
        combined = xr.concat(all_datasets, dim="sample_id")
        
        # Save to final output file
        print(f"Saving to {FINAL_OUTPUT}...")
        combined.to_netcdf(FINAL_OUTPUT)
        
        # Close the combined dataset
        combined.close()
        
        print(f"Successfully concatenated {len(all_datasets)} samples into {FINAL_OUTPUT}")
        return True
        
    except Exception as e:
        print(f"Error concatenating samples: {str(e)}")
        return False

def monitor(interval=300):
    """Monitor progress and concatenate completed samples."""
    last_concat_count = 0
    
    try:
        while True:
            # Check progress
            progress = check_progress()
            print_report(progress)
            
            # Check if we have new completed samples
            if progress['total_completed'] > last_concat_count:
                if os.path.exists(FINAL_OUTPUT):
                    print(f"\nUpdating concatenated file with {progress['total_completed'] - last_concat_count} new samples...")
                    # For simplicity, we'll regenerate the whole file
                    # A more complex approach would append only new samples
                else:
                    print("\nCreating initial concatenated file...")
                
                success = concatenate_all_samples_hdf5(progress['completed'])
                if success:
                    last_concat_count = progress['total_completed']
            
            # Check if all done
            if progress['total_completed'] == TOTAL_SAMPLES:
                print("\nAll samples have been completed and concatenated!")
                break
                
            print(f"\nNext update in {interval//60} minutes. Press Ctrl+C to exit.")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        # Just check once
        progress = check_progress()
        print_report(progress)
        
        if progress['completed']:
            response = input(f"Concatenate {len(progress['completed'])} completed samples? (y/n): ")
            if response.lower() in ('y', 'yes'):
                concatenate_all_samples_hdf5(progress['completed'])
    else:
        # Monitor continuously
        interval = 300
        if len(sys.argv) > 1:
            try:
                interval = int(sys.argv[1]) * 60
            except ValueError:
                print(f"Invalid interval: {sys.argv[1]}. Using default 5 minutes.")
                
        monitor(interval)

if __name__ == "__main__":
    main()
