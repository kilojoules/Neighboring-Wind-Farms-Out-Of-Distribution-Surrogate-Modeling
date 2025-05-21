#!/usr/bin/env python3

import os
import glob
import sys
import time
from collections import defaultdict
import h5py
import xarray as xr
import numpy as np

# Configuration
RESULT_DIR = "/work/users/juqu/Neighboring-Wind-Farms-Out-Of-Distribution-Surrogate-Modeling/results/wake"
OUTPUT_DIR = "/work/users/juqu/Neighboring-Wind-Farms-Out-Of-Distribution-Surrogate-Modeling/results/concatenated"
FINAL_OUTPUT = os.path.join(OUTPUT_DIR, "all_samples.h5")  # HDF5 output
TOTAL_SAMPLES = 5000
FILES_PER_SAMPLE = 10

def check_progress():
    """Check progress of sample processing and return completed samples."""
    all_files = glob.glob(os.path.join(RESULT_DIR, "res_*_*.nc"))
    sample_counts = defaultdict(int)
    for f in all_files:
        basename = os.path.basename(f)
        try:
            sample_id = int(basename.split('_')[1])
            sample_counts[sample_id] += 1
        except:
            continue

    completed = []
    incomplete = []
    for sample_id, count in sample_counts.items():
        if count >= FILES_PER_SAMPLE - 1:
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

    if progress['completed']:
        sample_str = str(progress['completed'][:10])
        if len(progress['completed']) > 10:
            sample_str = sample_str[:-1] + ", ...]"
        print(f"\nCompleted samples: {sample_str}")

    if progress['incomplete']:
        print("\nIncomplete samples (sample_id, files_count):")
        for i, (sample_id, count) in enumerate(progress['incomplete'][:5]):
            print(f"  Sample {sample_id}: {count}/{FILES_PER_SAMPLE} files")
        if len(progress['incomplete']) > 5:
            print(f"  ... and {len(progress['incomplete']) - 5} more")

    print("=" * 50)

import h5py
import os
import glob
import xarray as xr
import numpy as np

def concatenate_all_samples_hdf5(completed_samples, output_file):
    """Concatenate all completed samples into a single HDF5 file with hot start."""
    if not completed_samples:
        print("No completed samples to concatenate.")
        return False

    try:
        try:
            if os.path.exists(output_file):
                print("Hot start: Appending to existing HDF5 file.")
                with h5py.File(output_file, 'a') as hf:
                    existing_sample_ids = []
                    if 'sample_id' in hf:
                        existing_sample_ids = hf['sample_id'][:]
                    else:
                        existing_sample_ids = []

                    new_samples = [sid for sid in completed_samples if sid not in existing_sample_ids]
            else:
                print("Creating new HDF5 file.")
                with h5py.File(output_file, 'w') as hf:
                    new_samples = completed_samples
        except OSError as e:
            if "truncated file" in str(e):
                print(f"Corrupted file found. Deleting {output_file} and recreating.")
                os.remove(output_file)
                with h5py.File(output_file, 'w') as hf:
                    new_samples = completed_samples
            else:
                raise

        if not new_samples:
            print("No new samples to add.")
            return True

        with h5py.File(output_file, 'a') as hf:
            all_sample_datasets = []
            for i, sample_id in enumerate(new_samples):
                if i % 100 == 0:
                    print(f"Processing sample {i+1}/{len(new_samples)}")

                sample_files = sorted(glob.glob(os.path.join(RESULT_DIR, f"res_{sample_id}_*.nc")))

                if len(sample_files) != FILES_PER_SAMPLE:
                    print(f"Warning: Sample {sample_id} has {len(sample_files)} files, expected {FILES_PER_SAMPLE}")
                    continue

                try:
                    sample_datasets = [xr.open_dataset(f) for f in sample_files]
                    sample_concat = xr.concat(sample_datasets, dim="time")
                    sample_concat = sample_concat.assign_coords(sample_id=sample_id)
                    sample_concat = sample_concat.expand_dims("sample_id")
                    all_sample_datasets.append(sample_concat)

                    for ds in sample_datasets:
                        ds.close()

                except Exception as e:
                    print(f"Error processing sample {sample_id}: {str(e)}")
                    continue

            if not all_sample_datasets:
                print("No datasets were successfully processed.")
                return False

            combined = xr.concat(all_sample_datasets, dim="sample_id")

            # Store the combined xarray dataset into the hdf5 file.
            for var_name, data_array in combined.data_vars.items():
                if var_name in hf:
                    # Append if the variable already exists
                    hf[var_name].resize((hf[var_name].shape[0] + combined.sample_id.size, *data_array.shape[1:]))
                    hf[var_name][-combined.sample_id.size:, ...] = data_array.values
                else:
                    # Create if the variable doesn't exist
                    hf.create_dataset(var_name, data=data_array.values)

            # Store the coordinates.
            for coord_name, coord_array in combined.coords.items():
                if coord_name in hf:
                    hf[coord_name].resize((hf[coord_name].shape[0] + combined.sample_id.size,))
                    hf[coord_name][-combined.sample_id.size:] = coord_array.values
                else:
                    hf.create_dataset(coord_name, data=coord_array.values)

            print(f"Successfully concatenated {len(new_samples)} samples into {output_file}")
            return True

    except Exception as e:
        print(f"Error concatenating samples: {str(e)}")
        return False

def main():
    """Main function to monitor and collect data."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    progress = check_progress()
    print_report(progress)

    if progress['completed']:
        response = input(f"Concatenate {len(progress['completed'])} completed samples? (y/n): ")
        if response.lower() in ('y', 'yes'):
            concatenate_all_samples_hdf5(progress['completed'], FINAL_OUTPUT)

if __name__ == "__main__":
    main()
