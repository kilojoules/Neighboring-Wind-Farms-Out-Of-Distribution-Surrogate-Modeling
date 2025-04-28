#!/usr/bin/env python3

import os
import glob
import sys
import time
from collections import defaultdict
import h5py
import xarray as xr
import numpy as np

# --- Configuration ---
RESULT_DIR = "/work/users/juqu/Neighboring-Wind-Farms-Out-Of-Distribution-Surrogate-Modeling/results/wake"
OUTPUT_DIR = "/work/users/juqu/Neighboring-Wind-Farms-Out-Of-Distribution-Surrogate-Modeling/results/concatenated"
FINAL_OUTPUT = os.path.join(OUTPUT_DIR, "all_samples.h5")  # HDF5 output
TOTAL_SAMPLES = 5000
FILES_PER_SAMPLE = 10
# --- End Configuration ---

def find_progress(result_dir, total_samples, files_per_sample):
    """
    Scans the result directory to find completed, incomplete, and missing samples.
    """
    print(f"DEBUG: Scanning directory: {result_dir}")
    all_files = glob.glob(os.path.join(result_dir, "res_*_*.nc"))
    print(f"DEBUG: Found {len(all_files)} total .nc files.")

    sample_counts = defaultdict(int)
    processed_files = 0
    skipped_files = 0
    for f in all_files:
        basename = os.path.basename(f)
        parts = basename.split('_')
        # Expected format: res_{sample_id}_{anything_else}.nc
        if len(parts) >= 3 and parts[0] == 'res' and parts[1].isdigit():
            try:
                sample_id = int(parts[1])
                sample_counts[sample_id] += 1
                processed_files += 1
            except ValueError:
                print(f"DEBUG: Skipping file with non-integer sample ID: {basename}")
                skipped_files += 1
        else:
            print(f"DEBUG: Skipping file with unexpected name format: {basename}")
            skipped_files += 1

    print(f"DEBUG: Parsed {processed_files} files, skipped {skipped_files} files.")

    completed_ids = []
    incomplete_info = []
    all_found_ids = set(sample_counts.keys())

    # Use range starting from 1 up to TOTAL_SAMPLES + 1 for sample IDs
    expected_ids = set(range(1, total_samples + 1))

    for sample_id in sorted(all_found_ids):
         count = sample_counts[sample_id]
         if count >= files_per_sample - 1:
             completed_ids.append(sample_id)
         else:
             incomplete_info.append((sample_id, count))

    missing_ids = sorted(list(expected_ids - all_found_ids))
    # Ensure completed/incomplete lists are also sorted if needed (already are here)
    # completed_ids.sort() # Already sorted if iterating through sorted keys
    # incomplete_info.sort() # Already sorted if iterating through sorted keys

    total_completed = len(completed_ids)
    percent_complete = (total_completed / total_samples) * 100 if total_samples > 0 else 0

    progress = {
        "completed": completed_ids,
        "incomplete": incomplete_info,
        "missing": missing_ids,
        "total_completed": total_completed,
        "total_incomplete": len(incomplete_info),
        "total_missing": len(missing_ids),
        "percent_complete": percent_complete
    }
    print(f"DEBUG: Progress check complete. Found {total_completed} completed samples.")
    return progress

def print_progress_report(progress, total_samples, files_per_sample):
    """Prints a formatted progress report."""
    print("\n" + "=" * 50)
    print(f"PROGRESS REPORT - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    print(f"Target Samples: {total_samples}")
    print(f"Files per Sample: {files_per_sample}")
    print("-" * 50)
    print(f"Completed: {progress['total_completed']} / {total_samples} samples ({progress['percent_complete']:.1f}%)")
    print(f"Incomplete: {progress['total_incomplete']} samples")
    print(f"Missing: {progress['total_missing']} samples (expected IDs not found at all)")

    if progress['completed']:
        sample_str = str(progress['completed'][:10])
        if len(progress['completed']) > 10:
            sample_str = sample_str[:-1] + ", ...]"
        print(f"\nCompleted sample IDs: {sample_str}")

    if progress['incomplete']:
        print("\nIncomplete samples (sample_id, files_count):")
        for i, (sample_id, count) in enumerate(progress['incomplete'][:5]):
            print(f"  Sample {sample_id}: {count}/{files_per_sample} files")
        if len(progress['incomplete']) > 5:
            print(f"  ... and {len(progress['incomplete']) - 5} more")

    if progress['missing'] and len(progress['missing']) <= 10: # Only show missing if few
         print(f"\nMissing sample IDs: {progress['missing']}")
    elif progress['missing']:
         print(f"\nMissing sample IDs: {progress['missing'][:5]} ... and {len(progress['missing'])-5} more")


    print("=" * 50)

def delete_incomplete_files(incomplete_list, result_dir):
    deleted_files_count = 0
    deleted_samples_count = 0
    print(f"Scanning for files to delete in: {result_dir}")

    for sample_id, file_count in incomplete_list:
        # Construct the file pattern for this incomplete sample ID
        file_pattern = os.path.join(result_dir, f"res_{sample_id}_*.nc")
        # Find all files matching the pattern
        files_to_delete = glob.glob(file_pattern)

        if not files_to_delete:
            print(f"  - Sample {sample_id}: No files found matching pattern (unexpected). Skipping.")
            continue

        print(f"  - Sample {sample_id}: Found {len(files_to_delete)} files to delete.")
        deleted_samples_count += 1
        for filepath in files_to_delete:
            try:
                os.remove(filepath)
                # print(f"    Deleted: {os.path.basename(filepath)}") # Uncomment for verbose output
                deleted_files_count += 1
            except OSError as e:
                print(f"    ERROR deleting {os.path.basename(filepath)}: {e}")

    print(f"\nDeletion Summary:")
    print(f"  - Attempted to delete files for {deleted_samples_count} incomplete samples.")
    print(f"  - Successfully deleted {deleted_files_count} files.")


def concatenate_to_hdf5(completed_ids, result_dir, output_hdf5_file, files_per_sample):
    """
    Concatenates completed samples into an HDF5 file using separate groups
    for each sample to handle potentially varying dimensions.
    """
    print(f"\nDEBUG: Starting concatenation process for {len(completed_ids)} completed IDs.")
    print(f"DEBUG: Output file target: {output_hdf5_file}")
    print(f"DEBUG: Using separate HDF5 groups per sample.")

    if not completed_ids:
        print("No completed samples provided to concatenate.")
        return

    samples_to_process = []
    existing_groups = set() # Store names of existing groups (e.g., 'sample_1')

    if os.path.exists(output_hdf5_file):
        print(f"DEBUG: Output file '{output_hdf5_file}' exists. Checking for existing sample groups (Hot Start).")
        try:
            with h5py.File(output_hdf5_file, 'r') as hf:
                # List existing groups that look like sample groups
                for name in hf:
                    if isinstance(hf[name], h5py.Group) and name.startswith('sample_'):
                         existing_groups.add(name)
            print(f"DEBUG: Found {len(existing_groups)} existing sample groups in the HDF5 file.")
        except OSError as e:
             print(f"ERROR: Could not read existing HDF5 file '{output_hdf5_file}': {e}")
             print("Suggestion: Delete the existing file and rerun to recreate it.")
             return # Stop concatenation if file is unreadable

        # Determine which samples actually need processing
        samples_to_process = []
        for sid in completed_ids:
            group_name = f"sample_{sid}"
            if group_name not in existing_groups:
                samples_to_process.append(sid)

        samples_to_process.sort() # Keep order consistent
        print(f"DEBUG: Comparing {len(completed_ids)} completed IDs with {len(existing_groups)} existing groups.")
        print(f"DEBUG: Found {len(samples_to_process)} new samples to add as groups.")

    else:
        print(f"DEBUG: Output file '{output_hdf5_file}' does not exist. Creating new file.")
        samples_to_process = sorted(list(completed_ids)) # Process all completed if file is new

    if not samples_to_process:
        print("All completed samples are already present as groups in the HDF5 file. No new samples to add.")
        return

    print(f"DEBUG: Will process and add {len(samples_to_process)} samples as new groups.")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_hdf5_file), exist_ok=True)

    # Use a context manager for the HDF5 file, always open in append mode ('a')
    # as 'w' would delete existing groups even if we only wanted to add new ones.
    # If the file doesn't exist, 'a' will create it.
    try:
        with h5py.File(output_hdf5_file, 'a') as hf:
            for i, sample_id in enumerate(samples_to_process):
                group_name = f"sample_{sample_id}"
                print(f"Processing sample {i+1}/{len(samples_to_process)} (ID: {sample_id}) -> Group: '{group_name}'")

                # Check if group somehow already exists (shouldn't based on logic above, but safe)
                if group_name in hf:
                    print(f"WARNING: Group '{group_name}' already exists. Skipping sample {sample_id}.")
                    continue

                # Find files for this specific sample
                sample_file_pattern = os.path.join(result_dir, f"res_{sample_id}_*.nc")
                sample_files = sorted(glob.glob(sample_file_pattern))

                if len(sample_files) < files_per_sample - 1:
                    print(f"WARNING: Sample {sample_id}: Expected {files_per_sample} files, found {len(sample_files)}. Skipping.")
                    continue

                # Load and concatenate data for this single sample using xarray
                try:
                    datasets_for_sample = [xr.open_dataset(f) for f in sample_files]
                    # Combine along time dimension
                    sample_combined = xr.concat(datasets_for_sample, dim="time")
                    # Close the individual files
                    for ds in datasets_for_sample:
                        ds.close()

                    # No need to add sample_id as coordinate or expand dims here,
                    # the group name itself identifies the sample.

                except Exception as e:
                    print(f"ERROR: Failed to load or concatenate files for sample {sample_id}: {e}. Skipping.")
                    continue

                # --- Write data to a new group in HDF5 ---
                try:
                    sample_group = hf.create_group(group_name)
                    print(f"DEBUG: Created group '{group_name}'")

                    # Process data variables
                    for var_name, data_array in sample_combined.data_vars.items():
                        print(f"DEBUG: Writing variable '{var_name}' with shape {data_array.shape}")
                        sample_group.create_dataset(var_name, data=data_array.values, chunks=True, compression="gzip") # Added compression

                    # Process coordinates
                    for coord_name, coord_array in sample_combined.coords.items():
                         print(f"DEBUG: Writing coordinate '{coord_name}' with shape {coord_array.shape}")
                         sample_group.create_dataset(coord_name, data=coord_array.values) # No need for chunks/maxshape here usually

                except Exception as e:
                    print(f"ERROR: Failed writing data to group '{group_name}' for sample {sample_id}: {e}")
                    # Optional: Attempt to delete the potentially incomplete group?
                    if group_name in hf:
                         del hf[group_name]
                         print(f"DEBUG: Removed potentially incomplete group '{group_name}'")
                    continue # Skip to next sample

        print(f"\nSuccessfully processed {len(samples_to_process)} samples into {output_hdf5_file}")

    except Exception as e:
        # Catch broader errors during file operations or processing
        print(f"\nAn unexpected error occurred during HDF5 file operation: {e}")
        print("The HDF5 file might be incomplete or corrupted.")



def main():
    """Main function to run the progress check and concatenation."""
    print("--- Starting Script ---")
    if not os.path.isdir(RESULT_DIR):
        print(f"ERROR: Result directory not found: {RESULT_DIR}")
        sys.exit(1)

    # 1. Check Progress
    progress_data = find_progress(RESULT_DIR, TOTAL_SAMPLES, FILES_PER_SAMPLE)

    # 2. Print Report
    print_progress_report(progress_data, TOTAL_SAMPLES, FILES_PER_SAMPLE)

    # (Inside main function, after print_progress_report)
    if progress_data['incomplete']:
        print(f"\nWARNING: Found {progress_data['total_incomplete']} incomplete samples.")
        try:
            # Use input() to ask for confirmation
            confirm = input("Do you want to delete all .nc files for these incomplete samples? (yes/no): ").lower()
        except EOFError:
            confirm = 'no' # Default to no if input is redirected
    
        if confirm == 'yes':
            print("Proceeding with deletion...")
            # Call a new function or add logic here to delete
            delete_incomplete_files(progress_data['incomplete'], RESULT_DIR)
        else:
            print("Deletion skipped.")

    # 3. Ask to Concatenate
    if progress_data['completed']:
        try:
            response = 'y'
            #response = input(f"Concatenate {progress_data['total_completed']} completed samples? (y/n): ").lower()
        except EOFError: # Handle cases where input is piped or unavailable
             response = 'n'
             print("\nNo input detected, skipping concatenation.")

        if response in ('y', 'yes'):
            concatenate_to_hdf5(
                progress_data['completed'],
                RESULT_DIR,
                FINAL_OUTPUT,
                FILES_PER_SAMPLE
            )
        else:
            print("Concatenation skipped by user.")
    else:
        print("No completed samples found to concatenate.")

    print("--- Script Finished ---")

if __name__ == "__main__":
    main()
