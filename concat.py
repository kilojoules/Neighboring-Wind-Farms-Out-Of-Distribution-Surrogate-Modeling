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

#Modify the monitor function to use the new hdf5 function.
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
                if os.path.exists(FINAL_OUTPUT.replace(".nc", ".h5")):
                    print(f"\nUpdating concatenated file with {progress['total_completed'] - last_concat_count} new samples...")
                else:
                    print("\nCreating initial concatenated file...")

                success = concatenate_all_samples_hdf5(progress['completed'], FINAL_OUTPUT.replace(".nc", ".h5"))
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

#modify the main function.
def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        # Just check once
        progress = check_progress()
        print_report(progress)

        if progress['completed']:
            response = input(f"Concatenate {len(progress['completed'])} completed samples? (y/n): ")
            if response.lower() in ('y', 'yes'):
                concatenate_all_samples_hdf5(progress['completed'], FINAL_OUTPUT.replace(".nc", ".h5"))
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
