import numpy as np
import xarray as xr
import argparse
def generate_samples(n_samples, n_farms=9, dist_flag='uniform', lambda_years=2, output_file='samples.nc'):
    """
    Generate wind farm samples with uncertainties.
    Parameters:
    - n_samples: Total number of samples to generate.
    - n_farms: Number of farms per sample.
    - dist_flag: 'uniform' or 'exponential' for construction day sampling.
    - lambda_years: Scale parameter for the exponential distribution (in years).
    - output_file: Name of the output NetCDF file.
    Returns:
    - Saves the dataset to a NetCDF file.
    """
    # Seed for reproducibility
    np.random.seed(42)
    # Generate uncertainties
    rated_power = np.random.randint(10, 16, (n_samples, n_farms))  # MW
    rotor_diameter = 240 * np.sqrt(rated_power / 15)
    hub_height = rotor_diameter / 240 * 150
    ss_seed = np.random.randint(1, 100, (n_samples, n_farms))
    # Construction day sampling
    if dist_flag == 'uniform':
        construction_day = np.random.uniform(0, 3653, (n_samples, n_farms)).astype(int)  # 10 years in days
    elif dist_flag == 'exponential':
        construction_day = (np.random.exponential(scale=lambda_years * 365, size=(n_samples, n_farms))).astype(int)
    else:
        raise ValueError("Invalid dist_flag. Choose 'uniform' or 'exponential'.")
    # Create the dataset
    ds = xr.Dataset(
        {
            'rated_power': (['sample', 'farm'], rated_power),
            'rotor_diameter': (['sample', 'farm'], rotor_diameter),
            'hub_height': (['sample', 'farm'], hub_height),
            'construction_day': (['sample', 'farm'], construction_day),
            'ss_seed': (['sample', 'farm'], ss_seed),
        },
        coords={
            'sample': np.arange(n_samples),
            'farm': np.arange(n_farms),
        }
    )
    # Save to NetCDF
    ds.to_netcdf(output_file)
    print(f"Dataset saved to: {output_file}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate wind farm samples with uncertainties.")
    parser.add_argument("--n_samples", type=int, default=5000, help="Total number of samples to generate.")
    parser.add_argument("--dist_flag", type=str, default="uniform", choices=["uniform", "exponential"],
                        help="Distribution for construction day: 'uniform' or 'exponential'.")
    parser.add_argument("--lambda_years", type=float, default=2,
                        help="Scale parameter for exponential distribution (in years).")
    parser.add_argument("--output_file", type=str, default="samples.nc", help="Name of the output NetCDF file.")
    args = parser.parse_args()
    generate_samples(
        n_samples=args.n_samples,
        dist_flag=args.dist_flag,
        lambda_years=args.lambda_years,
        output_file=args.output_file
    )
