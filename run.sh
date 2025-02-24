#!/bin/bash
#####3#SBATCH --output="/work/users/ernim/niwecasestudy/logs/%j.log"
########SBATCH --error="/work/users/ernim/niwecasestudy/logs/%j.err"
#SBATCH --partition=windfatq
#SBATCH --job-name="neighbors"
#SBATCH --time=2-00:00:00
#SBATCH --ntasks-per-core 1
#SBATCH --ntasks-per-node 32
#SBATCH --nodes=1
#SBATCH --exclusive

. ~/.bashrc
conEnv
conda activate niwe
python generate_samps.py --dist_flag uniform --output_file uniform.nc
python generate_samps.py --dist_flag exponential --lambda_years 1 --output_file exponential_1yr.nc
python generate_samps.py --dist_flag exponential --lambda_years 2 --output_file exponential_2yr.nc
python preprocess_evaluations.py

