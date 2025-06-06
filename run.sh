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
#conEnv
eval "$(pixi shell-hook)"


python nn_train.py \
    --train_data_path "/work/users/juqu/Neighboring-Wind-Farms-Out-Of-Distribution-Surrogate-Modeling/final_datasets_for_modeling/train_data.npz" \
    --valid_data_path "/work/users/juqu/Neighboring-Wind-Farms-Out-Of-Distribution-Surrogate-Modeling/final_datasets_for_modeling/valid_data.npz" \
    --base_project_dir "/work/users/juqu/Neighboring-Wind-Farms-Out-Of-Distribution-Surrogate-Modeling" \
    --output_dir_name "model_optimization_results" \
    --data_context_name "Exp_1yr_Trained" \
    --model_variant_name "FFNN_ArchSearch_Initial" \
    --n_trials 100 \
    --cv_folds 5 \
    --optuna_jobs 1 \
    --max_epochs 400 \
    --patience 25 \
    --epoch_print_freq 20 \
    --min_layers 2 \
    --max_layers 6 \
    --min_initial_neurons 32 \
    --max_initial_neurons 256 \
    --neuron_step 16 \
    --min_final_neurons 8 \
    --max_final_neurons 128 \
    --abs_min_neurons_per_layer 8

#conda activate niwe
#python preprocess3.py 
#python preprocess_h5.py uniform_results
#python preprocess_h5.py exponential_2yr_results
#python collect4.py 1
#python evaluate_samples.py --start 115 --end 200 --output exponential_1yr_results --input exponential_1yr_samples.nc
#evaluate_samples --start 1  --end 5000
#python precompute_farm_layouts.py --seeds 50 --processes 30 --output re_precomputed_layouts.h5
#python generate_samps.py --dist_flag uniform --output_file uniform.nc
#python generate_samps.py --dist_flag exponential --lambda_years 1 --output_file exponential_1yr.nc
#python generate_samps.py --dist_flag exponential --lambda_years 2 --output_file exponential_2yr.nc
#python preprocess_evaluations.py

