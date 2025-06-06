welcome to the repo!

FILE                         PURPOSE
++++                         +++++++
run.sh                       run data generation scripts

- generate_samps.py            generate wind farm simulation samples given input distributions
- precompute_farm_layouts.py   precompute a database of smart start optimization results
- evaluate_sample.py & evaluate_samples.py: These are the core simulation scripts. evaluate_sample.py processes a single sample , while evaluate_samples.py processes a range of samples by calling it repeatedl
- collect.py: This script monitors the output directories from the simulation st
- preprocess.py: This script converts the raw, time-series simulation data into a feature matrix (X) and a target vector (Y).
- normalize_all_datasets.py: This script standardizes the features and target variable.
- data_manager.py: This script creates the final data splits for modeling. 
