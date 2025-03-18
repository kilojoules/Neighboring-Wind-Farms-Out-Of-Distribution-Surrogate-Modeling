welcome to the repo!

FILE                         PURPOSE
++++                         +++++++
run.sh                       run data generation scripts

generate_samps.py            generate wind farm simulation samples given input distributions
precompute_farm_layouts.py   precompute a database of smart start optimization results
preprocess_evaluations.py    compute 1 or 10 year revenue (with constant electricity price), normalize inputs/outputs

xgbs.sh                      run xgboost sweep for several assumptions (discount rate, 1 versus 10 years revenue)
xgboost_sweep.py             xgboost sweep with hyperparameter grid definition
