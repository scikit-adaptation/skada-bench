# Set the config file for base estimator experiments for No DA Source Only
python benchmark_utils/generate_base_estim_config.py

# Run base estimator experiments for No DA Source Only. Store the results in `results_base_estimators/`
benchopt run --config config/find_best_base_estimators_per_dataset.yml --no-plot --no-html

# Clean the results and store them in `results_base_estimators/results_base_estim_experiments.csv`
python visualize/convert_benchopt_output_to_readable_csv.py --domain source --directory outputs --output results_base_estimators --file_name results_base_estim_experiments

# Find the best base estimator and best SVC per dataset and store them in `config/best_base_estimators.yml`
python benchmark_utils/extract_best_base_estim.py

# Update the config file per dataset with the best base estimator as final estimator
python benchmark_utils/generate_config_per_dataset.py