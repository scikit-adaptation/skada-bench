python benchmark_utils/generate_base_estim_config.py
benchopt run --config config/find_best_base_estimators_per_dataset.yml
python visualize/convert_benchopt_output_to_readable_csv.py --domain source --directory outputs --output results
python benchmark_utils/extract_best_base_estim.py
python benchmark_utils/generate_config_per_dataset.py