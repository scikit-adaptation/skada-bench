# DA-Bench: Benchmarking Unsupervised Domain Adaptation Methods with Realistic Validation
Welcome to the official implementation of [DA-Bench: Benchmarking Unsupervised Domain Adaptation Methods with Realistic Validation](https://arxiv.org/abs).

To reproduce the results in this paper, you have three options:
- **Visualize stored results (Step 3):** with results in `/visualize/cleaned_outputs/`.
- **Run bench (Step 2):** with config files in `/config/datasets`.
- **Start from Scratch (Step 1):** Generate new config files and proceed from there.

We provided all the necessary files for running each step without the need for
running the previous one.

If you want to add new solvers, dataset or scorers just follow the instructions in the
CONTRIBUTE.md file.

## Requirements

To install the necessary requirements, run the following commands:

```bash
pip install -r requirements.txt              # Install dependencies
```

## Running the Benchmark

### Step 1: Finding the Best Base Estimators for Each Dataset

#### 1.1 Generate the Configuration File

Generate the config file for selecting base estimator on source:

```bash
python benchmark_utils/generate_base_estim_config.py
```

This generates `config/find_best_base_estimators_per_dataset.yml`.

#### 1.2 Run base estimator selection bench

Run base estimator experiments and store the results:

```bash
benchopt run --config config/find_best_base_estimators_per_dataset.yml --output results_base_estimators --no-plot --no-html
```

This generates `outputs/results_base_estimators`.

#### 1.3 Extract the results

Extract the results and store them in a CSV file `results_base_estimators/`:

```bash
python visualize/convert_benchopt_output_to_readable_csv.py --domain source --directory outputs --output results_base_estimators --file_name results_base_estim_experiments
```

This generates `results_base_estimators/results_base_estim_experiments.csv`.

#### 1.4 Find the best base estimators

Find the best base estimator per dataset and store them in `config/best_base_estimators.yml`:

```bash
python benchmark_utils/extract_best_base_estim.py
```

This generates `config/best_base_estimators.yml`.

#### 1.5 Generate Configurations with Best Base Estimators

Update the config file per dataset with the best base estimator:

```bash
python benchmark_utils/generate_config_per_dataset.py
```

This generates a config file for each dataset in `config/datasets/`.

### Step 2: Launch benchmark for each Dataset

To launch the benchmark for each dataset, run the following command:

```bash
benchopt run --config dataset.yml --timeout 3h --output output_dataset --no-plot --no-html
```

- `dataset.yml`: Config file of the specified dataset.
- `output_dataset`: Name of the output result parquet/csv.

#### Example: Simulated Dataset

```bash
benchopt run --config config/datasets/Simulated.yml --timeout 3h --output output_simulated --no-plot --no-html
```

> **Note:** In the paper results, the timeout was set to 3 hours.
> The `benchopt` framework supports running benchmarks in parallel on a SLURM cluster. For more details, refer to the [Benchopt user guide](https://benchopt.github.io/user_guide/advanced.html).

## Step 3: Displaying Results

Convert the `benchopt` output into a CSV format:

```bash
python visualize/convert_benchopt_output_to_readable_csv.py --directory outputs --domain target --file_name output_readable_dataset
```

This generates `visualize/cleaned_outputs/output_readable_dataset.csv`. This csv file can then be used by anyone to plot the benchmarking results.

### Visualization Commands

In the `visualize` folder, run the following commands to generate various results and plots:

- **Main Result Table:**
  ```bash
  python plot_results_all_datasets.py --csv-file cleaned_outputs/results_real_datasets_experiments.csv --csv-file-simulated cleaned_outputs/results_simulated_datasets_experiments.csv
  ```
- **Individual Tables per Dataset:**
  ```bash
  python plot_results_per_dataset.py --csv-file cleaned_outputs/results_real_datasets_experiments.csv --dataset BCI
  ```
- **Cross-val Score vs. Accuracy for Different Scorers:**
  ```bash
  python plot_inner_score_vs_acc.py --csv-file cleaned_outputs/results_real_datasets_experiments.csv
  ```
- **Accuracy of DA Methods using Unsupervised Scorers vs. Supervised Scorers:**
  ```bash
  python plot_supervised_vs_unsupervised.py --csv-file cleaned_outputs/results_real_datasets_experiments.csv
  ```
- **Change in Accuracy of DA Methods with Best Unsupervised Scorer vs. Supervised Scorer:**
  ```bash
  python plot_boxplot.py  --csv-file cleaned_outputs/results_real_datasets_experiments.csv
  ```
- **Mean Computing Time for Training and Testing Each Method:**
  ```bash
  python visualize/get_computational_time.py --directory outputs
  ```

All the generated tables and plots can be found in the `visualize` folder.

> **Note:** For the `get_computational_time` script, you need to give directly
> benchopt outputs which are not provided due to size limits (all other results
> are provided).


Happy benchmarking!
