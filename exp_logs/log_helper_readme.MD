## Experiment Log Helper

### Goal:

The purpose of this script is to facilitate the logging of running and finished experiments, in the context of utilizing the Benchopt framework. The main objectives are:

1. Log the dataset/solver combination that is either running or finished.
2. Provide a helper function to parse the log file and generate Benchopt commands to retrieve results from the cache and relaunch unfinished experiments.

### Usage:

1. Run your experiment using Benchopt, it will generate a csv file in the `./exp_logs` folder.

2. Run the Python script.

```python
python experiment_log_analyzer.py
```

3. The script will output the status of finished and running experiments, along with the corresponding Benchopt commands for both scenarios.

4. The generated Benchopt commands will be written to separate Bash scripts for executing finished and unfinished experiments. These 2 batch files can be found in `../` i.e in the `skada-bench` folder.

### Example:

```bash
# To retrieve finished experiments
./run_finished_exps.sh

# To relaunch unfinished experiments
./run_unfinished_exps.sh
```

### Folder Details:
- `experiment_log.csv`: CSV file containing experiment status information.
- `log_helper.py`: Contains the main script that will generate the bash files.
- `../run_finished_exps.sh`: Bash script to execute finished experiments.
- `../run_unfinished_exps.sh`: Bash script to retrieve unfinished experiments.
