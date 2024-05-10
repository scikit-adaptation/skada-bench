import pandas as pd
import argparse


BENCHOPT_RUNNING_FORMAT = 'benchopt run -d "{}" -s {} --output "{}" {}'

def check_experiment_status(csv_file):
    df = pd.read_csv(csv_file)
    finished_experiments = df[df['Status'] == 'Finished']
    running_experiments = df[df['Status'] == 'Running']
    return finished_experiments, running_experiments

def generate_benchopt_commands(experiments, slurm_yaml=None):
    commands = []

    # Groupby per dataset
    for idx, exp in experiments.iterrows():
        output_filename = f"output_{exp['Dataset']}_{exp['Solver']}"
        slurm_option = f"--slurm {slurm_yaml}" if slurm_yaml else ""
        commands.append(
            BENCHOPT_RUNNING_FORMAT.format(exp['Dataset'], exp['Solver'], output_filename, slurm_option)
        )
    return commands


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Benchopt commands for running or finished experiments.")
    parser.add_argument("csv_file", help="Path to the CSV file containing experiment status information.")
    parser.add_argument("--slurm", default=None, help="Path to the slurm yaml file.")
    args = parser.parse_args()

    csv_file = args.csv_file
    finished, running = check_experiment_status(csv_file)

    print("Finished experiments:")
    for idx, exp in finished.iterrows():
        print(exp.tolist())

    print("\nRunning experiments:")
    for idx, exp in running.iterrows():
        print(exp.tolist())


    finished_commands = generate_benchopt_commands(finished, args.slurm)
    finished_bash_file = "../run_finished_exps.sh"
    print("\nBenchopt commands for finished experiments:")
    with open(finished_bash_file, "w") as file:

        file.write('echo "Running finished experiments"\n')
        for cmd in finished_commands:
            print(cmd)
            file.write(cmd + "\n")
        
        file.write('\n\necho "All commands executed successfully"\n')


    running_commands = generate_benchopt_commands(running, args.slurm)
    running_bash_file = '../run_unfinished_exps.sh'
    print("\nBenchopt commands for running experiments:")
    with open(running_bash_file, "w") as file:
        for cmd in running_commands:
            print(cmd)
            file.write(cmd + "\n")
