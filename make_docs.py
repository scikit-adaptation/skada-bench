import argparse
import subprocess
import os
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_command(command, description):
    """Execute a command and log its output."""
    logging.info(f"Starting: {description}")
    try:
        result = subprocess.run(
            command,
            check=True,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logging.info(f"Successfully completed: {description}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error during {description}:")
        logging.error(f"Command output: {e.output}")
        logging.error(f"Command stderr: {e.stderr}")
        return False

def generate_figures_and_tables(
        deep_real_csv_folder,
        output_folder,
        shallow_real_csv_folder,
        shallow_simulated_csv_folder,
    ):
    """Generate all required figures and tables."""
    commands = [
        # Deep learning results unsupervised
        f"python visualize/plot_results_all_datasets_deep.py --csv-folder {deep_real_csv_folder} "
        "--scorer-selection unsupervised --score accuracy --output-format markdown "
        f"--output-folder {output_folder}",

        # Deep learning results supervised
        f"python visualize/plot_results_all_datasets_deep.py --csv-folder {deep_real_csv_folder} "
        "--scorer-selection supervised --score accuracy --output-format markdown "
        f"--output-folder {output_folder}",

        # Shallow results unsupervised
        f"python visualize/plot_results_all_datasets.py --csv-folder {shallow_real_csv_folder} "
        f"--csv-folder-simulated {shallow_simulated_csv_folder} --scorer-selection unsupervised "
        f"--score accuracy --output-format markdown --output-folder {output_folder}",

        # Shallow results supervised
        f"python visualize/plot_results_all_datasets.py --csv-folder {shallow_real_csv_folder} "
        f"--csv-folder-simulated {shallow_simulated_csv_folder} --scorer-selection supervised "
        f"--score accuracy --output-format markdown --output-folder {output_folder}",
        
        # f"python visualize/plot_results_all_datasets_deep.py --csv-folder {csv_folder} "
        # "--scorer-selection supervised --score accuracy --output-format latex",
        
        # # All datasets results
        # f"python visualize/plot_results_all_datasets.py --csv-file {csv_file_real} "
        # f"--csv-file-simulated {csv_file_simulated} --scorer-selection unsupervised "
        # "--output-format latex",
        
        # # Inner score vs accuracy scatter plot
        # f"python visualize/plot_inner_score_vs_acc.py --csv-file {csv_file_real}",
        
        # # Boxplot generation
        # f"python visualize/plot_boxplot.py --csv-file {csv_file_real}",
        
        # # Supervised vs unsupervised comparison
        # f"python visualize/plot_supervised_vs_unsupervised.py --csv-file {csv_file_real}",
        
        # # Per-dataset results
        # f"python visualize/plot_results_per_dataset.py --dataset simulated "
        # f"--csv-file {csv_file_real} --csv-file-simulated {csv_file_simulated}"
    ]
    
    for cmd in commands:
        if not run_command(cmd, "Figure/table generation"):
            return False
    return True

def build_sphinx_docs():
    """Build Sphinx documentation."""
    return run_command(
        "sphinx-build -M html docs/source/ docs/build/",
        "Sphinx documentation build"
    )

def main():
    parser = argparse.ArgumentParser(
        description="Generate figures, tables and build documentation"
    )
    parser.add_argument(
        "--deep-real-csv-folder",
        type=str,
        help="Path to the csv folder containing results for deep real data",
        default="./outputs/deep_results/real"
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        help="Path to the output folder",
        default="./docs/source/experiment_results"
    )
    parser.add_argument(
        "--shallow-real-csv-folder",
        type=str,
        help="Path to the csv folder containing results for shallow real data",
        default="./outputs/shallow_results/real"
    )
    parser.add_argument(
        "--shallow-simulated-csv-folder",
        type=str,
        help="Path to the csv folder containing results for shallow simulated data",
        default="./outputs/shallow_results/simulated"
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip figure and table generation"
    )
    args = parser.parse_args()

    # Ensure input paths exist
    for path in [args.deep_real_csv_folder, args.output_folder, args.shallow_real_csv_folder, args.shallow_simulated_csv_folder]:
    #for path in [args.csv_folder, args.csv_file_real, args.csv_file_simulated]:
        if not os.path.exists(path):
            logging.error(f"Path does not exist: {path}")
            return False

    # Create output directories if they don't exist
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)

    success = True
    if not args.skip_figures:
        success = generate_figures_and_tables(
            args.deep_real_csv_folder,
            args.output_folder,
            args.shallow_real_csv_folder,
            args.shallow_simulated_csv_folder,
        )
        if not success:
            logging.error("Failed to generate figures and tables")
            return False

    if not build_sphinx_docs():
        logging.error("Failed to build Sphinx documentation")
        return False

    logging.info("Documentation build completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)