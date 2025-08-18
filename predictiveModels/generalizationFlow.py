# Import required libraries and functions
import pandas as pd
import numpy as np
from pathlib import Path

# Import functions from individual scripts
from smoteDataGeneration import main as smote_main
from gaussianDataGeneration import main as copula_main
from ctganDataGeneration import main as ctgan_main
from modelTesting import evaluate_models

# Set up working directories
BASE_EXECUTIONS_DIR = Path(__file__).resolve().parent.parent / "executions"
BASE_EXECUTIONS_DIR.mkdir(exist_ok=True)

# Set the number of runs desired
TARGET_RUNS = 45

# Set false if only the average results of current saved runs is required, true if new executions are wanted
EXECUTE_NEW_RUNS = True

# Function to read all runs executed and saved in base executions folder
def get_existing_runs():
    runs = [d for d in BASE_EXECUTIONS_DIR.iterdir() if d.is_dir() and d.name.startswith("run_")]
    return sorted(runs)

# Function to get the next execution index, based on the existing last run
def next_run_index():
    runs = get_existing_runs()
    if not runs:
        return 1
    last = max([int(run.name.split("_")[1]) for run in runs])
    return last + 1

# Set up folders/directories to save new results
def create_run_directories(run_name):
    run_dir = BASE_EXECUTIONS_DIR / run_name

    # Create folders for each technique
    for technique in ["real", "smote", "copula", "ctgan"]:
        (run_dir / technique / "datasets").mkdir(parents=True, exist_ok=True)
        (run_dir / technique / "plots").mkdir(parents=True, exist_ok=True)
        (run_dir / technique / "generationRankings").mkdir(parents=True, exist_ok=True)
        (run_dir / technique / "metrics").mkdir(parents=True, exist_ok=True)
        (run_dir / technique / "confusion").mkdir(parents=True, exist_ok=True)

    return run_dir

# Function to run the designed pipeline for the specified splitting seed
def run_pipeline_for_seed(seed):
    run_name = f"run_{seed:03d}"
    run_dir = create_run_directories(run_name)

    # Generate each synthetic dataset for the current split
    smote_main(random_state=seed, output_dir=run_dir / "smote")
    copula_main(random_state=seed, output_dir=run_dir / "copula")
    ctgan_main(random_state=seed, output_dir=run_dir / "ctgan")

    # Evaluate the four models, each one with a different training dataset and the same testing one
    evaluate_models(output_dir=run_dir, random_state=seed)

# Aggregate confusion matrices and metrics across runs (mean and standard deviation)
def aggregate_results():
    summary = {}
    
    # Get existing runs
    runs = get_existing_runs()
    if not runs:
        print("There are no previous executions to aggregate.")
        return

    # Get metrics and confusion matrices of each technique
    for technique in ["real", "smote", "copula", "ctgan"]:
        metrics_list = []
        confusion_list = []

        # Explore every run existing for each technique
        for run in runs:
            run_path = run / technique

            # Get metrics
            metrics_file = run_path / "metrics" / "metrics.csv"
            if metrics_file.exists():
                df_metrics = pd.read_csv(metrics_file)
                metrics_list.append(df_metrics.iloc[0].to_dict())
            
            # Get confusion matrix
            confusion_file = run_path / "confusion" / "confusion_matrix.csv"
            if confusion_file.exists():
                df_confusion = pd.read_csv(confusion_file)
                confusion_list.append(df_confusion.values)

        # Build the average results
        if metrics_list:
            df_all = pd.DataFrame(metrics_list)
            mean_metrics = df_all.mean().to_dict()
            std_metrics = df_all.std().to_dict()
            mean_confusion = np.mean(confusion_list, axis=0) if confusion_list else None
            summary[technique] = {
                "mean_metrics": mean_metrics,
                "std_metrics": std_metrics,
                "mean_confusion": mean_confusion
            }

            # Show the mean and standard deviation of each metric
            print(f"\n🔷 Aggregated metrics for {technique.upper()} (through {len(metrics_list)} runs)")
            for k, v in mean_metrics.items():
                print(f" - {k}: {v:.4f} (±{std_metrics[k]:.4f})")

            # Show the mean confusion matrix
            if mean_confusion is not None:
                print("\n🔷 Average confusion matrix:")
                print(np.round(mean_confusion, 2))
        else:
            print(f"\n🔷 Metrics not found for {technique.upper()}")

    # Save summary of results
    summary_dir = BASE_EXECUTIONS_DIR / "summary"
    summary_dir.mkdir(exist_ok=True)
    for technique, data in summary.items():
        metrics_df = pd.DataFrame([data["mean_metrics"], data["std_metrics"]], index=["mean", "std"])
        metrics_df.to_csv(summary_dir / f"{technique}_summary_metrics.csv")

        if data["mean_confusion"] is not None:
            pd.DataFrame(data["mean_confusion"]).to_csv(summary_dir / f"{technique}_summary_confusion.csv", index=False)

# Main execution
if __name__ == "__main__":
    
    # Calculate aggregated results
    aggregate_results()

    # If desired, start executing new runs
    if EXECUTE_NEW_RUNS:
        start_index = next_run_index()

        # Target runs is the limit
        for seed in range(start_index, TARGET_RUNS + 1):
            print(f"\n🔷 Executing pipeline for random_state={seed}")
            run_pipeline_for_seed(seed)


