# Import required libraries and functions
import pandas as pd
import time
import sys
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from itertools import product

# Import the auxiliary utility functions
from dataGeneration.utilsDataGeneration import (
    load_and_preprocess_dataset,
    plot_histograms,
    plot_correlation_matrices,
    plot_pca_comparison,
    compute_jsd_for_columns,
    correlation_difference,
    centroid_distance_pca
)

# Configure dataset path and preprocessing parameters
FILE_PATH = ROOT_DIR / "data" / "ai4i2020.xlsx"
TARGET = "Machine failure"
DROP_COLS = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
CATEGORICAL = ["Type"]
NUMERIC_COLS = ["Air temperature [K]", "Process temperature [K]",
                "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]

# Function to apply SMOTE for synthetic data generation
def generate_smote_data(X, y, feature_columns, target_column, s_strategy, k, random_state):
    
    # Initialize SMOTE with the chosen parameters
    smote = SMOTE(sampling_strategy=s_strategy, k_neighbors=k, random_state=random_state)
    
    # Generate synthetic minority samples
    X_synth, y_synth = smote.fit_resample(X, y)
    
    # Build the final dataframe
    df_smote = pd.DataFrame(X_synth, columns=feature_columns)
    df_smote[target_column] = y_synth
    return df_smote


# Function to evaluate the quality of datasets generated with multiple parameter combinations
def best_smote_data(X, y, feature_columns, target_column, sampling_strategies, k_neighbors_list, random_state=42):
    
    results = []
    
    # Build the real dataset (features + target) used for comparison
    df_real = X.copy()
    df_real[target_column] = y.values

    # Generate all combinations of sampling strategies and k-neighbors
    combinations = list(product(sampling_strategies, k_neighbors_list))
    for sampling_strategy, k in combinations:
        try:
            # Generate synthetic data using the current combination
            df_synth = generate_smote_data(X, y, feature_columns, target_column, sampling_strategy, k, random_state)
            
            # Compute divergence and correlation metrics
            jsd_vals = compute_jsd_for_columns(df_real, df_synth, feature_columns)
            jsd_mean = sum(jsd_vals.values()) / len(jsd_vals)
            corr_diff = correlation_difference(df_real, df_synth, feature_columns)
            frob = corr_diff["Frobenius"]
            pca_dist = centroid_distance_pca(df_real, df_synth, feature_columns)

            # Store the results
            results.append({
                "sampling_strategy": sampling_strategy,
                "k_neighbors": k,
                "jsd_mean": jsd_mean,
                "frobenius_corr": frob,
                "pca_centroid_dist": pca_dist
            })

        # Skip combinations that are not compatible
        except Exception:
            continue

    # Return metrics dataframe
    return pd.DataFrame(results)


# Main execution
def main(random_state=42, output_dir=None):

    # If executed from the generalization script, redirect output paths
    if output_dir is None:
        output_dir = ROOT_DIR / "executions" / "individual" / "smote"
    else:
        output_dir = Path(output_dir)
    
    # Create directories if they do not exist
    datasets_path = output_dir / "datasets"
    rankings_path = output_dir / "generationRankings"
    plots_path    = output_dir / "plots"

    datasets_path.mkdir(parents=True, exist_ok=True)
    rankings_path.mkdir(parents=True, exist_ok=True)
    plots_path.mkdir(parents=True, exist_ok=True)
   
    # Measure execution time for the generation workflow    
    start_time = time.time()

    # Define the parameter grid for SMOTE
    sampling_strategies = [0.25, 0.3, 0.35]
    k_neighbors_list = [2, 3, 4, 5]

    if not FILE_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {FILE_PATH}")

    # Load and preprocess the dataset
    X, y, df_clean = load_and_preprocess_dataset(str(FILE_PATH), TARGET, DROP_COLS, CATEGORICAL)

    # Split the real dataset into 80-20; only the 80% is used for synthetic data generation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    # Build a consistent real dataset (features + target)
    df_train = X_train.copy()
    df_train[TARGET] = y_train.values

    # Evaluate all parameter combinations
    df_avg = best_smote_data(
        X_train, y_train,
        X.columns.tolist(), TARGET,
        sampling_strategies, k_neighbors_list,
        random_state=random_state
    )

    # Generate rankings based on metrics from each combination
    df_avg["rank_jsd"] = df_avg["jsd_mean"].rank(method="min").astype(int)
    df_avg["rank_frobenius"] = df_avg["frobenius_corr"].rank(method="min").astype(int)
    df_avg["rank_pca"] = df_avg["pca_centroid_dist"].rank(method="min").astype(int)
    
    # Calculate final score for each combination
    df_avg["total_rank_score"] = (
        df_avg["rank_jsd"] + df_avg["rank_frobenius"] + df_avg["rank_pca"]
    )
    
    # Generate the global ranking based on the total score
    df_avg = df_avg.sort_values(by="total_rank_score").reset_index(drop=True)
    df_avg["total_rank"] = df_avg.index + 1

    # Save the ranking to a CSV file
    df_avg.to_csv(rankings_path / "ranking_smote.csv", index=False, sep=';')

    # Generate the final dataset with best parameters
    best_row = df_avg.iloc[0]
    best_sampling = best_row["sampling_strategy"]
    best_k = int(best_row["k_neighbors"])

    df_smote_final = generate_smote_data(
        X_train, y_train,
        X.columns.tolist(), TARGET,
        s_strategy=best_sampling, k=best_k, random_state=random_state
    )

    # Save the final synthetic dataset to the corresponding folder (now from the best rank directly)
    df_smote_final.to_csv(datasets_path / "synthetic_smote.csv", index=False)

    # Generate and save comparative histograms
    plot_histograms(df_train, df_smote_final, NUMERIC_COLS, title_label="SMOTE", save_dir=str(plots_path))

    # Generate and save comparative correlation matrix
    plot_correlation_matrices(
        df_train[NUMERIC_COLS],
        df_smote_final[NUMERIC_COLS],
        "SMOTE",
        save_path=str(plots_path / "correlation_smote.png")
    )

    # Generate and save PCA comparison plots
    plot_pca_comparison(
        df_train,
        df_smote_final,
        NUMERIC_COLS,
        "SMOTE",
        save_path=str(plots_path / "pca_smote.png")
    )

    # Compute evaluation metrics for the final dataset (same as rank 1 now)
    jsd = compute_jsd_for_columns(df_train, df_smote_final, NUMERIC_COLS)
    corr = correlation_difference(df_train, df_smote_final, NUMERIC_COLS)
    dist = centroid_distance_pca(df_train, df_smote_final, NUMERIC_COLS)

    # Display execution summary on the console
    print(f"\n🔷 Total time spent: {time.time() - start_time:.2f} seconds")
    class_balance = df_smote_final[TARGET].value_counts(normalize=True)
    print(f"\n🔷 Class balance (synthetic dataset):\n{class_balance}")
    print("\n🔷 Final synthetic dataset metrics:\n")
    for col in NUMERIC_COLS:
        print(f"-{col:<25} JSD: {jsd[col]:.4f}")
    print(f"\n-MAE correlation: {corr['MAE']:.4f} | Frobenius: {corr['Frobenius']:.4f}")
    print(f"\n-PCA centroid distance: {dist:.4f}")
    print(f"\n🔷 Plots successfully saved in: {str(plots_path)}")


# This script can also be executed individually and the results will be saved in an isolated folder
if __name__ == "__main__":
    main()
