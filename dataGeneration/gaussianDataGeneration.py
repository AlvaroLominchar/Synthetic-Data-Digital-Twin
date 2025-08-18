# Import required libraries and functions
import pandas as pd
import os
import time
from sklearn.model_selection import train_test_split
from itertools import product
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import Metadata

# Ignore SDV library warning
import warnings
warnings.filterwarnings("ignore", message="We strongly recommend saving the metadata using 'save_to_json'")

# Import the auxiliary utility functions
from utilsDataGeneration import (
    load_and_preprocess_dataset,
    plot_histograms,
    plot_correlation_matrices,
    plot_pca_comparison,
    compute_jsd_for_columns,
    correlation_difference,
    centroid_distance_pca
)

# Configure dataset path and preprocessing parameters
FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ai4i2020.xlsx")
TARGET = "Machine failure"
DROP_COLS = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
CATEGORICAL = ["Type"]
NUMERIC_COLS = ["Air temperature [K]", "Process temperature [K]",
                "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]

# Set up working directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))

# Function to generate synthetic data using Gaussian Copula (minority class only)
def generate_copula_data(df_train, target_column, num_samples, distribution, random_state):
    # Filter minority class and convert target to boolean
    df_pos = df_train[df_train[target_column] == 1].copy()
    df_pos[target_column] = df_pos[target_column].astype(bool)

    # Define metadata and set target as boolean
    metadata = Metadata.detect_from_dataframe(df_pos)
    metadata.update_column(column_name=target_column, sdtype="boolean")

    # Initialize Gaussian Copula model with chosen distribution
    model = GaussianCopulaSynthesizer(metadata, default_distribution=distribution)
    model.fit(df_pos)

    # Generate synthetic samples for the minority class
    synth_pos = model.sample(num_rows=num_samples)

    # Combine real training data with synthetic minority samples
    df_combined = pd.concat([df_train, synth_pos], ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
    df_combined = df_combined.apply(pd.to_numeric, errors="coerce").dropna()

    return df_combined


# Function to evaluate multiple parameter combinations for Gaussian Copula
def best_copula_data(df_train, target_column, num_samples_list, distribution_list):
    results = []

    # Generate all combinations of distributions and sample sizes
    combinations = list(product(distribution_list, num_samples_list))
    for dist, n_samples in combinations:
        try:
            # Generate synthetic data using current parameters
            df_synth = generate_copula_data(df_train, target_column, n_samples, dist, random_state=42)

            # Compute divergence and correlation metrics
            jsd_vals = compute_jsd_for_columns(df_train, df_synth, NUMERIC_COLS)
            jsd_mean = sum(jsd_vals.values()) / len(jsd_vals)
            corr_diff = correlation_difference(df_train, df_synth, NUMERIC_COLS)
            frob = corr_diff["Frobenius"]
            pca_dist = centroid_distance_pca(df_train, df_synth, NUMERIC_COLS)

            # Store results
            results.append({
                "distribution": dist,
                "num_samples": n_samples,
                "jsd_mean": jsd_mean,
                "frobenius_corr": frob,
                "pca_centroid_dist": pca_dist
            })

        # Skip failing combinations
        except Exception:
            continue

    return pd.DataFrame(results)


# Main execution
def main(random_state=42, output_dir=None):

    # If executed from the generalization script, redirect output paths
    if output_dir is not None:
        datasets_path = os.path.join(output_dir, "datasets")
        rankings_path = os.path.join(output_dir, "generationRankings")
        plots_path = os.path.join(output_dir, "plots")

    # Create directories if they do not exist
    os.makedirs(datasets_path, exist_ok=True)
    os.makedirs(rankings_path, exist_ok=True)
    os.makedirs(plots_path, exist_ok=True)

    # Measure execution time for the generation workflow
    start_time = time.time()

    # Define parameter grids specific for Gaussian Copula
    num_samples_list = [1661, 2048, 2434]  # Adjusted synthetic positives
    distribution_list = ["gaussian_kde", "norm", "truncnorm"]

    # Load and preprocess the dataset
    X, y, df_clean = load_and_preprocess_dataset(FILE_PATH, TARGET, DROP_COLS, CATEGORICAL)

    # Split the real dataset into 80-20; only the 80% is used for synthetic data generation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    # Reconstruct training dataframe with target column for Copula
    df_train = X_train.copy()
    df_train[TARGET] = y_train.values

    # Evaluate all parameter combinations
    df_avg = best_copula_data(
        df_train, TARGET,
        num_samples_list, distribution_list
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
    df_avg.to_csv(os.path.join(rankings_path, "ranking_copula.csv"), index=False, sep=';')

    # Extract the top combination from the ranking
    best_params = df_avg.iloc[0]
    best_distribution = best_params["distribution"]
    best_samples = int(best_params["num_samples"])

    # Generate final synthetic dataset using the best parameters
    df_copula_final = generate_copula_data(df_train, TARGET,
                                           best_samples, best_distribution, random_state)

    # Save the final synthetic dataset to the corresponding folder
    df_copula_final.to_csv(os.path.join(datasets_path, "synthetic_copula.csv"), index=False)

    # Generate and save comparative histograms
    plot_histograms(df_clean.iloc[y_train.index], df_copula_final, NUMERIC_COLS, title_label="Copula", save_dir=plots_path)

    # Generate and save comparative correlation matrix
    plot_correlation_matrices(df_clean.iloc[y_train.index][NUMERIC_COLS],
                              df_copula_final[NUMERIC_COLS],
                              "Copula",
                              save_path=os.path.join(plots_path, "correlation_copula.png"))

    # Generate and save PCA comparison plots
    plot_pca_comparison(df_clean.iloc[y_train.index],
                        df_copula_final,
                        NUMERIC_COLS,
                        "Copula",
                        save_path=os.path.join(plots_path, "pca_copula.png"))

    # Compute evaluation metrics for the final dataset
    jsd = compute_jsd_for_columns(df_clean.iloc[y_train.index], df_copula_final, NUMERIC_COLS)
    corr = correlation_difference(df_clean.iloc[y_train.index], df_copula_final, NUMERIC_COLS)
    dist = centroid_distance_pca(df_clean.iloc[y_train.index], df_copula_final, NUMERIC_COLS)

    # Display execution summary on the console
    print(f"\n🔷 Total time spent: {time.time() - start_time:.2f} seconds")
    class_balance = df_copula_final[TARGET].value_counts(normalize=True)
    print(f"\n🔷 Class balance (synthetic dataset):\n{class_balance}")
    print("\n🔷 Final synthetic dataset metrics:\n")
    for col in NUMERIC_COLS:
        print(f"-{col:<25} JSD: {jsd[col]:.4f}")
    print(f"\n-MAE correlation: {corr['MAE']:.4f} | Frobenius: {corr['Frobenius']:.4f}")
    print(f"\n-PCA centroid distance: {dist:.4f}")
    print(f"\n🔷 Plots successfully saved in: {plots_path}")


# This script can also be executed individually and the results will be saved in an isolated folder
if __name__ == "__main__":
    individual_path = os.path.join(PARENT_DIR, "executions", "individual", "copula")
    os.makedirs(os.path.join(individual_path, "datasets"), exist_ok=True)
    os.makedirs(os.path.join(individual_path, "plots"), exist_ok=True)
    os.makedirs(os.path.join(individual_path, "generationRankings"), exist_ok=True)
    main(output_dir=individual_path)

