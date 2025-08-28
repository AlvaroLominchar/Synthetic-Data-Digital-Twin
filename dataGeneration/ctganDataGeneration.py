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
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata

# Ignore SDV library warning
import warnings
warnings.filterwarnings("ignore", message="We strongly recommend saving the metadata using 'save_to_json'")

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

# Function to train a CTGAN model on the minority class
def train_ctgan_model(df_train, target_column, epochs, generator_dim, pac, batch_size):
    
    # Filter minority class and convert target to boolean
    df_pos = df_train[df_train[target_column] == 1].copy()
    df_pos[target_column] = df_pos[target_column].astype(bool)

    # Define metadata and set target as boolean
    metadata = Metadata.detect_from_dataframes({"synthetic_table": df_pos})
    metadata.update_column(table_name="synthetic_table", column_name=target_column, sdtype="boolean")
    metadata.validate()

    # Initialize and train CTGAN model
    ctgan = CTGANSynthesizer(
        metadata=metadata,
        epochs=epochs,
        batch_size=batch_size,
        generator_dim=generator_dim,
        discriminator_steps=1,
        pac=pac,
        log_frequency=True,
        verbose=True,
        cuda=True,
        enforce_min_max_values=True,
        enforce_rounding=True
    )

    ctgan.fit(df_pos)
    return ctgan


# Function to generate synthetic data using a trained CTGAN
def generate_ctgan_data(ctgan, df_train, target_column, n_samples, random_state):
    synth_pos = ctgan.sample(num_rows=n_samples)
    synth_pos = synth_pos.apply(pd.to_numeric, errors="coerce").dropna()
    df_combined = pd.concat([df_train, synth_pos], ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df_combined


# Function to evaluate the quality of datasets generated with multiple parameter combinations
def best_ctgan_data(df_train, target_column, n_samples_list, epochs_list, gen_dim_list, pac_batch_combinations):
    
    results = []
    datasets_generated = {}
    
    # Create combinations
    combinations = [
        (epochs, gen_dim, pac, batch)
        for epochs in epochs_list
        for gen_dim in gen_dim_list
        for pac, batch in pac_batch_combinations
    ]
    
    # Make a copy to work on it
    df_real = df_train.copy()

    # Loop for each combination
    for epochs, gen_dim, pac, batch_size in combinations:
        
        # Skip invalid combinations (batch size must be divisible by pac)
        if batch_size % pac != 0:
            continue

        try:
            # Train CTGAN with current parameters
            ctgan = train_ctgan_model(df_train, target_column, epochs, gen_dim, pac, batch_size)

            # Generate synthetic data for each sample size
            for n_samples in n_samples_list:
                df_synth = generate_ctgan_data(ctgan, df_train, target_column, n_samples, random_state=42)

                # Compute divergence and correlation metrics
                jsd_vals = compute_jsd_for_columns(df_real, df_synth, NUMERIC_COLS)
                jsd_mean = sum(jsd_vals.values()) / len(jsd_vals)
                corr_diff = correlation_difference(df_real, df_synth, NUMERIC_COLS)
                frob = corr_diff["Frobenius"]
                pca_dist = centroid_distance_pca(df_real, df_synth, NUMERIC_COLS)

                # Store results
                results.append({
                    "epochs": epochs,
                    "generator_dim": str(gen_dim),
                    "pac": pac,
                    "batch_size": batch_size,
                    "num_samples": n_samples,
                    "jsd_mean": jsd_mean,
                    "frobenius_corr": frob,
                    "pca_centroid_dist": pca_dist
                })
                
                # Store datasets
                key = (epochs, str(gen_dim), pac, batch_size, n_samples)
                datasets_generated[key] = df_synth

        # Skip failing combinations
        except Exception:
            continue

    return pd.DataFrame(results), datasets_generated


# Main execution
def main(random_state=42, output_dir=None):

    # If executed from the generalization script, redirect output paths
    if output_dir is None:
        output_dir = ROOT_DIR / "executions" / "individual" / "ctgan"
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

    # Define parameter grids specific for CTGAN
    epochs_list = [400, 600]
    gen_dim_list = [(64,64), (64, 128)]
    pac_batch_combinations = [
        (2, 64),
        (2, 128),
        (3, 96),
        (3, 192)
    ]
    num_samples_list = [1661, 2048, 2434]

    if not FILE_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {FILE_PATH}")

    # Load and preprocess the dataset
    X, y, df_clean = load_and_preprocess_dataset(str(FILE_PATH), TARGET, DROP_COLS, CATEGORICAL)

    # Split the real dataset into 80-20; only the 80% is used for synthetic data generation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    # Reconstruct training dataframe with target column for CTGAN
    df_train = X_train.copy()
    df_train[TARGET] = y_train.values

    # Evaluate all parameter combinations
    df_avg, datasets_generated = best_ctgan_data(
        df_train, TARGET,
        num_samples_list, epochs_list, gen_dim_list, pac_batch_combinations
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
    df_avg.to_csv(rankings_path / "ranking_ctgan.csv", index=False, sep=';')

    # Extract the top combination from the ranking
    best_params = df_avg.iloc[0]
    best_key = (
        int(best_params["epochs"]),
        best_params["generator_dim"],
        int(best_params["pac"]),
        int(best_params["batch_size"]),
        int(best_params["num_samples"])
    )

    # Save the best dataset
    df_ctgan_final = datasets_generated[best_key]

    # Save the final synthetic dataset to the corresponding folder
    df_ctgan_final.to_csv(datasets_path / "synthetic_ctgan.csv", index=False)

    # Generate and save comparative histograms
    plot_histograms(df_clean.iloc[y_train.index], df_ctgan_final, NUMERIC_COLS, title_label="CTGAN", save_dir=str(plots_path))

    # Generate and save comparative correlation matrix
    plot_correlation_matrices(df_clean.iloc[y_train.index][NUMERIC_COLS],
                          df_ctgan_final[NUMERIC_COLS],
                          "CTGAN",
                          save_path=str(plots_path / "correlation_ctgan.png"))

    # Generate and save PCA comparison plots
    plot_pca_comparison(df_clean.iloc[y_train.index],
                    df_ctgan_final,
                    NUMERIC_COLS,
                    "CTGAN",
                    save_path=str(plots_path / "pca_ctgan.png"))

    # Compute evaluation metrics for the final dataset
    jsd = compute_jsd_for_columns(df_clean.iloc[y_train.index], df_ctgan_final, NUMERIC_COLS)
    corr = correlation_difference(df_clean.iloc[y_train.index], df_ctgan_final, NUMERIC_COLS)
    dist = centroid_distance_pca(df_clean.iloc[y_train.index], df_ctgan_final, NUMERIC_COLS)

    # Display execution summary on the console
    print(f"\n🔷 Total time spent: {time.time() - start_time:.2f} seconds")
    class_balance = df_ctgan_final[TARGET].value_counts(normalize=True)
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
