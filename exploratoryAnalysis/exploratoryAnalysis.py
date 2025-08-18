# Import required libraries and functions
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Configure dataset path and preprocessing parameters
FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ai4i2020.xlsx")
TARGET = "Machine failure"
NUMERIC_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

# Set custom colors for visualizations
main_color = "#1A237E"
secondary_color = "#FF7F0E"

# Set up working directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
PLOTS_DIR = os.path.join(PARENT_DIR, "exploratoryAnalysis")
os.makedirs(PLOTS_DIR, exist_ok=True)


# Main execution
if __name__ == "__main__":
 
    # Load dataset in raw format (as downloaded)
    df_raw = pd.read_excel(FILE_PATH, header=None)
    
    # Separate columns by commas
    df_raw = df_raw[0].str.split(",", expand=True)
    
    # Set first row as headers
    df_raw.columns = df_raw.iloc[0]
    df_raw = df_raw.drop(index=0).reset_index(drop=True)
    
    # Drop each failure type column, out interest is on Machine Failure as target.
    df_raw = df_raw.drop(columns=["TWF", "HDF", "PWF", "OSF", "RNF"], errors="ignore")

    # Print some basic information
    print("\n🔷 Dataset dimensions:", df_raw.shape)
    print("\n🔷 General Information:")
    print(df_raw.info())
    print("\n🔷 Missing values per column:", df_raw.isnull().sum())
    print("\n🔷 Number of duplicated rows:", df_raw.duplicated().sum())

    # Class balance
    if TARGET in df_raw.columns:
        print("\n🔷 Class balance (Machine failure):")
        print(df_raw[TARGET].value_counts(normalize=True))
    else:
        print("\nTarget column not found.")

    # Convert all columns to numeric where possible, ignoring errors
    for col in df_raw.columns:
        try:
            df_raw[col] = pd.to_numeric(df_raw[col])
        except ValueError:
            pass
    
    df_numeric = df_raw


    # Describe numeric columns
    print("\n🔷 Statistic (numeric features):")
    print(df_numeric.describe())

    # Generate histograms for each single variable
    print("\n🔷 Generating histograms...")
    for col in NUMERIC_COLS:
        if col in df_numeric.columns:
            
            plt.figure(figsize=(7, 4))
            
            # Build histogram
            sns.histplot(df_numeric[col].astype(float), bins=50, kde=True, color=main_color)
            plt.title(f"{col} distribution")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.tight_layout()

            # Save
            filename = os.path.join(PLOTS_DIR, f"hist_{col.replace(' ', '_')}.png")
            plt.savefig(filename, dpi=300)
            plt.close()

    # Generate correlation matrix
    print("\n🔷 Generating correlation matrix...")
    numeric_cols_existing = [c for c in NUMERIC_COLS if c in df_numeric.columns]
    plt.figure(figsize=(8, 6))
    
    # Build the matrix
    sns.heatmap(
        df_numeric[numeric_cols_existing].corr(),
        cmap='Blues',
        annot=True,
        fmt=".2f"
    )
    plt.title("Correlation Matrix - Numeric features")
    plt.tight_layout()

    # Save
    filename_corr = os.path.join(PLOTS_DIR, "correlation_matrix.png")
    plt.savefig(filename_corr, dpi=300)
    plt.close()

    # Generate PCA Analysis
    if TARGET in df_numeric.columns:
        print("\n🔷 Generating PCA plot...")
        
        # Standardize numerical variables
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_numeric[numeric_cols_existing])

        # Perform PCA Analysis
        pca = PCA(n_components=2)
        pca_components = pca.fit_transform(X_scaled)

        plt.figure(figsize=(8, 6))

        # Negative class
        sns.scatterplot(
            x=pca_components[df_numeric[TARGET] == 0, 0],
            y=pca_components[df_numeric[TARGET] == 0, 1],
            color=main_color,
            alpha=0.6,
            label="No failure"
        )

        # Positive class
        sns.scatterplot(
            x=pca_components[df_numeric[TARGET] == 1, 0],
            y=pca_components[df_numeric[TARGET] == 1, 1],
            color=secondary_color,
            alpha=0.8,
            label="Failure"
        )

        plt.title("PCA - 2D Projection of Numerical Variables")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend(title="Machine failure")
        plt.tight_layout()
        
        # Save
        filename_pca = os.path.join(PLOTS_DIR, "pca_projection.png")
        plt.savefig(filename_pca, dpi=300)
        plt.close()

print(f"\n🔷 Exploratory analysis completed on raw data. Plots saved in '{PLOTS_DIR}'.")


