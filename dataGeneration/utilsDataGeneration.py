# Import required libraries and functions
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import jensenshannon, euclidean


# Define main colors for plots
COLOR_REAL = "#1A237E"
COLOR_SYNTH = "#FF6F00"
COLOR_COMBINED = "#7CFC7C"

# This function is in charge of loading and preparing the dataset
def load_and_preprocess_dataset(file_path, target_column, drop_columns=[], categorical_columns=[]):
    
    # Load an Excel file without headers and split first column by commas
    df_raw = pd.read_excel(file_path, header=None)
    df_raw = df_raw[0].str.split(",", expand=True)
    
    # Use the first row as column headers
    df_raw.columns = df_raw.iloc[0]
    
    # Define the dataframe without headers and reset indexes
    df = df_raw.drop(index=0).reset_index(drop=True)
    
    # Drop unnecessary columns specified in 'drop_columns'
    df = df.drop(columns=drop_columns, errors='ignore')
    
    # Loop to convert categorical columns into integers with label encoding
    for col in categorical_columns:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col])
    
    # Convert all values to numeric and drop missing values
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    
    # Define feature matrix and target vector (integer)
    X = df.drop(columns=[target_column])
    y = df[target_column].astype(int)
    
    return X, y, df

# This function creates an histogram for each column of the dataset, comparing real and synthetic data
def plot_histograms(df_real, df_synth, columns, title_label, save_dir=None):
    
    # Loop for each variable
    for col in columns:
        
        # Create figure
        plt.figure(figsize=(7, 4))
        
        # Plot synthetic and real histograms
        sns.histplot(df_real[col], label='Real', color=COLOR_REAL, stat='density', bins=50, kde=True)
        sns.histplot(df_synth[col], label='Sintético', color=COLOR_SYNTH, stat='density', bins=50, kde=True)
        plt.title(f'Distribución comparada ({title_label}): {col}')
        plt.legend()
        plt.tight_layout()

    # Save
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            safe_col = col.replace(" ", "_").replace("[", "").replace("]", "")
            file_path = os.path.join(save_dir, f"hist_{safe_col}.png")
            plt.savefig(file_path)
            plt.close()
        else:
            plt.show()

# This function creates the two correlation matrices of real and synthetic data
def plot_correlation_matrices(df_real, df_synth, title_label, save_path=None):
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot correlation matrix for real data
    sns.heatmap(df_real.corr(), ax=axes[0], cmap='Blues', annot=True, fmt=".2f")
    axes[0].set_title('Correlación - Datos Reales')
    
    # Plot correlation matrix for synthetic data
    sns.heatmap(df_synth.corr(), ax=axes[1], cmap='Reds', annot=True, fmt=".2f")
    axes[1].set_title(f'Correlación - Datos {title_label}')
    plt.tight_layout()

    # Save
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# This function creates a PCA projection to compare real and synthetic data in 2D space 
def plot_pca_comparison(df_real, df_synth, columns, title_label, save_path=None):
    import os

    # Create copies of data
    df_real_copy = df_real[columns].copy()
    df_synth_copy = df_synth[columns].copy()

    # Join datasets in order to build combined PCA
    df_all = pd.concat([df_real_copy, df_synth_copy], ignore_index=True)
    X_scaled = StandardScaler().fit_transform(df_all)

    # Apply PCA
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)

    # Split components based on real and synthetic size.
    n_real = len(df_real_copy)
    real_pca = components[:n_real]
    synth_pca = components[n_real:]

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot real data
    axes[0].scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.7, color=COLOR_REAL, label="Real")
    axes[0].set_title("PCA - Datos Reales")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].grid(True)

    # Plot synthetic data
    axes[1].scatter(synth_pca[:, 0], synth_pca[:, 1], alpha=0.7, color=COLOR_SYNTH, label=title_label)
    axes[1].set_title(f"PCA - Datos {title_label}")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].grid(True)

    plt.tight_layout()

    # Save
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


# This function computes the Jensen-Shannon divergence for each column between real and synthetic data
def compute_jsd_for_columns(df_real, df_synth, columns, bins=50):
    
    results = {}
    for col in columns:
        
        # Create normalized histograms for real and synthetic data
        p_real, _ = np.histogram(df_real[col], bins=bins, density=True)
        p_synth, _ = np.histogram(df_synth[col], bins=bins, density=True)
        
        # Avoid division by zero by adding a small constant
        p_real += 1e-8
        p_synth += 1e-8
        
        # Compute Jensen-Shannon divergence
        jsd = jensenshannon(p_real, p_synth)
        results[col] = jsd
        
    return results

# This function compares the correlation matrices of real and synthetic data
def correlation_difference(real_df, synth_df, columns):
    
    # Compute correlation matrices
    corr_real = real_df[columns].corr().values
    corr_synth = synth_df[columns].corr().values
    
    # Calculate the Mean Absolute Error and Frobenius norm between matrices
    mae = np.mean(np.abs(corr_real - corr_synth))
    frob = np.linalg.norm(corr_real - corr_synth)
    
    return {"MAE": mae, "Frobenius": frob}

# This function calculates the Euclidean distance between PCA centroids of real and synthetic data
def centroid_distance_pca(df_real, df_synth, columns):
    
    # Standardize both datasets together
    scaler = StandardScaler()
    X_all = pd.concat([df_real[columns], df_synth[columns]])
    X_scaled = scaler.fit_transform(X_all)
    
    # Apply PCA to reduce to 2 components
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Compute centroids for real and synthetic samples
    n_real = len(df_real)
    centroid_real = X_pca[:n_real].mean(axis=0)
    centroid_synth = X_pca[n_real:].mean(axis=0)
    
    # Return Euclidean distance between centroids
    return euclidean(centroid_real, centroid_synth)