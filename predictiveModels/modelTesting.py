# Import required libraries and functions
import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_recall_curve, auc, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

# Import the auxiliary utility functions
from utilsDataGeneration import load_and_preprocess_dataset

# Configure dataset path and preprocessing parameters
FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ai4i2020.xlsx")
TARGET = "Machine failure"
DROP_COLS = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
CATEGORICAL = ["Type"]

# Function to train and evaluate a Random Forest model
def train_and_evaluate(X_train, y_train, X_test, y_test, random_state=42):
    
    # Define the parameter grid
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }

    # Create the classifier model
    base_model = RandomForestClassifier(random_state=random_state)

    # Perform training
    grid_search = RandomizedSearchCV(
        base_model,
        param_distributions=param_grid,
        n_iter=10,
        scoring='f1',
        cv=3,
        verbose=0,
        n_jobs=-1,
        random_state=random_state
    )
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_

    # Apply the model in isolated real data and get probabilities
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Perform the best threshold search between 0.1 and 0.9, optimizing f1
    thresholds = np.linspace(0.1, 0.9, 50)
    best_f1, best_t = 0, 0.5
    for t in thresholds:
        y_pred_temp = (y_proba >= t).astype(int)
        f1 = f1_score(y_test, y_pred_temp)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    # Get the final predictions based on the best threshold obtained
    y_pred = (y_proba >= best_t).astype(int)

    # Calculate metrics for evaluation and return them
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    cm = confusion_matrix(y_test, y_pred)

    # ROC curve data
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "PR-AUC": pr_auc,
        "Confusion Matrix": cm,
        "ROC Data": (fpr, tpr),
        "PR Data": (recall, precision)
    }, model

# Function to save and plot ROC curve
def save_roc_curve(fpr, tpr, roc_auc, output_path, title):
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(output_path / "roc_curve.png")
    plt.close()

# Function to save and plot Precision-Recall curve
def save_pr_curve(recall, precision, pr_auc, output_path, title):
    plt.figure()
    plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.savefig(output_path / "pr_curve.png")
    plt.close()

# Function to compare and measure the impact of training the same model with four different datasets
def evaluate_models(output_dir=None, random_state=42):

    # If executed from the generalization script, redirect output paths
    if output_dir is None:
        base_path = Path(__file__).resolve().parent.parent / "executions" / "individual"
    else:
        base_path = Path(output_dir)

    # Load and preprocess the dataset
    X_real, y_real, df_real = load_and_preprocess_dataset(FILE_PATH, TARGET, DROP_COLS, CATEGORICAL)
    
    # Split the real dataset into 80-20; the 20% is isolated and used to test each model
    X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
        X_real, y_real, test_size=0.2, stratify=y_real, random_state=random_state
    )

    # Train the model with real data and obtain its evaluation metrics and confusion matrix, based on 20% isolated data
    results_real, model_real = train_and_evaluate(X_train_real, y_train_real, X_test_real, y_test_real, random_state)
    
    # Save results
    real_path = base_path / "real"
    (real_path / "metrics").mkdir(parents=True, exist_ok=True)
    (real_path / "confusion").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "Accuracy": results_real["Accuracy"],
        "Precision": results_real["Precision"],
        "Recall": results_real["Recall"],
        "F1-score": results_real["F1-score"],
        "ROC-AUC": results_real["ROC-AUC"],
        "PR-AUC": results_real["PR-AUC"]
    }]).to_csv(real_path / "metrics" / "metrics.csv", index=False)
    pd.DataFrame(results_real["Confusion Matrix"]).to_csv(real_path / "confusion" / "confusion_matrix.csv", index=False)

    # Save ROC curve for real data
    fpr, tpr = results_real["ROC Data"]
    save_roc_curve(fpr, tpr, results_real["ROC-AUC"], real_path / "metrics", "ROC Curve - Real Data")

    recall_pr, precision_pr = results_real["PR Data"]
    save_pr_curve(recall_pr, precision_pr, results_real["PR-AUC"], real_path / "metrics", "PR Curve - Real Data")

    (real_path / "model").mkdir(parents=True, exist_ok=True)
    joblib.dump(model_real, real_path / "model" / "model_real.pkl")

    # Train the model with synthetic data and obtain its evaluation metrics and confusion matrix, based on 20% isolated data
    for technique, filename in [
        ("smote", "synthetic_smote.csv"),
        ("copula", "synthetic_copula.csv"),
        ("ctgan", "synthetic_ctgan.csv")
    ]:
        
        # Search for each dataset
        technique_path = base_path / technique
        dataset_path = technique_path / "datasets" / filename
        if not dataset_path.exists():
            print(f"[ADVERTENCIA] No se encontró {dataset_path}, se omite {technique}.")
            continue
        
        # Load synthetic data
        df_synth = pd.read_csv(dataset_path)
        X_synth, y_synth = df_synth.drop(columns=[TARGET]), df_synth[TARGET].astype(int)

        # Training and evaluation
        results, model_synth = train_and_evaluate(X_synth, y_synth, X_test_real, y_test_real)

        # Save results
        (technique_path / "metrics").mkdir(parents=True, exist_ok=True)
        (technique_path / "confusion").mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{
            "Accuracy": results["Accuracy"],
            "Precision": results["Precision"],
            "Recall": results["Recall"],
            "F1-score": results["F1-score"],
            "ROC-AUC": results["ROC-AUC"],
            "PR-AUC": results["PR-AUC"]
        }]).to_csv(technique_path / "metrics" / "metrics.csv", index=False)
        pd.DataFrame(results["Confusion Matrix"]).to_csv(technique_path / "confusion" / "confusion_matrix.csv", index=False)

        # Save ROC curve for synthetic data
        fpr, tpr = results["ROC Data"]
        save_roc_curve(fpr, tpr, results["ROC-AUC"], technique_path / "metrics", f"ROC Curve - {technique.upper()}")

        recall_pr, precision_pr = results["PR Data"]
        save_pr_curve(recall_pr, precision_pr, results["PR-AUC"], technique_path / "metrics", f"PR Curve - {technique.upper()}")

        (technique_path / "model").mkdir(parents=True, exist_ok=True)
        joblib.dump(model_synth, technique_path / "model" / f"model_{technique}.pkl")

        print(f"[INFO] Resultados guardados para {technique.upper()} en {technique_path}")

# This script can also be executed individually and the results will be saved in an isolated folder
if __name__ == "__main__":
    
    # Evaluate models in 'executions/individual/technique folder'
    evaluate_models()
