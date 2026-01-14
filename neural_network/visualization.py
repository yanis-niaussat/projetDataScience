import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# --- CONFIGURATION ---
DATA_PATH = "../training_matrix_sully.csv"
TARGETS = ['parc_chateau', 'centre_sully', 'gare_sully', 'caserne_pompiers']
INPUT_FEATURES = ['er', 'ks2', 'ks3', 'ks4', 'ks_fp', 'of', 'qmax', 'tm']
# Models are in tensorflow/keras/ folder relative to this script
MODEL_DIR = "tensorflow/keras"
# Scaler is in tensorflow/pkls/
SCALER_DIR = "tensorflow/pkls"

def load_data_and_prepare():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    X = df[INPUT_FEATURES]
    y = df[TARGETS]
    
    # Strict Protocol Split (Must match training!)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Load Scaler
    scaler_path = os.path.join(SCALER_DIR, "scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}. Please run tensorflow/keras_model.py first.")
        
    scaler = joblib.load(scaler_path)
    X_test_scaled = scaler.transform(X_test)
    
    return df, X_test_scaled, y_test

def plot_residuals(X_test_scaled, y_test, df_test_original):
    # Ensure Test Set aligns with original values for plotting vs features
    # Re-split original DF to get unscaled features for test set
    _, X_test_orig, _, _ = train_test_split(df_test_original[INPUT_FEATURES], df_test_original[TARGETS], test_size=0.2, random_state=42)
    
    display_names = {
        'parc_chateau': 'Parc_Chateau',
        'centre_sully': 'Centre_Sully',
        'gare_sully': 'Gare_Sully',
        'caserne_pompiers': 'Caserne_Pompiers'
    }

    for lieu in TARGETS:
        # Create Directory with Capitalized Name
        dir_name = display_names.get(lieu, lieu)
        save_dir = f"plots/{dir_name}"
        os.makedirs(save_dir, exist_ok=True)
        
        model_path = os.path.join(MODEL_DIR, f"keras_model_{lieu}.keras")
        if not os.path.exists(model_path): 
            print(f"Model not found for {lieu} at {model_path}")
            continue
            
        print(f"Loading model for {lieu}...")
        model = tf.keras.models.load_model(model_path)
        
        # Predict (Keras returns [N, 1], we need [N,])
        y_pred = model.predict(X_test_scaled, verbose=0).flatten()
        y_true = y_test[lieu]
        residuals = y_true - y_pred
        
        # 1. Residuals vs Predicted
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
        plt.axhline(0, color='r', linestyle='--', lw=2)
        plt.title(f"Residuals vs Predicted\n{lieu}")
        plt.xlabel("Predicted Water Level (m)")
        plt.ylabel("Residuals (m)")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{save_dir}/residuals_vs_predicted.png", dpi=300)
        plt.close()

        # 2. Error Distribution
        plt.figure(figsize=(8, 6))
        sns.histplot(residuals, kde=True, bins=30, color='orange')
        plt.axvline(0, color='r', linestyle='--')
        plt.title(f"Error Distribution\n{lieu}")
        plt.xlabel("Error (m)")
        plt.savefig(f"{save_dir}/error_distribution.png", dpi=300)
        plt.close()
        
        # 3. Actual vs Predicted (Individual)
        plt.figure(figsize=(8, 6))
        r2 = r2_score(y_true, y_pred)
        sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        plt.title(f"Actual vs Predicted (R2={r2:.3f})\n{lieu}")
        plt.xlabel("Actual (m)")
        plt.ylabel("Predicted (m)")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{save_dir}/actual_vs_predicted.png", dpi=300)
        plt.close()
        
        # 4. Residuals vs Qmax (PHYSICS CHECK)
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=X_test_orig['qmax'], y=residuals, alpha=0.6, hue=residuals.abs(), palette='coolwarm')
        plt.axhline(0, color='black', linestyle='--', lw=1)
        plt.title(f"Residuals vs Flow Rate (Qmax)\n{lieu}")
        plt.xlabel("Qmax (m3/s)")
        plt.ylabel("Residuals (m)")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{save_dir}/residuals_vs_qmax.png", dpi=300)
        plt.close()
        
        print(f"Generated 4 plots for {lieu} in {save_dir}/")

def main():
    print("--- STARTING VISUALIZATION (KERAS) ---")
    df, X_test_scaled, y_test = load_data_and_prepare()
    
    # 1. Global Analysis
    print("Generating Feature Correlation Matrix...")
    os.makedirs("plots/Global", exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    corr = df[INPUT_FEATURES].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix of Input Features")
    plt.tight_layout()
    plt.savefig("plots/Global/feature_correlation.png", dpi=300)
    plt.close()
    
    # 2. Per-Location Analysis
    print("Generating Per-Location Plots...")
    plot_residuals(X_test_scaled, y_test, df)
    
    print("--- DONE ---")

if __name__ == "__main__":
    main()
