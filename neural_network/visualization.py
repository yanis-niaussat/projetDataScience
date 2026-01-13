import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# --- CONFIGURATION ---
DATA_PATH = "../dataset_final_Sully.csv"
TARGETS = ['Parc_Chateau', 'Centre_Sully', 'Gare_Sully', 'Caserne_Pompiers']
INPUT_FEATURES = ['er', 'ks2', 'ks3', 'ks4', 'ks_fp', 'of', 'qmax', 'tm']
MODEL_DIR = "."

def load_data_and_prepare():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    X = df[INPUT_FEATURES]
    y = df[TARGETS]
    
    # Strict Protocol Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Load Scaler
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    X_test_scaled = scaler.transform(X_test)
    
    return df, X_test_scaled, y_test

    plt.tight_layout()
    # plt.savefig("actual_vs_predicted.png", dpi=300) # Removed global plot
    # print("Saved actual_vs_predicted.png")

def plot_residuals(X_test_scaled, y_test, df_test_original):
    # Ensure Test Set aligns with original values for plotting vs features
    # This requires a bit of care if shuffling was involved, but X_test_scaled corresponds to y_test rows.
    # We need the unscaled features for "Residuals vs Qmax". 
    # Since we did a random split, we can't easily map back unless we split the original DF the same way.
    # Let's rely on the fact that train_test_split is deterministic with random_state=42.
    
    # Re-split original DF to get unscaled features for test set
    _, X_test_orig, _, _ = train_test_split(df_test_original[INPUT_FEATURES], df_test_original[TARGETS], test_size=0.2, random_state=42)
    
    for lieu in TARGETS:
        # Create Directory
        save_dir = f"plots/{lieu}"
        os.makedirs(save_dir, exist_ok=True)
        
        model_path = os.path.join(MODEL_DIR, f"mlp_model_{lieu}.pkl")
        if not os.path.exists(model_path): continue
            
        model = joblib.load(model_path)
        y_pred = model.predict(X_test_scaled)
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
        # This checks if the model fails for extreme events
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
    print("--- STARTING VISUALIZATION ---")
    df, X_test_scaled, y_test = load_data_and_prepare()
    
    # 1. Global Analysis
    print("Generating Feature Correlation Matrix...")
    # Global plot can stay in root or move to a 'Global' folder. Let's keep it root for now or 'Global'
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
    # We pass 'df' (the original loaded dataframe) to extract unscaled features for Qmax plot
    plot_residuals(X_test_scaled, y_test, df)
    
    print("--- DONE ---")

if __name__ == "__main__":
    main()
