
import pandas as pd
import joblib
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

DATA_PATH = "../training_matrix_sully.csv"
TARGETS = ['parc_chateau', 'centre_sully', 'gare_sully', 'caserne_pompiers']
INPUT_FEATURES = ['er', 'ks2', 'ks3', 'ks4', 'ks_fp', 'of', 'qmax', 'tm']
MODEL_DIR = "tensorflow/keras"
SCALER_DIR = "tensorflow/pkls"

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found")
        return

    df = pd.read_csv(DATA_PATH)
    X = df[INPUT_FEATURES]
    y = df[TARGETS]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Load Scaler
    scaler = joblib.load(os.path.join(SCALER_DIR, "scaler.pkl"))
    X_test_scaled = scaler.transform(X_test)
    
    print("--- MODEL PERFORMANCE METRICS ---")
    for lieu in TARGETS:
        model_path = os.path.join(MODEL_DIR, f"keras_model_{lieu}.keras")
        if not os.path.exists(model_path):
            print(f"Skipping {lieu}, model not found.")
            continue
            
        model = tf.keras.models.load_model(model_path)
        y_pred = model.predict(X_test_scaled, verbose=0).flatten()
        y_true = y_test[lieu]
        
        
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        
        # Classification Metrics (Flood Detection Analysis)
        # We classify "High Water" as being in the top 10% of historical data for this location
        threshold = np.percentile(y_true, 90)
        y_true_class = (y_true > threshold).astype(int)
        y_pred_class = (y_pred > threshold).astype(int)
        
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
        acc = accuracy_score(y_true_class, y_pred_class)
        cm = confusion_matrix(y_true_class, y_pred_class)
        
        print(f"LOCATION: {lieu}")
        print(f"  R2 Score: {r2:.4f}")
        print(f"  RMSE:     {rmse:.4f}")
        print(f"  MAE:      {mae:.4f}")
        print(f"  --- Classification (Threshold >= {threshold:.2f}m) ---")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Confusion Matrix: \n{cm}")
        print("-" * 30)

if __name__ == "__main__":
    main()
