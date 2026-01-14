import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os
import random

# --- REPRODUCIBILITY SETUP ---
def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def main():
    set_seeds(42)
    
    # --- 1. CONFIGURATION ---
    DATA_PATH = "/home/yanis/Documents/projetDataScience/boosting/training_matrix_sully.csv"
    TARGETS = ['parc_chateau', 'centre_sully', 'gare_sully', 'caserne_pompiers']
    INPUT_FEATURES = ['er', 'ks2', 'ks3', 'ks4', 'ks_fp', 'of', 'qmax', 'tm']
    
    print("--- STARTING TENSORFLOW/KERAS FLOOD PREDICTION ---")
    
    # --- 2. DATA LOADING ---
    # if not os.path.exists(DATA_PATH):
        # raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please run matrixBuilder.py first.")
        
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    X = df[INPUT_FEATURES]
    y = df[TARGETS]
    
    # --- 3. PROTOCOL (STRICT) ---
    print("Applying Standard Protocol (Split 80/20, Random State 42)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # SCALING
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save Scaler for visualization
    joblib.dump(scaler, "scaler.pkl")
    print("Scaler saved to scaler.pkl")
    
    # --- 4. MODELING LOOP ---
    results = {}
    
    for lieu in TARGETS:
        print(f"\nTraining Keras MLP for TARGET: {lieu}")
        
        y_train_col = y_train[lieu]
        y_test_col = y_test[lieu]
        
        # Keras Model Definition
        # Equivalent to MLPRegressor(hidden_layer_sizes=(100, 100), activation='relu')
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(1)  # Output layer for regression
        ])
        
        # Compile
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Early Stopping to prevent overfitting and mimic sklearn's automatic convergence check
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=20, 
            restore_best_weights=True,
            verbose=0
        )
        
        # Train
        history = model.fit(
            X_train_scaled, y_train_col,
            validation_split=0.1, # Use 10% of train for validation (similar to sklearn's validation_fraction)
            epochs=300,           # Sufficient epochs
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1           # Silent training
        )
        
        # --- 5. EVALUATION ---
        y_pred = model.predict(X_test_scaled, verbose=1).flatten()
        
        r2 = r2_score(y_test_col, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_col, y_pred))
        
        print(f"  -> R2 Score: {r2:.4f}")
        print(f"  -> RMSE:     {rmse:.4f}")
        
        results[lieu] = {
            'r2': r2,
            'rmse': rmse
        }
        
        # Save model in Keras format
        model.save(f"keras_model_{lieu}.keras")

    print("\n--- FINAL SUMMARY (KERAS) ---")
    for lieu, metrics in results.items():
        print(f"{lieu:<20} | R2: {metrics['r2']:.4f} | RMSE: {metrics['rmse']:.4f}")

if __name__ == "__main__":
    main()
