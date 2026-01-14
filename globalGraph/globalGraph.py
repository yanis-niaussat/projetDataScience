import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- TENSORFLOW SETUP ---
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    print("Warning: TensorFlow not found.")
    tf = None

# --- 1. CONFIGURATION ---
model_files = {
    'Ridge':          'modele_Ridge/pickle/ridge/ridge_gare_sully.pkl',
    'Lasso':          'modele_Ridge/pickle/lasso/lasso_gare_sully.pkl',
    'Random Forest':  'randomforest/pickels/RF_GareSully.pkl',
    'XGBoost':        'boosting/xgboost_gare_sully.pkl',
    'Neural Net':     'neural_network/nn_gare_sully.keras' 
}

matrix_file = 'training_matrix_sully.csv'

# --- 2. LOAD DATA ---
print("Loading data...")
try:
    df = pd.read_csv(matrix_file)
except FileNotFoundError:
    print(f"ERROR: '{matrix_file}' not found.")
    exit()

features = ['er', 'ks2', 'ks3', 'ks4', 'ks_fp', 'of', 'qmax', 'tm']
target = 'gare_sully'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Fit on train
print("Scaler ready.")

# --- 3. HELPER: TRAIN NN ON THE FLY ---
def train_neural_net_on_the_fly(X_tr, y_tr):
    print("  > Training new Neural Network locally (Fallback)...")
    if tf is None: return None
    
    # Simple 2-layer Architecture (Standard for this physics)
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_tr.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1) # Linear output for regression
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # Quick train (50 epochs is enough for a smooth curve)
    model.fit(X_tr, y_tr, epochs=50, batch_size=32, verbose=0)
    print("  > Local Neural Network trained.")
    return model

# --- 4. CREATE SCENARIO ---
n_steps = 300
qmax_range = np.linspace(df['qmax'].min(), df['qmax'].max() * 1.1, n_steps)

mean_values = X_train.mean().values 
scenario_array = np.tile(mean_values, (n_steps, 1)) 
qmax_idx = features.index('qmax')
scenario_array[:, qmax_idx] = qmax_range

scenario_scaled_array = scaler.transform(scenario_array)

# --- 5. PREDICT & PLOT ---
plt.figure(figsize=(12, 8))

plt.scatter(df['qmax'], df[target], color='gray', alpha=0.15, s=15, label='Real Simulations')

styles = {
    'Ridge':         {'color': 'red',    'ls': '-',  'lw': 6, 'alpha': 0.3},
    'Lasso':         {'color': 'orange', 'ls': '--', 'lw': 2, 'alpha': 1.0},
    'Random Forest': {'color': 'green',  'ls': '-',  'lw': 2, 'alpha': 0.9},
    'XGBoost':       {'color': 'blue',   'ls': '-',  'lw': 3, 'alpha': 0.9},
    'Neural Net':    {'color': 'purple', 'ls': '-.', 'lw': 2, 'alpha': 1.0}
}

print("\n--- PREDICTING ---")

for name, filename in model_files.items():
    model = None
    is_keras = False
    
    try:
        # A. LOAD MODEL
        if 'Neural' in name:
            if tf is None: continue
            try:
                # Try loading file
                model = load_model(filename, compile=False)
                is_keras = True
            except Exception as e:
                # FALLBACK: Train new one if load fails
                print(f"  ! Could not load {filename} ({e}). Retraining...")
                model = train_neural_net_on_the_fly(X_train_scaled, y_train)
                is_keras = True
        else:
            model = joblib.load(filename)
            is_keras = False
            
        if model is None: continue

        # B. PREDICT
        # RF uses scaled data too (based on your confirmation)
        input_data = scenario_scaled_array
        
        preds = model.predict(input_data)
        
        if is_keras:
            preds = preds.flatten()

        # C. PLOT
        s = styles.get(name, {'color': 'black'})
        plt.plot(qmax_range, preds, 
                 label=name, 
                 color=s.get('color'), 
                 linestyle=s.get('ls', '-'), 
                 linewidth=s.get('lw', 2),
                 alpha=s.get('alpha', 1.0))
        
        print(f"  > {name}: OK")
        
    except FileNotFoundError:
        print(f"  ! Warning: '{filename}' not found.")
    except Exception as e:
        print(f"  ! Error with {name}: {e}")

# --- 6. FINISH GRAPH ---
plt.title(f"Vulnerability Analysis: {target.replace('_', ' ').title()}", fontsize=16)
plt.xlabel("River Flow $Q_{max}$ ($m^3/s$)", fontsize=14)
plt.ylabel("Water Height (m)", fontsize=14)

plt.axvline(x=5000, color='black', linestyle=':', alpha=0.5)
plt.text(5050, 0.5, "Tipping Point", fontsize=10, style='italic')
plt.axhline(0, color='black', linewidth=1)

plt.legend(fontsize=12)
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.savefig('graph_comparison_5models_final.png', dpi=300)
plt.show()

print("\nDone. Graph saved as 'graph_comparison_5models_final.png'")