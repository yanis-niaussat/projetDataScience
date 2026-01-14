import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

model_files = {
    'Ridge (Linear)': 'modele_Ridge/pickle/ridge_gare_sully.pkl',       # Check your filename
    'Random Forest':  'randomforest/pickels/RF_GareSully.pkl',   # Or 'model_rf_gare_sully.pkl'
    'XGBoost':        'boosting/xgboost_gare_sully.pkl',     # Check your filename
    # 'Neural Net':   'model_nn_gare_sully.pkl'
}

matrix_file = 'training_matrix_sully.csv'

print("Loading data and preparing scenario...")
try:
    df = pd.read_csv(matrix_file)
except FileNotFoundError:
    print(f"ERROR: '{matrix_file}' not found.")
    exit()

features = ['er', 'ks2', 'ks3', 'ks4', 'ks_fp', 'of', 'qmax', 'tm']

n_steps = 300
qmax_range = np.linspace(df['qmax'].min(), df['qmax'].max() * 1.1, n_steps)

scenario_data = {}
for col in features:
    scenario_data[col] = np.full(n_steps, df[col].mean())
scenario_data['qmax'] = qmax_range

scenario_df = pd.DataFrame(scenario_data)

scaler = StandardScaler()
scaler.fit(df[features])
scenario_scaled = scaler.transform(scenario_df)
scenario_scaled_df = pd.DataFrame(scenario_scaled, columns=features)

plt.figure(figsize=(12, 8))

plt.scatter(df['qmax'], df['gare_sully'], color='gray', alpha=0.15, s=15, label='Real Simulations')

colors = {'Ridge (Linear)': 'red', 'Random Forest': 'green', 'XGBoost': 'blue', 'Neural Net': 'purple'}

print("\n--- LOADING MODELS ---")
for name, filename in model_files.items():
    try:
        print(f"Processing {name}...")
        
        model = joblib.load(filename)
        
        if 'Ridge' in name or 'Neural' in name:
            input_data = scenario_scaled_df
        else:
            input_data = scenario_df
            
        try:
            preds = model.predict(input_data)
        except Exception as e:
            print(f"  > Standard predict failed ({e}). Retrying with Numpy array...")
            input_data_np = input_data.values
            preds = model.predict(input_data_np)

        plt.plot(qmax_range, preds, label=name, color=colors.get(name, 'black'), linewidth=2.5)
        
    except FileNotFoundError:
        print(f"  ! ERROR: File '{filename}' not found. Check the name.")
    except Exception as e:
        print(f"  ! ERROR with {name}: {e}")

plt.title("Vulnerability Analysis: Gare de Sully", fontsize=16)
plt.xlabel("River Flow $Q_{max}$ ($m^3/s$)", fontsize=14)
plt.ylabel("Water Height (m)", fontsize=14)

plt.axvline(x=5000, color='black', linestyle=':', alpha=0.5)
plt.text(5050, 0.5, "Tipping Point", fontsize=10, style='italic')

plt.legend(fontsize=12)
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.savefig('graph_comparison_pkl.png', dpi=300)
plt.show()

print("\nDone. Graph saved as 'graph_comparison_pkl.png'")