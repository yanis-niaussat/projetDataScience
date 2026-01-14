import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Suppression des avertissements de version bruyants
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- CONFIGURATION ---
target_suffix = "gare_sully" 
target_rf_name = "GareSully"
matrix_file = 'training_matrix_sully.csv'

# Chemins des fichiers (tels que vous les avez définis)
model_files = {
    'Ridge':          f'modele_Ridge/pickle/ridge/ridge_{target_suffix}.pkl',
    'Lasso':          f'modele_Ridge/pickle/lasso/lasso_{target_suffix}.pkl',
    'Random Forest':  f'randomforest/pickels/RF_{target_rf_name}.pkl',
    'XGBoost':        f'boosting/xgboost_{target_suffix}.pkl',
    'Neural Net':     f'neural_network/tensorflow/keras_model_{target_suffix}.keras' 
}

# --- 1. CHARGEMENT DES DONNÉES (CRUCIAL POUR LE FALLBACK) ---
print("Chargement des données...")
if not os.path.exists(matrix_file):
    print(f"ERREUR CRITIQUE: '{matrix_file}' introuvable. Impossible de continuer.")
    exit()

df = pd.read_csv(matrix_file)
features = ['er', 'ks2', 'ks3', 'ks4', 'ks_fp', 'of', 'qmax', 'tm']
target = target_suffix

X = df[features]
y = df[target]

# On refait le split pour avoir les données d'entrainement si besoin de réentraîner
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. GESTION DES SCALERS ---
print("Configuration des Scalers...")
scalers = {}

# On essaie de charger, sinon on en crée un neuf
try: 
    scalers['XGBoost'] = joblib.load("boosting/scaler.pkl")
except: 
    print("  > Scaler XGBoost manquant ou incompatible -> Création d'un nouveau.")
    s = StandardScaler()
    s.fit(X_train)
    scalers['XGBoost'] = s

try: 
    s_rl = joblib.load("modele_Ridge/pickle/ridge_lasso_scaler.pkl")
    scalers['Ridge'] = s_rl
    scalers['Lasso'] = s_rl
except:
    print("  > Scaler Ridge/Lasso manquant -> Création d'un nouveau.")
    s = StandardScaler()
    s.fit(X_train)
    scalers['Ridge'] = s
    scalers['Lasso'] = s

# Neural Net a souvent son propre scaler
try: scalers['Neural Net'] = joblib.load("neural_network/tensorflow/scaler.pkl")
except: pass # On gérera plus tard

# --- 3. CHARGEMENT / RÉPARATION DES MODÈLES ---
loaded_models = {}

print("\n--- CHARGEMENT DES MODÈLES (AVEC AUTO-RÉPARATION) ---")

# A. XGBOOST
try:
    import xgboost as xgb
    # On essaie de charger le pickle
    try:
        model = joblib.load(model_files['XGBoost'])
        loaded_models['XGBoost'] = model
        print("  > XGBoost : Chargé (Pickle)")
    except:
        print("  ! XGBoost Pickle incompatible. Ré-entraînement express...")
        # Ré-entraînement rapide
        model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        # Attention au scaling pour XGB si utilisé précédemment
        X_tr_sc = scalers['XGBoost'].transform(X_train)
        model.fit(X_tr_sc, y_train)
        loaded_models['XGBoost'] = model
        print("  > XGBoost : Ré-entraîné et prêt.")
except ImportError:
    print("  ! Module XGBoost non installé.")

# B. RANDOM FOREST (Souvent problématique entre versions)
try:
    try:
        model = joblib.load(model_files['Random Forest'])
        loaded_models['Random Forest'] = model
        print("  > Random Forest : Chargé")
    except Exception as e:
        print(f"  ! RF incompatible ({e}). Ré-entraînement...")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train) # RF utilise souvent Raw data
        loaded_models['Random Forest'] = model
        print("  > Random Forest : Ré-entraîné et prêt.")
except: pass

# C. RIDGE & LASSO
for m_name, m_class in [('Ridge', Ridge), ('Lasso', Lasso)]:
    try:
        model = joblib.load(model_files[m_name])
        loaded_models[m_name] = model
        print(f"  > {m_name} : Chargé")
    except:
        print(f"  ! {m_name} incompatible. Ré-entraînement...")
        model = m_class(alpha=1.0) # Alpha par défaut ou ajusté si connu
        X_tr_sc = scalers[m_name].transform(X_train)
        model.fit(X_tr_sc, y_train)
        loaded_models[m_name] = model
        print(f"  > {m_name} : Ré-entraîné et prêt.")

# D. NEURAL NET (Tensorflow)
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    path_nn = model_files['Neural Net']
    if os.path.exists(path_nn):
        try:
            model = load_model(path_nn, compile=False)
            loaded_models['Neural Net'] = model
            print("  > Neural Net : Chargé")
        except: print("  ! Neural Net : Erreur de chargement.")
    else:
        print("  ! Neural Net : Fichier introuvable (Ignoré).")
except: pass


# --- 4. PRÉPARATION SCÉNARIO ---
n_steps = 300
qmax_range = np.linspace(df['qmax'].min(), df['qmax'].max() * 1.1, n_steps)

# On fige les autres paramètres à la moyenne
mean_values = X_train.mean().values 
scenario_raw = np.tile(mean_values, (n_steps, 1)) 
qmax_idx = features.index('qmax')
scenario_raw[:, qmax_idx] = qmax_range
scenario_df = pd.DataFrame(scenario_raw, columns=features)

# --- 5. PLOT ---
plt.figure(figsize=(12, 8))
plt.scatter(df['qmax'], df[target], color='gray', alpha=0.15, s=15, label='Simulations Réelles')

styles = {
    'Ridge':         {'color': 'red',    'ls': '-',  'lw': 4, 'alpha': 0.5},
    'Lasso':         {'color': 'orange', 'ls': '--', 'lw': 2, 'alpha': 1.0},
    'Random Forest': {'color': 'green',  'ls': '-',  'lw': 2, 'alpha': 0.9},
    'XGBoost':       {'color': 'blue',   'ls': '-',  'lw': 3, 'alpha': 0.9},
    'Neural Net':    {'color': 'purple', 'ls': '-.', 'lw': 2, 'alpha': 1.0}
}

print("\n--- GÉNÉRATION DU GRAPHIQUE ---")

for name, model in loaded_models.items():
    try:
        # Préparation Input (Scaling ou Raw)
        if name in scalers and name != 'Random Forest':
            # .values pour éviter le warning "Feature names seen at fit..."
            input_data = scalers[name].transform(scenario_df.values) 
            # Si le modèle a été ré-entraîné sur DataFrame, on peut remettre en DF, 
            # mais souvent array passe partout. Pour sûreté avec XGBoost ré-entrainé:
            if name == 'XGBoost':
                input_data = pd.DataFrame(input_data, columns=features)
        else:
            # Random Forest (Raw)
            input_data = scenario_df

        # Prédiction
        preds = model.predict(input_data)
        
        if hasattr(preds, "flatten"): preds = preds.flatten()
        
        # Filtre physique (pas d'eau négative) pour graph propre
        preds = np.maximum(preds, 0)
        
        s = styles.get(name, {'color': 'black'})
        plt.plot(qmax_range, preds, label=name, **s)
        
    except Exception as e:
        print(f"  ! Erreur d'affichage pour {name}: {e}")

# Finitions
plt.title(f"Vulnérabilité aux Crues : {target.replace('_', ' ').title()}", fontsize=16)
plt.xlabel("Débit du Fleuve $Q_{max}$ ($m^3/s$)", fontsize=14)
plt.ylabel("Hauteur d'eau (m)", fontsize=14)
plt.axvline(x=5500, color='black', linestyle=':', alpha=0.5, label="Crue Typique")
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

output_file = 'graph_comparison_5models_final.png'
plt.savefig(output_file, dpi=300)
print(f"\nSuccès ! Graphique sauvegardé sous : {output_file}")
# plt.show() # Décommenter si exécution locale avec écran