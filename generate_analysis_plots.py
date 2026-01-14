import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# --- CONFIG ---
os.makedirs("assets/graphs", exist_ok=True)
TARGETS = ['parc_chateau', 'centre_sully', 'gare_sully', 'caserne_pompiers']
FEATURES = ['er', 'ks2', 'ks3', 'ks4', 'ks_fp', 'of', 'qmax', 'tm']

# --- 1. CHARGEMENT DONNÉES ---
print("Chargement des données...")
try:
    df = pd.read_csv('training_matrix_sully.csv')
    X = df[FEATURES]
    y = df[TARGETS]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
except:
    print("⚠️ Fichier CSV manquant, on passera la génération ML.")
    df = None

# --- 2. CHARGEMENT SCALERS ---
scalers = {}
try: scalers['xgb'] = joblib.load("boosting/scaler.pkl")
except: pass
try: scalers['ridge'] = joblib.load("modele_Ridge/pickle/ridge_lasso_scaler.pkl")
except: pass
# Le Lasso utilise le même scaler que le Ridge, donc on utilisera scalers['ridge']

# --- 3. CHARGEMENT MODÈLES ---
print("Chargement des modèles...")
model_store = {t: {} for t in TARGETS}

for t in TARGETS:
    # A. XGBoost
    try: model_store[t]['XGBoost'] = joblib.load(f"boosting/xgboost_{t}.pkl")
    except: pass
    
    # B. Random Forest (Attention aux noms de fichiers)
    # Ex: RF_GareSully.pkl
    t_camel = t.replace('_',' ').title().replace(' ','')
    try: model_store[t]['Random Forest'] = joblib.load(f"randomforest/pickels/RF_{t_camel}.pkl")
    except: pass

    # C. Ridge
    try: model_store[t]['Ridge'] = joblib.load(f"modele_Ridge/pickle/ridge/ridge_{t}.pkl")
    except: pass
    
    # D. Lasso (Ajouté)
    try: model_store[t]['Lasso'] = joblib.load(f"modele_Ridge/pickle/lasso/lasso_{t}.pkl")
    except: pass

# --- 4. FONCTION D'ASSEMBLAGE (POUR NEURAL NET) ---
def assemble_existing_plots(model_name, plot_filename_pattern):
    """Réutilise les images existantes du Neural Network"""
    print(f"♻️ Assemblage des images existantes pour {model_name}...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    fig.suptitle(f"{model_name} (Données pré-calculées)", fontsize=16)
    
    found_any = False
    for i, t in enumerate(TARGETS):
        path = f"neural_network/plots/{t}/{plot_filename_pattern}"
        ax = axes[i]
        if os.path.exists(path):
            img = mpimg.imread(path)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(t.replace('_', ' ').title(), fontsize=12)
            found_any = True
        else:
            ax.text(0.5, 0.5, "Image non trouvée", ha='center')
            ax.axis('off')

    if found_any:
        safe_name = model_name.replace(" ", "")
        output_name = "1_pred_vs_actual" if "actual" in plot_filename_pattern else "2_residuals_vs_qmax"
        save_path = f"assets/graphs/{output_name}_{safe_name}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        print(f"  -> Assemblé : {save_path}")
    plt.close()

# --- 5. GÉNÉRATION ML (POUR LES AUTRES) ---
def generate_ml_plots(model_name):
    if df is None: return
    print(f"📊 Génération calculée pour : {model_name}...")
    
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 12))
    axes1 = axes1.flatten()
    fig1.suptitle(f"Performance : {model_name} (Pred vs Actual)", fontsize=16)
    
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))
    axes2 = axes2.flatten()
    fig2.suptitle(f"Sécurité : {model_name} (Erreurs vs Débit)", fontsize=16)
    
    has_data = False
    for i, t in enumerate(TARGETS):
        model = model_store[t].get(model_name)
        if model:
            # Préparation des données (Scaling)
            X_in = X_test.copy()
            
            # Logique de Scaling
            if model_name == 'XGBoost' and 'xgb' in scalers:
                X_in = pd.DataFrame(scalers['xgb'].transform(X_test), columns=FEATURES)
            elif (model_name == 'Ridge' or model_name == 'Lasso') and 'ridge' in scalers:
                # Ridge et Lasso partagent le même scaler
                X_in = pd.DataFrame(scalers['ridge'].transform(X_test), columns=FEATURES)
            # Random Forest utilise X_test (raw) par défaut
            
            try:
                pred = model.predict(X_in)
                if hasattr(pred, "flatten"): pred = pred.flatten()
                
                y_true = y_test[t]
                residuals = y_true - pred
                has_data = True
                r2 = r2_score(y_true, pred)
                
                # Plot 1: Pred vs Actual
                ax1 = axes1[i]
                sns.scatterplot(x=y_true, y=pred, ax=ax1, alpha=0.5, color='teal')
                m = max(y_true.max(), pred.max())
                ax1.plot([0, m], [0, m], 'r--')
                ax1.set_title(f"{t} (R2={r2:.2f})")
                ax1.set_xlabel("Réalité")
                ax1.set_ylabel("Prédiction")
                
                # Plot 2: Residuals
                ax2 = axes2[i]
                sc = ax2.scatter(X_test['qmax'], residuals, c=residuals, cmap='coolwarm', alpha=0.6)
                ax2.axhline(0, color='k', ls='--')
                ax2.set_title(f"{t} - Erreur vs Qmax")
                ax2.set_ylabel("Erreur (m)")
                
            except Exception as e:
                print(f"  ! Erreur {model_name} / {t} : {e}")

    if has_data:
        safe_name = model_name.replace(" ", "")
        fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig1.savefig(f"assets/graphs/1_pred_vs_actual_{safe_name}.png")
        
        fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig2.savefig(f"assets/graphs/2_residuals_vs_qmax_{safe_name}.png")
        print(f"  -> Sauvegardé pour {model_name}")
    plt.close('all')

# --- MAIN EXECUTION ---

# 1. Modèles à calculer (Ajout de Lasso et RF explicite)
for m in ['XGBoost', 'Random Forest', 'Ridge', 'Lasso']:
    generate_ml_plots(m)

# 2. Neural Net (Assemblage des images existantes)
assemble_existing_plots('Neural Net', 'actual_vs_predicted.png')
assemble_existing_plots('Neural Net', 'residuals_vs_qmax.png')

print("\n✅ Terminé ! Tous les graphiques (XGB, RF, Ridge, Lasso, NN) sont prêts.")