import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib
import numpy as np
import os
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# --- 1. CHARGEMENT ET PRÉPARATION ---
print("Chargement des données...")
df = pd.read_csv("training_matrix_sully.csv")

features = ['er', 'ks2', 'ks3', 'ks4', 'ks_fp', 'of', 'qmax', 'tm']
targets = ['parc_chateau', 'centre_sully', 'gare_sully', 'caserne_pompiers']

X = df[features]
y = df[targets]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=features)

# --- 2. CONFIGURATION DE LA CROSS-VALIDATION COMMUNE ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)

models = {}
scores_r2_cv_display = {}
scores_rmse_cv_display = {}

# --- 3. ENTRAÎNEMENT ET VALIDATION CROISÉE ---
print("\n--- DÉBUT DE L'ENTRAÎNEMENT XGBOOST ---")

for lieu in targets:
    print(f"\n🔄 Analyse pour : {lieu}")
    
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    
    # Cross-Validation
    cv_results = cross_validate(
        model, X_train_scaled, y_train[lieu], 
        cv=kf, 
        scoring=['r2', 'neg_root_mean_squared_error'],
        n_jobs=-1
    )
    
    r2_cv = cv_results['test_r2'].mean()
    rmse_cv = -cv_results['test_neg_root_mean_squared_error'].mean()
    
    scores_r2_cv_display[lieu] = r2_cv
    scores_rmse_cv_display[lieu] = rmse_cv

    print(f"   📊 R2 moyen (CV) : {r2_cv:.4f}")

    # Entraînement final
    model.fit(X_train_scaled, y_train[lieu])
    models[lieu] = model
    
    # Sauvegarde du modèle .pkl au même endroit
    joblib.dump(model, f"xgboost_{lieu}.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("✅ Scaler sauvegardé sous 'scaler.pkl'")

# --- 4. ANALYSE ET SAUVEGARDE DES GRAPHIQUES ---
print("\n--- GÉNÉRATION ET SAUVEGARDE DES GRAPHIQUES ---")
for lieu_analyse in targets:
    print(f"💾 Sauvegarde du graphique pour {lieu_analyse}...")
    
    plt.figure(figsize=(10, 6)) # Définit une taille propre
    
    try:
        # Importance native
        xgb.plot_importance(
            models[lieu_analyse],
            importance_type='weight',
            title=f"Importance des variables - {lieu_analyse}",
        )
    except Exception:
        # Repli sur permutation importance
        r = permutation_importance(models[lieu_analyse], X_test_scaled, y_test[lieu_analyse], n_repeats=5)
        importances = pd.Series(r.importances_mean, index=features)
        importances.sort_values().plot(kind="barh")
        plt.title(f"Permutation importance - {lieu_analyse}")

    plt.tight_layout()
    
    # --- LA LIGNE CLÉ : SAUVEGARDE ---
    # Enregistre en PNG avec une bonne résolution (dpi)
    plt.savefig(f"importance_{lieu_analyse}.png", dpi=300)
    
    # Optionnel : décommente la ligne suivante si tu veux quand même les voir s'afficher
    # plt.show() 
    
    plt.close()

# --- 5. RÉSUMÉ FINAL ---
print("\n" + "="*50)
print(f"{'LIEU':<20} | {'R2 MOYEN (CV)':<15} | {'RMSE (m)':<10}")
print("-"*50)
for lieu in targets:
    print(f"{lieu.upper():<20} | {scores_r2_cv_display[lieu]:.4f}          | {scores_rmse_cv_display[lieu]:.4f}")
print("="*50)

print("\nC'est prêt ! Tu as maintenant les fichiers .pkl ET les images .png dans ton dossier.")