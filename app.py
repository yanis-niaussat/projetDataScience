import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONFIGURATION ---
st.set_page_config(page_title="Simulateur Inondation Sully", page_icon="🌊", layout="wide")
st.title("🌊 Prédiction de Crue - Sully-sur-Loire")

# --- CHARGEMENT DES ASSETS ---
@st.cache_resource
def load_assets():
    targets = ['parc_chateau', 'centre_sully', 'gare_sully', 'caserne_pompiers']
    models = {}
    base_dir = Path(__file__).parent
    
    # Adapte le chemin selon ton dossier réel
    models_dir = base_dir / "boosting" 

    for t in targets:
        # Chargement des modèles XGBoost
        model_path = models_dir / f"xgboost_{t}.pkl"
        if model_path.exists():
            models[t] = joblib.load(str(model_path))
        else:
            st.error(f"Modèle introuvable : {model_path}")
            return None, None, None

    # Chargement du Scaler
    scaler_path = models_dir / "scaler.pkl"
    if scaler_path.exists():
        scaler = joblib.load(str(scaler_path))
    else:
        st.error("Scaler introuvable")
        return None, None, None
        
    return models, scaler, targets

models, scaler, targets = load_assets()

if not models:
    st.stop()

# --- SIDEBAR : PARAMÈTRES HYDRAULIQUES ---
st.sidebar.header("⚙️ Paramètres de la Crue")

# Tooltips pour expliquer les variables physiques
qmax = st.sidebar.slider("Débit de pointe (Qmax)", 3000.0, 25000.0, 5500.0, step=100.0, help="Débit maximum du fleuve en m3/s")
tm = st.sidebar.slider("Durée Montée (tm)", 2000.0, 25000.0, 14000.0, help="Vitesse à laquelle la crue arrive")
st.sidebar.markdown("---")
st.sidebar.subheader("Rugosité (Frottement)")
ks_fp = st.sidebar.slider("Lit Majeur (Ks_fp)", 5.0, 20.0, 15.0, help="Végétation dans la plaine inondable (plus bas = plus dense)")
ks2 = st.sidebar.slider("Zone 2 (Ks2)", 18.0, 38.0, 28.0)
ks3 = st.sidebar.slider("Zone 3 (Ks3)", 27.0, 47.0, 37.0)
ks4 = st.sidebar.slider("Zone 4 (Ks4)", 18.0, 38.0, 28.0)
er = st.sidebar.slider("Érosion (er)", 0.0, 1.0, 0.5)
of = st.sidebar.slider("Facteur Ouvrage (of)", -0.2, 0.2, 0.0)

# Dataframe pour la prédiction
data = {
    'er': er, 'ks2': ks2, 'ks3': ks3, 'ks4': ks4, 
    'ks_fp': ks_fp, 'of': of, 'qmax': qmax, 'tm': tm
}
input_df = pd.DataFrame(data, index=[0])

# --- PRÉDICTIONS ---
input_scaled = scaler.transform(input_df)
input_scaled_df = pd.DataFrame(input_scaled, columns=input_df.columns)

results = {}
for t in targets:
    pred = models[t].predict(input_scaled_df)[0]
    results[t] = max(0.0, pred)

# --- AFFICHAGE PRINCIPAL (TABS) ---
tab1, tab2 = st.tabs(["📊 Tableau de Bord", "🧠 Analyse & Sensibilité"])

with tab1:
    # --- INDICATEURS CLÉS (KPI) ---
    st.markdown("### 🚨 Statut des Zones Critiques")
    cols = st.columns(4)
    for i, t in enumerate(targets):
        val = results[t]
        label = t.replace("_", " ").title()
        
        with cols[i]:
            if val < 0.05:
                st.metric(label=label, value=f"{val:.2f} m", delta="Sec", delta_color="normal")
            elif val < 0.50: # Seuil ajusté pour l'exemple
                st.metric(label=label, value=f"{val:.2f} m", delta="Risque Faible", delta_color="off")
            else:
                st.metric(label=label, value=f"{val:.2f} m", delta="Inondation", delta_color="inverse")

    # --- GRAPHIQUE BARRES ---
    st.markdown("### Comparaison des Hauteurs d'eau")
    fig, ax = plt.subplots(figsize=(10, 4))
    labels_plot = [t.replace("_", "\n").title() for t in targets]
    colors = ['#2a9d8f' if val < 0.5 else '#e76f51' for val in results.values()]
    
    bars = ax.bar(labels_plot, results.values(), color=colors)
    ax.set_ylabel("Hauteur d'eau (m)")
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label="Seuil Vigilance") # Ligne de seuil
    ax.legend()
    
    # Annotations
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.05, f"{height:.2f}m", ha='center', va='bottom', fontweight='bold')
    
    st.pyplot(fig)

with tab2:
    st.markdown("### 📈 Analyse de Sensibilité (Focus Qmax)")
    st.info("Ce graphique montre comment la hauteur d'eau évoluerait si le débit (Qmax) augmentait, toutes choses égales par ailleurs.")
    
    # --- CALCUL DE LA COURBE DE SENSIBILITÉ ---
    # On génère une plage de Qmax de 3000 à 10000
    q_range = np.linspace(3000, 10000, 50)
    
    # On crée une matrice où tout est fixe sauf Qmax
    temp_df = pd.concat([input_df]*50, ignore_index=True)
    temp_df['qmax'] = q_range
    
    # On scale
    temp_scaled = scaler.transform(temp_df)
    temp_scaled_df = pd.DataFrame(temp_scaled, columns=input_df.columns)
    
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    
    for t in targets:
        preds = models[t].predict(temp_scaled_df)
        # On force à 0 si négatif
        preds = [max(0, p) for p in preds]
        ax2.plot(q_range, preds, label=t.replace("_", " ").title())
    
    # Point actuel
    ax2.axvline(x=qmax, color='red', linestyle='--', label=f"Simulation actuelle ({qmax} m3/s)")
    
    ax2.set_xlabel("Débit Qmax (m3/s)")
    ax2.set_ylabel("Hauteur d'eau prédite (m)")
    ax2.set_title("Réponse des zones inondables face à la montée du débit")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    st.pyplot(fig2)

    # --- IMPORTANCE DES VARIABLES (STATIC) ---
    st.markdown("### 🧬 Quels facteurs influencent le plus chaque zone ?")
    st.write("Basé sur l'entraînement XGBoost (Graphiques générés par `model_xgboost.py`)")
    
    col_img1, col_img2 = st.columns(2)
    
    # Affichage des images si elles existent
    # Assurez-vous que les images sont dans le dossier racine ou boosting
    import os
    for i, t in enumerate(targets):
        # Chemin hypothétique, à adapter
        img_path = f"importance_{t}.png" 
        if os.path.exists(img_path):
            if i % 2 == 0:
                col_img1.image(img_path, caption=f"Importance - {t}", use_column_width=True)
            else:
                col_img2.image(img_path, caption=f"Importance - {t}", use_column_width=True)
        else:
            # Fallback text if image missing
            pass