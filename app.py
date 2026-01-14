import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# --- CONFIGURATION ---
st.set_page_config(page_title="FloodRisk: Sully-sur-Loire", page_icon="🌊", layout="wide")

# --- CUSTOM CSS FOR PRESENTATION MODE ---
st.markdown("""
<style>
    .metric-container {background-color: #f0f2f6; padding: 10px; border-radius: 10px;}
    .stAlert {padding: 0.5rem;}
</style>
""", unsafe_allow_html=True)

# --- 1. LOADING ASSETS ---
@st.cache_resource
def load_models_and_scaler():
    base_path = Path("boosting") # Ensure this matches your folder structure
    
    targets = ['parc_chateau', 'centre_sully', 'gare_sully', 'caserne_pompiers']
    models = {}
    
    # Try loading XGBoost models (fallback to others if needed)
    try:
        scaler = joblib.load(base_path / "scaler.pkl")
        for t in targets:
            models[t] = joblib.load(base_path / f"xgboost_{t}.pkl")
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

    return models, scaler, targets

# --- 1. CHARGEMENT MULTI-MODÈLES ---
@st.cache_resource
def load_all_models():
    # Chemins basés sur ton repo
    paths = {
        "XGBoost": Path("boosting"),
        "RandomForest": Path("randomforest/pickels"),  # Attention au dossier "pickels" vs "pickles"
        "Ridge": Path("modele_Ridge") # Si tu as exporté ces modèles
    }
    
    targets = ['parc_chateau', 'centre_sully', 'gare_sully', 'caserne_pompiers']
    model_store = {t: {} for t in targets} # Structure: {'parc_chateau': {'XGBoost': model, 'RF': model...}}
    
    # Charger le Scaler (nécessaire pour Ridge/Lasso/NeuralNet, pas pour RF/XGB en théorie mais utile si standardisé)
    scaler = joblib.load(paths["XGBoost"] / "scaler.pkl")

    # Charger XGBoost
    for t in targets:
        try:
            model_store[t]["XGBoost"] = joblib.load(paths["XGBoost"] / f"xgboost_{t}.pkl")
        except: pass

    # Charger Random Forest (Adapter les noms de fichiers si nécessaire, ex: RF_CasernePompiers.pkl)
    rf_names = {
        'parc_chateau': 'RF_ParcChateau.pkl',
        'centre_sully': 'RF_CentreSully.pkl',
        'gare_sully': 'RF_GareSully.pkl',
        'caserne_pompiers': 'RF_CasernePompiers.pkl'
    }
    for t, filename in rf_names.items():
        try:
            model_store[t]["Random Forest"] = joblib.load(paths["RandomForest"] / filename)
        except: pass

    return model_store, scaler, targets

models, scaler, targets = load_models_and_scaler()
model_store, scaler, targets = load_all_models()

# Mock Coordinates for the Map (Approximate for Sully-sur-Loire POIs)
poi_coords = {
    'parc_chateau': [47.7668, 2.3780],
    'centre_sully': [47.7680, 2.3750],
    'gare_sully': [47.7620, 2.3800],
    'caserne_pompiers': [47.7650, 2.3700]
}

# --- SIDEBAR: SIMULATION CONTROL ---
st.sidebar.header("🎛️ Control Panel")
st.sidebar.caption("Adjust hydraulic parameters to simulate a flood event.")

# Grouping inputs logically
with st.sidebar.expander("💧 Hydraulics (River)", expanded=True):
    qmax = st.slider("Qmax (Flow Rate m3/s)", 3000, 10000, 5500, step=100, help="Peak flow rate of the Loire river.")
    tm = st.slider("Tm (Duration s)", 20000, 60000, 43000)

with st.sidebar.expander("🌳 Topography (Roughness)", expanded=False):
    ks_fp = st.slider("Ks Flood Plain (Vegetation)", 10, 60, 30, help="Lower value = Denser vegetation = More friction")
    ks2 = st.slider("Ks Zone 2", 10, 60, 30)
    ks3 = st.slider("Ks Zone 3", 10, 60, 30)
    ks4 = st.slider("Ks Zone 4", 10, 60, 30)
    er = st.slider("Erosion Coeff", 0.0, 1.0, 0.5)
    of = st.slider("Obstacle Factor", -0.5, 0.5, 0.0)

# Prepare Input Vector
input_data = pd.DataFrame([[er, ks2, ks3, ks4, ks_fp, of, qmax, tm]], 
                          columns=['er', 'ks2', 'ks3', 'ks4', 'ks_fp', 'of', 'qmax', 'tm'])

# --- SIDEBAR ---
st.sidebar.header("🧠 Moteur de Prédiction")
model_choice = st.sidebar.selectbox(
    "Choisir le modèle :",
    ["Ensemble (Moyenne)", "XGBoost", "Random Forest"],
    help="L'ensemble fait la moyenne des modèles pour plus de robustesse."
)

# --- PRÉDICTIONS ---
preds = {}

# On prépare les données (Scaled pour certains, Raw pour d'autres si besoin, ici on simplifie avec scaled partout si entraîné ainsi)
input_scaled = scaler.transform(input_data)
input_scaled_df = pd.DataFrame(input_scaled, columns=input_data.columns)

for t in targets:
    models_available = model_store[t]
    
    if model_choice == "Ensemble (Moyenne)":
        # Moyenne de tous les modèles chargés pour ce lieu
        val_xgb = models_available.get("XGBoost").predict(input_scaled_df)[0] if "XGBoost" in models_available else 0
        val_rf = models_available.get("Random Forest").predict(input_data)[0] if "Random Forest" in models_available else 0
        val = (val_xgb + val_rf) / 2
        
    elif model_choice == "XGBoost":
        val = models_available["XGBoost"].predict(input_scaled_df)[0]
        
    elif model_choice == "Random Forest":
        val = models_available["Random Forest"].predict(input_data)[0]

    preds[t] = max(0.0, val)

# --- MAIN DASHBOARD ---
st.title("🌊 Smart Flood Defense: Sully-sur-Loire")
st.markdown("### Real-time Impact Assessment")

# TAB STRUCTURE
tab1, tab2, tab3 = st.tabs(["🚀 Simulation & Map", "📈 Sensitivity Analysis", "🧠 Model Transparency"])

# --- TAB 1: SIMULATION ---
with tab1:
    # 1. KPI Metrics Row
    cols = st.columns(4)
    map_data = []
    
    for i, t in enumerate(targets):
        level = preds[t]
        name = t.replace("_", " ").title()
        
        # Color Logic
        if level < 0.1: status, color = "Safe", "normal"
        elif level < 1.0: status, color = "Warning", "off"
        else: status, color = "CRITICAL", "inverse"

        with cols[i]:
            st.metric(label=name, value=f"{level:.2f} m", delta=status, delta_color=color)
        
        # Prepare data for map
        map_data.append({
            "name": name,
            "lat": poi_coords[t][0],
            "lon": poi_coords[t][1],
            "water_level": level,
            "size": (level + 0.5) * 100, # Bubble size based on flood
            "color": [255, 0, 0, 200] if level > 1.0 else [0, 255, 0, 150]
        })

    st.markdown("---")
    
    # 2. Map & Chart Split
    c1, c2 = st.columns([3, 2])
    
    with c1:
        st.subheader("📍 Impact Map")
        df_map = pd.DataFrame(map_data)
        st.map(df_map, latitude="lat", longitude="lon", size="size", zoom=13)

    with c2:
        st.subheader("📊 Comparative Levels")
        # Bar Chart
        fig, ax = plt.subplots(figsize=(5,4))
        colors = ['green' if x < 1.0 else 'red' for x in preds.values()]
        ax.bar(list(preds.keys()), list(preds.values()), color=colors)
        ax.set_ylabel("Water Level (m)")
        ax.set_xticklabels([t.replace("_", "\n").title() for t in targets], rotation=45)
        st.pyplot(fig)

# --- TAB 2: SENSITIVITY (DYNAMICS) ---
with tab2:
    st.subheader("📉 How does Flow Rate (Qmax) impact flooding?")
    st.write("This simulation holds all parameters constant (vegetation, topography) and only varies the River Flow.")
    
    # Create the curve data
    q_range = np.linspace(3000, 12000, 50)
    sensitivity_data = []
    
    # Repeat the current user input 50 times
    temp_df = pd.concat([input_data]*50, ignore_index=True)
    temp_df['qmax'] = q_range # Overwrite Qmax column
    
    # Scale
    temp_scaled = scaler.transform(temp_df)
    temp_scaled_df = pd.DataFrame(temp_scaled, columns=input_data.columns)
    
    # Predict for all targets
    chart_data = pd.DataFrame(index=q_range)
    for t in targets:
        p = models[t].predict(temp_scaled_df)
        chart_data[t] = np.maximum(p, 0)
        
    st.line_chart(chart_data)
    st.caption(f"Current Simulation Point: Qmax = {qmax} m3/s")

# --- TAB 3: MODEL TRANSPARENCY (THE WHY) ---
with tab3:
    st.header("🔍 Model Interpretation & Checking")
    
    st.markdown("### 1. Quels facteurs influencent la crue ?")
    st.info("Comparaison de ce que les modèles 'regardent' pour faire leur prédiction.")
    
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Vision Globale (Random Forest)**")
        st.image("randomforest/graphs/graph1_global_importance.png", use_column_width=True)
    with c2:
        st.write("**Vision Locale (XGBoost - Parc Château)**")
        st.image("boosting/importance_parc_chateau.png", use_column_width=True)

    st.markdown("### 2. Fiabilité des modèles")
    st.write("Performance sur les données de test (Graphiques générés par l'équipe)")
    
    lieu_valid = st.select_slider("Choisir un lieu pour voir la précision :", options=targets)
    
    img_path = f"neural_network/plots/{lieu_valid}/actual_vs_predicted.png"
    if os.path.exists(img_path):
        st.image(img_path, caption=f"Prédiction vs Réalité ({lieu_valid})")
    else:
        st.warning("Graphique de validation manquant pour ce lieu.")