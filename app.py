import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
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

# --- HELPER: SAFE IMAGE ---
BASE_DIR = Path(__file__).parent

def safe_image(path, caption=None, **kwargs):
    # Convert string to Path object
    p = Path(path)
    # If not absolute, make it relative to BASE_DIR
    if not p.is_absolute():
        p = BASE_DIR / p
        
    if p.exists():
        st.image(str(p), caption=caption, **kwargs)
    else:
        st.warning(f"Image not found: {p.name} (Placeholder)")

# --- HELPER: ROBUST PREDICTION ---
def safe_scalar_predict(model, X):
    """Ensures prediction returns a single float scalar."""
    try:
        p = model.predict(X)
        # Convert to numpy array and flatten if it's multidimensional (like (N,1) from Keras)
        p = np.array(p).flatten()
        if len(p) > 0:
            return float(p[0])
    except Exception as e:
        # st.warning(f"Prediction error: {e}") 
        pass
    return 0.0

def safe_vector_predict(model, X):
    """Ensures prediction returns a 1D array of floats."""
    try:
        p = model.predict(X)
        return np.array(p).flatten()
    except Exception as e:
        return np.zeros(X.shape[0])

# --- 1. CHARGEMENT MULTI-MODÈLES ET SCALERS ---
@st.cache_resource
def load_all_models():
    # Chemins absolus basés sur l'emplacement du script
    BASE_DIR = Path(__file__).parent
    paths = {
        "XGBoost": BASE_DIR / "boosting",
        "RandomForest": BASE_DIR / "randomforest/pickels",
        "Ridge": BASE_DIR / "modele_Ridge/pickle",
        "NeuralNet": BASE_DIR / "neural_network/tensorflow/pkls"
    }
    
    targets = ['parc_chateau', 'centre_sully', 'gare_sully', 'caserne_pompiers']
    model_store = {t: {} for t in targets}
    scalers = {}

    # 1. Scalers
    try: scalers["XGBoost"] = joblib.load(paths["XGBoost"] / "scaler.pkl")
    except Exception as e: print(f"XGB Scaler failed: {e}")
    
    # RandomForest
    try: scalers["RandomForest"] = joblib.load(paths["RandomForest"] / "rf_scaler.pkl")
    except Exception as e: print(f"RF Scaler failed: {e}") 

    try: scalers["Ridge"] = joblib.load(paths["Ridge"] / "ridge_lasso_scaler.pkl")
    except Exception as e: print(f"Ridge Scaler failed: {e}")

    try: scalers["NeuralNet"] = joblib.load(paths["NeuralNet"] / "scaler.pkl")
    except Exception as e: print(f"NN Scaler failed: {e}")

    # 2. Models
    # XGBoost
    for t in targets:
        try: model_store[t]["XGBoost"] = joblib.load(paths["XGBoost"] / f"xgboost_{t}.pkl")
        except Exception as e: print(f"Failed to load XGBoost {t}: {e}")

    # Random Forest
    rf_names = {
        'parc_chateau': 'RF_ParcChateau.pkl',
        'centre_sully': 'RF_CentreSully.pkl',
        'gare_sully': 'RF_GareSully.pkl',
        'caserne_pompiers': 'RF_CasernePompiers.pkl'
    }
    for t, filename in rf_names.items():
        try: model_store[t]["RandomForest"] = joblib.load(paths["RandomForest"] / filename)
        except Exception as e: st.error(f"Failed to load RandomForest {t}: {e}")

    # Ridge - Models are in a subfolder 'ridge'
    for t in targets:
        try: model_store[t]["Ridge"] = joblib.load(paths["Ridge"] / f"ridge/ridge_{t}.pkl")
        except Exception as e: st.error(f"Failed to load Ridge {t}: {e}")

    # NeuralNet
    for t in targets:
        try: 
            model_store[t]["NeuralNet"] = joblib.load(paths["NeuralNet"] / f"keras_model_{t}.pkl")
        except Exception as e: 
            st.error(f"Failed to load NeuralNet {t}: {e}")
    
    return model_store, scalers, targets

model_store, scalers, targets = load_all_models()

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
st.sidebar.header("Moteur de Prédiction")
model_choice = st.sidebar.selectbox(
    "Choisir le modèle :",
    ["Ensemble (Moyenne)", "XGBoost", "RandomForest", "Ridge", "NeuralNet"],
    help="L'ensemble fait la moyenne des modèles pour plus de robustesse."
)

# --- PRÉDICTIONS ---
preds = {}

# Helper to scale if scaler exists, else raw
def get_scaled(name, df):
    if name in scalers and scalers[name] is not None:
        try:
            return pd.DataFrame(scalers[name].transform(df), columns=df.columns)
        except Exception as e:
            # Silently return raw if scaling fails
            return df
    return df

input_xgb = get_scaled("XGBoost", input_data)
input_rf = get_scaled("RandomForest", input_data)
input_ridge = get_scaled("Ridge", input_data)
input_nn = get_scaled("NeuralNet", input_data)

for t in targets:
    models_available = model_store.get(t, {})
    val = 0.0
    
    if model_choice == "Ensemble (Moyenne)":
        count = 0
        total = 0.0
        
        if "XGBoost" in models_available:
            total += safe_scalar_predict(models_available["XGBoost"], input_xgb)
            count += 1
        if "RandomForest" in models_available:
            total += safe_scalar_predict(models_available["RandomForest"], input_rf)
            count += 1
        if "Ridge" in models_available:
            total += safe_scalar_predict(models_available["Ridge"], input_ridge)
            count += 1
        if "NeuralNet" in models_available:
            total += safe_scalar_predict(models_available["NeuralNet"], input_nn)
            count += 1
            
        val = total / count if count > 0 else 0.0
        
    elif model_choice == "XGBoost" and "XGBoost" in models_available:
        val = safe_scalar_predict(models_available["XGBoost"], input_xgb)
        
    elif model_choice == "RandomForest" and "RandomForest" in models_available:
        val = safe_scalar_predict(models_available["RandomForest"], input_rf)

    elif model_choice == "Ridge" and "Ridge" in models_available:
        val = safe_scalar_predict(models_available["Ridge"], input_ridge)

    elif model_choice == "NeuralNet" and "NeuralNet" in models_available:
        val = safe_scalar_predict(models_available["NeuralNet"], input_nn)

    preds[t] = max(0.0, val)

# --- MAIN DASHBOARD ---
st.title("🌊 Smart Flood Defense: Sully-sur-Loire")
st.markdown("### Real-time Impact Assessment")

# TAB STRUCTURE
tab1, tab2, tab3 = st.tabs(["Simulation & Map", "Sensitivity Analysis", "Model Transparency"])

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
        ax.set_xticks(range(len(targets)))
        ax.set_xticklabels([t.replace("_", "\n").title() for t in targets], rotation=45)
        st.pyplot(fig)

# --- TAB 2: SENSITIVITY (DYNAMICS) ---
with tab2:
    st.subheader("📉 How does Flow Rate (Qmax) impact flooding?")
    st.write(f"Simulation based on: **{model_choice}**")
    
    # Create the curve data
    q_range = np.linspace(3000, 12000, 50)
    
    # Repeat the current user input 50 times
    temp_df = pd.concat([input_data]*50, ignore_index=True)
    temp_df['qmax'] = q_range # Overwrite Qmax column
    
    # Pre-calculate scaled versions for the sensitivity loop
    temp_xgb = get_scaled("XGBoost", temp_df)
    temp_rf = get_scaled("RandomForest", temp_df)
    temp_ridge = get_scaled("Ridge", temp_df)
    temp_nn = get_scaled("NeuralNet", temp_df)
    
    # Predict for all targets
    chart_data = pd.DataFrame(index=q_range)
    
    for t in targets:
        models_available = model_store.get(t, {})
        
        # Default to 0 values
        p = np.zeros(50)
        
        if model_choice == "Ensemble (Moyenne)":
            v_xgb = safe_vector_predict(models_available["XGBoost"], temp_xgb) if "XGBoost" in models_available else np.zeros(50)
            v_rf = safe_vector_predict(models_available["RandomForest"], temp_rf) if "RandomForest" in models_available else np.zeros(50)
            v_ridge = safe_vector_predict(models_available["Ridge"], temp_ridge) if "Ridge" in models_available else np.zeros(50)
            v_nn = safe_vector_predict(models_available["NeuralNet"], temp_nn) if "NeuralNet" in models_available else np.zeros(50)
            p = (v_xgb + v_rf + v_ridge + v_nn) / 4
            
        elif model_choice == "XGBoost" and "XGBoost" in models_available:
            p = safe_vector_predict(models_available["XGBoost"], temp_xgb)
                
        elif model_choice == "RandomForest" and "RandomForest" in models_available:
             p = safe_vector_predict(models_available["RandomForest"], temp_rf)

        elif model_choice == "Ridge" and "Ridge" in models_available:
             p = safe_vector_predict(models_available["Ridge"], temp_ridge)

        elif model_choice == "NeuralNet" and "NeuralNet" in models_available:
             p = safe_vector_predict(models_available["NeuralNet"], temp_nn)

        chart_data[t] = np.maximum(p, 0)
        
    st.line_chart(chart_data)
    st.caption(f"Current Simulation Point: Qmax = {qmax} m3/s")

# --- TAB 3: MODEL TRANSPARENCY (THE WHY) ---
with tab3:
    st.header("🔍 Model Interpretation & Performance")
    
    # 1. GLOBAL PERFORMANCE
    st.markdown("## 1. Global Model Performance")
    st.info("Comparaison des performances des différents modèles sur l'ensemble des cibles.")
    
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Comparaison Globale (R2 Score)**")
        safe_image("randomforest/graphs/graph_comparison_5models_final.png", 
                  caption="Comparaison des performances (5 modèles)", use_column_width=True)
    with c2:
        st.write("**Précision Globale**")
        safe_image("randomforest/graphs/graph_global_accuracy.png", 
                  caption="Précision globale par modèle", use_column_width=True)

    st.markdown("---")

    # 2. FEATURE IMPORTANCE
    st.markdown("## 2. Feature Importance (Explicabilité)")
    st.write("Quels paramètres influencent le plus les prédictions ?")
    
    # Global & Ridge
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Vision Globale (Random Forest)**")
        safe_image("randomforest/graphs/graph1_global_importance.png", 
                  caption="Importance moyenne des variables (RF)", use_column_width=True)
    with c2:
        st.write("**Stabilité des Coefficients (Ridge/Lasso)**")
        safe_image("modele_Ridge/graphs/ridge_lasso_paths.png", 
                  caption="Chemin de régularisation", use_column_width=True)
        
    # Local Decomposition
    st.markdown("#### Focus Local : Influence par Lieu")
    location_choice = st.selectbox("Choisir le lieu pour l'explicabilité :", targets, key="loc_imp")
    
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**XGBoost Importance ({location_choice})**")
        safe_image(f"boosting/importance_{location_choice}.png", use_column_width=True)
    with c2:
        st.write("**Comparaison Sectorielle (RF)**")
        safe_image("randomforest/graphs/graph4_location_importances.png", use_column_width=True)

    st.markdown("---")

    # 3. PHYSICS CONSISTENCY
    st.markdown("## 3. Cohérence Physique")
    st.write("Les modèles respectent-ils les lois de l'hydraulique ? (ex: le niveau monte si Qmax augmente)")
    safe_image("physics_analysis/reynolds_vs_water_level.png", use_column_width=True)
    st.write("On peut voir que les données sont cohérentes avec la loi de Reynolds. Les lois de l'hydraulique sont respectées.")
    st.markdown("---")
    
    # 4. NEURAL NETWORK DEEP DIVE
    st.markdown("## 4. Focus: Deep Learning Diagnostics")
    st.write("Analyse détaillée des performances du **Réseau de Neurones** (Meilleur modèle complexe).")
    
    loc_nn = st.selectbox("Choisir le lieu pour le diagnostic :", targets, key="loc_nn", format_func=lambda x: x.replace("_", " ").title())
    
    folder_map = {
        'parc_chateau': 'Parc_Chateau',
        'centre_sully': 'Centre_Sully',
        'gare_sully': 'Gare_Sully',
        'caserne_pompiers': 'Caserne_Pompiers'
    }
    target_folder = folder_map.get(loc_nn, "Global")
    
    c1, c2 = st.columns(2)
    with c1:
        safe_image(f"neural_network/plots/{target_folder}/actual_vs_predicted.png", caption="Actual vs Predicted")
        safe_image(f"neural_network/plots/{target_folder}/residuals_vs_qmax.png", caption="Residuals vs Qmax (Physique)")
    with c2:
        safe_image(f"neural_network/plots/{target_folder}/residuals_vs_predicted.png", caption="Residuals Distribution")
        safe_image(f"neural_network/plots/{target_folder}/error_distribution.png", caption="Error Histogram")