import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go # Assure-toi d'avoir fait cet import en haut

# --- CONFIGURATION ---
st.set_page_config(page_title="FloodRisk: Sully-sur-Loire", page_icon="🌊", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .metric-container {background-color: #f0f2f6; padding: 10px; border-radius: 10px;}
    .stAlert {padding: 0.5rem;}
</style>
""", unsafe_allow_html=True)

# --- 1. CHARGEMENT DES DONNÉES (POUR LE BENCHMARK) ---
@st.cache_data
def load_data():
    csv_path = "training_matrix_sully.csv"
    if not os.path.exists(csv_path):
        return None, None, None, None, None
    
    df = pd.read_csv(csv_path)
    features = ['er', 'ks2', 'ks3', 'ks4', 'ks_fp', 'of', 'qmax', 'tm']
    targets = ['parc_chateau', 'centre_sully', 'gare_sully', 'caserne_pompiers']
    
    X = df[features]
    y = df[targets]
    
    # Split identique au Notebook pour garantir la validité du test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, features

# --- 2. CHARGEMENT DES MODÈLES ET SCALERS ---
@st.cache_resource
def load_all_assets():
    targets = ['parc_chateau', 'centre_sully', 'gare_sully', 'caserne_pompiers']
    model_store = {t: {} for t in targets}
    scalers = {}
    
    # --- A. CHARGEMENT DES SCALERS ---
    # On essaie de charger les scalers spécifiques, sinon on utilisera un fallback
    try:
        scalers['xgb'] = joblib.load("boosting/scaler.pkl")
    except: pass
    
    try:
        scalers['ridge'] = joblib.load("modele_Ridge/pickle/ridge_lasso_scaler.pkl")
    except: pass
    
    try:
        # Neural Net Scaler
        scalers['nn'] = joblib.load("neural_network/tensorflow/scaler.pkl")
    except: pass

    # --- B. CHARGEMENT DES MODÈLES ---
    
    # 1. XGBoost
    for t in targets:
        try:
            model_store[t]["XGBoost"] = joblib.load(f"boosting/xgboost_{t}.pkl")
        except: pass

    # 2. Random Forest (Noms de fichiers CamelCase)
    rf_names = {
        'parc_chateau': 'RF_ParcChateau.pkl',
        'centre_sully': 'RF_CentreSully.pkl',
        'gare_sully': 'RF_GareSully.pkl',
        'caserne_pompiers': 'RF_CasernePompiers.pkl'
    }
    for t, filename in rf_names.items():
        try:
            path = Path("randomforest/pickels") / filename
            model_store[t]["Random Forest"] = joblib.load(path)
        except: pass

    # 3. Ridge & Lasso
    for t in targets:
        try:
            model_store[t]["Ridge"] = joblib.load(f"modele_Ridge/pickle/ridge/ridge_{t}.pkl")
            model_store[t]["Lasso"] = joblib.load(f"modele_Ridge/pickle/lasso/lasso_{t}.pkl")
        except: pass
        
    # 4. Neural Network (Keras)
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        for t in targets:
            try:
                path = f"neural_network/tensorflow/keras_model_{t}.keras"
                if os.path.exists(path):
                    model_store[t]["Neural Net"] = load_model(path, compile=False)
            except: pass
    except ImportError:
        pass # Pas de tensorflow installé

    return model_store, scalers, targets

# Initialisation
X_train, X_test, y_train, y_test, feature_names = load_data()
model_store, scalers, targets = load_all_assets()

# Mock Coordinates
poi_coords = {
    'parc_chateau': [47.7668, 2.3780],
    'centre_sully': [47.7680, 2.3750],
    'gare_sully': [47.7620, 2.3800],
    'caserne_pompiers': [47.7650, 2.3700]
}

# --- DANS LA SIDEBAR ---
with st.sidebar.expander("💧 Hydraulique (Loire)", expanded=True):
    # AVANT : qmax = st.slider("Qmax ...", 3000, 10000, 5500, step=100)
    # APRÈS : On monte à 25 000 pour chercher les points de rupture de la Caserne
    qmax = st.slider("Qmax (Débit m3/s)", 2000, 25000, 5500, step=100, help="Débit de la Loire. > 10000 = Crue majeure.")
    tm = st.slider("Tm (Durée s)", 20000, 80000, 43000)

with st.sidebar.expander("🌳 Topographie (Rupture & Friction)", expanded=False):
    st.caption("Paramètres de Digue")
    # Clarifier que 'er' est critique pour la Caserne
    er = st.slider("Coeff Erosion (Brèche)", 0.0, 1.0, 0.5, help="1.0 = La digue cède totalement.")
    of = st.slider("Facteur Obstacle", -0.5, 0.5, 0.0)
    
    st.caption("Friction (Ks)")
    ks_fp = st.slider("Ks Plaine", 10, 60, 30)
    ks2 = st.slider("Ks Zone 2", 10, 60, 30)
    ks3 = st.slider("Ks Zone 3", 10, 60, 30)
    ks4 = st.slider("Ks Zone 4", 10, 60, 30)

# Vecteur d'entrée utilisateur
input_data = pd.DataFrame([[er, ks2, ks3, ks4, ks_fp, of, qmax, tm]], columns=feature_names)

# Choix du modèle actif pour la simulation
st.sidebar.markdown("---")
st.sidebar.header("🧠 Modèle Actif")
model_choice = st.sidebar.selectbox(
    "Moteur de prédiction :",
    ["Ensemble (Moyenne)", "XGBoost", "Random Forest", "Ridge", "Neural Net"],
    index=0
)

# --- FONCTION DE PRÉDICTION UNIFIÉE ---
def predict_value(model_name, target, input_df):
    """Gère le scaling et la prédiction selon le type de modèle"""
    models_loc = model_store.get(target, {})
    model = models_loc.get(model_name)
    
    if model is None:
        return 0.0
        
    # Gestion des Scalers spécifiques
    X_in = input_df.copy()
    
    if model_name in ["XGBoost"] and 'xgb' in scalers:
        X_in = pd.DataFrame(scalers['xgb'].transform(input_df), columns=input_df.columns)
    elif model_name in ["Ridge", "Lasso"] and 'ridge' in scalers:
        X_in = pd.DataFrame(scalers['ridge'].transform(input_df), columns=input_df.columns)
    elif model_name == "Neural Net" and 'nn' in scalers:
        X_in = pd.DataFrame(scalers['nn'].transform(input_df), columns=input_df.columns)
    # Random Forest utilise souvent les données brutes ("raw")
        
    try:
        pred = model.predict(X_in)
        # Gestion des formats de sortie (array, float, tensor)
        if hasattr(pred, "flatten"): 
            val = pred.flatten()[0]
        elif isinstance(pred, list):
            val = pred[0]
        else:
            val = float(pred) # Cas simple
        return max(0.0, val)
    except Exception as e:
        return 0.0

# Calcul des prédictions pour l'affichage principal
preds_current = {}
for t in targets:
    if model_choice == "Ensemble (Moyenne)":
        v1 = predict_value("XGBoost", t, input_data)
        v2 = predict_value("Random Forest", t, input_data)
        preds_current[t] = (v1 + v2) / 2
    else:
        preds_current[t] = predict_value(model_choice, t, input_data)


# --- MAIN INTERFACE ---
st.title("🌊 Smart Flood Defense: Sully-sur-Loire")

# ONGLETS
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Simulation", "🏆 Le Tournoi (Benchmark)", "🧠 Analyse & Transparence", "📉 Sensibilité"])

# --- TAB 1: SIMULATION (PRODUIT FINI) ---
with tab1:
    st.markdown("### 📡 Surveillance Temps Réel")
    
    # 1. KPIs
    cols = st.columns(4)
    map_data = []
    
    for i, t in enumerate(targets):
        level = preds_current[t]
        name = t.replace("_", " ").title()
        
        if level < 0.5: status, color = "Sûr", "normal"
        elif level < 1.5: status, color = "Vigilance", "off"
        else: status, color = "DANGER", "inverse"

        with cols[i]:
            st.metric(label=name, value=f"{level:.2f} m", delta=status, delta_color=color)
            
        map_data.append({
            "lat": poi_coords[t][0], "lon": poi_coords[t][1],
            "size": (level + 0.5) * 100,
            "color": [200, 0, 0, 200] if level > 1.0 else [0, 200, 0, 150]
        })
    
    st.divider()
    
    # 2. Carte et Graphique
    c1, c2 = st.columns([3, 2])
    with c1:
        st.map(pd.DataFrame(map_data), latitude="lat", longitude="lon", size="size", color="color", zoom=13)
    with c2:
        st.caption("Niveau d'eau comparatif")
        st.bar_chart(pd.Series(preds_current).rename("Hauteur (m)"))


# --- TAB 2: LE TOURNOI (BENCHMARK) ---
with tab2:
    st.header("🥊 Comparaison des Modèles")
    
    if X_test is None:
        st.error("Fichier de données 'training_matrix_sully.csv' manquant. Impossible de lancer le tournoi.")
    else:
        st.markdown("""
        **Qui est le meilleur ?** Nous évaluons chaque modèle sur les données de test (200 simulations jamais vues).
        """)
        
       # --- DANS TAB 2 (Le Tournoi) ---
if st.button("🏁 Lancer le Benchmark en direct"):
    
    # ASTUCE VITESSE : On prend un sous-échantillon si le test est trop gros (>100 lignes)
    if len(X_test) > 100:
        X_eval_sample = X_test.sample(100, random_state=42)
        y_eval_sample = y_test.loc[X_eval_sample.index]
    else:
        X_eval_sample = X_test
        y_eval_sample = y_test

    scores = []
    # On enlève la barre de progression détaillée qui ralentit l'app
    with st.spinner('Calcul des scores en cours... (Cela peut prendre quelques secondes)'):
        
        for t in targets:
            models_loc = model_store[t]
            y_true = y_eval_sample[t]
            
            for m_name, model in models_loc.items():
                # On prépare les données (Scaling)
                X_in = X_eval_sample.copy()
                
                # Gestion des scalers (identique à avant)
                if m_name == "XGBoost" and 'xgb' in scalers:
                    X_in = pd.DataFrame(scalers['xgb'].transform(X_eval_sample), columns=feature_names)
                elif m_name in ["Ridge", "Lasso"] and 'ridge' in scalers:
                    X_in = pd.DataFrame(scalers['ridge'].transform(X_eval_sample), columns=feature_names)
                elif m_name == "Neural Net" and 'nn' in scalers:
                    X_in = pd.DataFrame(scalers['nn'].transform(X_eval_sample), columns=feature_names)
                
                try:
                    p = model.predict(X_in)
                    if hasattr(p, "flatten"): p = p.flatten()
                    r2 = r2_score(y_true, p)
                    scores.append({"Lieu": t, "Modèle": m_name, "R2 Score": r2})
                except Exception:
                    pass
    
    # Affichage du résultat
    df_scores = pd.DataFrame(scores)
    st.subheader("Classement par Précision (R²)")
    st.bar_chart(df_scores, x="Lieu", y="R2 Score", color="Modèle", stack=False)
    st.success("Benchmark terminé !")

    st.divider()
    
    # GLOBAL GRAPH INTERACTIF
    st.subheader("⚔️ Duel de Comportement (Global Graph)")
    st.info("On fait varier **Qmax** (le débit) en figeant les autres paramètres. Observez la fluidité des courbes.")
    
    target_plot = st.selectbox("Choisir le lieu pour le duel :", targets, index=2)
    
    # Génération scénario
    steps = 50
    q_vals = np.linspace(3000, 10000, steps)
    base_scen = pd.DataFrame([X_train.mean()] * steps, columns=feature_names)
    base_scen['qmax'] = q_vals
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    models_loc = model_store[target_plot]
    colors = {'XGBoost': 'blue', 'Random Forest': 'green', 'Ridge': 'red', 'Lasso': 'orange', 'Neural Net': 'purple'}
    
    for m_name in ["Ridge", "XGBoost", "Random Forest", "Neural Net"]:
        if m_name in models_loc:
            # Predict manually via helper logic reused
            ys = []
            # On doit faire une boucle ou vectoriser. 
            # Pour simplifier l'appel vectorisé :
            X_scen = base_scen.copy()
            
            if m_name == "XGBoost" and 'xgb' in scalers:
                X_scen = pd.DataFrame(scalers['xgb'].transform(base_scen), columns=feature_names)
            elif m_name in ["Ridge", "Lasso"] and 'ridge' in scalers:
                X_scen = pd.DataFrame(scalers['ridge'].transform(base_scen), columns=feature_names)
            elif m_name == "Neural Net" and 'nn' in scalers:
                X_scen = pd.DataFrame(scalers['nn'].transform(base_scen), columns=feature_names)
                
            try:
                p = models_loc[m_name].predict(X_scen)
                if hasattr(p, "flatten"): p = p.flatten()
                ax.plot(q_vals, p, label=m_name, color=colors.get(m_name, 'grey'), linewidth=2)
            except: pass

    ax.set_xlabel("Débit Qmax (m3/s)")
    ax.set_ylabel("Hauteur d'eau (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


# --- TAB 3: ANALYSE (FEATURE IMPORTANCE) ---
with tab3:
    st.header("🔍 Pourquoi ça inonde ?")
    st.markdown("Interprétation des boîtes noires.")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("1. Vision Globale (Random Forest)")
        st.write("Le Random Forest permet de classer les variables par importance moyenne.")
        # Utilisation de l'image existante
        img_rf = "randomforest/graphs/graph1_global_importance.png"
        if os.path.exists(img_rf):
            st.image(img_rf, use_column_width=True)
        else:
            st.warning("Image RF non trouvée.")
            
    with c2:
        st.subheader("2. Sélection de Variables (Lasso)")
        st.write("Le Lasso a 'tué' les variables inutiles (coefficients à zéro).")
        # Utilisation de l'image existante
        img_lasso = "modele_Ridge/graphs/comparaison_ridge_lasso.png"
        if os.path.exists(img_lasso):
            st.image(img_lasso, use_column_width=True, caption="Comparaison Ridge vs Lasso")
        else:
            st.warning("Image Lasso non trouvée.")
            
    st.info("Conclusion commune : **Qmax** (Débit) est le facteur n°1, suivi par la topographie locale.")

# --- TAB 4: SENSIBILITÉ (DYNAMIQUE) ---
with tab4:
    st.header("📉 Analyse de Sensibilité")
    st.write("Comment la hauteur d'eau évolue si on change uniquement le débit, avec VOS paramètres actuels ?")
    
    # On reprend input_data (défini par les sliders)
    q_range = np.linspace(3000, 12000, 100)
    
    # On duplique les inputs utilisateur
    sensi_df = pd.concat([input_data] * 100, ignore_index=True)
    sensi_df['qmax'] = q_range
    
    # On prédit avec le modèle choisi
    # On utilise notre fonction predict_value mais il faut boucler car elle prend un df simple
    # Optimisation : appel direct au modèle
    target_sensi = st.selectbox("Lieu à observer :", targets)
    
    # Utilisation du modèle actif
    y_sensi = []
    
    # Scaling manuel rapide pour affichage fluide
    if model_choice == "XGBoost" and 'xgb' in scalers:
        X_in = scalers['xgb'].transform(sensi_df)
        model = model_store[target_sensi].get("XGBoost")
        if model: y_sensi = model.predict(X_in)
    elif model_choice == "Random Forest":
        model = model_store[target_sensi].get("Random Forest")
        if model: y_sensi = model.predict(sensi_df)
    else:
        # Fallback boucle (plus lent mais sûr)
        y_sensi = [predict_value(model_choice, target_sensi, row.to_frame().T) for _, row in sensi_df.iterrows()]
    
    if len(y_sensi) > 0:
        chart_data = pd.DataFrame({"Qmax": q_range, "Hauteur d'eau": y_sensi})
        st.line_chart(chart_data, x="Qmax", y="Hauteur d'eau")
        
    st.caption(f"Point de simulation actuel : Qmax={qmax}")

# --- TAB 4: ANALYSE APPROFONDIE ---
with tab4:
    st.header("🔬 Audit de Performance")
    
    # SOUS-ONGLETS POUR L'ANALYSE
    subtab1, subtab2, subtab3 = st.tabs(["🎯 Précision", "⚠️ Sécurité (Résidus)", "🕸️ Synthèse (Radar)"])
    
    # 1. PRÉCISION (Scatter Plot)
    with subtab1:
        st.subheader("La Diagonale de Vérité")
        st.markdown("Comparaison : **Réalité (Axe X)** vs **Prédiction IA (Axe Y)**. Plus les points sont proches de la ligne rouge, meilleur est le modèle.")
        
        if os.path.exists("assets/graphs/1_pred_vs_actual.png"):
            st.image("assets/graphs/1_pred_vs_actual.png", use_column_width=True)
        else:
            st.warning("Graphique non généré. Lancez 'generate_analysis_plots.py'.")
            
    # 2. SÉCURITÉ (Résidus)
    with subtab2:
        st.subheader("Analyse des Erreurs Critiques")
        st.markdown("""
        **Question cruciale :** Le modèle se trompe-t-il quand la crue est violente ?
        * **Points Rouges ( > 0)** : Le modèle *sous-estime* le danger (Dangereux ❌).
        * **Points Bleus ( < 0)** : Le modèle *sur-estime* le danger (Fausse alerte mais Sûr ✅).
        """)
        
        if os.path.exists("assets/graphs/2_residuals_vs_qmax.png"):
            st.image("assets/graphs/2_residuals_vs_qmax.png", use_column_width=True)
        else:
            st.warning("Graphique non généré.")

    # 3. RADAR CHART (Interactif)
    with subtab3:
        st.subheader("Synthèse Multi-Critères")
        
        # Données (Estimations basées sur tes tests)
        categories = ['Précision (R2)', 'Robustesse (Physique)', 'Vitesse', 'Sécurité', 'Simplicité']
        
        fig = go.Figure()

        # XGBoost (Le Compétiteur)
        fig.add_trace(go.Scatterpolar(
              r=[0.99, 0.95, 0.90, 0.92, 0.70],
              theta=categories,
              fill='toself',
              name='XGBoost'
        ))
        
        # Random Forest (La Référence)
        fig.add_trace(go.Scatterpolar(
              r=[0.98, 0.85, 0.80, 0.90, 0.80],
              theta=categories,
              fill='toself',
              name='Random Forest'
        ))
        
        # Ridge (Le Baseline)
        fig.add_trace(go.Scatterpolar(
              r=[0.85, 0.60, 1.00, 0.70, 0.95],
              theta=categories,
              fill='toself',
              name='Ridge (Linéaire)'
        ))

        fig.update_layout(
          polar=dict(
            radialaxis=dict(
              visible=True,
              range=[0, 1]
            )),
          showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)
        st.caption("Note : Scores normalisés de 0 à 1 basés sur l'étude comparative.")