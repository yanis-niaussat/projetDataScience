
# 🚨 PROJET DATA SCIENCE : Modélisation Inondation Loire-Sully (IRSN)

**Objectif :** Remplacer un simulateur hydraulique lent (Telemac) par un modèle d'IA instantané.
**Deadline :** Vendredi prochain (Urgence absolue).
**Stratégie :** "Le Grand Tournoi des Familles". On compare les 3 grandes approches du ML (Linéaire, Ensembliste, Connexionniste).

---

## 📍 ÉTAPE 0 : LA "GOLDEN DATA" (URGENT - CE WEEK-END)

Avant de coder quoi que ce soit, nous devons générer un fichier unique **`dataset_final_Sully.csv`** propre et partagé.

### 1. Structure du Dataset
* **Entrées (X - 8 colonnes) :** Extraites du nom des fichiers (`er`, `ks2`, `ks3`, `ks4`, `ks_fp`, `of`, `qmax`, `tm`).
* **Sorties (Y - 4 colonnes) :** Hauteurs d'eau extraites des matrices CSV aux coordonnées suivantes (Validées hydrauliquement) :

| Point d'intérêt (Target) | Indice Ligne (X) | Indice Colonne (Y) | Description |
| :--- | :---: | :---: | :--- |
| **Parc_Chateau** | 27 | 50 | Zone critique (Bord de Loire) |
| **Centre_Sully** | 18 | 42 | Centre-ville |
| **Gare_Sully** | 16 | 28 | Zone urbaine intermédiaire |
| **Caserne_Pompiers** | 12 | 11 | Zone "sèche" / éloignée |

> **⚠️ Attention :** Les indices sont donnés pour une matrice Python/R standard. Si vous utilisez `pandas` ou `numpy`, vérifiez bien l'orientation (Ligne=X, Colonne=Y).

---

## ⚙️ ÉTAPE 1 : LE PROTOCOLE TECHNIQUE (LOI MARTIALE)

Pour que nous puissions comparer nos résultats Mercredi, **tout le monde doit utiliser EXACTEMENT ce code de départ.** Copiez-collez ceci dans vos Notebooks respectifs.

### 1. Split Train/Test (Inviolable)
On ne touche **JAMAIS** au `X_test` pour régler les modèles. C'est le juge de paix final.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Chargement
df = pd.read_csv("dataset_final_Sully.csv")
X = df[['er', 'ks2', 'ks3', 'ks4', 'ks_fp', 'of', 'qmax', 'tm']]
y = df[['Parc_Chateau', 'Centre_Sully', 'Gare_Sully', 'Caserne_Pompiers']]

# SPLIT : random_state=42 est OBLIGATOIRE pour qu'on ait les mêmes données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SCALING : Important pour Neural Net et Lasso/Ridge
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

```

### 2. Stratégie de Modélisation (Boucle simple)

Ne faites pas de "Multi-Output" natif complexe. Faites une boucle simple sur les 4 lieux.

```python
targets = ['Parc_Chateau', 'Centre_Sully', 'Gare_Sully', 'Caserne_Pompiers']
results = {}

for lieu in targets:
    print(f"Training for {lieu}...")
    # Sélection de la colonne cible
    y_train_col = y_train[lieu]
    
    # Votre modèle ici (Exemple)
    # model.fit(X_train_scaled, y_train_col)
    
    # Sauvegarde
    # results[lieu] = model

```

---

## 👥 ÉTAPE 2 : RÉPARTITION DES RÔLES

Chacun est responsable d'une famille d'algorithmes.

### 👤 Fatima : "Linear Expert" (Statistique & Cours)

* **Mission :** Prouver qu'on a écouté le prof et offrir de l'interprétabilité.
* **Algorithmes :**
1. **Linear Regression :** Score de référence (Baseline).
2. **Ridge () :** Gérer la corrélation entre les variables `ks`.
3. **Lasso () :** **CRUCIAL.** Identifier les variables inutiles (coef = 0).


* **Livrable clé :** "Quelles variables physiques le Lasso a-t-il supprimées ?"
* **Librairie :** `sklearn.linear_model`

### 👤 Marius : "Bagging Expert" (Robustesse)

* **Mission :** Fournir un modèle stable, robuste et difficile à prendre en défaut (Overfitting faible).
* **Algorithmes :**
1. **Random Forest Regressor :** La valeur sûre.


* **Approche :** Méthode ensembliste parallèle (moyenne des arbres).
* **Livrable clé :** Le graphique de **Feature Importance** (Quelle variable physique cause l'inondation ?).
* **Librairie :** `sklearn.ensemble`

### 👤 Tom : "Boosting Expert" (Performance Pure)

* **Mission :** Aller chercher la précision maximale (Compétition Kaggle).
* **Algorithmes :**
1. **XGBoost :** Le challenger agressif.


* **Approche :** Méthode ensembliste séquentielle (correction des erreurs précédentes).
* **Attention :** Bien installer la librairie (`pip install xgboost`).
* **Librairie :** `xgboost`

### 👤 Yanis : "Neural Expert" (Deep Learning & Viz)

* **Mission :** Capturer les non-linéarités complexes avec une approche connexionniste.
* **Algorithmes :**
1. **MLP Regressor :** Réseau de neurones (ex: 2 couches cachées de 100 neurones). Solver='adam'.


* **Mission Transverse :** Mercredi, tu récupères les prédictions des 3 autres sur le `X_test` et tu traces les graphiques comparatifs ("Réel vs Prédit").
* **Librairie :** `sklearn.neural_network`

---

## 📅 ÉTAPE 3 : LA ROADMAP DE SURVIE

| Timing | Action | Responsable |
| --- | --- | --- |
| **Ce Weekend** | Générer `dataset_final_Sully.csv` et le mettre sur le Drive/Git. | **Membre 1** |
| **Lundi** | Coder son modèle (V1) qui tourne sans erreur. | **Tous** |
| **Mardi** | **Tuning (Cross-Validation).** Chacun optimise ses hyperparamètres sur `X_train`. | **Tous** |
| **Mercredi** | **LE GRAND MERGE.** On met tout dans un tableau comparatif. On fige les résultats. | **Membre 4** (Lead) |
| **Jeudi** | Rédaction Rapport & Slides. (Interprétation Physique > Code). | **Tous** |
| **Vendredi** | **Rendu / Soutenance.** | **Tous** |

---

## 📝 STRUCTURE DU RAPPORT (Suggestion)

1. **Contexte :** Pourquoi l'IA ? (Accélérer Telemac).
2. **Data Engineering :** Validation hydraulique des points (coordonnées).
3. **Approche Statistique (Cours) :** Lasso/Ridge (Sélection de variables).
4. **Approche Ensembliste (Le Match) :** Bagging (Random Forest) vs Boosting (XGBoost). Qui gagne ?
5. **Approche Connexionniste :** Réseau de Neurones (Capacité de généralisation).
6. **Conclusion :** Tableau final des scores. Recommandation pour l'IRSN.
* *Ouverture : "Pour aller plus loin, nous pourrions tester le Kriging (Processus Gaussiens) pour quantifier l'incertitude."*



🚀 **Allez l'équipe, on vise le 18/20 !**

```