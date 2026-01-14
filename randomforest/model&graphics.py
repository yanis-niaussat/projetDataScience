import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv('training_matrix_sully.csv')

input_cols = ['er','ks2','ks3','ks4','ks_fp','of','qmax','tm']
target_cols = ['parc_chateau', 'centre_sully', 'gare_sully', 'caserne_pompiers']

def model_filename(loc_name: str) -> str:
    """Builds a stable filename like RF_ParcChateau.pkl from a target column name."""
    parts = [p for p in loc_name.split('_') if p]
    pascal = ''.join(part.capitalize() for part in parts)
    return f"RF_{pascal or loc_name}.pkl"

X = df[input_cols]
y = df[target_cols]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("Training Random Forest (Multi-Output)...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'sully_flood_model_final.pkl')
print("Model saved as 'sully_flood_model_final.pkl'")
print(f"Global Accuracy (R²): {model.score(X_test, y_test):.4f}")
joblib.dump(scaler, 'rf_scaler.pkl')

print("Training and saving per-location models...")
for loc in target_cols:
    loc_model = RandomForestRegressor(n_estimators=100, random_state=42)
    loc_model.fit(X_scaled, df[loc])
    fname = model_filename(loc)
    joblib.dump(loc_model, fname)
    print(f"Saved {fname}")

plt.figure(figsize=(10, 6))
importances = pd.Series(model.feature_importances_, index=input_cols).sort_values(ascending=False)
importances.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Global Factor: What drives flooding in Sully?")
plt.ylabel("Importance Score")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('graph1_global_importance.png')
print("Saved graph1_global_importance.png")
plt.show()

predictions = model.predict(X_test)
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()

for i, loc in enumerate(target_cols):
    ax = axes[i]
    y_real = y_test.iloc[:, i]
    y_pred = predictions[:, i]
    
    ax.scatter(y_real, y_pred, alpha=0.5, color='purple')
    m_max = max(y_real.max(), y_pred.max())
    ax.plot([0, m_max], [0, m_max], 'r--', lw=2, label='Perfect')
    
    ax.set_title(f"Accuracy: {loc}")
    ax.set_xlabel("Real (m)")
    ax.set_ylabel("Predicted (m)")
    ax.grid(True)

plt.tight_layout()
plt.savefig('graph2_accuracy_grid.png')
print("Saved graph2_accuracy_grid.png")
plt.show()

plt.figure(figsize=(12, 7))
colors = ['teal', 'orange', 'green', 'red']
for i, loc in enumerate(target_cols):
    plt.scatter(df['qmax'], df[loc], alpha=0.4, s=15, label=loc, color=colors[i])

plt.title("Vulnerability Analysis: River Flow vs. Water Levels")
plt.xlabel("River Flow Qmax (m3/s)")
plt.ylabel("Water Height (m)")
plt.legend(markerscale=3)
plt.grid(True, alpha=0.5)
plt.savefig('graph3_physics_curves.png')
print("Saved graph3_physics_curves.png")
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, loc in enumerate(target_cols):
    local_model = RandomForestRegressor(n_estimators=50, random_state=42)
    local_model.fit(X_scaled, df[loc])
    
    local_imp = pd.Series(local_model.feature_importances_, index=input_cols).sort_values(ascending=False)
    
    ax = axes[i]
    local_imp.plot(kind='bar', color='lightgreen', edgecolor='black', ax=ax)
    ax.set_title(f"Drivers for: {loc}")
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', linestyle='--')

plt.tight_layout()
plt.savefig('graph4_location_importances.png')
print("Saved graph4_location_importances.png")
plt.show()

print("\nAll Done.")