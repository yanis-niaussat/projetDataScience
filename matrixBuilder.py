import pandas as pd
import os
import re

# --- CONFIGURATION ---
data_folder = 'data'
output_file = 'dataset_final_Sully.csv'

# Coordonnées validées (Format : LIGNE R / X_Map, COLONNE R / Y_Map)
# Basé sur ton analyse : Château (27, 50) correspond aux indices matriciels [27, 50]
targets = {
    'Parc_Chateau':     (27, 50),
    'Centre_Sully':     (18, 42),
    'Gare_Sully':       (16, 28),
    'Caserne_Pompiers': (12, 11)
}

# --- PROCESSING ---
print(f"Scanning '{data_folder}'...")
files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

dataset = []
# On s'assure que les colonnes soient dans le bon ordre
input_names = ['er','ks2','ks3','ks4','ks_fp','of','qmax','tm']
target_names = list(targets.keys())

for filename in files:
    try:
        # 1. Extraction des INPUTS (depuis le nom de fichier)
        name_clean = filename.replace('.csv', '')
        if '=' not in name_clean: continue
        
        # Nettoyage robuste du suffixe "_maxH_sully"
        _, values_str = name_clean.split('=')
        val_parts = values_str.split(',')
        
        # Regex pour ne garder que le chiffre du dernier paramètre (vire "_maxH...")
        val_parts[-1] = re.match(r'^-?\d+(\.\d+)?', val_parts[-1]).group(0)
        
        inputs = [float(v) for v in val_parts]

        # 2. Extraction des OUTPUTS (depuis la grille)
        file_path = os.path.join(data_folder, filename)
        
        # index_col=0 est CRUCIAL car la colonne A contient les numéros de ligne "1,2,3..."
        df = pd.read_csv(file_path, index_col=0) 
        
        outputs = []
        for name, (coord_x_row, coord_y_col) in targets.items():
            # R est en base 1, Python en base 0 -> on fait -1
            row_idx = coord_x_row - 1
            col_idx = coord_y_col - 1
            
            val = df.iloc[row_idx, col_idx]
            outputs.append(val)

        dataset.append(inputs + outputs)

    except Exception as e:
        print(f"Skipping {filename}: {e}")

# --- SAVE ---
final_df = pd.DataFrame(dataset, columns=input_names + target_names)

print("\n--- RAPPORT FINAL ---")
print(f"Lignes extraites : {len(final_df)}")
print("Moyenne des hauteurs d'eau (Vérif > 0) :")
print(final_df[target_names].mean())

final_df.to_csv(output_file, index=False)
print(f"Sauvegardé dans {output_file}")