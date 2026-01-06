import pandas as pd
import os
import re

# --- Configuration ---
data_folder = 'data'
poi_file_path = 'src/LoireSully_model/Sully.poi'

# 1. Setup: Get POI coordinates and Map Info
pois = {}
map_bounds = {}

with open(poi_file_path, 'r') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'): continue
        
        if line.startswith('.'):
            parts = line.split('=')[1].split(':')
            map_bounds['xmin'], map_bounds['ymin'] = map(float, parts[0].split(','))
            map_bounds['cols'], map_bounds['rows'] = map(int, parts[1].split(','))
            map_bounds['xmax'], map_bounds['ymax'] = map(float, parts[2].split(','))
        else:
            name, coords = line.split('=')
            pois[name] = tuple(map(float, coords.split(',')))

# Pre-calculate Grid Indices (Adapted for your coordinates)
poi_indices = {}
for name, (x, y) in pois.items():
    # Calculate percentage across the map
    x_pct = (x - map_bounds['xmin']) / (map_bounds['xmax'] - map_bounds['xmin'])
    y_pct = (y - map_bounds['ymin']) / (map_bounds['ymax'] - map_bounds['ymin'])
    
    # Calculate Indices
    col_idx = int(x_pct * map_bounds['cols'])
    
    # ADAPTATION: Removed "1 - y_pct". 
    # Your coordinates indicate the grid starts from the bottom (Y=0).
    row_idx = int(y_pct * map_bounds['rows'])
    
    # Clamp to ensure we stay within 0-63
    col_idx = max(0, min(col_idx, map_bounds['cols'] - 1))
    row_idx = max(0, min(row_idx, map_bounds['rows'] - 1))

    poi_indices[name] = (row_idx, col_idx)

# 2. Main Loop
dataset = []
column_names = []

files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
print(f"Found {len(files)} files. Processing...")

for filename in files:
    try:
        # A. Parse Filename
        name_clean = filename.replace('.csv', '')
        if '=' not in name_clean: continue
        
        keys_str, values_str = name_clean.split('=')
        val_parts = values_str.split(',')
        val_parts[-1] = re.match(r'^-?\d+(\.\d+)?', val_parts[-1]).group(0)
        
        input_values = [float(v) for v in val_parts]
        
        if not column_names:
            input_keys = keys_str.split(',')
            sorted_poi_names = sorted(poi_indices.keys())
            column_names = input_keys + sorted_poi_names

        # B. Read Grid
        file_path = os.path.join(data_folder, filename)
        df = pd.read_csv(file_path, index_col=0)

        # C. Extract Values
        poi_values = []
        for name in sorted_poi_names:
            r, c = poi_indices[name]
            # Accessing matrix: [row, col]
            val = df.iloc[r, c]
            poi_values.append(val)

        dataset.append(input_values + poi_values)

    except Exception as e:
        print(f"Skipping {filename}: {e}")

# 3. Create DataFrame
final_df = pd.DataFrame(dataset, columns=column_names)
print("Matrix shape:", final_df.shape)
print(final_df.head())
final_df.to_csv('training_dataset.csv', index=False)