import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def calculate_and_plot_reynolds():
    # Load dataset
    script_dir = Path(__file__).parent
    csv_path = script_dir.parent / 'training_matrix_sully.csv'
    df = pd.read_csv(csv_path)
    
    # Constants for approximation
    # Loire Width approx 300m at Sully
    WIDTH = 300.0 
    # Kinematic Viscosity of water (m2/s)
    NU = 1e-6 
    
    # Reynolds Number Calculation
    # Re = (Velocity * Characteristic Length) / Nu
    # Approximating for wide open channel:
    # V = Q / A ~ Q / (Width * Depth)
    # L = Hydraulic Diameter ~ 4 * Hydraulic Radius ~ 4 * Depth (or just Depth approx)
    # Using L = Depth (h) and V = Q / (W * h)
    # Re = (Q / (W * h)) * h / Nu = Q / (W * Nu)
    # This simplification shows Re is primarily driven by Flow Rate Q in a wide channel context
    # irrespective of the specific local depth (if we assume full section flow).
    
    df['Reynolds'] = df['qmax'] / (WIDTH * NU)
    
    targets = ['parc_chateau', 'centre_sully', 'gare_sully', 'caserne_pompiers']
    
    # Create Plot
    plt.figure(figsize=(12, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, target in enumerate(targets):
        # Sort by Reynolds for clean line/scatter
        df_sorted = df.sort_values(by='Reynolds')
        
        plt.scatter(df_sorted['Reynolds'], df_sorted[target], 
                    alpha=0.5, s=20, label=target.replace('_', ' ').title(), color=colors[i])
        
        # Add a trend line (smooth)
        z = np.polyfit(df_sorted['Reynolds'], df_sorted[target], 3)
        p = np.poly1d(z)
        plt.plot(df_sorted['Reynolds'], p(df_sorted['Reynolds']), color=colors[i], linestyle='--', linewidth=4)

    plt.title('Physics Check: Water Level vs Reynolds Number', fontsize=16)
    plt.xlabel('Reynolds Number (Re)', fontsize=12)
    plt.ylabel('Water Level (m)', fontsize=12, alpha=0.2)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Scientific notation for X axis
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    output_path = 'reynolds_vs_water_level.png'
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")
    print("Reynolds Statistics:")
    print(df['Reynolds'].describe())

if __name__ == "__main__":
    calculate_and_plot_reynolds()
