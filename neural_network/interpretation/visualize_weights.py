import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# --- CONFIGURATION ---
# Adjust paths relative to this script location (neural_network/interpretation/)
MODEL_DIR = ".." 
INPUT_FEATURES = ['er', 'ks2', 'ks3', 'ks4', 'ks_fp', 'of', 'qmax', 'tm']
TARGET_LOC = 'Parc_Chateau' # Focusing on the main critical zone for deep analysis

def plot_neural_weights():
    model_path = os.path.join(MODEL_DIR, f"mlp_model_{TARGET_LOC}.pkl")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    # Sklearn MLP weights are in model.coefs_
    # coefs_[0] is Input (8) -> Hidden 1 (100)
    weights_layer1 = model.coefs_[0] # Shape (8, 100)
    
    save_dir = "plots"
    os.makedirs(save_dir, exist_ok=True)

    # 1. Heatmap (features vs neurons)
    plt.figure(figsize=(15, 6))
    sns.heatmap(weights_layer1, yticklabels=INPUT_FEATURES, cmap="viridis", center=0)
    plt.title(f"Neural Weights Heatmap (Layer 1)\n{TARGET_LOC}")
    plt.xlabel("Hidden Neurons (0-99)")
    plt.savefig(f"{save_dir}/weights_heatmap.png", dpi=300)
    plt.close()
    print(f"Saved {save_dir}/weights_heatmap.png")
    
    # 2. Derived Feature Importance (Mean Absolute Weight)
    # This approximates how much "input signal" enters the network from each feature
    importance = np.mean(np.abs(weights_layer1), axis=1)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/neural_feature_importance.png", dpi=300)
    plt.close()
    print(f"Saved {save_dir}/neural_feature_importance.png")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/neural_feature_importance.png", dpi=300)
    plt.close()
    print(f"Saved {save_dir}/neural_feature_importance.png")

    # 3. Network Graph Visualization (NetworkX)
    try:
        import networkx as nx
        print("Generating Full Network Graph...")
        
        G = nx.DiGraph()
        
        # We need to handle multiple layers: Input(8) -> H1(100) -> H2(100) -> Output(1)
        # To avoid hairballs, we visualize the 'Strongest Path' architecture
        # showing top active neurons in each layer.
        
        n_show_h1 = 12
        n_show_h2 = 12
        
        # --- NODES ---
        # Layer 0: Inputs (All)
        for i, feature in enumerate(INPUT_FEATURES):
            G.add_node(feature, layer=0, pos=(0, -i)) # Negative Y to draw top-down or just scale
            
        # Layer 1: H1 (Top N by average input weight)
        # argsort gives indices of sorted values. [-N:] takes last N (highest).
        h1_activity = np.mean(np.abs(model.coefs_[0]), axis=0) # (100,)
        top_h1_indices = np.argsort(h1_activity)[-n_show_h1:]
        
        for i, idx in enumerate(top_h1_indices):
            G.add_node(f"H1_{idx}", layer=1, pos=(1, -i * (len(INPUT_FEATURES)/n_show_h1)))
            
        # Layer 2: H2 (Top N by average input weight from H1)
        # Note: model.coefs_[1] is (100, 100) -> rows=H1, cols=H2
        h2_activity = np.mean(np.abs(model.coefs_[1]), axis=0) # (100,)
        top_h2_indices = np.argsort(h2_activity)[-n_show_h2:]
        
        for i, idx in enumerate(top_h2_indices):
            G.add_node(f"H2_{idx}", layer=2, pos=(2, -i * (len(INPUT_FEATURES)/n_show_h2)))
            
        # Layer 3: Output (All)
        G.add_node("Output", layer=3, pos=(3, -len(INPUT_FEATURES)/2))
        
        # --- EDGES ---
        edges = []
        weights = []
        
        # L0 -> L1
        w_l0 = model.coefs_[0]
        for r, feature in enumerate(INPUT_FEATURES):
            for i, h1_idx in enumerate(top_h1_indices):
                w = w_l0[r, h1_idx]
                G.add_edge(feature, f"H1_{h1_idx}", weight=w)
                edges.append((feature, f"H1_{h1_idx}"))
                weights.append(w)

        # L1 -> L2
        w_l1 = model.coefs_[1]
        for i, h1_idx in enumerate(top_h1_indices):
            for j, h2_idx in enumerate(top_h2_indices):
                w = w_l1[h1_idx, h2_idx]
                G.add_edge(f"H1_{h1_idx}", f"H2_{h2_idx}", weight=w)
                edges.append((f"H1_{h1_idx}", f"H2_{h2_idx}"))
                weights.append(w)
                
        # L2 -> L3
        w_l2 = model.coefs_[2]
        for j, h2_idx in enumerate(top_h2_indices):
                w = w_l2[h2_idx, 0] # Output is 1 neuron at index 0
                G.add_edge(f"H2_{h2_idx}", "Output", weight=w)
                edges.append((f"H2_{h2_idx}", "Output"))
                weights.append(w)

        # Draw
        plt.figure(figsize=(16, 10))
        pos = nx.get_node_attributes(G, 'pos')
        
        # Colors
        edge_colors = ['red' if w > 0 else 'blue' for w in weights]
        # Normalize widths better
        max_w = max([abs(w) for w in weights])
        edge_widths = [(abs(w)/max_w) * 3 for w in weights] 
        
        nx.draw_networkx_nodes(G, pos, node_size=600, node_color='lightgray', edgecolors='black')
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')
        
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors, width=edge_widths, alpha=0.5, arrows=True, arrowsize=10)
        
        plt.title(f"Full Deep Network Architecture (Strongest Paths)\nFeatures -> H1({n_show_h1}) -> H2({n_show_h2}) -> Output")
        plt.axis('off')
        plt.savefig(f"{save_dir}/full_network_graph.png", dpi=300)
        plt.close()
        print(f"Saved {save_dir}/full_network_graph.png")
        
    except ImportError:
        print("NetworkX not installed. Skipping graph.")
    except Exception as e:
        print(f"Error drawing graph: {e}") 
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    plot_neural_weights()
