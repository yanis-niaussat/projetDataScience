import pandas as pd

def main():
    df = pd.read_csv("../dataset_final_Sully.csv")
    targets = ['Parc_Chateau', 'Centre_Sully', 'Gare_Sully', 'Caserne_Pompiers']
    
    print("--- TARGET DISTRIBUTION ANALYSIS ---")
    for t in targets:
        print(f"\nTarget: {t}")
        print(f"  Min: {df[t].min()}")
        print(f"  Max: {df[t].max()}")
        print(f"  Unique Values: {df[t].nunique()}")
        
        # Check for 0.0 or most common value
        top_counts = df[t].value_counts().head(5)
        print("  Top 5 Most Frequent Values:")
        print(top_counts)
        
        if top_counts.iloc[0] > len(df) * 0.1:
            print("  [!] WARNING: Highly repetitive values detected. This creates lines in residual plots.")

if __name__ == "__main__":
    main()
