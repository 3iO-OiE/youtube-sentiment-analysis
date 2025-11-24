import pandas as pd
import requests
from pathlib import Path

def download_reddit_dataset():
    """Télécharge le dataset Reddit depuis GitHub"""
    
    url = "https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv"
    
    # Créer le dossier si nécessaire
    data_dir = Path("data/raw")
    
    print("Téléchargement du dataset...")
    response = requests.get(url)
    response.raise_for_status()
    
    # Sauvegarder
    output_path = data_dir / "reddit.csv"
    with open(output_path, 'wb') as f:
        f.write(response.content)
    
    # Charger et afficher les statistiques
    df = pd.read_csv(output_path)
    
    print(f"\n Dataset téléchargé: {output_path}")
    print(f"\n Statistiques du dataset:")
    print(f"  - Nombre total de commentaires: {len(df)}")
    print(f"  - Colonnes: {list(df.columns)}")
    print(f"\n Distribution des labels:")
    print(df['category'].value_counts())
    print(f"\n Statistiques des longueurs de texte:")
    df['text_length'] = df['clean_comment'].astype(str).str.len()
    print(df['text_length'].describe())
    
    return df

if __name__ == "__main__":
    df = download_reddit_dataset()