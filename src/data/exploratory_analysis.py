import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_dataset(data_path):
    """Analyse exploratoire des données"""
    
    df = pd.read_csv(data_path)
    
    # Créer le dossier pour les visualisations
    viz_dir = Path("logs/visualizations")
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Distribution des classes
    plt.figure(figsize=(10, 6))
    label_names = {0: 'Négatif', 1: 'Neutre', 2: 'Positif'}
    counts = df['label'].value_counts().sort_index()
    
    plt.subplot(1, 2, 1)
    counts.plot(kind='bar', color=['#e74c3c', '#95a5a6', '#2ecc71'])
    plt.title('Distribution des Sentiments')
    plt.xlabel('Sentiment')
    plt.ylabel('Nombre de commentaires')
    plt.xticks([0, 1, 2], ['Négatif', 'Neutre', 'Positif'], rotation=0)
    
    # 2. Distribution des longueurs
    plt.subplot(1, 2, 2)
    df['text_length'] = df['text'].str.len()
    df['text_length'].hist(bins=50, edgecolor='black')
    plt.title('Distribution de la Longueur des Textes')
    plt.xlabel('Nombre de caractères')
    plt.ylabel('Fréquence')
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'data_exploration.png', dpi=300, bbox_inches='tight')
    print(f" Visualisation sauvegardée: {viz_dir / 'data_exploration.png'}")
    
    # Statistiques détaillées
    print("\n STATISTIQUES DÉTAILLÉES")
    print("="*50)
    print(f"\nNombre total d'exemples: {len(df)}")
    print(f"\nDistribution des classes:")
    for label, count in counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {label_names[label]}: {count} ({percentage:.1f}%)")
    
    print(f"\n Longueur des textes:")
    print(df['text_length'].describe())
    
    # Vérifier le déséquilibre
    max_count = counts.max()
    min_count = counts.min()
    imbalance_ratio = max_count / min_count
    print(f"\n Ratio de déséquilibre: {imbalance_ratio:.2f}")
    
    if imbalance_ratio > 1.5:
        print(" Classes déséquilibrées détectées!")
    else:
        print(" Classes relativement équilibrées")

if __name__ == "__main__":
    analyze_dataset("data/processed/cleaned_data.csv")