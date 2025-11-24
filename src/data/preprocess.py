import pandas as pd
import re
import nltk
from pathlib import Path

# Télécharger les ressources NLTK
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

class TextCleaner:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Nettoie un texte"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        
        # Supprimer les URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Supprimer les mentions (@username)
        text = re.sub(r'@\w+', '', text)
        
        # Supprimer les hashtags mais garder le texte
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Garder les lettres, chiffres et espaces de base
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        # Supprimer les espaces multiples
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def process_dataset(self, input_path, output_path):
        """Traite le dataset complet"""
        print(" Chargement des données...")
        df = pd.read_csv(input_path)
        
        print(" Nettoyage des textes...")
        df['text'] = df['clean_comment'].apply(self.clean_text)
        
        # Mapper les labels: -1 -> 0 (neg), 0 -> 1 (neutral), 1 -> 2 (pos)
        df['label'] = df['category'].map({-1: 0, 0: 1, 1: 2})
        
        # Filtrer les textes vides
        df = df[df['text'].str.len() > 5]
        
        # Sélectionner uniquement les colonnes nécessaires
        df_clean = df[['text', 'label']].copy()
        
        # Sauvegarder
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(output_path, index=False)
        
        print(f"\n Données nettoyées sauvegardées: {output_path}")
        print(f"\n Nombre de commentaires après nettoyage: {len(df_clean)}")
        print(f"\n Distribution finale des labels:")
        print(df_clean['label'].value_counts().sort_index())
        
        return df_clean

if __name__ == "__main__":
    cleaner = TextCleaner()
    df = cleaner.process_dataset(
        "data/raw/reddit.csv",
        "data/processed/cleaned_data.csv"
    )