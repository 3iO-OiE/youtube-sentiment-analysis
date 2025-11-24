from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path

def create_train_test_split(input_path, test_size=0.2, random_state=42):
    """Crée un split train/test reproductible"""
    
    print(" Chargement des données...")
    df = pd.read_csv(input_path)
    
    print(f" Création du split (test_size={test_size})...")
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['label']  # Maintenir la proportion des classes
    )
    
    # Sauvegarder
    output_dir = Path("data/processed")
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n Split créé avec succès!")
    print(f" Train set: {len(train_df)} exemples ({train_path})")
    print(f" Test set: {len(test_df)} exemples ({test_path})")
    
    print(f"\n Distribution dans train set:")
    print(train_df['label'].value_counts().sort_index())
    
    print(f"\n Distribution dans test set:")
    print(test_df['label'].value_counts().sort_index())

if __name__ == "__main__":
    create_train_test_split("data/processed/cleaned_data.csv")