import joblib
import numpy as np
from pathlib import Path
import time

def test_model_performance():
    """Teste les performances du modèle"""
    
    print(" TESTS DU MODÈLE")
    print("="*60)
    
    # Charger le modèle
    models_dir = Path("models")
    vectorizer = joblib.load(models_dir / "tfidf_vectorizer.joblib")
    model = joblib.load(models_dir / "sentiment_model.joblib")
    
    # Cas de test
    test_cases = [
        # Textes positifs
        ("This is absolutely amazing! Best video ever!", "Positif"),
        ("Love it! Keep up the great work!", "Positif"),
        
        # Textes négatifs
        ("This is terrible and boring", "Négatif"),
        ("Waste of time, very disappointing", "Négatif"),
        
        # Textes neutres
        ("It's okay, nothing special", "Neutre"),
        ("Just another video", "Neutre"),
        
        # Cas limites
        ("", "?"),  # Texte vide
        ("a" * 1000, "?"),  # Texte très long
        ("😊😊😊", "?"),  # Emojis seulement
        ("Français mélangé with English", "?"),  # Langues mélangées
    ]
    
    print("\n Test des prédictions:")
    correct = 0
    total_valid = 0
    
    for text, expected in test_cases:
        if expected != "?":
            total_valid += 1
        
        try:
            X = vectorizer.transform([text])
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0]
            sentiment = {0: "Négatif", 1: "Neutre", 2: "Positif"}[pred]
            confidence = np.max(proba)
            
            status = "✅" if sentiment == expected or expected == "?" else "❌"
            if sentiment == expected:
                correct += 1
            
            print(f"{status} '{text[:50]}...' -> {sentiment} ({confidence:.2f})")
            
        except Exception as e:
            print(f" Erreur: {e}")
    
    if total_valid > 0:
        accuracy = (correct / total_valid) * 100
        print(f"\n Précision sur cas de test: {accuracy:.1f}% ({correct}/{total_valid})")
    
    # Test de temps d'inférence
    print("\n Test de temps d'inférence:")
    test_batch = ["This is a test comment"] * 50
    
    start = time.time()
    X = vectorizer.transform(test_batch)
    _ = model.predict(X)
    inference_time = (time.time() - start) * 1000
    
    print(f"  Temps pour 50 commentaires: {inference_time:.2f}ms")
    print(f"  Critère: < 100ms ({'✅' if inference_time < 100 else '❌'})")

if __name__ == "__main__":
    test_model_performance()