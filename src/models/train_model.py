import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib
from pathlib import Path
import time
import matplotlib.pyplot as plt
import seaborn as sns

class SentimentModelTrainer:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.best_model_name = None
        
    def load_data(self, train_path, test_path):
        """Charge les données train/test"""
        print(" Chargement des données...")
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        
        self.X_train = self.train_df['text']
        self.y_train = self.train_df['label']
        self.X_test = self.test_df['text']
        self.y_test = self.test_df['label']
        
        print(f" Train: {len(self.X_train)} exemples")
        print(f" Test: {len(self.X_test)} exemples")
    
    def create_vectorizer(self):
        """Crée et fit le vectoriseur TF-IDF"""
        print("\n Création du vectoriseur TF-IDF...")
        
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),  # Unigrammes et bigrammes
            min_df=2,            # Ignorer les termes très rares
            max_df=0.95,         # Ignorer les termes trop fréquents
            strip_accents='unicode',
            lowercase=True
        )
        
        self.X_train_vec = self.vectorizer.fit_transform(self.X_train)
        self.X_test_vec = self.vectorizer.transform(self.X_test)
        
        print(f" Vocabulaire: {len(self.vectorizer.vocabulary_)} mots")
        print(f" Matrice train: {self.X_train_vec.shape}")
    
    def train_logistic_regression(self):
        """Entraîne une régression logistique avec optimisation"""
        print("\n Entraînement Logistic Regression...")
        
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'solver': ['lbfgs', 'saga'],
            'max_iter': [200]
        }
        
        lr = LogisticRegression(random_state=42)
        
        grid_search = GridSearchCV(
            lr,
            param_grid,
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train_vec, self.y_train)
        
        print(f" Meilleurs paramètres: {grid_search.best_params_}")
        print(f" Meilleur score CV: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def train_random_forest(self):
        """Entraîne un Random Forest"""
        print("\n Entraînement Random Forest...")
        
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        
        rf.fit(self.X_train_vec, self.y_train)
        return rf
    
    def train_svm(self):
        """Entraîne un SVM"""
        print("\n Entraînement SVM...")
        
        svm = SVC(
            kernel='linear',
            C=1.0,
            random_state=42,
            probability=True
        )
        
        svm.fit(self.X_train_vec, self.y_train)
        return svm
    
    def evaluate_model(self, model, model_name):
        """Évalue un modèle"""
        print(f"\n Évaluation {model_name}...")
        
        # Prédictions
        y_pred = model.predict(self.X_test_vec)
        
        # Métriques
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        print(f"\n{'='*60}")
        print(f"Modèle: {model_name}")
        print(f"{'='*60}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score (weighted): {f1:.4f}")
        
        print(f"\n Classification Report:")
        target_names = ['Négatif', 'Neutre', 'Positif']
        print(classification_report(self.y_test, y_pred, target_names=target_names))
        
        # Matrice de confusion
        cm = confusion_matrix(self.y_test, y_pred)
        
        return {
            'model': model,
            'name': model_name,
            'accuracy': accuracy,
            'f1_score': f1,
            'confusion_matrix': cm
        }
    
    def plot_confusion_matrix(self, cm, model_name):
        """Visualise la matrice de confusion"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Négatif', 'Neutre', 'Positif'],
            yticklabels=['Négatif', 'Neutre', 'Positif']
        )
        plt.title(f'Matrice de Confusion - {model_name}')
        plt.ylabel('Vrai Label')
        plt.xlabel('Prédiction')
        
        viz_dir = Path("logs/visualizations")
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(viz_dir / f'confusion_matrix_{model_name.replace(" ", "_")}.png', 
                    dpi=300, bbox_inches='tight')
        print(f" Matrice sauvegardée")
    
    def measure_inference_time(self, model, n_samples=50):
        """Mesure le temps d'inférence"""
        print(f"\n Mesure du temps d'inférence...")
        
        sample_texts = self.X_test.sample(n=n_samples, random_state=42)
        sample_vec = self.vectorizer.transform(sample_texts)
        
        start_time = time.time()
        _ = model.predict(sample_vec)
        inference_time = (time.time() - start_time) * 1000  # en ms
        
        print(f" Temps pour {n_samples} commentaires: {inference_time:.2f}ms")
        print(f" Temps moyen par commentaire: {inference_time/n_samples:.2f}ms")
        
        return inference_time
    
    def train_and_compare(self):
        """Entraîne et compare plusieurs modèles"""
        models_results = []
        
        # Logistic Regression
        lr_model = self.train_logistic_regression()
        lr_results = self.evaluate_model(lr_model, "Logistic Regression")
        self.plot_confusion_matrix(lr_results['confusion_matrix'], "Logistic Regression")
        lr_results['inference_time'] = self.measure_inference_time(lr_model)
        models_results.append(lr_results)
        
        # Random Forest
        rf_model = self.train_random_forest()
        rf_results = self.evaluate_model(rf_model, "Random Forest")
        self.plot_confusion_matrix(rf_results['confusion_matrix'], "Random Forest")
        rf_results['inference_time'] = self.measure_inference_time(rf_model)
        models_results.append(rf_results)
        
        # SVM
        svm_model = self.train_svm()
        svm_results = self.evaluate_model(svm_model, "SVM")
        self.plot_confusion_matrix(svm_results['confusion_matrix'], "SVM")
        svm_results['inference_time'] = self.measure_inference_time(svm_model)
        models_results.append(svm_results)
        
        # Sélectionner le meilleur modèle
        print("\n" + "="*60)
        print(" COMPARAISON DES MODÈLES")
        print("="*60)
        
        for result in models_results:
            print(f"\n{result['name']}:")
            print(f"  Accuracy: {result['accuracy']:.4f}")
            print(f"  F1-Score: {result['f1_score']:.4f}")
            print(f"  Temps d'inférence: {result['inference_time']:.2f}ms")
        
        # Choisir le meilleur basé sur F1-score et temps
        best_model = max(models_results, key=lambda x: x['f1_score'])
        
        print(f"\n Meilleur modèle: {best_model['name']}")
        print(f"   F1-Score: {best_model['f1_score']:.4f}")
        print(f"   Accuracy: {best_model['accuracy']:.4f}")
        
        self.model = best_model['model']
        self.best_model_name = best_model['name']
        
        return models_results
    
    def save_model(self):
        """Sauvegarde le modèle et le vectoriseur"""
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Sauvegarder le vectoriseur
        vectorizer_path = models_dir / "tfidf_vectorizer.joblib"
        joblib.dump(self.vectorizer, vectorizer_path)
        print(f"\n Vectoriseur sauvegardé: {vectorizer_path}")
        
        # Sauvegarder le modèle
        model_path = models_dir / "sentiment_model.joblib"
        joblib.dump(self.model, model_path)
        print(f" Modèle sauvegardé: {model_path}")
        
        # Sauvegarder les métadonnées
        metadata = {
            'model_type': self.best_model_name,
            'accuracy': float(accuracy_score(self.y_test, self.model.predict(self.X_test_vec))),
            'f1_score': float(f1_score(self.y_test, self.model.predict(self.X_test_vec), average='weighted')),
            'n_features': len(self.vectorizer.vocabulary_),
            'classes': {0: 'Négatif', 1: 'Neutre', 2: 'Positif'}
        }
        
        import json
        metadata_path = models_dir / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f" Métadonnées sauvegardées: {metadata_path}")

def main():
    trainer = SentimentModelTrainer()
    
    # Charger les données
    trainer.load_data(
        "data/processed/train.csv",
        "data/processed/test.csv"
    )
    
    # Créer le vectoriseur
    trainer.create_vectorizer()
    
    # Entraîner et comparer les modèles
    results = trainer.train_and_compare()
    
    # Sauvegarder le meilleur modèle
    trainer.save_model()
    
    print("\n Entraînement terminé avec succès!")

if __name__ == "__main__":
    main()