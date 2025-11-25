---
title: YouTube Sentiment Analysis API
emoji: 📊
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 7860
---

# 🎬 YouTube Sentiment Analysis - Analyse de Sentiment des Commentaires YouTube

Système MLOps complet permettant l'analyse automatique du sentiment des commentaires YouTube en temps réel via une extension Chrome connectée à une API cloud.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📋 Table des Matières

- [Description du Projet](#description-du-projet)
- [Architecture Technique](#architecture-technique)
- [Fonctionnalités](#fonctionnalités)
- [Installation et Utilisation](#installation-et-utilisation)
- [API Endpoints](#api-endpoints)
- [Extension Chrome](#extension-chrome)
- [Performance du Modèle](#performance-du-modèle)
- [Démonstration](#démonstration)

## 🎯 Description du Projet

### Problématique

Les créateurs de contenu YouTube reçoivent des centaines, voire des milliers de commentaires sur leurs vidéos. L'analyse manuelle de ces commentaires est :
- ⏰ Chronophage et peu scalable
- 🤔 Subjective et biaisée
- 📊 Difficile à quantifier pour des décisions stratégiques

### Solution

Notre système offre une solution complète comprenant :
- 🤖 **Modèle ML** entraîné sur 36K+ commentaires avec 87.9% d'accuracy
- 🚀 **API REST FastAPI** déployée sur Hugging Face Spaces
- 🌐 **Extension Chrome** pour analyse en temps réel
- 📈 **Visualisations** interactives et statistiques détaillées

## 🏗️ Architecture Technique

```
┌─────────────────┐
│  Extension      │
│  Chrome         │◄──── Utilisateur visite YouTube
└────────┬────────┘
         │ Extrait commentaires
         ▼
┌─────────────────┐
│  API FastAPI    │
│  (Cloud)        │◄──── POST /predict_batch
└────────┬────────┘
         │ Traitement ML
         ▼
┌─────────────────┐
│  Modèle ML      │
│  TF-IDF + LR    │◄──── Vectorisation + Classification
└────────┬────────┘
         │ Prédictions
         ▼
┌─────────────────┐
│  Résultats      │
│  + Stats        │◄──── Retour à l'extension
└─────────────────┘
```

### Stack Technique

| Composant | Technologies |
|-----------|-------------|
| **Frontend** | JavaScript, HTML5, CSS3, Chrome Extension API |
| **Backend** | FastAPI, Python 3.10+, Uvicorn |
| **ML** | scikit-learn, TF-IDF, Logistic Regression |
| **Déploiement** | Docker, Hugging Face Spaces |
| **Version Control** | Git, GitHub |

## ✨ Fonctionnalités

### Extension Chrome
- ✅ Extraction automatique des commentaires YouTube
- 📊 Statistiques globales en temps réel
- 🎨 Mode sombre/clair
- 🔍 Filtres par sentiment (Positif/Neutre/Négatif)
- 📋 Export des résultats
- ⚡ Performance : analyse de 50 commentaires en <2s

### API
- 🏥 Health check endpoint
- 🔄 Traitement par batch
- 📊 Statistiques détaillées
- 🛡️ Validation automatique des données
- 📝 Documentation interactive Swagger

## 🚀 Installation et Utilisation

### Prérequis

- Python 3.10 ou supérieur
- Git
- Google Chrome
- Compte Hugging Face (pour déploiement)

### 1️⃣ Installation Locale de l'API

```bash
# Cloner le repository
git clone https://github.com/votre-username/youtube-sentiment-analysis.git
cd youtube-sentiment-analysis

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'API localement
uvicorn app_api:app --reload --host 0.0.0.0 --port 8000
```

L'API sera accessible sur `http://localhost:8000`

### 2️⃣ Installation de l'Extension Chrome

1. Ouvrir Chrome et aller dans `chrome://extensions/`
2. Activer le **Mode développeur** (coin supérieur droit)
3. Cliquer sur **Charger l'extension non empaquetée**
4. Sélectionner le dossier `chrome-extension/`
5. L'extension apparaît dans la barre d'outils

### 3️⃣ Configuration de l'Extension

1. Ouvrir l'extension
2. Dans le champ "URL de l'API", entrer :
   - Local : `http://localhost:8000`
   - Production : `https://votre-space.hf.space`
3. Cliquer sur **Sauvegarder**

### 4️⃣ Utilisation

1. Visiter n'importe quelle vidéo YouTube
2. Cliquer sur l'icône de l'extension
3. Cliquer sur **Analyser les Commentaires**
4. Consulter les résultats et statistiques

## 📡 API Endpoints

### GET `/`
Informations générales sur l'API

**Réponse :**
```json
{
  "message": "YouTube Sentiment Analysis API",
  "version": "1.0.0",
  "endpoints": ["/health", "/predict_batch"]
}
```

### GET `/health`
Vérification de l'état de l'API et du modèle

**Réponse :**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "vectorizer_loaded": true,
  "model_type": "LogisticRegression",
  "vocabulary_size": 5000
}
```

### POST `/predict_batch`
Analyse de sentiment pour un batch de commentaires

**Requête :**
```json
{
  "comments": [
    {"text": "This video is amazing! Great content!"},
    {"text": "Terrible experience, waste of time"},
    {"text": "It's okay, nothing special"}
  ]
}
```

**Réponse :**
```json
{
  "predictions": [
    {
      "text": "This video is amazing! Great content!",
      "sentiment": "Positif",
      "confidence": 0.99
    },
    {
      "text": "Terrible experience, waste of time",
      "sentiment": "Négatif",
      "confidence": 0.97
    },
    {
      "text": "It's okay, nothing special",
      "sentiment": "Neutre",
      "confidence": 0.85
    }
  ],
  "statistics": {
    "negative_percentage": 33.33,
    "neutral_percentage": 33.33,
    "positive_percentage": 33.33,
    "average_confidence": 0.94
  },
  "total_comments": 3
}
```

### Exemple Python

```python
import requests

# URL de l'API
url = "https://votre-space.hf.space/predict_batch"

# Données à envoyer
data = {
    "comments": [
        {"text": "This video is amazing! Great content!"},
        {"text": "Terrible experience, waste of time"},
        {"text": "It's okay, nothing special"}
    ]
}

# Requête POST
response = requests.post(url, json=data)

# Afficher les résultats
results = response.json()
print(f"Total: {results['total_comments']} commentaires")
print(f"Positifs: {results['statistics']['positive_percentage']}%")
print(f"Négatifs: {results['statistics']['negative_percentage']}%")
```

### Exemple cURL

```bash
curl -X POST "https://votre-space.hf.space/predict_batch" \
  -H "Content-Type: application/json" \
  -d '{
    "comments": [
      {"text": "This video is amazing!"},
      {"text": "Terrible content"},
      {"text": "It'\''s okay"}
    ]
  }'
```

## 🌐 Extension Chrome

### Fonctionnalités Détaillées

#### 1. Statistiques Globales
- Pourcentage de chaque sentiment
- Graphique circulaire interactif
- Nombre total de commentaires analysés

#### 2. Liste des Commentaires
- Affichage du texte complet
- Badge de sentiment (Positif/Neutre/Négatif)
- Score de confiance du modèle
- Icône visuelle par sentiment

#### 3. Filtres
- **Tous** : Affiche tous les commentaires
- **Positifs** : Filtre uniquement les commentaires positifs
- **Neutres** : Filtre uniquement les commentaires neutres
- **Négatifs** : Filtre uniquement les commentaires négatifs

#### 4. Mode Sombre
- Basculement automatique selon les préférences système
- Toggle manuel disponible
- Persistance de la préférence

#### 5. Export
- Copie des résultats en format texte structuré
- Inclut les statistiques et la liste complète

## 🎯 Performance du Modèle

### Métriques

| Métrique | Valeur |
|----------|--------|
| **Accuracy** | 87.90% |
| **F1-Score (Négatif)** | 0.79 |
| **F1-Score (Neutre)** | 0.92 |
| **F1-Score (Positif)** | 0.89 |
| **Temps d'inférence (50 cmt)** | 40ms |

### Dataset

- **Source** : Reddit Sentiment Analysis
- **Taille** : 36,454 commentaires
- **Distribution** :
  - Négatifs : 8,241 (22.6%)
  - Neutres : 12,454 (34.2%)
  - Positifs : 15,759 (43.2%)

### Modèle

- **Vectorisation** : TF-IDF (5000 features, n-grams 1-2)
- **Algorithme** : Logistic Regression (C=1.0, L2 regularization)
- **Classes** : 
  - 0 : Négatif
  - 1 : Neutre
  - 2 : Positif

## 📸 Démonstration

### 1. Interface de l'Extension

![Extension Chrome](screenshots/extension_interface.png)

L'interface affiche les statistiques globales avec un graphique circulaire et la liste des commentaires filtrables.

### 2. Statistiques Détaillées

![Statistiques](screenshots/statistics.png)

Visualisation des pourcentages de chaque sentiment avec nombres absolus.

### 3. Liste des Commentaires

![Liste Commentaires](screenshots/comments_list.png)

Chaque commentaire est affiché avec son sentiment, sa confiance et une icône colorée.

### 4. Mode Sombre

![Mode Sombre](screenshots/dark_mode.png)

Interface adaptée pour une utilisation confortable la nuit.

### 5. Résultats API

![API Response](screenshots/api_response.png)

Exemple de réponse JSON structurée de l'API.

### 6. Tests Réussis

![Tests](screenshots/tests_passed.png)

Tous les tests unitaires et d'intégration passent avec succès.

## 🐳 Déploiement Docker

### Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app_api.py .
COPY models/ ./models/

EXPOSE 7860

CMD ["uvicorn", "app_api:app", "--host", "0.0.0.0", "--port", "7860"]
```

### Build et Run

```bash
# Build l'image
docker build -t youtube-sentiment-api .

# Run le container
docker run -p 7860:7860 youtube-sentiment-api
```

## 📊 Structure du Projet

```
youtube-sentiment-analysis/
├── data/
│   ├── raw/                 # Données brutes
│   └── processed/           # Données nettoyées
├── models/
│   ├── sentiment_model.joblib
│   └── tfidf_vectorizer.joblib
├── src/
│   ├── data/               # Scripts de traitement
│   ├── models/             # Scripts d'entraînement
│   ├── api/                # Code API
│   └── utils/              # Fonctions utilitaires
├── chrome-extension/
│   ├── manifest.json
│   ├── popup.html
│   ├── popup.js
│   └── styles.css
├── tests/
│   ├── test_api.py
│   └── test_model.py
├── app_api.py              # Application FastAPI
├── Dockerfile
├── requirements.txt
└── README.md
```

##  Configuration Avancée

### Variables d'Environnement

```bash
# .env
API_URL=https://3xpe-youtube-sentiment-api.hf.space
MODEL_PATH=./models/sentiment_model.joblib
VECTORIZER_PATH=./models/tfidf_vectorizer.joblib
MAX_BATCH_SIZE=100
CACHE_ENABLED=true
```

### Personnalisation du Modèle

Pour réentraîner le modèle avec vos propres données :

```python
from src.models.train_model import train_sentiment_model

# Entraîner avec nouveau dataset
train_sentiment_model(
    data_path='data/processed/cleaned_data.csv',
    output_dir='models/',
    vocab_size=5000,
    ngram_range=(1, 2)
)
```



⭐ Si ce projet vous a été utile, n'hésitez pas à lui donner une étoile !
