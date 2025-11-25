---
title: YouTube Sentiment Analysis API
emoji: 📊
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 7860
---

# YouTube Sentiment Analysis API

API FastAPI pour analyser le sentiment des commentaires YouTube.

## Endpoints

- `GET /` - Informations sur l'API
- `GET /health` - Vérification de l'état
- `POST /predict_batch` - Analyse de commentaires

## Exemple d'utilisation
```python
import requests

url = "https://YOUR-SPACE.hf.space/predict_batch"

data = {
    "comments": [
        {"text": "This video is amazing!"},
        {"text": "Terrible content"},
        {"text": "It's okay"}
    ]
}

response = requests.post(url, json=data)
print(response.json())
```

## Modèle

- Vectoriseur: TF-IDF
- Classificateur: Logistic Regression
- Classes: Négatif (0), Neutre (1), Positif (2)
````