from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict
import joblib
import numpy as np
from pathlib import Path
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modèles Pydantic
class Comment(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)

class PredictionBatch(BaseModel):
    comments: List[Comment] = Field(..., min_items=1, max_items=100)

class SentimentPrediction(BaseModel):
    text: str
    sentiment: str
    confidence: float
    label: int

class BatchPredictionResponse(BaseModel):
    predictions: List[SentimentPrediction]
    statistics: Dict[str, float]
    total_comments: int

# Initialisation de l'application
app = FastAPI(
    title="YouTube Sentiment Analysis API",
    description="API pour analyser le sentiment des commentaires YouTube",
    version="1.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifier les domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales pour le modèle
vectorizer = None
model = None
SENTIMENT_LABELS = {0: "Négatif", 1: "Neutre", 2: "Positif"}

def load_models():
    """Charge le modèle et le vectoriseur au démarrage"""
    global vectorizer, model
    
    try:
        models_dir = Path("models")
        
        logger.info("Chargement du vectoriseur...")
        vectorizer = joblib.load(models_dir / "tfidf_vectorizer.joblib")
        
        logger.info("Chargement du modèle...")
        model = joblib.load(models_dir / "sentiment_model.joblib")
        
        logger.info(" Modèles chargés avec succès!")
        
    except Exception as e:
        logger.error(f" Erreur lors du chargement des modèles: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Événement au démarrage de l'application"""
    load_models()

@app.get("/")
async def root():
    """Route racine"""
    return {
        "message": "YouTube Sentiment Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Vérifier l'état de l'API",
            "/predict_batch": "Analyser un batch de commentaires"
        }
    }

@app.get("/health")
async def health_check():
    """Vérifie l'état de l'API et du modèle"""
    
    if vectorizer is None or model is None:
        raise HTTPException(
            status_code=503,
            detail="Modèles non chargés"
        )
    
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
        "model_type": type(model).__name__
    }

@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(batch: PredictionBatch):
    """
    Analyse un batch de commentaires et retourne les sentiments
    
    Args:
        batch: Liste de commentaires à analyser
    Returns:
        Prédictions avec statistiques globales
    """

    if vectorizer is None or model is None:
        raise HTTPException(
            status_code=503,
            detail="Modèles non chargés"
        )

    try:
        # Extraire les textes
        texts = [comment.text for comment in batch.comments]
        
        # Vectorisation
        X = vectorizer.transform(texts)
        
        # Prédictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        # Construire les résultats
        results = []
        sentiment_counts = {0: 0, 1: 0, 2: 0}
        
        for i, (text, pred, proba) in enumerate(zip(texts, predictions, probabilities)):
            confidence = float(np.max(proba))
            sentiment = SENTIMENT_LABELS[pred]
            
            results.append(SentimentPrediction(
                text=text,
                sentiment=sentiment,
                confidence=confidence,
                label=int(pred)
            ))
            
            sentiment_counts[pred] += 1
        
        # Calculer les statistiques
        total = len(texts)
        statistics = {
            "negative_percentage": round((sentiment_counts[0] / total) * 100, 2),
            "neutral_percentage": round((sentiment_counts[1] / total) * 100, 2),
            "positive_percentage": round((sentiment_counts[2] / total) * 100, 2),
            "average_confidence": round(float(np.mean([r.confidence for r in results])), 4)
        }
        
        logger.info(f" Analysé {total} commentaires avec succès")
        
        return BatchPredictionResponse(
            predictions=results,
            statistics=statistics,
            total_comments=total
        )
        
    except Exception as e:
        logger.error(f" Erreur lors de la prédiction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prédiction: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
