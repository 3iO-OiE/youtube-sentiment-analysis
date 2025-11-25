import requests
import time
import traceback

API_URL = "https://3xpe-youtube-sentiment-api.hf.space"
# API_URL = "http://localhost:8000"
def test_health_endpoint():
    """Teste le endpoint /health"""
    print("\n🔍 Test du endpoint /health")
    
    try:
        response = requests.get(f"{API_URL}/health")
        print(f"  Status Code: {response.status_code}")
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        print(f"  Response: {data}")
        
        assert data['status'] == 'healthy', f"Status is not healthy: {data.get('status')}"
        assert 'model_type' in data, "model_type not in response"
        
        print("  ✅ Health check OK")
        print(f"  Modèle: {data.get('model_type')}")
        
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        traceback.print_exc()
        raise

def test_predict_batch_endpoint():
    """Teste le endpoint /predict_batch"""
    print("\n🔍 Test du endpoint /predict_batch")
    
    try:
        # Test 1: Batch normal
        payload = {
            "comments": [
                {"text": "Amazing video!"},
                {"text": "This is terrible"},
                {"text": "It's okay"}
            ]
        }
        
        print(f"  Envoi de {len(payload['comments'])} commentaires...")
        
        start = time.time()
        response = requests.post(f"{API_URL}/predict_batch", json=payload)
        response_time = (time.time() - start) * 1000
        
        print(f"  Status Code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"  ❌ Réponse API: {response.text}")
            raise Exception(f"API returned {response.status_code}")
        
        data = response.json()
        print(f"  Clés de réponse: {list(data.keys())}")
        
        # Vérifications détaillées
        assert 'predictions' in data, "Clé 'predictions' manquante"
        assert 'statistics' in data, "Clé 'statistics' manquante"
        
        print(f"  Nombre de prédictions: {len(data['predictions'])}")
        assert len(data['predictions']) == 3, f"Expected 3 predictions, got {len(data['predictions'])}"
        
        print(f"  ✅ Batch normal OK (temps: {response_time:.0f}ms)")
        print(f"  Stats: {data['statistics']}")
        
        # Afficher quelques prédictions
        print("\n  Exemples de prédictions:")
        for pred in data['predictions']:
            print(f"    - {pred['sentiment']}: \"{pred['text'][:30]}...\" (conf: {pred['confidence']:.2f})")
        
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        traceback.print_exc()
        raise

def run_all_tests():
    """Lance tous les tests"""
    print("="*60)
    print("🧪 TESTS DE L'API")
    print("="*60)
    
    try:
        # Test de connexion basique
        print("\n🔌 Test de connexion à l'API...")
        try:
            response = requests.get(API_URL, timeout=5)
            print(f"  ✅ API accessible sur {API_URL}")
        except requests.exceptions.ConnectionError:
            print(f"  ❌ Impossible de se connecter à {API_URL}")
            print("  Vérifiez que l'API est lancée avec:")
            print("  uvicorn src.api.app:app --port 8000")
            return
        
        test_health_endpoint()
        test_predict_batch_endpoint()
        
        print("\n" + "="*60)
        print("✅ TOUS LES TESTS PASSÉS")
        print("="*60)
        
    except Exception as e:
        print(f"\n" + "="*60)
        print(f"❌ TEST ÉCHOUÉ")
        print(f"Erreur: {str(e)}")
        print("="*60)

if __name__ == "__main__":
    run_all_tests()