import requests
import json

def test_api():
    """Teste l'API localement"""
    
    base_url = "http://localhost:8000"
    
    # Test 1: Health check
    print(" Test 1: Health Check")
    response = requests.get(f"{base_url}/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    
    # Test 2: Prédiction batch
    print("\n Test 2: Batch Prediction")
    test_comments = {
        "comments": [
            {"text": "This is absolutely amazing! Love it!"},
            {"text": "Terrible experience, very disappointing"},
            {"text": "It's okay, nothing special"},
            {"text": "Best video ever! Subscribed!"},
            {"text": "Boring and useless content"}
        ]
    }
    
    response = requests.post(
        f"{base_url}/predict_batch",
        json=test_comments
    )
    
    print(f"Status: {response.status_code}")
    result = response.json()
    
    print("\n Statistiques:")
    print(json.dumps(result['statistics'], indent=2))
    
    print("\n Prédictions:")
    for pred in result['predictions']:
        print(f"  {pred['sentiment']}: {pred['text'][:50]}... (conf: {pred['confidence']:.2f})")

if __name__ == "__main__":
    test_api()