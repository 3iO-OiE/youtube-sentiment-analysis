FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les requirements
COPY requirements-prod.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copier les fichiers nécessaires
COPY app_api.py .
COPY models/ ./models/

# Exposer le port
EXPOSE 7860

# Commande de démarrage
CMD ["uvicorn", "app_api:app", "--host", "0.0.0.0", "--port", "7860"]