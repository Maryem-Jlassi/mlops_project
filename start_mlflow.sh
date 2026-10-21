#!/bin/bash
# Lance l'interface MLflow en arrière-plan avec SQLite

cd ~/ml_project
source venv/bin/activate

# Tuer une éventuelle instance existante
pkill -f "mlflow ui" 2>/dev/null

# Lancer l'UI MLflow
nohup mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000 > mlflow.log 2>&1 &

echo "✅ MLflow UI lancée sur http://127.0.0.1:5000"
echo "📝 Logs : mlflow.log"
