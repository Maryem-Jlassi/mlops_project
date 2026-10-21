"""API FastAPI pour exposer la fonction predict du modèle Churn."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="Churn Prediction API")

MODEL_PATH = "classifier.joblib"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
print(f"Modèle chargé depuis {MODEL_PATH}")

# Les 9 features exactes attendues par le modèle
FEATURES = [
    "CreditScore", "Gender", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
]


class PredictionInput(BaseModel):
    CreditScore: float
    Gender: int          # 0 = Female, 1 = Male (selon ton encoder)
    Age: float
    Tenure: float
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float


@app.post("/predict")
def predict(data: PredictionInput):
    try:
        # Construire le DataFrame avec les 9 features dans le bon ordre
        input_df = pd.DataFrame([data.model_dump()])[FEATURES]
        prediction = model.predict(input_df)[0]
        return {"prediction": int(prediction)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/")
def root():
    return {"message": "Churn Prediction API is running. Go to /docs"}
