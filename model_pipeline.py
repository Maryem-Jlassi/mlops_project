# model_pipeline.py
"""
Pipeline ML modulaire pour la prédiction du churn client.
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def prepare_data(csv_path, test_size=0.2, random_state=1):
    """
    Charge, nettoie, encode et découpe les données.

    Parameters
    ----------
    csv_path : str
        Chemin vers le fichier CSV.
    test_size : float
        Proportion du jeu de test.
    random_state : int
        Graine aléatoire.

    Returns
    -------
    x_train, x_test, y_train, y_test
    """
    df = pd.read_csv(csv_path)

    # Encodage de Gender
    encoder = LabelEncoder()
    df["Gender"] = encoder.fit_transform(df["Gender"])

    # Suppression colonnes inutiles
    df = df.drop(["Surname", "Geography"], axis=1)

    # Séparation features / cible
    x = df.drop(["Exited"], axis=1)
    y = df["Exited"]

    # Suppression identifiants
    x = x.drop(columns=["RowNumber", "CustomerId"])

    # Split train/test
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    return x_train, x_test, y_train, y_test


def train_model(x_train, y_train, n_estimators=100, random_state=42):
    """
    Entraîne un RandomForestClassifier.
    """
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test):
    """
    Évalue le modèle et retourne les métriques.
    """
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Confusion matrix:\n", matrix)
    print("Classification report:\n", report)

    return {
        "accuracy": accuracy,
        "confusion_matrix": matrix,
        "classification_report": report,
    }


def save_model(model, path="classifier.joblib"):
    """Sauvegarde le modèle entraîné avec joblib."""
    joblib.dump(model, path)
    print(f"Modèle sauvegardé dans {path}")


def load_model(path="classifier.joblib"):
    """Charge un modèle sauvegardé."""
    model = joblib.load(path)
    print(f"Modèle chargé depuis {path}")
    return model
