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
import mlflow
import mlflow.sklearn

# Configuration MLflow (utilise une base SQLite locale)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("churn-prediction")

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


def train_model(x_train, y_train, **hyperparams):
    """Entraîner le modèle avec tracking MLflow."""
    from sklearn.ensemble import RandomForestClassifier

    # Valeurs par défaut si non fournies
    n_estimators = hyperparams.get("n_estimators", 100)
    max_depth = hyperparams.get("max_depth", None)
    min_samples_split = hyperparams.get("min_samples_split", 2)

    # Démarrer un run MLflow
    with mlflow.start_run(run_name="random_forest_training"):

        # 1. Logger les paramètres
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("random_state", 42)

        # 2. Créer et entraîner le modèle
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
        )
        model.fit(x_train, y_train)

        # 3. Enregistrer le modèle dans MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name="churn_classifier",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
            )

        print(f"✅ Modèle loggé dans MLflow avec {n_estimators} arbres")
        return model

def evaluate_model(model, x_test, y_test):
    """Évaluer le modèle et logger les métriques dans MLflow."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # Prédictions
    y_pred = model.predict(x_test)

    # Calcul des métriques
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Logger dans MLflow
    with mlflow.start_run(run_name="model_evaluation"):
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Sauvegarder le rapport de classification comme artefact
        from sklearn.metrics import classification_report
        report = classification_report(y_test, y_pred)
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")

    print(f"✅ Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    print(report)
    return accuracy, precision, recall, f1
def save_model(model, path="classifier.joblib"):
    """Sauvegarde le modèle entraîné avec joblib."""
    joblib.dump(model, path)
    print(f"Modèle sauvegardé dans {path}")


def load_model(path="classifier.joblib"):
    """Charge un modèle sauvegardé."""
    model = joblib.load(path)
    print(f"Modèle chargé depuis {path}")
    return model
