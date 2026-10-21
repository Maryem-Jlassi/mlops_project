"""Pipeline Prefect - Atelier 3 MLOps."""

import subprocess
import argparse
from prefect import flow, task

# Import direct depuis model_pipeline (plus simple et pas de conflit)
from model_pipeline import (
    prepare_data as _prepare_data,
    train_model as _train_model,
    evaluate_model as _evaluate_model,
    save_model as _save_model,
    load_model as _load_model,
)
from main import CSV_PATH, MODEL_PATH

PROJECT_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]


# ============================================================
#  TASKS
# ============================================================
@task
def start_api():
    """Lancer l'API FastAPI avec uvicorn."""
    subprocess.run(
        ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"],
        check=False,
    )

@task
def clone_repo():
    """Clone the remote Git repository."""
    import os
    import shutil

    dest = "/tmp/ml_project_remote"  # nosec B108
    if os.path.exists(dest):
        shutil.rmtree(dest)

    subprocess.run(  # nosec B603 B607
        ["git", "clone", "git@github.com:Maryem-Jlassi/mlops_project.git", dest],
        check=True,
    )

@task
def install_dependencies():
    """Installer les dépendances."""
    subprocess.run(  # nosec B603 B607
        ["pip", "install", "-r", "requirements.txt"], check=True
    )


@task
def prepare_data():
    """Préparation des données."""
    return _prepare_data(CSV_PATH)


@task
def train_model(x_train, y_train):
    """Entraînement du modèle."""
    return _train_model(x_train, y_train)


@task
def save_model(model):
    """Sauvegarder le modèle."""
    return _save_model(model, MODEL_PATH)


@task
def load_model():
    """Charger le modèle."""
    return _load_model(MODEL_PATH)


@task
def evaluate_model(model, x_test, y_test):
    """Évaluer le modèle."""
    return _evaluate_model(model, x_test, y_test)


@task
def format_code():
    """Formater le code avec black."""
    subprocess.run(["black", *PROJECT_FILES], check=False)  # nosec B603 B607


@task
def quality_code():
    """Analyser la qualité du code."""
    subprocess.run(["pylint", *PROJECT_FILES], check=False)  # nosec B603 B607


@task
def security_code():
    """Analyser la sécurité du code."""
    subprocess.run(["bandit", "-r", *PROJECT_FILES], check=False)  # nosec B603 B607


@task
def run_unit_tests():
    """Exécuter les tests unitaires."""
    subprocess.run(["pytest", "tests/", "-v"], check=False)  # nosec B603 B607


# ============================================================
#  FLOWS
# ============================================================

@flow(name="ml-pipeline-api")
def api_flow():
    """Flow de lancement de l'API FastAPI."""
    start_api()

@flow(name="ml-pipeline-code")
def code_flow():
    """Flow code : clone + install + format + quality + security + tests."""
    clone_repo()  # ⬅️ PREMIÈRE étape
    install_dependencies()
    format_code()
    quality_code()
    security_code()
    run_unit_tests()


@flow(name="ml-pipeline-all")
def all_flow():
    """Flow all : code + préparation + train + save + evaluate."""
    code_flow()
    x_train, x_test, y_train, y_test = prepare_data()
    model = train_model(x_train, y_train)
    save_model(model)
    evaluate_model(model, x_test, y_test)


@flow(name="ml-pipeline-train")
def train_flow():
    """Flow train : préparation + entraînement + sauvegarde."""
    x_train, _, y_train, _ = prepare_data()
    model = train_model(x_train, y_train)
    save_model(model)
    return model


@flow(name="ml-pipeline-evaluate")
def evaluate_flow():
    """Flow evaluate : chargement + évaluation."""
    _, x_test, _, y_test = prepare_data()
    model = load_model()
    evaluate_model(model, x_test, y_test)


# ============================================================
#  CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flow",
        choices=["all", "entrainement", "evaluate", "code", "api"],
        required=True,
    )
    args = parser.parse_args()

    {
        "all": all_flow,
        "entrainement": train_flow,
        "evaluate": evaluate_flow,
        "code": code_flow,
        "api": api_flow,
    }[args.flow]()
