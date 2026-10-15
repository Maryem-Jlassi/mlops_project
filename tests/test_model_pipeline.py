import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from model_pipeline import save_model, load_model


def test_save_and_load_model(tmp_path):
    """Vérifie qu'on peut sauvegarder puis recharger un modèle."""
    model = RandomForestClassifier(n_estimators=2)
    x = np.random.rand(20, 3)
    y = np.random.randint(0, 2, 20)
    model.fit(x, y)

    path = os.path.join(tmp_path, "m.pkl")
    save_model(model, path)
    loaded = load_model(path)

    assert loaded is not None
    assert hasattr(loaded, "predict")
