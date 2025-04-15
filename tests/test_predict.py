from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from src.predict import Predictor

@patch("src.predict.Path.exists", return_value=True)  # 👈 ¡Mock del exists!
@patch("src.predict.joblib.load")
@patch("src.predict.pd.read_csv")
def test_predict_latest_funciona(mock_read_csv, mock_joblib_load, mock_exists):
    # Simula joblib.load() para modelo, escaladores y feature_names
    modelo_mock = MagicMock()
    modelo_mock.predict.return_value = np.array([0.1, 0.2])

    scaler_X_mock = MagicMock()
    scaler_X_mock.transform.return_value = np.array([[1, 2], [3, 4]])

    scaler_y_mock = MagicMock()
    scaler_y_mock.inverse_transform.return_value = np.array([[30], [35]])

    mock_joblib_load.side_effect = [
        modelo_mock,      # model
        scaler_X_mock,    # scaler_X
        scaler_y_mock,    # scaler_y
        ["feat1", "feat2"]  # feature_names
    ]

    # Simula df con features
    df_mock = pd.DataFrame({
        "date": ["2025-04-15", "2025-04-16"],
        "feat1": [0.5, 0.6],
        "feat2": [0.1, 0.2]
    })
    mock_read_csv.return_value = df_mock

    predictor = Predictor()
    resultado = predictor.predict_latest()

    assert not resultado.empty
    assert "prediccion_aprobacion" in resultado.columns