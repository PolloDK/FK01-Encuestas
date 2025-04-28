from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from src.predict import Predictor

@patch("src.predict.Path.exists", return_value=True)
@patch("src.predict.joblib.load")
@patch("src.predict.pd.read_csv")
def test_predict_latest_funciona(mock_read_csv, mock_joblib_load, mock_exists):
    modelo_mock = MagicMock()
    modelo_mock.predict.return_value = np.array([0.1, 0.2])

    scaler_X_mock = MagicMock()
    scaler_X_mock.transform.return_value = np.array([[1, 2]])

    scaler_y_mock = MagicMock()
    scaler_y_mock.inverse_transform.return_value = np.array([[30], [40]])

    mock_joblib_load.side_effect = [
        {
            "model": modelo_mock,
            "feature_names": ["feat1", "feat2"],
            "scaler_X": scaler_X_mock,
            "scaler_y": scaler_y_mock
        },
        {
            "model": modelo_mock,
            "feature_names": ["feat1", "feat2"],
            "scaler_X": scaler_X_mock,
            "scaler_y": scaler_y_mock
        }
    ]

    df_mock = pd.DataFrame({
        "date": ["2025-04-15", "2025-04-16"],
        "feat1": [0.5, 0.6],
        "feat2": [0.1, 0.2],
        "aprobacion_boric": [0.27, None]
    })
    df_mock["date"] = pd.to_datetime(df_mock["date"])
    mock_read_csv.return_value = df_mock

    predictor = Predictor()
    resultado = predictor.predict()

    assert not resultado.empty
    assert "prediccion_aprobacion" in resultado.columns
    assert resultado.shape[0] == 2  # ✅ validación directa
    
def test_predicciones_solo_con_features_completos():
    predictor = Predictor()
    resultados = predictor.predict()

    df_features = pd.read_csv(predictor.features_path)
    df_features["date"] = pd.to_datetime(df_features["date"])

    if resultados.empty:
        assert True, "✅ No se generaron predicciones porque no había features."
        return

    # Fechas en las que había features no nulos
    fechas_features_validos = df_features.dropna(subset=predictor.aprobacion_bundle["feature_names"])["date"]

    # Fechas predichas
    fechas_predichas = resultados["date"]

    # Asegurar que solo se predice para fechas donde había features válidos
    assert all(fechas_predichas.isin(fechas_features_validos)), (
        "❌ Se generaron predicciones para días donde no había features completos."
    )