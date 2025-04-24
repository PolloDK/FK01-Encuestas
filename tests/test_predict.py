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
    
def test_no_predicciones_despues_de_ultima_aprobacion():
    predictor = Predictor()
    resultados = predictor.predict()

    # Cargar los features con los datos de aprobación reales
    df_features = pd.read_csv(predictor.features_path)
    df_features["date"] = pd.to_datetime(df_features["date"])
    if resultados.empty or "date" not in resultados.columns:
        assert True, "✅ No se generaron predicciones como se esperaba."
        return

    # Última fecha donde hay dato real de aprobación
    ultima_fecha_real = df_features[df_features["aprobacion_boric"].notna()]["date"].max()

    # Fechas en las que se generaron predicciones
    fechas_predichas = resultados["date"]

    # Validar que no haya ninguna predicción más allá de la última fecha válida
    assert all(fechas_predichas <= ultima_fecha_real), (
        f"❌ Se encontraron predicciones con fecha posterior a la última fecha de aprobación real: {ultima_fecha_real}"
    )