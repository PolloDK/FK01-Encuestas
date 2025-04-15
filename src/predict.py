import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from src.config import FEATURES_DATASET_PATH, MODEL_DIR, PREDICTIONS_PATH
from src.logger import get_logger

logger = get_logger(__name__, "predict.log")


class Predictor:
    def __init__(self, features_path=FEATURES_DATASET_PATH, model_dir=MODEL_DIR):
        self.features_path = Path(features_path)
        self.model_dir = Path(model_dir)

        modelo_path = self.model_dir / "modelo_xgb.pkl"
        if not modelo_path.exists():
            logger.warning(f"No se encontró el modelo entrenado en {modelo_path}. Se omite la predicción.")
            self.model = None
            return

        try:
            self.model = joblib.load(modelo_path)
            self.scaler_X = joblib.load(self.model_dir / "scaler_X.pkl")
            self.scaler_y = joblib.load(self.model_dir / "scaler_y.pkl")
            self.feature_names = joblib.load(self.model_dir / "feature_names.pkl")
            logger.info("Modelo y escaladores cargados correctamente.")
        except Exception as e:
            logger.error(f"Error al cargar modelo o escaladores: {e}")
            self.model = None

    def predict_latest(self):
        if self.model is None:
            logger.warning("Predicción omitida porque no hay modelo cargado.")
            return pd.DataFrame()

        try:
            df = pd.read_csv(self.features_path)
            logger.info(f"Features cargados desde {self.features_path}")
        except Exception as e:
            logger.error(f"No se pudo cargar el archivo de features: {e}")
            return pd.DataFrame()

        df = df.dropna(subset=self.feature_names)
        if df.empty:
            logger.warning("No hay datos suficientes para realizar la predicción (faltan features completos).")
            return pd.DataFrame()

        X = df[self.feature_names]
        if X.empty:
            logger.warning("No hay filas con features válidos para predecir.")
            return pd.DataFrame()

        try:
            X_scaled = self.scaler_X.transform(X)
            y_pred_scaled = self.model.predict(X_scaled)
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

            df_result = df[["date"]].copy()
            df_result["prediccion_aprobacion"] = y_pred
            logger.info(f"{len(df_result)} predicciones generadas.")
            return df_result
        except Exception as e:
            logger.error(f"Error durante la predicción: {e}")
            return pd.DataFrame()


if __name__ == "__main__":
    predictor = Predictor()
    resultados = predictor.predict_latest()

    if resultados.empty:
        logger.warning("No se generaron predicciones. Guardando CSV vacío con columnas.")
        resultados = pd.DataFrame(columns=["date", "prediccion_aprobacion"])

    resultados.to_csv(PREDICTIONS_PATH, index=False)
    print(f"📈 Predicción guardada en {PREDICTIONS_PATH}")
