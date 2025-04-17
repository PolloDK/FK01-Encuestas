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

        # === Aprobación ===
        self.model_aprob = self._load_model("modelo_xgb.pkl")
        self.scaler_X_aprob = self._load_model("scaler_X.pkl")
        self.scaler_y_aprob = self._load_model("scaler_y.pkl")
        self.feature_names_aprob = self._load_model("feature_names.pkl")

        # === Desaprobación ===
        self.model_desaprob = self._load_model("modelo_xgb_desaprobacion.pkl")
        self.scaler_X_desaprob = self._load_model("scaler_X_desaprobacion.pkl")
        self.scaler_y_desaprob = self._load_model("scaler_y_desaprobacion.pkl")
        self.feature_names_desaprob = self._load_model("feature_names_desaprobacion.pkl")

    def _load_model(self, filename):
        path = self.model_dir / filename
        if not path.exists():
            logger.warning(f"No se encontró el archivo: {filename}")
            return None
        try:
            return joblib.load(path)
        except Exception as e:
            logger.error(f"Error al cargar {filename}: {e}")
            return None

    def predict(self):
        try:
            df = pd.read_csv(self.features_path)
            logger.info(f"Features cargados desde {self.features_path}")
            print(f"✅ Features cargados: {df.shape}")
        except Exception as e:
            logger.error(f"No se pudo cargar el archivo de features: {e}")
            return pd.DataFrame()

        resultados = pd.DataFrame()

        # === Predicción de aprobación ===
        if self.model_aprob and self.scaler_X_aprob and self.scaler_y_aprob and self.feature_names_aprob:
            df_aprob = df.copy()
            try:
                columnas_excluir_aprob = [
                    'week_start', 'aprobacion_boric', 'desaprobacion_boric',
                    'disapproval_rolling_7d', 'disapproval_lag_7d',
                    'disapproval_diff', 'disapproval_pct_change'
                ]
                df_aprob = df_aprob.drop(columns=columnas_excluir_aprob, errors='ignore')
                #print(f"🔍 Variables disponibles para aprobación: {df_aprob.columns.tolist()}")
                features_presentes_aprob = [f for f in self.feature_names_aprob if f in df_aprob.columns]
                #print(f"✅ Usando variables para aprobación: {features_presentes_aprob}")
                X_aprob = df_aprob[features_presentes_aprob]
                X_aprob = self.scaler_X_aprob.transform(X_aprob)
                y_aprob_scaled = self.model_aprob.predict(X_aprob)
                y_aprob = self.scaler_y_aprob.inverse_transform(y_aprob_scaled.reshape(-1, 1)).flatten()

                df_aprob_result = df_aprob[["date"]].copy()
                df_aprob_result["prediccion_aprobacion"] = y_aprob
                resultados = df_aprob_result if resultados.empty else resultados.merge(df_aprob_result, on="date", how="outer")
                logger.info(f"Predicciones de aprobación generadas: {len(y_aprob)}")
            except Exception as e:
                logger.error(f"Error al generar predicción de aprobación: {e}")

        # === Predicción de desaprobación ===
        if self.model_desaprob and self.scaler_X_desaprob and self.scaler_y_desaprob and self.feature_names_desaprob:
            df_desaprob = df.copy()
            try:
                columnas_excluir_desaprob = [
                    'week_start', 'aprobacion_boric', 'desaprobacion_boric',
                    'approval_rolling_7d', 'approval_lag_7d',
                    'approval_diff', 'approval_pct_change'
                ]
                df_desaprob = df_desaprob.drop(columns=columnas_excluir_desaprob, errors='ignore')
                #print(f"🔍 Variables disponibles para desaprobación: {df_desaprob.columns.tolist()}")
                features_presentes_desaprob = [f for f in self.feature_names_desaprob if f in df_desaprob.columns]
                #print(f"✅ Usando variables para desaprobación: {features_presentes_desaprob}")
                X_desaprob = df_desaprob[features_presentes_desaprob]
                X_desaprob = self.scaler_X_desaprob.transform(X_desaprob)
                y_desaprob_scaled = self.model_desaprob.predict(X_desaprob)
                y_desaprob = self.scaler_y_desaprob.inverse_transform(y_desaprob_scaled.reshape(-1, 1)).flatten()

                df_desaprob_result = df_desaprob[["date"]].copy()
                df_desaprob_result["prediccion_desaprobacion"] = y_desaprob
                resultados = df_desaprob_result if resultados.empty else resultados.merge(df_desaprob_result, on="date", how="outer")
                logger.info(f"Predicciones de desaprobación generadas: {len(y_desaprob)}")
            except Exception as e:
                logger.error(f"Error al generar predicción de desaprobación: {e}")

        return resultados.sort_values("date")


if __name__ == "__main__":
    predictor = Predictor()
    resultados = predictor.predict()

    if resultados.empty:
        logger.warning("No se generaron predicciones. Guardando CSV vacío con columnas.")
        resultados = pd.DataFrame(columns=["date", "prediccion_aprobacion", "prediccion_desaprobacion"])

    print(f"💾 Guardando archivo en: {PREDICTIONS_PATH}")
    resultados.to_csv(PREDICTIONS_PATH, index=False)
    print(f"📈 Predicción guardada en {PREDICTIONS_PATH}")
