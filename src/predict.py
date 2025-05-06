import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from src.azure_blob import read_csv_blob, write_csv_blob
from src.azure_blob import download_blob
from config import FEATURES_DATASET_PATH, MODEL_DIR, PREDICTIONS_PATH
from logger import get_logger

logger = get_logger(__name__, "predict.log")

class Predictor:
    def __init__(self, features_path=FEATURES_DATASET_PATH, model_dir=MODEL_DIR):
        self.features_path = Path(features_path)
        self.model_dir = Path(model_dir)
        # === Aprobación ===
        self.aprobacion_bundle = self._load_model("modelo_aprobacion.pkl")
        # === Desaprobación ===
        self.desaprobacion_bundle = self._load_model("modelo_desaprobacion.pkl")

    def _load_model(self, filename):
        path = self.model_dir / filename
        if not path.exists():
            print(f"📥 Modelo {filename} no encontrado localmente. Descargando desde Azure...")
            try:
                download_blob(f"models/{filename}", str(path))
            except Exception as e:
                logger.error(f"❌ Error al descargar modelo desde Azure: {e}")
                return None

    def predict(self):
        print("🚀 Entrando a método `predict()`")
        try:
            df = read_csv_blob(str(self.features_path))
            if df.empty or df.shape[0] == 0 or df.shape[1] == 0:
                logger.warning(f"⚠️ El archivo {self.features_path} fue cargado pero está vacío.")
                return pd.DataFrame()
            logger.info(f"✅ Features cargados desde {self.features_path}: {df.shape}")
            print(f"✅ Features cargados: {df.shape}")
        except Exception as e:
            logger.error(f"❌ No se pudo cargar el archivo de features: {e}")
            print(f"❌ ERROR leyendo features_dataset.csv: {e}")
            return pd.DataFrame()

        resultados = pd.DataFrame()

        # === Predicción de aprobación ===
        if self.aprobacion_bundle:
            try:
                df_aprob = df.copy()
                
                # Obtengo los nombres de las features, el modelo y los scalers a utilizar:
                feature_names = self.aprobacion_bundle["feature_names"]
                model = self.aprobacion_bundle.get("model")
                scaler_X = self.aprobacion_bundle.get("scaler_X", None)
                scaler_y = self.aprobacion_bundle.get("scaler_y", None)
                
                #if self.aprobacion_bundle:
                #    print("✅ Validación bundle:")
                #    print("Features seleccionadas:", self.aprobacion_bundle.get("feature_names"))
                #    print("Modelo entrenado:", model)
                #    print("🔍 Bundle keys:", self.aprobacion_bundle.keys())

                # Asegura que la columna date esté en formato reconocible de fecha
                df_aprob["date"] = pd.to_datetime(df_aprob["date"], errors="coerce")

                # Asegura que todas las columnas requeridas estén en el dataset
                missing_cols = [col for col in feature_names if col not in df_aprob.columns]
                if missing_cols:
                    logger.error(f"❌ Faltan columnas en el DataFrame para predicción de aprobación: {missing_cols}")
                    return resultados


                # Construimos X y filtramos NaNs
                X_aprob_raw = df_aprob[feature_names]
                #print(X_aprob_raw.tail(10))
                #print("📊 Total de NaNs por columna:")
                #print(X_aprob_raw.isna().sum().sort_values(ascending=False))
                logger.info(f"🔍 Filas antes de filtrar NaNs (aprobación): {X_aprob_raw.shape[0]}")
                mask_notna = X_aprob_raw.notna().all(axis=1)
                X_aprob_raw = X_aprob_raw[mask_notna]
                #print("🧪 Columnas en X_aprob_raw:", X_aprob_raw.columns.tolist())
                #print(X_aprob_raw.tail(10))
                df_aprob = df_aprob.loc[X_aprob_raw.index].copy()
                logger.info(f"🧹 Filas después de filtrar NaNs (aprobación): {X_aprob_raw.shape[0]}")
                
                #print("🧪 Columnas en X_aprob_raw:")
                #print(X_aprob_raw.columns.tolist())
                #print("📌 Columnas que espera el modelo:")
                #print(feature_names)

                if X_aprob_raw.empty:
                    logger.warning("⚠️ No hay suficientes datos para predecir aprobación.")
                    return resultados

                X_aprob_scaled = scaler_X.transform(X_aprob_raw)
                y_aprob_scaled = model.predict(X_aprob_scaled)
                y_aprob = (
                    scaler_y.inverse_transform(y_aprob_scaled.reshape(-1, 1)).flatten()
                    if scaler_y else y_aprob_scaled
                )
                
                # Guardar resultados
                df_aprob_result = df_aprob[["date"]].copy()
                df_aprob_result["prediccion_aprobacion"] = y_aprob
                resultados = (
                    df_aprob_result if resultados.empty
                    else resultados.merge(df_aprob_result, on="date", how="outer")
                )
                logger.info(f"📈 Predicciones de aprobación generadas: {len(y_aprob)}")

            except Exception as e:
                logger.error(f"❌ Error al predecir aprobación: {e}")

        # === Predicción de desaprobación ===
        if self.desaprobacion_bundle:
            try:
                df_desaprob = df.copy()

                feature_names = self.desaprobacion_bundle["feature_names"]
                model = self.desaprobacion_bundle.get("model")
                scaler_X = self.desaprobacion_bundle.get("scaler_X", None)
                scaler_y = self.desaprobacion_bundle.get("scaler_y", None)

                #print("✅ Validación bundle (desaprobación):")
                #print("Features seleccionadas:", feature_names)
                #print("Modelo entrenado:", model)
                #print("🔍 Bundle keys:", self.desaprobacion_bundle.keys())

                df_desaprob["date"] = pd.to_datetime(df_desaprob["date"], errors="coerce")

                missing_cols = [col for col in feature_names if col not in df_desaprob.columns]
                if missing_cols:
                    logger.error(f"❌ Faltan columnas en el DataFrame para desaprobación: {missing_cols}")
                    return resultados

                X_desaprob_raw = df_desaprob[feature_names]
                logger.info(f"🔍 Filas antes de filtrar NaNs (desaprobación): {X_desaprob_raw.shape[0]}")
                X_desaprob_raw = X_desaprob_raw.dropna()
                df_desaprob = df_desaprob.loc[X_desaprob_raw.index].copy()
                logger.info(f"🧹 Filas después de filtrar NaNs (desaprobación): {X_desaprob_raw.shape[0]}")

                #print("🧪 Columnas en X_desaprob_raw:")
                #print(X_desaprob_raw.columns.tolist())
                #print("📌 Columnas que espera el modelo:")
                #print(feature_names)

                if X_desaprob_raw.empty:
                    logger.warning("⚠️ No hay suficientes datos para predecir desaprobación.")
                    return resultados

                # Escalamiento
                X_desaprob_scaled = scaler_X.transform(X_desaprob_raw)

                # Predicción
                y_desaprob_scaled = model.predict(X_desaprob_scaled)
                y_desaprob = (
                    scaler_y.inverse_transform(y_desaprob_scaled.reshape(-1, 1)).flatten()
                    if scaler_y else y_desaprob_scaled
                )
                #print("🔍 scaler_y center_:", getattr(scaler_y, "center_", "N/A"))
                #print("🔍 scaler_y scale_:", getattr(scaler_y, "scale_", "N/A"))
                #print("🔍 y_desaprob_scaled (sample):", y_desaprob_scaled[:5])
                #print("🔍 y_desaprob inverso (sample):", y_desaprob[:5])
                #print("🔁 Predicciones desaprobación (últimos 5):", y_desaprob[-5:])
                #print("📅 Fechas predichas:", df_desaprob["date"].tail().tolist())
                #print("🎯 Escala predicha (antes de inversa):", y_desaprob_scaled[:5])
                #print("🎯 Escala final (después de inversa):", y_desaprob[:5])
                df_desaprob_result = df_desaprob[["date"]].copy()
                df_desaprob_result["prediccion_desaprobacion"] = y_desaprob

                resultados = (
                    df_desaprob_result if resultados.empty
                    else resultados.merge(df_desaprob_result, on="date", how="outer")
                )
                logger.info(f"📉 Predicciones de desaprobación generadas: {len(y_desaprob)}")

            except Exception as e:
                logger.error(f"❌ Error al predecir desaprobación: {e}")



        if resultados.empty:
            logger.warning("No se generaron predicciones. Guardando CSV vacío con columnas.")
            return pd.DataFrame(columns=["date", "prediccion_aprobacion", "prediccion_desaprobacion"])

        return resultados.sort_values("date")

if __name__ == "__main__":
    
    predictor = Predictor()
    resultados = predictor.predict()

    if resultados.empty:
        logger.warning("No se generaron predicciones. Guardando CSV vacío con columnas.")
        resultados = pd.DataFrame(columns=["date", "prediccion_aprobacion", "prediccion_desaprobacion"])

    print(f"💾 Guardando archivo en: {PREDICTIONS_PATH}")
    write_csv_blob(resultados, PREDICTIONS_PATH)
    print(f"📈 Predicción guardada en {PREDICTIONS_PATH}")