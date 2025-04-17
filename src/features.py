import pandas as pd
import numpy as np
import os
from src.logger import get_logger
logger = get_logger(__name__, "features.log")

class FeatureEngineer:
    def __init__(self, input_path, encuestas_path, output_path):
        self.input_path = input_path
        self.encuestas_path = encuestas_path
        self.output_path = output_path

    def weighted_avg(self, values, weights):
        return (values * weights).sum() / weights.sum() if weights.sum() != 0 else values.mean()

    def run(self):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        logger.info("Inicio de feature engineering")

        try:
            df = pd.read_csv(self.input_path)
            df_encuestas = pd.read_csv(self.encuestas_path)
            logger.info(f"Archivos cargados: {self.input_path}, {self.encuestas_path}")
        except Exception as e:
            logger.error(f"Error al cargar archivos: {e}")
            return

        df["createdAt"] = pd.to_datetime(df["createdAt"])
        df["date"] = pd.to_datetime(df["createdAt"].dt.date)
        df_encuestas["date"] = pd.to_datetime(df_encuestas["date"])

        if os.path.exists(self.output_path):
            df_existing = pd.read_csv(self.output_path)
            df_existing["date"] = pd.to_datetime(df_existing["date"])
            processed_dates = df_existing["date"].unique()
            df = df[~df["date"].isin(processed_dates)].copy()
            logger.info(f"{len(processed_dates)} días ya procesados. Se omiten.")
        else:
            df_existing = None

        if df.empty:
            logger.warning("No hay nuevos días para procesar.")
            return

        logger.info(f"Procesando {df['date'].nunique()} días nuevos.")

        # === Cálculos ===
        try:
            df_daily = df.groupby("date", as_index=False).agg({
                "score_positive": "mean",
                "score_negative": "mean",
                "score_neutral": "mean",
                "retweetCount": "mean",
                "replyCount": "mean",
                "likeCount": "mean",
                "quoteCount": "mean",
                **{f"robertuito_{i}": "mean" for i in range(768)}
            })

            engagement_vars = ["retweetCount", "replyCount", "likeCount", "quoteCount"]
            weighted_features = []
            for var in engagement_vars:
                df_w = df.groupby("date").apply(lambda x: pd.Series({
                    f"weighted_positive_{var}": self.weighted_avg(x["score_positive"], x[var]),
                    f"weighted_negative_{var}": self.weighted_avg(x["score_negative"], x[var]),
                    f"weighted_neutral_{var}": self.weighted_avg(x["score_neutral"], x[var])
                })).reset_index()
                weighted_features.append(df_w)

            df_weighted_all = weighted_features[0]
            for df_w in weighted_features[1:]:
                df_weighted_all = df_weighted_all.merge(df_w, on="date", how="left")

            df_daily = df_daily.merge(df_weighted_all, on="date", how="left")
            df_daily = df_daily.set_index("date").resample("D").ffill().reset_index()

            df_encuestas["week_start"] = df_encuestas["date"] - pd.to_timedelta(df_encuestas["date"].dt.weekday, unit="D")
            df_daily["week_start"] = df_daily["date"] - pd.to_timedelta(df_daily["date"].dt.weekday, unit="D")

            df_final = df_daily.merge(df_encuestas[["week_start", "aprobacion_boric", "desaprobacion_boric"]], on="week_start", how="left")
            df_final = df_final.sort_values("date")

            if df_final.empty:
                logger.warning("No se generaron features. DataFrame final vacío.")
                return

            if ("aprobacion_boric" not in df_final.columns or df_final["aprobacion_boric"].isna().all()) and \
               ("desaprobacion_boric" not in df_final.columns or df_final["desaprobacion_boric"].isna().all()):
                logger.warning("No hay datos de aprobación o desaprobación para los días procesados.")
                return

            df_final["approval_rolling_7d"] = df_final["aprobacion_boric"].rolling(window=7, min_periods=1).mean()
            df_final["approval_lag_7d"] = df_final["aprobacion_boric"].shift(7)
            df_final["approval_diff"] = df_final["aprobacion_boric"].diff()
            df_final["approval_pct_change"] = df_final["aprobacion_boric"].pct_change()

            df_final["disapproval_rolling_7d"] = df_final["desaprobacion_boric"].rolling(window=7, min_periods=1).mean()
            df_final["disapproval_lag_7d"] = df_final["desaprobacion_boric"].shift(7)
            df_final["disapproval_diff"] = df_final["desaprobacion_boric"].diff()
            df_final["disapproval_pct_change"] = df_final["desaprobacion_boric"].pct_change()

            for lag in range(1, 8):
                df_final[f"score_positive_lag_{lag}"] = df_final["score_positive"].shift(lag)
                df_final[f"score_negative_lag_{lag}"] = df_final["score_negative"].shift(lag)
                df_final[f"score_neutral_lag_{lag}"] = df_final["score_neutral"].shift(lag)

            df_final["score_negative_rolling7"] = df_final["score_negative"].rolling(window=7, min_periods=3).mean()
            df_final["score_negative_rolling3"] = df_final["score_negative"].rolling(window=3, min_periods=3).mean()
            df_final["sentiment_net"] = df_final["score_positive"] - df_final["score_negative"]
            df_final["sentiment_net_rolling7"] = df_final["sentiment_net"].rolling(window=7, min_periods=3).mean()

            if df_existing is not None:
                df_final = pd.concat([df_existing, df_final], ignore_index=True).drop_duplicates(subset=["date"])

            print("📊 df_final shape:", df_final.shape)
            print("📅 Fechas únicas en df_final:", df_final["date"].nunique())
            print("📁 Guardando en:", self.output_path)
            df_final.to_csv(self.output_path, index=False, encoding="utf-8")
            logger.info(f"Features guardados en: {self.output_path}")
            logger.info(f"Días nuevos procesados: {df['date'].nunique()}")

        except Exception as e:
            logger.error(f"Error durante feature engineering: {e}")

if __name__ == "__main__":
    input_path = "data/processed_data.csv"
    encuestas_path = "data/encuestas.csv"
    output_path = "data/features_dataset.csv"

    engineer = FeatureEngineer(input_path, encuestas_path, output_path)
    engineer.run()