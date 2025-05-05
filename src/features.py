import pandas as pd
import numpy as np
from src.azure_blob import read_csv_blob, write_csv_blob
from sklearn.preprocessing import RobustScaler
from src.logger import get_logger
from src.config import PROCESSED_DATA_PATH, ENCUESTAS_PATH, FEATURES_DATASET_PATH

logger = get_logger(__name__, "features.log")

class FeatureEngineer:
    def __init__(self, input_path, encuestas_path, output_path):
        self.input_path = input_path
        self.encuestas_path = encuestas_path
        self.output_path = output_path

    def weighted_avg(self, values, weights):
        return (values * weights).sum() / weights.sum() if weights.sum() != 0 else values.mean()

    def run(self):
        logger.info("Inicio de feature engineering")

        try:
            df = read_csv_blob(self.input_path)
            df_encuestas = read_csv_blob(self.encuestas_path)
            logger.info(f"Archivos cargados: {self.input_path}, {self.encuestas_path}")
        except Exception as e:
            logger.error(f"Error al cargar archivos: {e}")
            return

        df["createdAt"] = pd.to_datetime(df["createdAt"])
        df["date"] = df["createdAt"].dt.floor("D").dt.tz_localize(None)
        df_encuestas["date"] = pd.to_datetime(df_encuestas["date"]).dt.tz_localize(None)

        try:
            df_existing = read_csv_blob(self.output_path)
            df_existing["date"] = pd.to_datetime(df_existing["date"])
            fechas_nuevas = df["date"].unique()
            df_existing = df_existing[~df_existing["date"].isin(fechas_nuevas)]
        except FileNotFoundError:
            df_existing = None

        if df.empty:
            logger.warning("No hay nuevos días para procesar.")
            return

        logger.info(f"Procesando {df['date'].nunique()} días nuevos.")

        try:
            # === Agregación diaria ===
            engagement_vars = ["retweetCount", "replyCount", "likeCount", "quoteCount"]
            scaler = RobustScaler()
            df[engagement_vars] = scaler.fit_transform(df[engagement_vars])

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

            # === Agregación ponderada por engagement ===
            weighted_features = []
            for var in engagement_vars:
                df_w = df.groupby("date", group_keys=False).apply(
                    lambda x: pd.Series({
                        f"weighted_positive_{var}": self.weighted_avg(x["score_positive"], x[var]),
                        f"weighted_negative_{var}": self.weighted_avg(x["score_negative"], x[var]),
                        f"weighted_neutral_{var}": self.weighted_avg(x["score_neutral"], x[var])
                    })
                ).reset_index()
                weighted_features.append(df_w)

            df_weighted_all = weighted_features[0]
            for df_w in weighted_features[1:]:
                df_weighted_all = df_weighted_all.merge(df_w, on="date", how="left")

            df_daily = df_daily.merge(df_weighted_all, on="date", how="left")

            # === Expansión de encuestas ===
            start_date = df_encuestas["date"].min()
            end_date = df_encuestas["date"].max() + pd.Timedelta(days=6)
            df_cadem_expandido = pd.DataFrame({"date": pd.date_range(start=start_date, end=end_date, freq="D")})

            df_cadem_expandido["semana_objetivo"] = df_cadem_expandido["date"].apply(
                lambda d: d + pd.to_timedelta(6 - d.weekday(), unit="D")
            )
            df_encuestas_ren = df_encuestas.rename(columns={"date": "semana_objetivo"})
            df_cadem_expandido = df_cadem_expandido.merge(df_encuestas_ren, on="semana_objetivo", how="left")

            df_daily = df_daily.merge(
                df_cadem_expandido[["date", "aprobacion_boric", "desaprobacion_boric"]],
                on="date", how="left"
            )

            # === Variables derivadas ===
            df_daily = df_daily.sort_values("date").reset_index(drop=True)

            df_daily["approval_rolling_7d"] = df_daily["aprobacion_boric"].shift(1).rolling(window=7, min_periods=7).mean().fillna(method="ffill") 
            df_daily["approval_lag_7d"] = df_daily["aprobacion_boric"].shift(7)
            df_daily["approval_lag_14d"] = df_daily["aprobacion_boric"].shift(14)

            df_daily["disapproval_rolling_7d"] = df_daily["desaprobacion_boric"].shift(1).rolling(window=7, min_periods=7).mean().fillna(method="ffill") 
            df_daily["disapproval_lag_7d"] = df_daily["desaprobacion_boric"].shift(7)
            df_daily["disapproval_lag_14d"] = df_daily["desaprobacion_boric"].shift(14)

            for lag in range(1, 8):
                df_daily[f"score_positive_lag_{lag}"] = df_daily["score_positive"].shift(lag)
                df_daily[f"score_negative_lag_{lag}"] = df_daily["score_negative"].shift(lag)
                df_daily[f"score_neutral_lag_{lag}"] = df_daily["score_neutral"].shift(lag)

            df_daily["score_negative_rolling7"] = df_daily["score_negative"].rolling(window=7, min_periods=7).mean()
            df_daily["score_negative_rolling3"] = df_daily["score_negative"].rolling(window=3, min_periods=3).mean()

            df_daily["sentiment_net"] = df_daily["score_positive"] - df_daily["score_negative"]
            df_daily["sentiment_net_rolling3"] = df_daily["sentiment_net"].rolling(window=3, min_periods=3).mean()
            df_daily["sentiment_net_rolling7"] = df_daily["sentiment_net"].rolling(window=7, min_periods=7).mean()
            df_daily["sentiment_net_rolling14"] = df_daily["sentiment_net"].rolling(window=14, min_periods=14).mean()
            df_daily["sentiment_net_change"] = df_daily["sentiment_net"] - df_daily["sentiment_net"].shift(1)

            df_daily["date"] = pd.to_datetime(df_daily["date"]).dt.date

            if df_existing is not None:
                df_daily = pd.concat([df_existing, df_daily], ignore_index=True).drop_duplicates(subset=["date"])

            write_csv_blob(df_daily, self.output_path)
            logger.info(f"✅ Features guardados en: {self.output_path}")
            logger.info(f"Días nuevos procesados: {df['date'].nunique()}")

        except Exception as e:
            logger.error(f"Error durante feature engineering: {e}")

if __name__ == "__main__":
    input_path = PROCESSED_DATA_PATH
    encuestas_path = ENCUESTAS_PATH
    output_path = FEATURES_DATASET_PATH

    engineer = FeatureEngineer(input_path, encuestas_path, output_path)
    engineer.run()
