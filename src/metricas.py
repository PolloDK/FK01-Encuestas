import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, date
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import os
import re
from tqdm import tqdm
from src.config import PROCESSED_DATA_PATH, WORDCLOUD_PATH, PREDICTIONS_PATH, FEATURES_DATASET_PATH
from src.azure_blob import read_csv_blob, write_csv_blob, blob_exists, upload_image_blob
from src.logger import get_logger
logger = get_logger(__name__, "metricas.log")

def clasificar_sentimiento(row):
    scores = {
        "positivo": row["score_positive"],
        "negativo": row["score_negative"],
        "neutro": row["score_neutral"]
    }
    max_score = max(scores.values())

    # Manejo de empates: si hay más de un score con el mismo valor máximo
    candidatos = [k for k, v in scores.items() if v == max_score]
    if len(candidatos) > 1:
        return "neutro"  # asignamos neutro en caso de empate
    return candidatos[0]

def calcular_metricas():
    logger.info("Iniciando cálculo de métricas")

    try:
        df_pred = read_csv_blob(PREDICTIONS_PATH)
        logger.info("Archivo de predicciones cargado.")
    except Exception as e:
        logger.error(f"No se pudo cargar predicciones_diarias.csv: {e}")
        return

    try:
        df_features = read_csv_blob(FEATURES_DATASET_PATH)
        df_features["date"] = pd.to_datetime(df_features["date"], errors="coerce")
        negatividad_por_dia = df_features.groupby(df_features['date'].dt.date)['score_negative'].mean().reset_index()
        negatividad_por_dia.columns = ['date', 'indice_negatividad']
        negatividad_por_dia["date"] = pd.to_datetime(negatividad_por_dia["date"])
        logger.info("Índice de negatividad calculado.")
    except Exception as e:
        logger.error(f"Error en cálculo de índice de negatividad: {e}")
        return

    try:
        print("Comenzando cálculo de % tweets negativos")
        df_raw = read_csv_blob(PROCESSED_DATA_PATH)

        if "createdAt" not in df_raw.columns:
            raise ValueError("❌ La columna 'createdAt' no está presente en processed_data.csv")

        # Forzar conversión de fechas
        df_raw["createdAt"] = pd.to_datetime(df_raw["createdAt"], errors="coerce")

        # Verifica si hay muchas fechas mal parseadas
        invalid_dates = df_raw["createdAt"].isna().sum()
        df_raw = df_raw.dropna(subset=["score_positive", "score_negative", "score_neutral"])
        #print("se eliminó NaNs")
        df_raw["date_only"] = df_raw["createdAt"].dt.date
        #print(f"🔍 Filas restantes antes de clasificar: {len(df_raw)}")
        
        # Vectorizado sin .apply
        scores = df_raw[["score_positive", "score_negative", "score_neutral"]]
        df_raw["sentimiento_clasificado"] = scores.idxmax(axis=1).str.replace("score_", "")

        empates = scores.eq(scores.max(axis=1), axis=0).sum(axis=1) > 1
        df_raw.loc[empates, "sentimiento_clasificado"] = "neutro"

        conteos = df_raw.groupby("date_only")["sentimiento_clasificado"].value_counts().unstack(fill_value=0).reset_index()
        conteos.columns.name = None

        # Asegurar que todas las clases estén presentes
        for col in ["positive", "negative", "neutral"]:
            if col not in conteos.columns:
                conteos[col] = 0

        conteos = conteos.rename(columns={
            "positive": "tweets_positivos",
            "negative": "tweets_negativos",
            "neutral": "tweets_neutros"
        })
        conteos = conteos.rename(columns={"date_only": "date"})
        conteos = conteos.drop(columns=["neutro"], errors="ignore")

        # Validar que existan todas
        cols_esperadas = ["tweets_negativos", "tweets_positivos", "tweets_neutros"]
        faltantes = [col for col in cols_esperadas if col not in conteos.columns]

        if faltantes:
            print(f"❌ Faltan columnas esperadas para cálculo: {faltantes}")
            raise ValueError(f"No se puede calcular métricas porque faltan columnas: {faltantes}")

        # Si todo bien, continúa
        conteos["total_tweets"] = conteos[cols_esperadas].sum(axis=1)
        conteos["porcentaje_tweets_negativos"] = conteos["tweets_negativos"] / conteos["total_tweets"]
        conteos["date"] = pd.to_datetime(conteos["date"])
        logger.info("Porcentaje de tweets negativos calculado.")
        #print("🧪 Conteos (últimas filas):")
        #print(conteos.tail())
        #print("📊 Columnas finales:", conteos.columns.tolist())
        missing_cols = [col for col in ["tweets_negativos", "tweets_positivos", "tweets_neutros"] if col not in conteos.columns]
        if missing_cols:
            print(f"⚠️ Faltan columnas para cálculo de totales: {missing_cols}")
    except Exception as e:
        logger.error(f"Error en cálculo de % tweets negativos: {e}")
        return

    try:
        df_metricas = pd.merge(negatividad_por_dia, conteos, on="date", how="outer")
        df_actualizado = pd.merge(df_pred, df_metricas, on="date", how="left")
        write_csv_blob(df_actualizado, PREDICTIONS_PATH)
        logger.info("Archivo de predicciones_diarias.csv actualizado con métricas.")
    except Exception as e:
        logger.error(f"Error al actualizar archivo final: {e}")
        print(f"❌ ERROR en merge final: {e}")
        
def generar_wordcloud_diario():
    today = datetime.today().date()
    generar_wordcloud_para_fecha(today)


def generar_wordcloud_para_fecha(target_date):
    os.makedirs(WORDCLOUD_PATH, exist_ok=True)

    df = read_csv_blob(PROCESSED_DATA_PATH)
    if "createdAt" not in df.columns or "text" not in df.columns:
        logger.warning("El archivo no tiene las columnas requeridas: createdAt y text")
        return

    df["createdAt"] = pd.to_datetime(df["createdAt"], errors='coerce')
    df = df.dropna(subset=["createdAt", "text"])
    df_target = df[df["createdAt"].dt.date == target_date]

    if df_target.empty:
        logger.warning(f"No hay tweets disponibles para {target_date}")
        return

    text = " ".join(df_target["text"].dropna().astype(str))
    # 🔍 Lista de palabras o frases a eliminar (convertidas a minúsculas)
    frases_excluir = ["gabriel boric", "presidente boric"]
    palabras_excluir = ["boric", "gabriel", "presidente", "chile", "gobierno"]

    # Eliminar frases primero
    for frase in frases_excluir:
        text = text.replace(frase, "")

    # Eliminar palabras individuales usando regex con límites de palabra
    for palabra in palabras_excluir:
        text = re.sub(rf"\b{re.escape(palabra)}\b", "", text)

    wordcloud = WordCloud(
        width=800, height=400,
        background_color='white',
        colormap='plasma',
        max_words=200
    ).generate(text)

    buffer = BytesIO()
    wordcloud.to_image().save(buffer, format="PNG")
    upload_image_blob(buffer.getvalue(), f"wordclouds/wordcloud_{target_date}.png")
    logger.info(f"📤 Wordcloud subido a Azure: wordclouds/wordcloud_{target_date}.png")


def generar_wordclouds_historicos():
    df = read_csv_blob(PROCESSED_DATA_PATH)
    df = df.dropna(subset=["createdAt", "text"])

    # Convertir a datetime naive (sin timezone)
    df["createdAt"] = pd.to_datetime(df["createdAt"], errors="coerce")
    df["createdAt"] = df["createdAt"].dt.tz_localize(None)

    fecha_inicio = pd.to_datetime("2024-10-01")
    fecha_fin = pd.to_datetime(date.today())

    df = df[(df["createdAt"] >= fecha_inicio) & (df["createdAt"] <= fecha_fin)]

    dias_unicos = df["createdAt"].dt.date.unique()

    for i, fecha in enumerate(tqdm(sorted(dias_unicos), desc="Wordclouds")):
        generar_wordcloud_para_fecha(fecha)

def generar_wordclouds_pendientes():
    fecha_inicio = datetime.strptime("2024-10-01", "%Y-%m-%d").date()
    fecha_hoy = datetime.today().date()
    fechas_totales = [fecha_inicio + timedelta(days=i) for i in range((fecha_hoy - fecha_inicio).days + 1)]

    # ✅ Filtrar solo las fechas que realmente faltan
    fechas_pendientes = [fecha for fecha in fechas_totales if not blob_exists(f"wordclouds/wordcloud_{fecha}.png")]

    if not fechas_pendientes:
        print("✅ Todas las wordclouds ya están generadas.")
        return

    print(f"🌀 Generando {len(fechas_pendientes)} wordclouds pendientes...")

    for fecha in tqdm(fechas_pendientes, desc="📊 Progreso"):
        try:
            generar_wordcloud_para_fecha(fecha)
        except Exception as e:
            print(f"⚠️ Error generando wordcloud para {fecha}: {e}")

def main():
    calcular_metricas()
    #generar_wordcloud_diario()
    #generar_wordclouds_historicos()  # Ejecuta solo si quieres correr todos
    #generar_wordclouds_pendientes()

if __name__ == "__main__":
    main()
