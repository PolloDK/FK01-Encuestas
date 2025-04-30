import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, date
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
import re
from tqdm import tqdm
from src.config import PROCESSED_DATA_PATH, WORDCLOUD_PATH, PREDICTIONS_PATH, FEATURES_DATASET_PATH
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
        df_pred = pd.read_csv(PREDICTIONS_PATH, parse_dates=["date"])
        logger.info("Archivo de predicciones cargado.")
    except Exception as e:
        logger.error(f"No se pudo cargar predicciones_diarias.csv: {e}")
        return

    try:
        df_features = pd.read_csv(FEATURES_DATASET_PATH, parse_dates=["date"])
        negatividad_por_dia = df_features.groupby(df_features['date'].dt.date)['score_negative'].mean().reset_index()
        negatividad_por_dia.columns = ['date', 'indice_negatividad']
        negatividad_por_dia["date"] = pd.to_datetime(negatividad_por_dia["date"])
        logger.info("Índice de negatividad calculado.")
    except Exception as e:
        logger.error(f"Error en cálculo de índice de negatividad: {e}")
        return

    try:
        print("Comenzando cálculo de % tweets negativos")
        df_raw = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["createdAt"])
        #print("Base cargada")
        df_raw = df_raw.dropna(subset=["score_positive", "score_negative", "score_neutral"])
        #print("se eliminó NaNs")
        df_raw["date_only"] = df_raw["createdAt"].dt.date
        #print(f"🔍 Filas restantes antes de clasificar: {len(df_raw)}")
        
        # Vectorizado sin .apply
        scores = df_raw[["score_positive", "score_negative", "score_neutral"]]
        df_raw["sentimiento_clasificado"] = scores.idxmax(axis=1).str.replace("score_", "")

        empates = scores.eq(scores.max(axis=1), axis=0).sum(axis=1) > 1
        df_raw.loc[empates, "sentimiento_clasificado"] = "neutro"
        print("✅ Clasificación completada")

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

        conteos["total_tweets"] = conteos[["tweets_negativos", "tweets_positivos", "tweets_neutros"]].sum(axis=1)
        conteos["porcentaje_tweets_negativos"] = conteos["tweets_negativos"] / conteos["total_tweets"]
        conteos["date"] = pd.to_datetime(conteos["date"])
        logger.info("Porcentaje de tweets negativos calculado.")
        #print("🧪 Conteos (últimas filas):")
        #print(conteos.tail())
        #print("📊 Columnas finales:", conteos.columns.tolist())
    except Exception as e:
        logger.error(f"Error en cálculo de % tweets negativos: {e}")
        return

    try:
        #print("📅 Fechas predicción:", df_pred["date"].tail())
        #print("📅 Fechas métricas:", df_metricas["date"].tail())
        #print("📊 Columnas métricas:", df_metricas.columns.tolist())
        #df_pred["date"] = pd.to_datetime(df_pred["date"]).dt.normalize()
        #conteos["date"] = pd.to_datetime(conteos["date"]).dt.normalize()
        #negatividad_por_dia["date"] = pd.to_datetime(negatividad_por_dia["date"]).dt.normalize()
        df_metricas = pd.merge(negatividad_por_dia, conteos, on="date", how="outer")
        df_actualizado = pd.merge(df_pred, df_metricas, on="date", how="left")
        df_actualizado.to_csv(PREDICTIONS_PATH, index=False, date_format="%Y-%m-%d")
        logger.info("Archivo de predicciones_diarias.csv actualizado con métricas.")
    except Exception as e:
        logger.error(f"Error al actualizar archivo final: {e}")
        
def generar_wordcloud_diario():
    today = datetime.today().date()
    generar_wordcloud_para_fecha(today)


def generar_wordcloud_para_fecha(target_date):
    os.makedirs(WORDCLOUD_PATH, exist_ok=True)

    df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["createdAt"])
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

    output_path = os.path.join(WORDCLOUD_PATH, f"wordcloud_{target_date}.png")
    wordcloud.to_file(output_path)
    logger.info(f"Wordcloud generado: {output_path}")


def generar_wordclouds_historicos():
    df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["createdAt"], low_memory=False)
    df = df.dropna(subset=["createdAt", "text"])

    # Convertir a datetime naive (sin timezone)
    df["createdAt"] = df["createdAt"].dt.tz_localize(None)

    fecha_inicio = pd.to_datetime("2024-10-01")
    fecha_fin = pd.to_datetime(date.today())

    df = df[(df["createdAt"] >= fecha_inicio) & (df["createdAt"] <= fecha_fin)]

    dias_unicos = df["createdAt"].dt.date.unique()

    for i, fecha in enumerate(tqdm(sorted(dias_unicos), desc="Wordclouds")):
        generar_wordcloud_para_fecha(fecha)

def generar_wordclouds_pendientes():
    carpeta_wordclouds = Path("data/wordclouds")  # ajusta si tu carpeta es distinta
    carpeta_wordclouds.mkdir(parents=True, exist_ok=True)

    fecha_inicio = datetime.strptime("2024-10-01", "%Y-%m-%d").date()
    fecha_hoy = datetime.today().date()
    fechas = [fecha_inicio + timedelta(days=i) for i in range((fecha_hoy - fecha_inicio).days + 1)]

    for fecha in tqdm(fechas, desc="🌀 Generando wordclouds pendientes"):
        archivo_wordcloud = carpeta_wordclouds / f"wordcloud_{fecha}.png"

        if not archivo_wordcloud.exists():
            try:
                generar_wordcloud_para_fecha(fecha)
            except Exception as e:
                print(f"⚠️ Error generando wordcloud para {fecha}: {e}")
        else:
            print(f"✅ Wordcloud ya existe para {fecha}")

def main():
    #calcular_metricas()
    #generar_wordcloud_diario()
    #generar_wordclouds_historicos()  # Ejecuta solo si quieres correr todos
    generar_wordclouds_pendientes()

if __name__ == "__main__":
    main()
