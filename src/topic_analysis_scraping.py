import os
import pandas as pd
from datetime import datetime
from apify_client import ApifyClient
from config import APIFY_API_KEY
import re
import nltk
import spacy
import emoji
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from dateutil.relativedelta import relativedelta

APIFY_API_KEY = os.getenv("APIFY_API_KEY") or "TU_API_KEY_AQUI"
client = ApifyClient(APIFY_API_KEY)

nltk.download("stopwords")
nltk.download("wordnet")

# Stopwords
stopwords_lda_extra = {
    "haber", "tener", "hacer", "decir", "poder", "dar", "ir", "poner", "ser", "estar", "querer"
}

spanish_stopwords = set(stopwords.words("spanish"))
custom_stopwords = {
    "q", "va", "ser", "tra", "sido", "vez", "hoy", "ahora", "nuevo", "así"
}

# Stopwords final para general (sin quitar los verbos)
stopwords_final_general = spanish_stopwords.union(custom_stopwords)

# Stopwords final para Topic Modeling (con los verbos)
stopwords_final_topic = stopwords_final_general.union(stopwords_lda_extra)

# Lematizer
lemmatizer = WordNetLemmatizer()
nlp = spacy.load("es_core_news_sm")


def scrape_tweets_candidate_paginate(usuario_cuenta: str, fecha_inicio: str, fecha_fin: str, max_items_per_month: int = 500, scrape_comments: bool = False):
    """
    Scrapea tweets del candidato en ventanas mensuales. Acumula en CSV.
    
    Args:
        usuario_cuenta (str): Ej. "GonzaloWinter"
        fecha_inicio (str): "YYYY-MM-DD"
        fecha_fin (str): "YYYY-MM-DD"
        max_items_per_month (int): máx tweets a scrapear por mes
        scrape_comments (bool): si se deben scrapear comentarios
    """
    fecha_inicio_dt = datetime.strptime(fecha_inicio, "%Y-%m-%d").date()
    fecha_fin_dt = datetime.strptime(fecha_fin, "%Y-%m-%d").date()

    if fecha_inicio_dt >= fecha_fin_dt:
        raise ValueError("❌ La fecha de inicio debe ser anterior a la fecha de fin.")

    current_start = fecha_inicio_dt

    while current_start < fecha_fin_dt:
        current_end = current_start + relativedelta(months=1)
        if current_end > fecha_fin_dt:
            current_end = fecha_fin_dt

        print(f"\n📅 Scrapeando tweets de {current_start} a {current_end}...")

        try:
            run_input = {
                "searchTerms": [f"from:{usuario_cuenta} since:{current_start} until:{current_end}"],
                "maxItems": max_items_per_month,
                "queryType": "Latest",
                "lang": "es"
            }

            run = client.actor("kaitoeasyapi~twitter-x-data-tweet-scraper-pay-per-result-cheapest").call(run_input=run_input)
            tweets = client.dataset(run["defaultDatasetId"]).list_items().items
            df_tweets = pd.DataFrame(tweets)

            tweets_path = f"data/tweets_{usuario_cuenta}_{fecha_inicio}_a_{fecha_fin}.csv"

            if df_tweets.empty:
                print("⚠️ No se encontraron tweets en esta ventana.")
            else:
                # Append seguro
                if os.path.exists(tweets_path):
                    df_existing = pd.read_csv(tweets_path)
                    df_combined = pd.concat([df_existing, df_tweets], ignore_index=True)
                    df_combined = df_combined.drop_duplicates(subset=["id"]).reset_index(drop=True)
                    print(f"✅ {len(df_tweets)} nuevos tweets. Total acumulado: {len(df_combined)}")
                else:
                    df_combined = df_tweets
                    print(f"🆕 {len(df_tweets)} tweets (nuevo archivo creado).")

                df_combined.to_csv(tweets_path, index=False)

                # Si scrape_comments=True, sacar comentarios SOLO de los tweets nuevos
                if scrape_comments:
                    tweet_ids = df_tweets["id"].tolist()
                    scrape_comments_candidate(usuario_cuenta, tweet_ids, df_tweets, fecha_inicio, fecha_fin)

        except Exception as e:
            print(f"❌ Error en scraping para ventana {current_start} a {current_end}: {e}")

        # Avanzar a la siguiente ventana
        current_start = current_end

def scrape_comments_candidate(usuario_cuenta: str, tweet_ids: list, df_tweets: pd.DataFrame, fecha_inicio: str, fecha_fin: str, max_items_per_tweet: int = 5):
    """
    Scrapea comentarios de los tweets dados. Guarda en CSV.

    Args:
        usuario_cuenta (str): cuenta del candidato
        tweet_ids (list): lista de IDs de tweets a los que sacar comentarios
        df_tweets (pd.DataFrame): DataFrame con los tweets originales
        fecha_inicio (str): para nombrar el CSV
        fecha_fin (str): para nombrar el CSV
        max_items_per_tweet (int): cantidad de comentarios a traer por tweet
    """
    all_comments = []

    for tweet_id in tweet_ids:
        print(f"💬 Buscando comentarios al tweet {tweet_id[:8]}...")

        reply_run = client.actor("kaitoeasyapi~twitter-x-data-tweet-scraper-pay-per-result-cheapest").call(
            run_input={
                "searchTerms": [f"conversation_id:{tweet_id}"],
                "maxItems": max_items_per_tweet,
                "queryType": "Latest",
                "lang": "es"
            }
        )

        comments = client.dataset(reply_run["defaultDatasetId"]).list_items().items

        # Enriquecer cada comentario con metadata del tweet original
        parent_row = df_tweets[df_tweets["id"] == tweet_id].iloc[0]

        for comment in comments:
            comment["parent_tweet_id"] = tweet_id
            comment["parent_text"] = parent_row.get("text")
            comment["parent_createdAt"] = parent_row.get("createdAt")

        all_comments.extend(comments)

    # Guardar si hay comentarios
    if all_comments:
        df_comments = pd.DataFrame(all_comments)
        comments_path = f"data/comments_{usuario_cuenta}_{fecha_inicio}_a_{fecha_fin}.csv"
        df_comments.to_csv(comments_path, index=False)
        print(f"✅ {len(df_comments)} comentarios guardados en {comments_path}")
    else:
        print("⚠️ No se encontraron comentarios.")

def clean_text(text, min_words=3, max_words=50, stopwords_set=stopwords_final_topic):
    if pd.isna(text):
        return None
    
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r"[^\w\sáéíóúñü]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    doc = nlp(text)
    words = [
        token.lemma_ for token in doc
        if token.lemma_ not in stopwords_set
        and not token.is_punct
        and not token.is_space
        and not token.is_digit
    ]
    
    if len(words) < min_words or len(words) > max_words:
        return None
    
    return " ".join(words)

def preprocess_tweets_csv(csv_path: str, text_column: str = "text") -> pd.DataFrame:
    """
    Preprocesa un CSV con tweets. Devuelve un DataFrame limpio y guarda el CSV preprocesado.

    Args:
        csv_path (str): Ruta al CSV de tweets.
        text_column (str): Nombre de la columna que contiene el texto.

    Returns:
        pd.DataFrame: DataFrame con columna `clean_text` y resto de metadata.
    """
    import os
    
    df = pd.read_csv(csv_path)
    
    # Validación de columna
    if text_column not in df.columns:
        raise ValueError(f"❌ La columna '{text_column}' no existe en el CSV.")

    print(f"🔍 Preprocesando {len(df)} tweets...")
    
    tqdm.pandas(desc="🧹 Limpiando texto")
    df["clean_text"] = df[text_column].astype(str).progress_apply(clean_text)
    
    # Eliminar los que quedaron vacíos tras limpieza
    df = df.dropna(subset=["clean_text"]).reset_index(drop=True)
    
    print(f"✅ Tweets restantes tras limpieza: {len(df)}")
    
    # Construir nombre de salida
    base_name = os.path.basename(csv_path).replace(".csv", "")
    output_path = os.path.join(os.path.dirname(csv_path), f"{base_name}_preprocessed.csv")
    
    df.to_csv(output_path, index=False)
    print(f"💾 CSV preprocesado guardado en: {output_path}")
    
    return df


if __name__ == "__main__":
    scrape_tweets_candidate_paginate("Parisi_oficial", "2008-01-01", "2018-01-01", max_items_per_month=500, scrape_comments=False)
    #preprocess_tweets_csv("data/tweets_Carolina_Toha_2018-01-01_a_2025-06-10.csv", text_column="text")