import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from scipy.special import softmax
import torch
import numpy as np
from tqdm import tqdm
from src.config import PREPROCESSED_PATH, SENTIMENT_DATA_PATH, EMBEDDING_DATA_PATH, PROCESSED_DATA_PATH
from src.logger import get_logger
logger = get_logger(__name__, "preprocessing.log")

nltk.download("stopwords")
nltk.download("wordnet")

spanish_stopwords = set(stopwords.words("spanish"))
custom_stopwords = {
    "q", "ver", "tan", "va", "ser", "cosa", "tra", "sido", "vez",
    "hoy", "ahora", "año", "día", "nuevo", "gente",
    "así", "solo", "parte", "mientras", "puede", "cómo", "hizo"
}
stopwords_final = spanish_stopwords.union(custom_stopwords)
lemmatizer = WordNetLemmatizer()


class TweetPreprocessor:
    def __init__(self, input_path):
        self.input_path = input_path
        self.tokenizer = AutoTokenizer.from_pretrained("pysentimiento/robertuito-sentiment-analysis")
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained("pysentimiento/robertuito-sentiment-analysis")
        self.embedding_tokenizer = AutoTokenizer.from_pretrained("pysentimiento/robertuito-base-uncased")
        self.embedding_model = AutoModel.from_pretrained("pysentimiento/robertuito-base-uncased").to(torch.device("cpu"))

    def clean_text(self, text):
        if pd.isna(text):
            return None
        
        #try:
        #    if detect(text) != "es":
        #        return None
        #except:
        #    return None
        
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = re.sub(r"@\S+", "", text)
        text = re.sub(r"#\S+", "", text)
        text = re.sub(r"[^a-záéíóúñü\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        words = text.split()
        words = [word for word in words if word not in stopwords_final]
        words = [lemmatizer.lemmatize(word) for word in words]
        if len(words) < 3 or len(words) > 50:
            return None
        return " ".join(words)

    def analyze_sentiment(self, text):
        if not isinstance(text, str) or text.strip() == "":
            return "Neutral", 0.0, 0.0, 0.0, 0.0
        try:
            encoded = self.tokenizer(text, return_tensors='pt')
            output = self.sentiment_model(**encoded)
            scores = softmax(output.logits.detach().numpy()[0])
            label_score = max(zip(["Negative", "Neutral", "Positive"], scores), key=lambda x: x[1])
            return label_score[0], label_score[1], scores[0], scores[1], scores[2]
        except:
            return "Neutral", 0.0, 0.0, 0.0, 0.0

    def get_embedding(self, text):
        if isinstance(text, float) and pd.isna(text):
            return np.zeros(768)
        inputs = self.embedding_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=50)
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def run_pipeline(self):
        if not os.path.isfile(self.input_path):
            logger.error(f"❌ No se encontró el archivo {self.input_path}.")
            return

        df_all = pd.read_csv(self.input_path, low_memory=False)
        df_all["createdAt"] = pd.to_datetime(df_all["createdAt"])

        if "processed" not in df_all.columns:
            df_all["processed"] = False

        df_nuevos = df_all[df_all["processed"] == False].copy()
        total_nuevos = len(df_nuevos)

        if total_nuevos == 0:
            logger.info("⏭️ No hay nuevos tweets para procesar.")
            return

        if total_nuevos < 500:
            logger.info(f"⏭️ Solo hay {total_nuevos} tweets nuevos. Se requiere al menos 500 para procesar en batch.")
            return

        logger.info(f"📥 {total_nuevos} tweets nuevos encontrados. Iniciando procesamiento en batches...")

        CHUNK_SIZE = 5000
        processed_chunks = []

        for i in range(0, total_nuevos, CHUNK_SIZE):
            chunk = df_nuevos.iloc[i:i + CHUNK_SIZE].copy()
            logger.info(f"🔄 Procesando tweets {i + 1} a {i + len(chunk)}")

            chunk["text"] = chunk["text"].astype(str).apply(self.clean_text)
            chunk = chunk.dropna(subset=["text"])
            if chunk.empty:
                logger.info(f"⏭️ Batch {i + 1} descartado por limpieza.")
                continue

            logger.info(f"🧼 {len(chunk)} tweets luego de limpieza.")

            tqdm.pandas(desc="🔍 Analizando sentimiento")
            resultados = chunk["text"].progress_apply(self.analyze_sentiment)

            try:
                chunk["sentiment_label"], chunk["score_label"], chunk["score_negative"], chunk["score_neutral"], chunk["score_positive"] = zip(*resultados)
            except ValueError:
                logger.warning(f"⚠️ Error desempaquetando resultados de sentimiento en batch {i + 1}.")
                continue

            tqdm.pandas(desc="🔗 Generando embeddings")
            embeddings_list = chunk["text"].progress_apply(self.get_embedding).tolist()
            robertuito_features = pd.DataFrame(embeddings_list, columns=[f"robertuito_{j}" for j in range(768)])
            chunk = chunk.reset_index(drop=True)
            df_chunk_processed = pd.concat([chunk, robertuito_features], axis=1)

            processed_chunks.append(df_chunk_processed)

        if not processed_chunks:
            logger.warning("⚠️ No se generaron chunks procesados.")
            return

        df_processed = pd.concat(processed_chunks, ignore_index=True)
        logger.info(f"✅ Total tweets procesados exitosamente: {len(df_processed)}")

        # === Guardar temporalmente
        df_processed.to_csv(PREPROCESSED_PATH, index=False)
        logger.info(f"💾 Texto limpio + sentimiento + embeddings guardado en: {PREPROCESSED_PATH}")

        # === Merge con archivo existente si corresponde
        if os.path.exists(PROCESSED_DATA_PATH):
            df_existing = pd.read_csv(PROCESSED_DATA_PATH)
            df_existing["createdAt"] = pd.to_datetime(df_existing["createdAt"])
            df_processed = pd.concat([df_existing, df_processed], ignore_index=True).drop_duplicates(subset=["id"])

        df_processed.to_csv(PROCESSED_DATA_PATH, index=False)
        logger.info(f"✅ Archivo final actualizado en: {PROCESSED_DATA_PATH}")

        # === Marcar como procesados en el original y guardar
        df_all.loc[df_all["id"].isin(df_processed["id"]), "processed"] = True
        df_all.to_csv(self.input_path, index=False)
        logger.info(f"📝 Flag 'processed' actualizado en {self.input_path}")