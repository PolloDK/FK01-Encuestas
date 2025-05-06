import pandas as pd
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
from src.config import PREPROCESSED_PATH, SENTIMENT_DATA_PATH, EMBEDDING_DATA_PATH, PROCESSED_DATA_PATH, RAW_DATA_PATH
from src.logger import get_logger
from src.azure_blob import read_csv_blob, write_csv_blob, append_csv_blob
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
        
        #text detection se puede activar si quieres filtrar idioma
        #try:
        #    if detect(text) != "es":
        #        return None
        #except:
        #    return None
        
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)     # quitar links
        text = re.sub(r"@\w+", "", text)                        # quitar menciones
        text = re.sub(r"[^\w\sáéíóúñü]", "", text)              # mantener palabras y acentos
        text = re.sub(r"\s+", " ", text).strip()

        words = text.split()
        words = [w for w in words if w not in stopwords_final]
        words = [lemmatizer.lemmatize(w) for w in words]
        if len(words) < 3 or len(words) > 50:
            return None
        return " ".join(words)
    
    def analyze_sentiment(self, text):
        if not isinstance(text, str) or text.strip() == "":
            return "Neutral", 0.0, 0.0, 0.0, 0.0
        try:
            encoded = self.tokenizer(text, return_tensors='pt')
            output = self.sentiment_model(**encoded)
            logits = output.logits.detach().numpy()

            if logits.shape[0] == 0:
                raise ValueError("Output logits vacíos")

            scores = softmax(logits[0])
            label_score = max(zip(["Negative", "Neutral", "Positive"], scores), key=lambda x: x[1])
            return label_score[0], label_score[1], scores[0], scores[1], scores[2]
        except Exception as e:
            logger.warning(f"⚠️ Error en análisis de sentimiento: {e}")
            return "Neutral", 0.0, 0.0, 0.0, 0.0
        
    def get_embedding(self, text):
        if isinstance(text, float) and pd.isna(text):
            return np.zeros(768)
        try:
            inputs = self.embedding_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=50)
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        except Exception as e:
            logger.warning(f"⚠️ Error generando embedding: {e}")
            return np.zeros(768)
    
    def run_pipeline(self) -> bool:
        try:
            df_all = read_csv_blob(self.input_path)
            df_all["createdAt"] = pd.to_datetime(df_all["createdAt"], errors="coerce")
        except FileNotFoundError:
            print(f"❌ No se encontró el archivo {self.input_path}.")
            logger.error(f"No se encontró el archivo {self.input_path}.")
            return False

        if "processed" not in df_all.columns:
            df_all["processed"] = False

        df_nuevos = df_all[df_all["processed"] == False].copy()
        if df_nuevos.empty:
            print("⏭️ No hay nuevos tweets para procesar.")
            logger.info("No hay nuevos tweets para procesar.")
            return False

        print(f"🔄 Procesando tweets ")

        # === 1. Limpieza de texto ===
        print(f"Comenzando limpieza de texto")
        df_nuevos["text"] = df_nuevos["text"].astype(str).apply(self.clean_text)
        df_nuevos = df_nuevos.dropna(subset=["text"])
        if df_nuevos.empty:
            print("⏭️ Todos los tweets fueron descartados tras limpieza.")
            logger.info("Todos los tweets fueron descartados tras limpieza.")
            return False

        write_csv_blob(df_nuevos, PREPROCESSED_PATH)
        print(f"💾 Guardado texto limpio en: {PREPROCESSED_PATH}")
        logger.info(f"💾 Texto limpio guardado en: {PREPROCESSED_PATH}")

        # === 2. Análisis de sentimiento ===
        print(f"Comenzando análisis de sentimiento de los tweets limpios")
        tqdm.pandas(desc="🔍 Analizando sentimiento")
        resultados = df_nuevos["text"].progress_apply(self.analyze_sentiment)

        df_sentiment = df_nuevos.copy()
        df_sentiment[["sentiment_label", "score_label", "score_negative", "score_neutral", "score_positive"]] = pd.DataFrame(resultados.tolist(), index=df_nuevos.index)

        write_csv_blob(df_sentiment, SENTIMENT_DATA_PATH)
        print(f"💾 Guardado análisis de sentimiento en: {SENTIMENT_DATA_PATH}")
        logger.info(f"💾 Sentimiento guardado en: {SENTIMENT_DATA_PATH}")

        # === 3. Embeddings ===
        print(f"Comenzando embedding de los tweets")
        tqdm.pandas(desc="🔗 Generando embeddings")
        embeddings_list = df_sentiment["text"].progress_apply(self.get_embedding).tolist()
        robertuito_features = pd.DataFrame(embeddings_list, columns=[f"robertuito_{i}" for i in range(768)])

        df_embedding = df_sentiment.reset_index(drop=True)
        df_embedding_final = pd.concat([df_embedding, robertuito_features], axis=1)

        write_csv_blob(df_embedding_final, EMBEDDING_DATA_PATH)
        print(f"💾 Embeddings guardados en: {EMBEDDING_DATA_PATH}")
        logger.info(f"💾 Embeddings guardados en: {EMBEDDING_DATA_PATH}")

        # === 4. Guardar todo en processed_data.csv
        print(f"Guardando nuevos tweets en {PROCESSED_DATA_PATH}")
        df_processed = df_embedding_final.copy()
        df_processed["id"] = df_processed["id"].astype(str)

        append_csv_blob(df_processed, PROCESSED_DATA_PATH)
        print(f"✅ Archivo final actualizado en: {PROCESSED_DATA_PATH}")
        logger.info(f"✅ Archivo final actualizado en: {PROCESSED_DATA_PATH}")

        # === 5. Actualizar raw_data con flag "processed"
        print(f"Actualizando flag 'processed'")
        df_all.loc[df_all["id"].isin(df_nuevos["id"]), "processed"] = True
        write_csv_blob(df_all, self.input_path)
        print(f"Tweets clasificados como 'processed' en {self.input_path}")
        logger.info(f"📝 Flag 'processed' actualizado en {self.input_path}")

        return True



if __name__ == "__main__":
    preprocessor = TweetPreprocessor(input_path=RAW_DATA_PATH)
    preprocessor.run_pipeline()
