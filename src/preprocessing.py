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
from src.config import RAW_DATA_PATH, PREPROCESSED_PATH, SENTIMENT_DATA_PATH, EMBEDDING_DATA_PATH, PROCESSED_DATA_PATH
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
    
    def run_pipeline(self):
        if not os.path.isfile(self.input_path):
            print(f"❌ No se encontró el archivo {self.input_path}.")
            logger.error(f"No se encontró el archivo {self.input_path}.")
            return

        df_all = pd.read_csv(self.input_path, low_memory=False)
        df_all["createdAt"] = pd.to_datetime(df_all["createdAt"])

        if "processed" not in df_all.columns:
            df_all["processed"] = False

        df_nuevos = df_all[df_all["processed"] == False].copy()
        if df_nuevos.empty:
            print("⏭️ No hay nuevos tweets para procesar.")
            logger.info("No hay nuevos tweets para procesar.")
            return

        print(f"🔄 Procesando tweets ")

        # === 1. Limpieza de texto ===
        print(f"Comenzando limpieza de texto")
        df_nuevos["text"] = df_nuevos["text"].astype(str).apply(self.clean_text)
        df_nuevos = df_nuevos.dropna(subset=["text"])
        if df_nuevos.empty:
            print("⏭️ Todos los tweets fueron descartados tras limpieza.")
            logger.info("Todos los tweets fueron descartados tras limpieza.")
            return

        df_nuevos.to_csv(PREPROCESSED_PATH, index=False)
        print(f"💾 Guardado texto limpio en: {PREPROCESSED_PATH}")
        logger.info(f"💾 Texto limpio guardado en: {PREPROCESSED_PATH}")

        # === 2. Análisis de sentimiento ===
        print(f"Comenzando análisis de sentimiento de los tweets limpios")
        tqdm.pandas(desc="🔍 Analizando sentimiento")
        resultados = df_nuevos["text"].progress_apply(self.analyze_sentiment)

        df_sentiment = df_nuevos.copy()
        df_sentiment[["sentiment_label", "score_label", "score_negative", "score_neutral", "score_positive"]] = pd.DataFrame(resultados.tolist(), index=df_nuevos.index)

        df_sentiment.to_csv(SENTIMENT_DATA_PATH, index=False)
        print(f"💾 Guardado análisis de sentimiento en: {SENTIMENT_DATA_PATH}")
        logger.info(f"💾 Sentimiento guardado en: {SENTIMENT_DATA_PATH}")

        # === 3. Embeddings ===
        print(f"Comenzando embedding de los tweets")
        tqdm.pandas(desc="🔗 Generando embeddings")
        embeddings_list = df_sentiment["text"].progress_apply(self.get_embedding).tolist()
        robertuito_features = pd.DataFrame(embeddings_list, columns=[f"robertuito_{i}" for i in range(768)])

        df_embedding = df_sentiment.reset_index(drop=True)
        df_embedding_final = pd.concat([df_embedding, robertuito_features], axis=1)

        df_embedding_final.to_csv(EMBEDDING_DATA_PATH, index=False)
        print(f"💾 Embeddings guardados en: {EMBEDDING_DATA_PATH}")
        logger.info(f"💾 Embeddings guardados en: {EMBEDDING_DATA_PATH}")

        # === 4. Guardar todo en processed_data.csv
        print(f"Guardando nuevos tweets en {PROCESSED_DATA_PATH}")
        df_processed = df_embedding_final.copy()
        df_processed["id"] = df_processed["id"].astype(str)

        if os.path.exists(PROCESSED_DATA_PATH):
            print(f"✅ Existe archivo en {PROCESSED_DATA_PATH}")
            logger.info("✅ Existe archivo processed_data.csv")

            try:
                # Leer sólo encabezado para obtener columnas
                with open(PROCESSED_DATA_PATH, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                columnas_existentes = first_line.split(",")
                
                # Asegurar que columnas estén en el mismo orden
                df_processed = df_processed[columnas_existentes]

                # Guardar en modo append sin encabezado
                df_processed.to_csv(PROCESSED_DATA_PATH, mode="a", header=False, index=False)
                print(f"✅ Nuevos datos agregados a {PROCESSED_DATA_PATH}")
                logger.info(f"✅ Nuevos datos agregados a {PROCESSED_DATA_PATH}")
            except Exception as e:
                print(f"⚠️ Error al guardar en {PROCESSED_DATA_PATH}: {e}")
                logger.error(f"⚠️ Error al guardar en {PROCESSED_DATA_PATH}: {e}")
        else:
            df_processed.to_csv(PROCESSED_DATA_PATH, index=False)
            print(f"📄 Archivo nuevo creado: {PROCESSED_DATA_PATH}")
            logger.info(f"📄 Archivo nuevo creado: {PROCESSED_DATA_PATH}")

        print(f"✅ Archivo final actualizado en: {PROCESSED_DATA_PATH}")
        logger.info(f"✅ Archivo final actualizado en: {PROCESSED_DATA_PATH}")

        # === 5. Actualizar raw_data con flag "processed"
        print(f"Actualizando flag 'processed'")
        df_all.loc[df_all["id"].isin(df_nuevos["id"]), "processed"] = True
        df_all.to_csv(self.input_path, index=False)
        print(f"Tweets clasificados como 'processed' en {self.input_path}")
        logger.info(f"📝 Flag 'processed' actualizado en {self.input_path}")


#if __name__ == "__main__":
#    preprocessor = TweetPreprocessor(input_path=RAW_DATA_PATH)
#    preprocessor.run_pipeline()
