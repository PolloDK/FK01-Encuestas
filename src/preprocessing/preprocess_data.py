import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords

# Descargamos stopwords:
nltk.download("stopwords")
spanish_stopwords = set(stopwords.words("spanish"))

# Rutas
csv_path = "data/raw_data.csv"
processed_csv_path = "data/processed_data.csv"

if not os.path.isfile(csv_path):
    print(f"❌ Error: No se encontró el archivo {csv_path}. Asegúrate de que el scraper haya obtenido datos.")
    exit()


df = pd.read_csv(csv_path)

print(f"📊 Se encontraron {len(df)} tweets. Iniciando limpieza...")

def clean_text(text):
    """ Función que limpia menciones, links, mayúsculas, 
    stopwords, vacíos y largos.
    """
    if pd.isna(text):
        return None 
    
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\S+", "", text)
    text = re.sub(r"[^a-záéíóúñü\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    words = [word for word in words if word not in spanish_stopwords]
    text = " ".join(words)
    
    # Largo y corto de tweets
    word_count = len(words)
    if word_count < 5 or word_count > 50:
        return None
    
    return text

df["text"] = df["text"].astype(str).apply(clean_text)
df = df.dropna(subset=["text"])
print(f"📊 Después de la limpieza, quedan {len(df)} tweets.")

df.to_csv(processed_csv_path, index=False, encoding="utf-8")

print(f"✅ Preprocesamiento completado. Datos guardados en '{processed_csv_path}'")
