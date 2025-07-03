import pandas as pd
import re

# Cargar tu CSV (ajusta el nombre si es necesario)
df = pd.read_csv("data/df_all_tweets_categorizado.csv")

# Función para eliminar 'el' y 'él' como palabras individuales
def remove_el_y_el_acento(text):
    return re.sub(r'\b(él|el)\b', '', text, flags=re.IGNORECASE).strip()

# Aplicar la función al campo 'clean_text'
df["clean_text"] = df["clean_text"].astype(str).apply(remove_el_y_el_acento)

# Opcional: eliminar dobles espacios después de la limpieza
df["clean_text"] = df["clean_text"].str.replace(r'\s+', ' ', regex=True).str.strip()

# Guardar como Parquet
df.to_csv("data/df_all_tweets_categorizado_limpio.csv")
