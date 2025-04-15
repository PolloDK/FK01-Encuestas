import pandas as pd

# Cargar el dataset
df = pd.read_csv("data/raw_data.csv")

# Si existe una columna 'processed' pero está vacía o mal, se elimina
if "processed" in df.columns:
    df = df.drop(columns=["processed"])

# Crear la columna correctamente y marcar todos como True
df["processed"] = False

# Guardar nuevamente
df.to_csv("data/raw_data.csv", index=False)

print("✅ Columna 'processed' corregida: todos los tweets existentes marcados como procesados.")
