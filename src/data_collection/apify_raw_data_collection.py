from apify_client import ApifyClient
import signal
import pandas as pd
import os
import sys

API_TOKEN = 'apify_api_FXff9JQtt1wIKIcMBaI1xiqVQMCBte0SeAlO'
file_path = "data/raw_data.csv"

stop_scraping = False

def signal_handler(sig, frame):
    """ Maneja cuando se quiere cortar el proceso de scraping
    sin interrumpir el script
    """
    global stop_scraping
    print("\n🛑 Deteniendo el scraper... Guardando tweets recolectados hasta ahora.")
    stop_scraping = True
    
signal.signal(signal.SIGINT, signal_handler)

client = ApifyClient(API_TOKEN)

print("🚀 Iniciando el scraper de tweets... Presiona Ctrl + C para detenerlo sin perder datos.")

# Definir fecha de inicio y fecha de fin.
SINCE_DATE = "2025-02-10_00:00:00_UTC" 
UNTIL_DATE = "2025-03-10_00:00:00_UTC" 

# Validación cantidad de tweets previo a la extracción
if os.path.isfile(file_path):
    df_old = pd.read_csv(file_path)
    tweets_before = len(df_old)
else:
    tweets_before = 0  # Si el archivo no existe, no hay tweets previos

print(f"📊 Tweets antes de la actualización: {tweets_before}")

# Definición de parámetros para el run del actor en Apify:
run_input = {
    "searchTerms": [
        f"Boric since:{SINCE_DATE} until:{UNTIL_DATE}",  
        f"Gabriel Boric since:{SINCE_DATE} until:{UNTIL_DATE}",
        f"Presidente Boric since:{SINCE_DATE} until:{UNTIL_DATE}"
    ],
    "maxItems": 5000,
    "queryType": "Latest",
    "lang": "es",
    "filter:verified": False,
    "filter:replies": False,
    "filter:quote": False,
    "min_retweets": 0, 
    "min_faves": 0,
}

print("🔄 Ejecutando el scraper en Apify...")
run = client.actor("kaitoeasyapi~twitter-x-data-tweet-scraper-pay-per-result-cheapest").call(run_input=run_input)
if "defaultDatasetId" not in run:
    print("❌ Error: No se encontró un dataset en la respuesta de Apify.")
    exit()
print("✅ Scraper ejecutado con éxito. Obteniendo tweets...")

## Buscarndo tweets con criterios
tweets = []
try:
    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        if stop_scraping:  # Si el usuario presionó Ctrl + C, detenemos la recolección
            break
        tweets.append(item)
except Exception as e:
    print(f"⚠️ Error durante la recolección: {e}")

if not tweets:
    print("⚠️ No se encontraron tweets con los criterios especificados.")
    sys.exit(1)

df = pd.DataFrame(tweets)
columnas_relevantes = ["id", "full_text", "created_at", "user_screen_name", "retweet_count", "favorite_count"]
df = df[columnas_relevantes] if set(columnas_relevantes).issubset(df.columns) else df

# Se agregan tweets al archivo de raw data:
if os.path.exists(file_path):
    print("📂 El archivo ya existe. Agregando nuevos tweets...")
    df.to_csv(file_path, mode='a', header=False, index=False, encoding="utf-8")
else:
    print("📂 Creando nuevo archivo y guardando tweets...")
    df.to_csv(file_path, mode='w', header=True, index=False, encoding="utf-8")

df_updated = pd.read_csv(file_path)
tweets_after = len(df_updated)

print(f"📊 Tweets después de la actualización: {tweets_after}")
print(f"📈 Se agregaron {tweets_after - tweets_before} nuevos tweets.")
print(f"📂 Tweets guardados en '{file_path}'.")
print("✅ Proceso completado con éxito.")