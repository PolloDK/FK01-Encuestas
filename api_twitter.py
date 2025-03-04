import tweepy
import json
import time

# Reemplaza con tus credenciales de la API de Twitter/X
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAANzfzgEAAAAASDWWeYTQmgXAVNCB7tF5eN%2B1gl8%3D91aJgMISU98h0QLPIZA0f2m5QMaQfqaIqJfLnEMZozuPlgqFZZ"

# Configurar el cliente de Twitter API v2
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# Palabras clave para buscar menciones al presidente de Chile
keywords = ["Boric", "Merluzo", "Gabriel Boric", "Presidente Boric"]
query = " OR ".join(keywords) + " -is:retweet lang:es"

# Función para obtener tweets
def get_tweets(query, max_results=400):  # Reducimos la cantidad de tweets
    try:
        response = client.search_recent_tweets(query=query, max_results=max_results, tweet_fields=["created_at", "author_id", "text"])
        
        if response.data:
            tweets = [{"id": tweet.id, "author_id": tweet.author_id, "created_at": str(tweet.created_at), "text": tweet.text} for tweet in response.data]
            return tweets
        else:
            return []
    except tweepy.TooManyRequests:
        print("⚠️ Has excedido el límite de la API. Esperando 60 segundos...")
        time.sleep(60)  # Esperar un minuto antes de intentar de nuevo
        return get_tweets(query, max_results)
    except Exception as e:
        print(f"❌ Error al obtener tweets: {e}")
        return []

# Obtener tweets
print("Buscando tweets...")
tweets = get_tweets(query)

# Guardar en un archivo JSON
if tweets:
    with open("data/tweets_boric.json", "w", encoding="utf-8") as f:
        json.dump(tweets, f, indent=4, ensure_ascii=False)
    print(f"✅ Se guardaron {len(tweets)} tweets en 'tweets_boric.json'")
else:
    print("⚠️ No se encontraron tweets.")
