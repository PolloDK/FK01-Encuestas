import tweepy
import json
import time
import csv

BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAANzfzgEAAAAASDWWeYTQmgXAVNCB7tF5eN%2B1gl8%3D91aJgMISU98h0QLPIZA0f2m5QMaQfqaIqJfLnEMZozuPlgqFZZ"
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# Palabras clave
keywords = ["Boric", "Merluzo", "Gabriel Boric", "Presidente Boric"]
query = " OR ".join(keywords) + " -is:retweet lang:es"

def get_tweets(date, max_results=100):
    start_time = f"{date}T00:00:00Z"
    end_time = f"{date}T23:59:59Z"
    
    try:
        response = client.search_recent_tweets(
            query=query,
            max_results=max_results,
            tweet_fields=["created_at", "author_id", "text"],
            start_time=start_time,
            end_time=end_time
        )
        
        if response.data:
            tweets = [{
                "id": tweet.id,
                "author_id": tweet.author_id,
                "created_at": str(tweet.created_at),
                "text": tweet.text
            } for tweet in response.data]
            return tweets
        else:
            return []
    except tweepy.TooManyRequests:
        print("⚠️ Has excedido el límite de la API. Esperando 60 segundos...")
        time.sleep(60)
        return get_tweets(date, max_results)
    except Exception as e:
        print(f"❌ Error al obtener tweets: {e}")
        return []

print("Buscando tweets...")
tweets = get_tweets(query)

data_path_csv = f"data/tweets_boric.csv"
if tweets:
    with open(data_path_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "author_id", "created_at", "text"])
        for tweet in tweets:
            writer.writerow([tweet["id"], tweet["author_id"], tweet["created_at"], tweet["text"]])
    
    print(f"✅ Se guardaron {len(tweets)} tweets en '{data_path_csv}'.")

