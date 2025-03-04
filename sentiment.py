import json
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax


modelo_es = "pysentimiento/robertuito-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(modelo_es)
model = AutoModelForSequenceClassification.from_pretrained(modelo_es)
labels = ['Negative', 'Neutral', 'Positive']

#date_input = input("Ingrese la fecha del archivo de tweets a analizar (YYYY-MM-DD): ")
file_path = f"data/tweets_boric.json"

if not os.path.exists(file_path):
    print(f"❌ No se encontró el archivo '{file_path}'. Asegúrate de extraer los tweets primero.")
    exit()

tweets = []
with open(file_path, "r", encoding="utf-8") as f:
    tweets = json.load(f)

def analyze_sentiment(text):
    tweet_words = []
    for word in text.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = 'http'
        tweet_words.append(word)
    
    tweet_proc = ' '.join(tweet_words)
    encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
    output = model(**encoded_tweet)
    scores = output.logits.detach().numpy()[0]
    scores = softmax(scores)
    
    sentiment_scores = {labels[i]: float(scores[i]) for i in range(len(scores))}
    sentiment_label = max(sentiment_scores, key=sentiment_scores.get) 
    
    return {"scores": sentiment_scores, "label": sentiment_label}

tweets_with_sentiment = []
for tweet in tweets:
    tweet["sentiment"] = analyze_sentiment(tweet["text"])
    tweets_with_sentiment.append(tweet)

output_path = f"data/sentiment_boric.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(tweets_with_sentiment, f, indent=4, ensure_ascii=False)