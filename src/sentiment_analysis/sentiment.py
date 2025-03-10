import pandas as pd
import json
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

modelo_es = "pysentimiento/robertuito-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(modelo_es)
model = AutoModelForSequenceClassification.from_pretrained(modelo_es)
labels = ['Negative', 'Neutral', 'Positive']

file_path = "data/processed_data.csv"  
df = pd.read_csv(file_path)

sentiment_counts = {"Negative": 0, "Neutral": 0, "Positive": 0}

def analyze_sentiment(text):
    if not isinstance(text, str) or text.strip() == "":
        return "Neutral" 

    tweet_words = []
    for word in text.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = 'http'
        tweet_words.append(word)

    tweet_proc = ' '.join(tweet_words)

    if len(tweet_proc.split()) < 3: 
        return "Neutral"

    try:
        encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
        output = model(**encoded_tweet)
        scores = output.logits.detach().numpy()[0]
        scores = softmax(scores)

        sentiment_scores = {labels[i]: float(scores[i]) for i in range(len(scores))}
        sentiment_label = max(sentiment_scores, key=sentiment_scores.get)

        sentiment_counts[sentiment_label] += 1
        return sentiment_label
    except Exception as e:
        print(f"⚠️ Error procesando el tweet: {str(e)}")
        return "Neutral" 


df["sentiment"] = [analyze_sentiment(text) for text in tqdm(df["text"], desc="🔍 Analizando sentimientos")]

output_path = "data/sentiment_data.csv"
df.to_csv(output_path, index=False, encoding="utf-8")

print(f"\n✅ Se guardaron los análisis de sentimiento en '{output_path}'.")
print("📊 Resumen de análisis de sentimiento:")
print(f"  - Negativos: {sentiment_counts['Negative']}")
print(f"  - Neutros: {sentiment_counts['Neutral']}")
print(f"  - Positivos: {sentiment_counts['Positive']}")