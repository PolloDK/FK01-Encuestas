# FK01-Encuestas

## 📊 Análisis de Tweets y Sentimiento sobre Boric

Este proyecto tiene como objetivo la recolección, procesamiento y análisis de tweets relacionados con Boric. Se incluyen modelos de procesamiento de lenguaje natural (NLP) para el análisis de sentimiento, tendencias y patrones en los datos.

---

### 📁 Estructura del Proyecto

FK01-ENCUESTAS/ │── data/ # Archivos de datos procesados y sin procesar │ ├── raw_data.csv # Tweets sin procesar │ ├── processed_data.csv # Tweets limpios │ ├── sentiment_data.csv # Tweets con análisis de sentimiento │ ├── sentiment_boric_tweets.csv # Tweets analizados en sentimiento │ │── notebooks/ # Jupyter Notebooks para análisis │ ├── Processed Data Analysis.ipynb │ │── src/ # Código fuente │ │── api/ │ │ ├── api_twitter.py # Conexión con la API de Twitter │ │ ├── apify_raw_data_collection.py # Recolector de tweets con Apify │ │ │ │── data_collection/ # Scripts de recolección de datos │ │── feature_engineering/ # Ingeniería de características │ │── model_evaluation/ # Evaluación de modelos │ │── model_training/ # Entrenamiento de modelos │ │── preprocessing/ # Preprocesamiento de datos │ │ ├── preprocess_data.py # Limpieza de tweets │ │ │ │── sentiment_analysis/ # Análisis de sentimiento │ │ ├── sentiment.py # Modelo de sentimiento │ │── README.md # Este archivo

---

## ⚙️ Instalación y Dependencias

### **1️⃣ Clonar el repositorio**
```bash
git clone <URL_DEL_REPOSITORIO>
cd FK01-ENCUESTAS