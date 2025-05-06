from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()

# ========================
# 📦 PATHS GENERALES
# ========================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Nombres de blobs (rutas lógicas en Azure)
RAW_DATA_PATH = "raw_data.csv"
PREPROCESSED_PATH = "preprocessed.csv"
SENTIMENT_DATA_PATH = "sentiment_analysis.csv"
PROCESSED_DATA_PATH = "processed_data.csv"
EMBEDDING_DATA_PATH = "embedding_data.csv"
FEATURES_DATASET_PATH = "features_dataset.csv"
PREDICTIONS_PATH = "predicciones_diarias.csv"
ENCUESTAS_PATH = "encuestas.csv"

# Model 
MODEL_DIR = Path("models")

# LOGS
LOGS_DIR = BASE_DIR / "logs"
TEST_LOG_PATH = LOGS_DIR / "tests.log"
RESUMEN_MD_PATH = LOGS_DIR / "resumen_diario.md"

# WORDCLOUD Y LOGO
WORDCLOUD_PATH = DATA_DIR / "wordclouds"
LOGO_PATH = BASE_DIR / "app/assets/logo_fk.png"

# ========================
# 🔑 API KEYS
# ========================

APIFY_API_KEY = os.getenv("APIFY_API_KEY")
