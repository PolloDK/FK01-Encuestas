import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (
    BASE_DIR,
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    FEATURES_DATASET_PATH,
    ENCUESTAS_PATH,
    PREDICTIONS_PATH,
    MODEL_DIR
)

from src.scraping import TweetScraper
from src.preprocessing import TweetPreprocessor
from src.features import FeatureEngineer
from src.modeling import ModelTrainer
from src.predict import Predictor
from src.metricas import calcular_metricas, generar_wordcloud_diario, generar_wordclouds_pendientes
from src.utils import generar_resumen_diario, enviar_resumen_por_email
import os
import subprocess
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    # Crear carpeta de logs si no existe
    os.makedirs("logs", exist_ok=True)
    log_path = "logs/tests.log"

    print("🔍 Ejecutando tests antes de iniciar el flujo...")
    # Ejecutar pytest en modo silencioso y guardar salida
    result = subprocess.run(
        ["pytest", str(BASE_DIR / "tests"), "--disable-warnings", "-q"],
        capture_output=True,
        text=True
    )

    # Timestamp para el log
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Agregar resultados al archivo de logs
    with open(log_path, "a") as f:
        f.write(f"\n--- Test Run at {timestamp} ---\n")
        f.write(result.stdout)
        if result.stderr:
            f.write("\n[STDERR]\n")
            f.write(result.stderr)

    if result.returncode != 0:
        print("❌ Tests fallaron. Abortando ejecución del pipeline.")
        print(result.stdout)
        print(result.stderr)
        return
    else:
        print("✅ Todos los tests pasaron correctamente.\n")

    print("🚀 Iniciando flujo diario de actualización de datos de Twitter...")

    # Scraping
    scraper = TweetScraper()
    scraper.scrapear_tweets_pendientes()
    print("✅ Scraping finalizado.")

    # Preprocesamiento + Sentimiento + Embeddings
    preprocessor = TweetPreprocessor(RAW_DATA_PATH)
    preprocessor.run_pipeline()
    print("✅ Preprocesamiento completado.")

    # Feature Engineering
    print(f"Comenzando con feature engineering")
    engineer = FeatureEngineer(
        input_path=PROCESSED_DATA_PATH,
        encuestas_path=ENCUESTAS_PATH,
        output_path=FEATURES_DATASET_PATH
    )
    engineer.run()
    print("✅ Feature engineering completado.")

    # Modelado con XGBoost (comentado si no deseas reentrenar)
    # top_vars = [
    #     'approval_rolling_7d',
    #     'approval_lag_7d',
    #     'sentiment_net_rolling7',
    #     'score_negative_rolling7',
    #     'score_negative_rolling3',
    #     'weighted_negative_quoteCount',
    #     'score_negative_lag_4',
    # ]
    # trainer = ModelTrainer(
    #     input_path=FEATURES_DATASET_PATH,
    #     model_output_dir=MODEL_DIR,
    #     top_vars=top_vars
    # )
    # trainer.train()
    # print("✅ Modelado completado.")

    # Predicción con el modelo entrenado
    print(f"Comenzando predicción")
    predictor = Predictor()
    if not os.path.exists(FEATURES_DATASET_PATH):
        print(f"⚠️ No se encontró el archivo {FEATURES_DATASET_PATH}. Se omite la predicción.")
        return
    predicciones = predictor.predict()
    predicciones.to_csv(PREDICTIONS_PATH, index=False)
    print(f"✅ Predicción generada y guardada en {PREDICTIONS_PATH}")

    # Cálculo de Métricas
    print(f"Comenzamos el cálculo de métricas")
    calcular_metricas()
    print(f"Comenzamos el armado de la wordcloud")
    generar_wordclouds_pendientes()
    
    # === Generar y enviar resumen diario ===
    print("Generando resumen diario...")
    resumen = generar_resumen_diario()
    enviar_resumen_por_email(contenido_md=resumen)
    


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error en la ejecución: {e}")
