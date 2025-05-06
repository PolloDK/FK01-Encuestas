import pandas as pd
from datetime import datetime, timedelta
from apify_client import ApifyClient
from src.config import APIFY_API_KEY, RAW_DATA_PATH
from src.logger import get_logger
from src.azure_blob import read_csv_blob, write_csv_blob

logger = get_logger(__name__, "scraping.log")

class TweetScraper:
    def __init__(self):
        self.client = ApifyClient(APIFY_API_KEY)

    def scrapear_tweets_pendientes(self) -> bool:
        hoy = datetime.today().date()
        logger.info(f"Scraping iniciado. Hoy es: {hoy}")

        nuevos_tweets = False  # <--- Flag para saber si hay novedades

        try:
            logger.info("📂 Buscando raw_data con tweets...")
            df_existing = read_csv_blob(RAW_DATA_PATH)
            df_existing["createdAt"] = pd.to_datetime(df_existing["createdAt"], errors="coerce")
            df_existing["date"] = pd.to_datetime(df_existing["createdAt"].dt.date)
            logger.info(f"✅ Base cargada con éxito.")
        except FileNotFoundError:
            df_existing = pd.DataFrame(columns=["id", "createdAt", "text"])
            df_existing["date"] = pd.to_datetime([]).date
            logger.warning("📂 No se encontró base existente. Se creará nueva.")

        if df_existing.empty:
            fechas_faltantes = pd.date_range(end=hoy, periods=3).date
        else:
            ultimo_dia = df_existing["date"].max()
            fechas_faltantes = pd.date_range(start=ultimo_dia + timedelta(days=1), end=hoy).date

        if len(fechas_faltantes) == 0:
            print("⏭️ No hay días pendientes por scrapear.")
            logger.info("⏭️ No hay días pendientes por scrapear.")
            return False

        logger.info(f"🔍 Días pendientes por scrapear: {list(fechas_faltantes)}")

        for dia in fechas_faltantes:
            print(f"\n🚀 Scrapeando tweets para el día {dia}...")
            try:
                run = self.client.actor("kaitoeasyapi~twitter-x-data-tweet-scraper-pay-per-result-cheapest").call(
                    run_input={
                        "searchTerms": [
                            f"Boric since:{dia} until:{dia + timedelta(days=1)}",
                            f"Gabriel Boric since:{dia} until:{dia + timedelta(days=1)}",
                            f"Presidente Boric since:{dia} until:{dia + timedelta(days=1)}"
                        ],
                        "maxItems": 30,
                        "queryType": "Top",
                        "lang": "es"
                    }
                )
                dataset_items = self.client.dataset(run["defaultDatasetId"]).list_items().items
                df_nuevos = pd.DataFrame(dataset_items)
                if df_nuevos.empty:
                    logger.warning(f"⚠️ No se encontraron tweets nuevos para {dia}.")
                    continue

                df_nuevos["createdAt"] = pd.to_datetime(df_nuevos["createdAt"])
                df_nuevos["date"] = pd.to_datetime(df_nuevos["createdAt"].dt.date)
                df_nuevos["processed"] = False

                df_existing = pd.concat([df_existing, df_nuevos], ignore_index=True).drop_duplicates(subset=["id"])
                nuevos_tweets = True  # <--- Se encontraron tweets nuevos
                print((f"✅ {len(df_nuevos)} tweets agregados para {dia}."))
                logger.info(f"✅ {len(df_nuevos)} tweets agregados para {dia}.")
            except Exception as e:
                print((f"❌ Error al scrapear para {dia}: {e}"))
                logger.error(f"❌ Error al scrapear para {dia}: {e}")

        if nuevos_tweets:
            print((f"Guardando nuevos tweets"))
            print(df_existing.tail())
            write_csv_blob(df_existing, RAW_DATA_PATH)
            logger.info(f"\n💾 Base actualizada con {len(df_existing)} registros.")
            logger.info("Scraping finalizado y guardado.")
            print(("Scraping finalizado y guardado."))
        return nuevos_tweets



#if __name__ == "__main__":
#    scraper = TweetScraper()
#    scraper.scrapear_tweets_pendientes()
