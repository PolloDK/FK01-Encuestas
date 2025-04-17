import pandas as pd
from datetime import datetime, timedelta
from apify_client import ApifyClient
from src.config import APIFY_API_KEY, RAW_DATA_PATH
from src.logger import get_logger

logger = get_logger(__name__, "scraping.log")

class TweetScraper:
    def __init__(self):
        self.client = ApifyClient(APIFY_API_KEY)

    def scrapear_tweets_pendientes(self):
        hoy = datetime.today().date()
        logger.info(f"Scraping iniciado. Hoy es: {hoy}")

        try:
            logger.info("📂 Buscando raw_data con tweets...")
            df_existing = pd.read_csv(RAW_DATA_PATH, low_memory=False)
            df_existing["createdAt"] = pd.to_datetime(df_existing["createdAt"])
            df_existing["date"] = df_existing["createdAt"].dt.date
            logger.info(f"✅ Base cargada con éxito.")
        except FileNotFoundError:
            df_existing = pd.DataFrame(columns=["id", "createdAt", "text"])
            df_existing["date"] = pd.to_datetime([])
            logger.warning("📂 No se encontró base existente. Se creará nueva.")

        if df_existing.empty:
            fechas_faltantes = pd.date_range(end=hoy, periods=3).date
        else:
            ultimo_dia = df_existing["date"].max()
            fechas_faltantes = pd.date_range(start=ultimo_dia + timedelta(days=1), end=hoy).date

        if len(fechas_faltantes) == 0:
            print("⏭️ No hay días pendientes por scrapear.")
            logger.info("⏭️ No hay días pendientes por scrapear.")
            return

        logger.info(f"🔍 Días pendientes por scrapear: {list(fechas_faltantes)}")

        for dia in fechas_faltantes:
            print(f"\n🚀 Scrapeando tweets para el día {dia}...")
            logger.info(f"\n🚀 Scrapeando tweets para el día {dia}...")
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
                        "lang": "es",
                        "filter:verified": False,
                        "filter:replies": False,
                        "filter:quote": False,
                    }
                )
                dataset_id = run["defaultDatasetId"]
                dataset_items = self.client.dataset(dataset_id).list_items().items
                df_nuevos = pd.DataFrame(dataset_items)
                if df_nuevos.empty:
                    print(f"⚠️ No se encontraron tweets nuevos para {dia}.")
                    logger.warning(f"⚠️ No se encontraron tweets nuevos para {dia}.")
                    continue

                df_nuevos["createdAt"] = pd.to_datetime(df_nuevos["createdAt"])
                df_nuevos["processed"] = False
                df_existing = pd.concat([df_existing, df_nuevos], ignore_index=True).drop_duplicates(subset=["id"])
                logger.info(f"✅ {len(df_nuevos)} tweets agregados para {dia}.")

            except Exception as e:
                print(f"❌ Error al scrapear para {dia}: {e}")
                logger.error(f"❌ Error al scrapear para {dia}: {e}")

        df_existing.to_csv(RAW_DATA_PATH, index=False)
        print(f"\n💾 Base actualizada con {len(df_existing)} registros.")
        print("Scraping finalizado y guardado.")
        logger.info(f"\n💾 Base actualizada con {len(df_existing)} registros.")
        logger.info("Scraping finalizado y guardado.")


#if __name__ == "__main__":
#    scraper = TweetScraper()
#    scraper.scrapear_tweets_pendientes()
