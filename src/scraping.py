import pandas as pd
from datetime import datetime
from apify_client import ApifyClient
from src.config import APIFY_API_KEY, RAW_DATA_PATH

from src.logger import get_logger
logger = get_logger(__name__, "scraping.log")

class TweetScraper:
    def __init__(self):
        self.client = ApifyClient(APIFY_API_KEY)

    def scrapear_tweets_del_dia(self):
        hoy = datetime.today().date()
        logger.info(f"Scraping iniciado para {hoy}")

        try:
            df_existing = pd.read_csv(RAW_DATA_PATH)
            df_existing["createdAt"] = pd.to_datetime(df_existing["createdAt"])
            df_existing["date"] = df_existing["createdAt"].dt.date
            logger.info("Base existente cargada correctamente.")
        except FileNotFoundError:
            df_existing = pd.DataFrame()
            logger.warning("No se encontró base existente. Se creará nueva.")

        if not df_existing.empty and hoy in df_existing["date"].unique():
            logger.info(f"Tweets del día {hoy} ya existen. Scraping omitido.")
            return

        logger.info(f"Scrapeando tweets del día {hoy} desde Apify...")
        try:
            run = self.client.actor("kaitoeasyapi~twitter-x-data-tweet-scraper-pay-per-result-cheapest").call(run_input={
                "searchTerms": [
                    f"Boric since:{hoy} until:{hoy}",
                    f"Gabriel Boric since:{hoy} until:{hoy}",
                    f"Presidente Boric since:{hoy} until:{hoy}"
                ],
                "maxItems": 100,
                "queryType": "Top",
                "lang": "es",
                "filter:verified": False,
                "filter:replies": False,
                "filter:quote": False,
            })
        except Exception as e:
            logger.error(f"Error durante llamada a Apify: {e}")
            return

        dataset_id = run["defaultDatasetId"]
        dataset_items = self.client.dataset(dataset_id).list_items().items
        df_nuevos = pd.DataFrame(dataset_items)

        if df_nuevos.empty:
            logger.warning("No se encontraron tweets nuevos para hoy.")
            return

        df_nuevos["createdAt"] = pd.to_datetime(df_nuevos["createdAt"])
        df_total = pd.concat([df_existing, df_nuevos], ignore_index=True).drop_duplicates(subset=["id"])
        df_total.to_csv(RAW_DATA_PATH, index=False)

        logger.info(f"{len(df_nuevos)} tweets nuevos guardados en {RAW_DATA_PATH}")
