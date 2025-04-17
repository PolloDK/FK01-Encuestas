from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import pandas as pd
from src.scraping import TweetScraper

@patch("src.scraping.ApifyClient")
@patch("pandas.read_csv")
def test_scrapear_tweets_ya_existen(mock_read_csv, mock_apify):
    hoy = datetime.today().date()
    df_simulado = pd.DataFrame({
        "id": [1, 2],
        "createdAt": [datetime.now() - timedelta(days=1), datetime.now()],
        "date": [hoy - timedelta(days=1), hoy]
    })
    mock_read_csv.return_value = df_simulado

    scraper = TweetScraper()
    with patch("pandas.DataFrame.to_csv"):
        scraper.scrapear_tweets_pendientes()

    # No se debe llamar a actor si ya están todos los días scrapeados
    mock_apify().actor().call.assert_not_called()

@patch("src.scraping.ApifyClient")
@patch("pandas.read_csv", side_effect=FileNotFoundError)
def test_scrapear_tweets_nueva_llamada(mock_read_csv, mock_apify):
    mock_dataset = MagicMock()
    mock_dataset.list_items.return_value.items = [{
        "id": "abc123",
        "createdAt": datetime.now().isoformat()
    }]
    mock_apify().actor().call.return_value = {"defaultDatasetId": "xyz"}
    mock_apify().dataset().list_items = mock_dataset.list_items

    scraper = TweetScraper()

    with patch("pandas.DataFrame.to_csv") as mock_save:
        scraper.scrapear_tweets_pendientes()
        mock_apify().actor().call.assert_called()
        mock_save.assert_called_once()