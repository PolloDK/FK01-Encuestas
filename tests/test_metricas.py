import pandas as pd
import numpy as np
import pytest
from src.metricas import clasificar_sentimiento
from unittest.mock import patch, MagicMock

# === Test 1: clasificación de sentimiento ===
def test_clasificar_sentimiento():
    fila_1 = {"score_positive": 0.2, "score_negative": 0.7, "score_neutral": 0.1}
    fila_2 = {"score_positive": 0.5, "score_negative": 0.5, "score_neutral": 0.5}
    fila_3 = {"score_positive": 0.6, "score_negative": 0.2, "score_neutral": 0.2}

    assert clasificar_sentimiento(fila_1) == "negativo"
    assert clasificar_sentimiento(fila_2) == "neutro"
    assert clasificar_sentimiento(fila_3) == "positivo"

# === Test 2: generación de WordCloud (sin guardar imagen) ===
@patch("src.metricas.WordCloud")
def test_wordcloud_generado(mock_wordcloud_class):
    instancia_wc = MagicMock()
    instancia_wc.generate.return_value = instancia_wc  # ⬅️ esto es clave
    mock_wordcloud_class.return_value = instancia_wc
    texto = "boric presidente boric economía chile cambio futuro boric chile"

    from src.metricas import generar_wordcloud_para_fecha

    with patch("pandas.read_csv") as mock_csv:
        mock_csv.return_value = pd.DataFrame({
            "createdAt": pd.to_datetime(["2025-04-15"] * 3),
            "text": [texto, texto, texto]
        })

        generar_wordcloud_para_fecha(pd.to_datetime("2025-04-15").date())

        assert instancia_wc.generate.called
        assert instancia_wc.to_file.called  # ✔️ ahora sí debería pasar