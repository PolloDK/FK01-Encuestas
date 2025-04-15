import pytest
from src.preprocessing import TweetPreprocessor
import numpy as np

# Usamos un archivo de prueba cualquiera (no se usará realmente en estos tests)
PRETEND_INPUT_PATH = "tests/fake_path.csv"
preprocessor = TweetPreprocessor(PRETEND_INPUT_PATH)

# 1. Test de limpieza de texto
def test_clean_text_remueve_urls_y_stopwords():
    texto = "Gabriel Boric miente otra vez en cadena nacional https://noticias.cl"
    resultado = preprocessor.clean_text(texto)

    assert resultado is not None, "El resultado no debería ser None para un texto válido"
    assert "http" not in resultado
    assert "@" not in resultado
    assert "#" not in resultado
    palabras = resultado.split()
    assert 3 <= len(palabras) <= 50, f"Texto limpio tiene {len(palabras)} palabras"

# 2. Test de texto vacío para sentiment
def test_analyze_sentiment_texto_vacio():
    resultado = preprocessor.analyze_sentiment("")
    assert resultado[0] == "Neutral"
    assert all(isinstance(x, float) for x in resultado[1:])

# 3. Test embedding dummy
def test_get_embedding_devuelve_vector_correcto():
    emb = preprocessor.get_embedding("hola mundo")
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (768,)