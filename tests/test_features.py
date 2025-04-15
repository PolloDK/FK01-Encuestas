import pandas as pd
import numpy as np
from src.features import FeatureEngineer

# Instancia dummy (no usamos rutas reales para estos tests)
fe = FeatureEngineer("input.csv", "encuestas.csv", "output.csv")

# === Test 1: promedio ponderado básico ===
def test_weighted_avg_basico():
    values = pd.Series([0.2, 0.5, 0.8])
    weights = pd.Series([1, 2, 3])

    resultado = fe.weighted_avg(values, weights)
    esperado = (0.2*1 + 0.5*2 + 0.8*3) / (1+2+3)

    assert np.isclose(resultado, esperado), f"Esperado {esperado}, pero fue {resultado}"

# === Test 2: promedio ponderado con pesos cero ===
def test_weighted_avg_pesos_cero():
    values = pd.Series([0.2, 0.5, 0.8])
    weights = pd.Series([0, 0, 0])

    resultado = fe.weighted_avg(values, weights)
    esperado = values.mean()

    assert np.isclose(resultado, esperado)

# === Test 3: integración mínima con DataFrames simulados ===
def test_merge_encuestas_diarias():
    df_daily = pd.DataFrame({
        "date": pd.to_datetime(["2025-04-14", "2025-04-15"]),
        "score_positive": [0.4, 0.6],
        "score_negative": [0.3, 0.2]
    })
    df_daily["week_start"] = df_daily["date"] - pd.to_timedelta(df_daily["date"].dt.weekday, unit="D")

    df_encuestas = pd.DataFrame({
        "date": pd.to_datetime(["2025-04-15"]),
        "aprobacion_boric": [31]
    })
    df_encuestas["week_start"] = df_encuestas["date"] - pd.to_timedelta(df_encuestas["date"].dt.weekday, unit="D")

    df_merged = df_daily.merge(df_encuestas[["week_start", "aprobacion_boric"]], on="week_start", how="left")

    assert "aprobacion_boric" in df_merged.columns
    assert df_merged["aprobacion_boric"].isna().sum() < len(df_merged)