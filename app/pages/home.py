import sys
import os
from pathlib import Path

# Agrega el directorio raíz del proyecto al sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]  # Sube 2 niveles desde app/pages/home.py
sys.path.append(str(ROOT_DIR))

import streamlit as st
import pandas as pd
import plotly.express as px
import os
from datetime import datetime, date
from src.azure_blob import container_client
from src.azure_blob import read_csv_blob, blob_exists
from io import BytesIO
from PIL import Image

@st.cache_data
def cargar_datos():
    df_pred = read_csv_blob("predicciones_diarias.csv")
    df_cadem = read_csv_blob("encuestas.csv")
    return df_pred.sort_values("date"), df_cadem.sort_values("date")

st.cache_data.clear()
def show_home():
    df_pred, df_cadem = cargar_datos()

    st.markdown("## 📈 Evolución de la Predicción vs Encuesta CADEM")

    # 🔹 Filtros de fecha + gráfico agrupados
    with st.container():
        st.subheader("Predicción de Aprobación Presidencial")
        min_date = df_pred["date"].min().date()
        max_date = df_pred["date"].max().date()

        col1, col2 = st.columns(2)
        with col1:
            fecha_inicio = st.date_input("📅 Desde", value=min_date, min_value=min_date, max_value=max_date)
        with col2:
            fecha_fin = st.date_input("📅 Hasta", value=max_date, min_value=min_date, max_value=max_date)

        if fecha_inicio > fecha_fin:
            st.error("❌ La fecha inicial no puede ser posterior a la fecha final.")
            st.stop()

        # Filtrar los datos
        df_pred_filtrado = df_pred[(df_pred["date"] >= pd.to_datetime(fecha_inicio)) & (df_pred["date"] <= pd.to_datetime(fecha_fin))]
        df_cadem_filtrado = df_cadem[(df_cadem["date"] >= pd.to_datetime(fecha_inicio)) & (df_cadem["date"] <= pd.to_datetime(fecha_fin))]

        # Fecha de predicción:
        df_pred_filtrado = df_pred_filtrado[df_pred_filtrado["date"] >= "2024-08-09"]
        df_cadem_filtrado = df_cadem_filtrado[df_cadem_filtrado["date"] >= "2024-08-09"]

        st.markdown("""
            <style>
            /* Cambia el color del título del expander */
            [data-testid="stExpander"] summary {
                color: #2697e1 !important;
                font-weight: 600;
            }

            /* Ícono del checkbox */
            [data-testid="stCheckbox"] svg {
                stroke: white !important;
                fill: #2697e1 !important;
            }

            /* Fondo del checkbox solamente */
            [data-testid="stCheckbox"] .st-cj {
                background-color: #2697e1 !important;
                border: 1px solid #2697e1 !important;
            }

            /* Ícono del checkbox */
            [data-testid="stCheckbox"] .st-cj svg {
                color: white !important;
            }
            /* Aplica el estilo solo a la fecha seleccionada del calendario */
            .stDateInput [data-baseweb="calendar"] div[aria-selected="true"] {
                background-color: #2697e1 !important;
                color: white !important;
                border-radius: 50% !important;
            }
            </style>
        """, unsafe_allow_html=True)

            
        with st.expander("Selecciona las series a mostrar"):
            pred_apr = st.checkbox("Predicción Aprobación", value=True)
            cadem_apr = st.checkbox("CADEM Aprobación", value=True)
            pred_des = st.checkbox("Predicción Desaprobación", value=True)
            cadem_des = st.checkbox("CADEM Desaprobación", value=True)

        # Luego armar la lista:
        seleccionadas = []
        if pred_apr: seleccionadas.append("Predicción Aprobación")
        if cadem_apr: seleccionadas.append("CADEM Aprobación")
        if pred_des: seleccionadas.append("Predicción Desaprobación")
        if cadem_des: seleccionadas.append("CADEM Desaprobación")

        # Gráfico
        fig = px.line()

        if "Predicción Aprobación" in seleccionadas:
            fig.add_scatter(
                x=df_pred_filtrado["date"],
                y=df_pred_filtrado["prediccion_aprobacion"],
                name="Predicción Aprobación",
                mode="lines+markers",
                line=dict(color="royalblue", width=2),
                marker=dict(size=5)
            )

        if "CADEM Aprobación" in seleccionadas:
            fig.add_scatter(
                x=df_cadem_filtrado["date"],
                y=df_cadem_filtrado["aprobacion_boric"],
                name="CADEM Aprobación",
                mode="lines+markers",
                line=dict(color="firebrick", dash="dash", width=2),
                marker=dict(size=6)
            )

        if "Predicción Desaprobación" in seleccionadas:
            fig.add_scatter(
                x=df_pred_filtrado["date"],
                y=df_pred_filtrado["prediccion_desaprobacion"],
                name="Predicción Desaprobación",
                mode="lines+markers",
                line=dict(color="darkgreen", width=2),
                marker=dict(size=5)
            )

        if "CADEM Desaprobación" in seleccionadas:
            fig.add_scatter(
                x=df_cadem_filtrado["date"],
                y=df_cadem_filtrado["desaprobacion_boric"],
                name="CADEM Desaprobación",
                mode="lines+markers",
                line=dict(color="orange", dash="dash", width=2),
                marker=dict(size=6)
            )

        fig.update_layout(
            dragmode=False,
            xaxis=dict(
                showticklabels=True,
                showline=False,
                showgrid=True,
                title=None
            ), 
            yaxis=dict(
                showticklabels=True,
                showline=False,
                showgrid=True,
                title=None
            ),
            hovermode="x unified",
            template="plotly_white",
            legend_title_text="Fuente",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        
    st.markdown("---")
    
 # 🔸 Selector de día para métricas y wordcloud
    fechas_disponibles = df_pred["date"].dropna().dt.date.unique()
    fechas_disponibles = sorted(fechas_disponibles)
    fecha_seleccionada = st.date_input("📆 Selecciona un día para ver métricas específicas", value=fechas_disponibles[-1], min_value=min_date, max_value=max_date)

    df_dia = df_pred[df_pred["date"].dt.date == fecha_seleccionada]
    if df_dia.empty:
        st.warning(f"⚠️ No hay datos para {fecha_seleccionada}")
        return

    # 🔹 Índice de negatividad
    st.markdown("### Índice de Negatividad")
    st.text('El índice de negatividad representa la probabilidad de que un tweet sea negativo.')

    col1, col2 = st.columns([1, 1])
    with col1:
        valor_neg = df_dia["indice_negatividad"].values[0]

        # Valor del día anterior
        fecha_dia_anterior = fecha_seleccionada - pd.Timedelta(days=1)
        df_dia_anterior = df_pred[df_pred["date"].dt.date == fecha_dia_anterior]

        if not df_dia_anterior.empty:
            valor_anterior = df_dia_anterior["indice_negatividad"].values[0]
            delta_neg = valor_neg - valor_anterior
        else:
            delta_neg = None

        st.markdown(f"##### Día: {fecha_seleccionada}")
        st.metric(
            label="",
            value=f"{valor_neg:.1%}",
            delta=f"{delta_neg * 100:+.1f} pp" if delta_neg is not None else "N/A",
            delta_color="inverse"
        )
        progreso_html = f"""
        <div style="background-color:#eee;border-radius:0.5rem;height:1.2rem;width:100%">
            <div style="background-color:#c62828;width:{valor_neg*100:.1f}%;
                        height:100%;border-radius:0.5rem;"></div>
        </div>
        """
        st.markdown(progreso_html, unsafe_allow_html=True)

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### Evolución últimos 7 días")
        df_ultimos7 = df_pred[df_pred["date"].dt.date <= fecha_seleccionada].dropna(subset=["indice_negatividad"]).sort_values("date").tail(7)
        fig_neg = px.line(df_ultimos7, x="date", y="indice_negatividad", markers=True, line_shape="linear")
        fig_neg.update_traces(line=dict(color="firebrick", width=3), marker=dict(color="firebrick"))
        fig_neg.update_layout(
            dragmode=False, 
            height=250, 
            margin=dict(l=10, r=10, t=30, b=10), 
            xaxis=dict(
                showticklabels=True,
                showline=False,
                showgrid=True,
                title=None
            ), 
            yaxis=dict(
                showticklabels=True,
                showline=False,
                showgrid=True,
                title=None
            ),
            template="simple_white")
        st.plotly_chart(fig_neg, use_container_width=True, config={"displayModeBar": False})

    # 🔻 % Tweets Negativos
    if "porcentaje_tweets_negativos" in df_dia.columns:
        st.markdown("### Proporción de Tweets Negativos")
        st.text('Proporción de tweets negativos según clasificación por score de sentimiento.')

        col1, col2 = st.columns([1, 1])
        with col1:
            valor_pct = df_dia["porcentaje_tweets_negativos"].values[0]

            # Valor del día anterior
            fecha_dia_anterior = fecha_seleccionada - pd.Timedelta(days=1)
            df_dia_anterior = df_pred[df_pred["date"].dt.date == fecha_dia_anterior]

            if not df_dia_anterior.empty and "porcentaje_tweets_negativos" in df_dia_anterior.columns:
                valor_anterior_pct = df_dia_anterior["porcentaje_tweets_negativos"].values[0]
                delta_pct = valor_pct - valor_anterior_pct
            else:
                delta_pct = None

            st.markdown(f"##### Día: {fecha_seleccionada}")
            st.metric(
                label="",
                value=f"{valor_pct:.1%}",
                delta=f"{delta_pct * 100:+.1f} pp" if delta_pct is not None else "N/A",
                delta_color="inverse"
            )
            progreso_html_pct = f"""
            <div style="background-color:#eee;border-radius:0.5rem;height:1.2rem;width:100%">
                <div style="background-color:#ff9800;width:{valor_pct*100:.1f}%;
                            height:100%;border-radius:0.5rem;"></div>
            </div>
            """
            st.markdown(progreso_html_pct, unsafe_allow_html=True)

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### Evolución últimos 7 días")
            df_pct_ultimos7 = df_pred[df_pred["date"].dt.date <= fecha_seleccionada].dropna(subset=["porcentaje_tweets_negativos"]).sort_values("date").tail(7)
            fig_pct = px.line(df_pct_ultimos7, x="date", y="porcentaje_tweets_negativos", markers=True, line_shape="linear")
            fig_pct.update_traces(line=dict(color="orange", width=3), marker=dict(color="orange"))
            fig_pct.update_layout(
                dragmode=False, 
                height=250, 
                margin=dict(l=10, r=10, t=30, b=10),
                xaxis=dict(
                    showticklabels=True,
                    showline=False,
                    showgrid=True,
                    title=None
                ), 
                yaxis=dict(
                    showticklabels=True,
                    showline=False,
                    showgrid=True,
                    title=None
                ),
                template="simple_white")
            st.plotly_chart(fig_pct, use_container_width=True, config={"displayModeBar": False})

    st.markdown("---")

    # 🔸 Word Cloud
    st.markdown("### Nube de Palabras del Día")
    col1, col2 = st.columns([1, 1])
    with col1:
        blob_path = f"wordclouds/wordcloud_{fecha_seleccionada}.png"
        if blob_exists(blob_path):
            blob_client = container_client.get_blob_client(blob_path)
            img_data = blob_client.download_blob().readall()
            img = Image.open(BytesIO(img_data))
            st.image(img, ...)
        else:
            st.warning(f"⚠️ La nube de palabras aún no está disponible para {fecha_seleccionada}.")

    with col2:
        st.markdown("#### Descripción")
        st.markdown(
            "La nube de palabras representa las palabras más frecuentes en los tweets "
            "relacionados con la aprobación presidencial. Las palabras más grandes "
            "indican mayor frecuencia."
        )