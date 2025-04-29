import streamlit as st

def show_about():

    st.markdown("""
    # Proyecto FK01 – Predicción de Aprobación Presidencial

    ---

    ## 🎯 Objetivos

    El proyecto tiene como propósito **predecir la aprobación presidencial de Gabriel Boric en Chile** en base a los **tweets que lo mencionan**.  
    Además, busca **analizar la variación diaria** de este indicador, vinculándola a los principales **acontecimientos** que ocurren durante la semana.

    Para lograrlo, se implementa un enfoque de **aprendizaje supervisado** utilizando técnicas de **Machine Learning** e **Inteligencia Artificial**.


    ## 🎯 Variable a Predecir

    En esta primera versión, el modelo predice dos indicadores:

    - ✅ **Aprobación presidencial** de Gabriel Boric
    - ❌ **Desaprobación presidencial** de Gabriel Boric

    La fuente de estos datos es la **Encuesta CADEM**, cuyos resultados se publican cada **domingo**.  
    El período disponible comprende desde **marzo de 2022** (inicio del mandato presidencial) hasta **la actualidad**.

    🔗 Para más información: [CADEM Plaza Pública](https://cadem.cl/plaza-publica/)


    ## ⚙️ Características de Predicción

    El modelo utiliza las siguientes series de características (**features**) para realizar las predicciones:

    1. 🧠 **Análisis de Sentimiento**  
    Utilizando el modelo **RoBERTuito** (adaptación de **BERT** para español), cada tweet es analizado para obtener:
    - Probabilidad de ser **positivo**
    - Probabilidad de ser **negativo**
    - Probabilidad de ser **neutro**

    2. 🔤 **Word Embedding**  
    Cada tweet es representado numéricamente mediante **vectores de embeddings**, capturando su **significado semántico**.

    3. 🛠️ **Feature Engineering**  
    Se construyen nuevas variables, incluyendo:
    - **Rezagos** y **ventanas móviles** de la aprobación y desaprobación.
    - **Scores de sentimiento ponderados** por engagement (retweets, likes, comentarios).
    - **Diferencias** y **tendencias** de sentimiento a lo largo del tiempo.

    ✅ El modelo fue entrenado respetando estrictamente la **secuencia temporal** de los datos.


    ## 🔄 Pipeline de Actualización Diaria

    Cada día, de manera automática, se ejecutan las siguientes etapas:

    1. 📥 **Scraping** de nuevos tweets mencionando al presidente.
    2. 🧠 **Análisis de Sentimiento** sobre los textos obtenidos.
    3. 🔤 **Generación de Word Embeddings**.
    4. 🛠️ **Construcción de nuevas variables** mediante Feature Engineering.
    5. 🔮 **Predicción** de la aprobación y desaprobación diaria.


    # 🚀 ¡Actualizando la aprobación presidencial día a día en base a la voz de Twitter!

    ---

    ### 🛠️ Nota Técnica
    Actualmente, las predicciones están disponibles **desde octubre de 2024 hasta el presente**.
    """, unsafe_allow_html=True)
