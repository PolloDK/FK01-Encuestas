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

    En esta primera versión, el modelo predice dos indicadores: la **aprobación presidencial** y la **desaprobación presidencial** de Gabriel Boric

    La fuente de estos datos es la **Encuesta CADEM**, cuyos resultados se publican cada **domingo**.  
    El período disponible comprende desde **marzo de 2022** (inicio del mandato presidencial) hasta **la actualidad**.

    🔗 Para más información: [CADEM Plaza Pública](https://cadem.cl/plaza-publica/)


    ## ⚙️ Características de Predicción

    El modelo utiliza las siguientes características para realizar las predicciones:

    1. 🧠 **Análisis de Sentimiento**  
    Utilizando el modelo **RoBERTuito** (adaptación de **BERT** para español), cada tweet es analizado para obtener: la probabilidad de que su contenido sea **positivo**
    la probabilidad de que su contenido sea  **negativo** y la probabilidad de que su contenido sea  **neutro**.

    2. 🔤 **Word Embedding**  
    Cada tweet es representado numéricamente mediante **vectores de embeddings**, capturando su **significado semántico**. Cada tweet es representado por un
    vector de **768 dimensiones**.
    Estos vectores son generados utilizando el modelo **RoBERTuito** y son utilizados como entradas para el modelo de predicción.
    El modelo RoBERTuito es un modelo de lenguaje basado en la arquitectura **Transformer** y ha sido preentrenado en un corpus de texto en español.

    3. 🛠️ **Feature Engineering**  
    Además de las antes mencionadas, se crean una serie de variables, incluyendo:
        - **Rezagos** y **ventanas móviles** de la aprobación y desaprobación.
        - **Scores de sentimiento ponderados** por engagement (retweets, likes, comentarios).
        - **Diferencias** y **tendencias** de sentimiento a lo largo del tiempo.

    ✅ El modelo fue entrenado respetando estrictamente la **secuencia temporal** de los datos.


    ## 🚀 Modelación y métricas de performance:
    
    Se generó un pipeline de preprocesamiento en donde se escalan los valores de las variables de entrada utilizando *Robust Scaler* ya que las variables no distribuyen normal y presentan outliers.
    El modelo en su versión v1.0.0, fue entrenado utilizando un modelo de **XGBoost** (modelo de árboles de decisión con estrategia boosing) respetando la secuencia temporal de los datos.
    El modelo fue entrenado utilizando un **70%** de los datos y validado con un **30%** de los datos restantes.
    Las métricas de performance obtenidas son las siguientes:
    - **Aprobación**: 
        - **MAE**: 0.012
        - **R2**: 0.639
    - **Desaprobación**:
        - **MAE**: 0.010
        - **R2**: 0.782

    ## 🔄 Pipeline de Actualización Diaria

    Cada día, de manera automática, se ejecutan las siguientes etapas:

    1. 📥 **Scraping** de nuevos tweets mencionando al presidente.
    2. 🧠 **Análisis de Sentimiento** sobre los textos obtenidos.
    3. 🔤 **Generación de Word Embeddings**.
    4. 🛠️ **Construcción de nuevas variables** mediante Feature Engineering.
    5. 🔮 **Predicción** de la aprobación y desaprobación diaria.

    ---

    ### 🛠️ Nota Técnica
    Actualmente, las predicciones están disponibles **desde octubre de 2024 hasta el presente**.
    """, unsafe_allow_html=True)
