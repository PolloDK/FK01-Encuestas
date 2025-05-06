# 📊 FK01-Encuestas – Aprobación Presidencial en Chile vía Twitter

Este proyecto predice la aprobación presidencial de Gabriel Boric a partir de tweets en español, utilizando scraping, análisis de sentimiento, embeddings, feature engineering y modelos de machine learning.

## 🧠 Tecnologías utilizadas

- Python 3.10  
- pandas, scikit-learn, joblib, transformers, torch, nltk  
- pysentimiento/robertuito para sentimiento y embeddings  
- pytest para testing automatizado   
- streamlit, wordcloud, matplotlib, markdown, dotenv  

## 📁 Estructura del proyecto

```
FK01-Encuestas/
├── app/                 
├── src/                 → Código fuente principal
│   ├── scraping.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── predict.py
│   ├── metricas.py
│   ├── utils.py
│   ├── logger.py
│   └── main.py
├── tests/               → Tests unitarios con pytest
├── .env.template        → Variables de entorno (plantilla)
├── .gitignore
├── README.md
├── requirements.txt
└── setup.sh             → Script de instalación automática
```

## ⚙️ Instalación rápida

```bash
git clone https://github.com/tu_usuario/FK01-Encuestas.git
cd FK01-Encuestas
bash setup.sh
cp .env.template .env     # Editar con tus credenciales reales
source .venv/bin/activate
```

## 🚀 Ejecución del pipeline

```bash
python src/main.py
```

Este script:
1. Ejecuta tests con `pytest`  
2. Hace scraping de tweets del día  
3. Limpia y analiza sentimientos  
4. Genera embeddings  
5. Calcula features diarios 
6. Predice aprobación y desaprobación.
7. Genera wordclouds e índice de negatividad.

## 🌐 Aplicación Web

Puedes acceder a la aplicación aquí:  
🔗 [Aplicación de Aprobación Presidencial](https://fk-economics-aprobacion-presidencial-chile.streamlit.app/)  

Desarrollada como parte del trabajo de análisis y visualización de datos de FK Economics.  

## 🔗 Enlaces Relevantes

- Página web oficial de FK Economics: [www.fkeconomics.com](https://www.fkeconomics.com/)  
- LinkedIn de FK Economics: [linkedin.com/company/fk-economics](https://www.linkedin.com/company/fkeconomics)  

## 📌 Créditos

Proyecto desarrollado por Cristian Rodríguez – FK Economics  
Contacto: crodriguez@fkeconomics.com  
