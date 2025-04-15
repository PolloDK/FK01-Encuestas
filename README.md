# 📊 FK01-Encuestas – Aprobación Presidencial en Chile vía Twitter

Este proyecto predice la aprobación presidencial de Gabriel Boric a partir de tweets en español, utilizando scraping, análisis de sentimiento, embeddings, feature engineering y modelos de machine learning (XGBoost). Automatizado diariamente con `cron` y reportes por correo.

## 🧠 Tecnologías utilizadas

- Python 3.10  
- pandas, scikit-learn, joblib, transformers, torch, nltk  
- pysentimiento/robertuito para sentimiento y embeddings  
- pytest para testing automatizado  
- cron para ejecución diaria  
- streamlit, wordcloud, matplotlib, markdown, dotenv  
- SMTP (Gmail) para envío de reportes  

## 📁 Estructura del proyecto

```
FK01-Encuestas/
├── app/                 → Archivos estáticos (logo, assets)
├── data/                → CSVs y archivos de datos (no versionados)
├── logs/                → Archivos de log diarios
├── models/              → Modelos y escaladores entrenados
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

## 🔐 Configura el archivo `.env`

```env
APIFY_API_KEY=tu_api_key_de_apify
EMAIL_REMITENTE=correo@gmail.com
EMAIL_CLAVE_APP=clave_app_generada_en_gmail
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
5. Calcula features diarios y agrega aprobación CADEM  
6. Predice aprobación usando XGBoost  
7. Genera wordclouds e índice de negatividad  
8. Envía un resumen por correo en HTML profesional  

## 🧪 Correr tests

```bash
pytest tests/ -v
```

## 📬 Resumen Diario

Cada día se genera un resumen Markdown + HTML que se envía por correo con:

- Resultados de tests  
- Aprobación presidencial estimada y variación  
- Índice de negatividad (barra roja)  
- % de tweets negativos (barra naranja)  
- Wordcloud del día  
- Logo institucional  

## ⏱ Automatización vía Cron

Para ejecutar automáticamente todos los días a medianoche:

```bash
crontab -e
```

Y agrega:

```cron
0 0 * * * cd /ruta/a/FK01-Encuestas && /ruta/a/python src/main.py >> logs/cron_main.log 2>&1
```

## 🧹 Buenas prácticas

- No subas `.env` ni `data/` al repo  
- Usa `.env.template` como referencia para despliegue  
- Usa `setup.sh` para instalar y preparar entorno  
- Usa logs para debug y análisis de ejecución  
- Corre `pytest` antes de cada push o cron  

## 📌 Créditos

Proyecto desarrollado por Cristian Rodríguez – FK Economics  
Contacto: crodriguez@fkeconomics.com  

## 📄 Licencia

Uso interno y académico. No redistribuir sin autorización.