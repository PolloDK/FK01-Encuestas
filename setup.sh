#!/bin/bash

echo "🚀 Iniciando setup del proyecto FK01-Encuestas..."

# === (Opcional) Crear y activar entorno virtual
if [ ! -d ".venv" ]; then
  echo "📦 Creando entorno virtual..."
  python3 -m venv .venv
fi

echo "✅ Activando entorno virtual..."
source .venv/bin/activate

# === Instalar dependencias
echo "📚 Instalando dependencias desde requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# === Crear carpetas necesarias
echo "📁 Verificando carpetas..."
mkdir -p data logs models app/assets tests

# === Descargar recursos de NLTK
echo "🔍 Descargando recursos NLTK..."
python -m nltk.downloader stopwords wordnet

# === Mensaje final
echo "✅ Setup completado. No olvides crear tu archivo .env con las credenciales necesarias."
echo "👉 Usa 'source .venv/bin/activate' para activar el entorno antes de correr el proyecto."