#!/bin/bash

echo "🐳 Iniciando setup Docker del proyecto FK01-Encuestas..."

# Build de la imagen
echo "📦 Construyendo imagen de Docker..."
docker build -t aprobacion_presidencial .

# Correr contenedor
echo "🚀 Iniciando contenedor en puerto 8501..."
docker run -d -p 8501:8501 aprobacion_presidencial

echo "✅ Docker desplegado. Abre tu navegador en http://localhost:8501"