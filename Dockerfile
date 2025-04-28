# Imagen base
FROM python:3.10-slim

# Directorio de trabajo
WORKDIR /app

# Copiar solo requirements primero
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código (solo lo necesario)
COPY app/ app/
COPY src/ src/
COPY models/ models/
COPY data/ data/
COPY logs/ logs/
COPY tests/ tests/
COPY .env.template .env.template

# Exponer puerto de Streamlit
EXPOSE 8501

# Comando para correr la app de Streamlit
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]