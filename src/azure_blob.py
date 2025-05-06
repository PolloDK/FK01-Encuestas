import os
from dotenv import load_dotenv
from azure.storage.blob import BlobClient
from azure.storage.blob import BlobServiceClient
import pandas as pd
from io import BytesIO
from tqdm import tqdm
import sys

# Cargar variables de entorno
load_dotenv()

AZURE_CONN_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_BLOB_CONTAINER = os.getenv("AZURE_BLOB_CONTAINER", "data")

# Cliente base
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONN_STR)
container_client = blob_service_client.get_container_client(AZURE_BLOB_CONTAINER)

def read_csv_blob(blob_name: str) -> pd.DataFrame:
    """Descarga un archivo CSV desde Azure Blob Storage con barra de progreso."""
    print(f"📦 Intentando abrir: {blob_name}")
    blob_client = container_client.get_blob_client(blob_name)

    if not blob_client.exists():
        raise FileNotFoundError(f"❌ No existe el blob: {blob_name}")

    blob_props = blob_client.get_blob_properties()
    total_size = blob_props.size
    print(f"📦 Tamaño del blob: {total_size / 1024 / 1024:.2f} MB")

    stream = blob_client.download_blob()
    buffer = BytesIO()

    # Barra de carga mientras se descarga
    with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"⬇️ Descargando {blob_name}") as pbar:
        for chunk in stream.chunks():
            buffer.write(chunk)
            pbar.update(len(chunk))

    buffer.seek(0)
    df = pd.read_csv(buffer, low_memory=False, parse_dates=["date"])
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce") 
    print(f"✅ CSV leído: {df.shape}")
    return df

def write_csv_blob(df: pd.DataFrame, blob_name: str) -> None:
    """Sube un DataFrame como CSV a Azure Blob Storage con feedback visual."""
    print(f"📤 Subiendo {blob_name}...")
    blob_client = container_client.get_blob_client(blob_name)
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    total_size = buffer.getbuffer().nbytes

    with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"⬆️ Subiendo {blob_name}") as pbar:
        def progress_hook(current, total):
            # ⚠️ Protege contra valores None
            if current is not None and isinstance(current, int):
                try:
                    pbar.update(current - pbar.n)
                except Exception:
                    pass  # evita errores inesperados por inconsistencias en el hook

        blob_client.upload_blob(
            buffer,
            overwrite=True,
            raw_response_hook=lambda resp: progress_hook(
                resp.context.get('upload_stream_current'), total_size
            )
        )

    print(f"✅ Archivo actualizado: {blob_name}")

def append_csv_blob(df_new: pd.DataFrame, blob_name: str):
    """Concatena nuevo contenido con lo ya existente y lo guarda de nuevo."""
    try:
        df_existing = read_csv_blob(blob_name)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True).drop_duplicates(subset=["id"])
    except FileNotFoundError:
        df_combined = df_new

    write_csv_blob(df_combined, blob_name)

def upload_image_blob(local_path_or_bytes, blob_path, content_type="image/png"):
    """Sube una imagen al contenedor de Azure Blob Storage."""
    if isinstance(local_path_or_bytes, str):
        with open(local_path_or_bytes, "rb") as f:
            data = f.read()
    else:
        data = local_path_or_bytes

    print(f"🖼️ Subiendo imagen a {blob_path}...")
    blob = blob_service_client.get_blob_client(container=AZURE_BLOB_CONTAINER, blob=blob_path)
    blob.upload_blob(data, overwrite=True, content_type=content_type)
    print(f"✅ Imagen subida: {blob_path}")

def blob_exists(blob_path: str) -> bool:
    """Verifica si un blob ya existe en el contenedor."""
    blob_client = blob_service_client.get_blob_client(container=AZURE_BLOB_CONTAINER, blob=blob_path)
    return blob_client.exists()

def download_blob_file(blob_name: str, local_path: str):
    """Descarga un archivo binario desde Azure Blob Storage a una ruta local."""
    blob_client = container_client.get_blob_client(blob_name)
    if not blob_client.exists():
        raise FileNotFoundError(f"❌ No se encontró el blob: {blob_name}")
    with open(local_path, "wb") as f:
        download_stream = blob_client.download_blob()
        f.write(download_stream.readall())
    print(f"📥 Descargado {blob_name} a {local_path}")