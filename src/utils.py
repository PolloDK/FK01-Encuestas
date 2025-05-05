import os
import pandas as pd
from datetime import datetime
import re
from markdown import markdown
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from src.azure_blob import read_csv_blob
from src.config import (
    TEST_LOG_PATH,
    PREDICTIONS_PATH,
    RESUMEN_MD_PATH,
    WORDCLOUD_PATH,
    LOGO_PATH
)


def generar_resumen_diario():
    resumen = []

    resumen.append(f"# 📝 Resumen Diario\n")
    resumen.append(f"**Fecha de ejecución:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # === TESTS ===
    resumen.append("## ✅ Resultados de Tests Automáticos\n")
    try:
        with open(TEST_LOG_PATH, "r") as f:
            logs = f.read()
        ultimo_bloque = logs.strip().split("--- Test Run at")[-1]
        total_tests = re.search(r"(\d+) passed", ultimo_bloque)
        failed_tests = re.search(r"(\d+) failed", ultimo_bloque)

        if failed_tests:
            resumen.append("- ❌ **Tests fallaron**")
            resumen.append(f"  - Tests fallidos: **{failed_tests.group(1)}**")
        elif total_tests:
            resumen.append("- ✅ **Todos los tests pasaron correctamente**")
            resumen.append(f"  - Tests ejecutados: **{total_tests.group(1)}**")
        else:
            resumen.append("- ⚠️ No se pudo interpretar el resultado de los tests.")
    except Exception as e:
        resumen.append(f"- ⚠️ Error al leer el log de tests: `{e}`")

    # === PREDICCIÓN ===
    resumen.append("\n## 📈 Predicción de Aprobación Presidencial\n")
    try:
        df_pred = read_csv_blob(PREDICTIONS_PATH)
        ultima = df_pred.sort_values("date").iloc[-1]
        penultima = df_pred.sort_values("date").iloc[-2] if len(df_pred) > 1 else None

        resumen.append(f"- 📅 Fecha: **{ultima['date'].date()}**")
        resumen.append(f"- 🔢 Aprobación estimada: **{ultima['prediccion_aprobacion']:.4f}**")

        if penultima is not None:
            cambio = ultima['prediccion_aprobacion'] - penultima['prediccion_aprobacion']
            signo = "+" if cambio >= 0 else ""
            resumen.append(f"- 📊 Cambio respecto al día anterior: **{signo}{cambio:.4f}**")
    except Exception as e:
        resumen.append(f"- ⚠️ Error al leer predicciones: `{e}`")

    resumen_str = "\n".join(resumen)
    os.makedirs(os.path.dirname(RESUMEN_MD_PATH), exist_ok=True)
    with open(RESUMEN_MD_PATH, "w", encoding="utf-8") as f:
        f.write(resumen_str)

    print("\n📝 Resumen diario generado:")
    print(resumen_str)

    return resumen_str


def enviar_resumen_por_email(contenido_md, destinatario="crodriguez@fkeconomics.com"):
    remitente = os.getenv("EMAIL_REMITENTE")
    clave_app = os.getenv("EMAIL_CLAVE_APP")
    nombre_remitente = "FK Economics Data"

    if not remitente or not clave_app:
        print("❌ Faltan EMAIL_REMITENTE o EMAIL_CLAVE_APP")
        return

    df_pred = read_csv_blob(PREDICTIONS_PATH).sort_values("date")
    ultima = df_pred.iloc[-1]
    penultima = df_pred.iloc[-2] if len(df_pred) > 1 else None
    fecha_str = ultima["date"].strftime("%Y-%m-%d")

    ind_neg = float(ultima.get("indice_negatividad", 0))
    pct_neg = float(ultima.get("porcentaje_tweets_negativos", 0)) * 100

    cambio_neg = ind_neg - float(penultima["indice_negatividad"]) if penultima is not None else 0
    cambio_pct = pct_neg - (float(penultima["porcentaje_tweets_negativos"]) * 100) if penultima is not None else 0

    flecha_neg = "🔺" if cambio_neg > 0 else "🔻"
    color_neg = "#c0392b" if cambio_neg > 0 else "#27ae60"

    flecha_pct = "🔺" if cambio_pct > 0 else "🔻"
    color_pct = "#c0392b" if cambio_pct > 0 else "#27ae60"

    wordcloud_path = WORDCLOUD_PATH / f"wordcloud_{fecha_str}.png"
    wordcloud_path = wordcloud_path if wordcloud_path.exists() else None

    html_render = markdown(contenido_md)
    cambio_aprob = ultima["prediccion_aprobacion"] - penultima["prediccion_aprobacion"]
    color_aprob = "#27ae60" if cambio_aprob > 0 else "#c0392b"
    flecha_aprob = "🔺" if cambio_aprob > 0 else "🔻"

    html_base = f"""<html><body style="font-family: Arial; padding: 30px;">
    <div style="max-width: 720px; margin:auto; background:#fff; border-radius:10px;">
      <div style="background:#f0f0f0; padding:20px 30px; display:flex; align-items:center;">
        <img src="cid:logo" style="height:50px; margin-right:20px;">
        <h2 style="margin:0;">FK Economics – Resumen Diario 📊</h2>
      </div>
      <div style="padding:30px;">
        <p><strong>Fecha:</strong> {fecha_str}</p>
        {html_render}
        <h3>📉 Índice de negatividad</h3>
        <div style="background:#eee; height:20px; border-radius:5px;">
          <div style="background:#e74c3c; width:{min(ind_neg*100,100):.1f}%; height:100%; border-radius:5px;"></div>
        </div>
        <p><strong>{ind_neg:.3f}</strong> <span style="color:{color_neg};">{flecha_neg} {cambio_neg:+.3f}</span></p>

        <h3>💬 % de Tweets Negativos</h3>
        <div style="background:#eee; height:20px; border-radius:5px;">
          <div style="background:orange; width:{min(pct_neg,100):.1f}%; height:100%; border-radius:5px;"></div>
        </div>
        <p><strong>{pct_neg:.2f}%</strong> <span style="color:{color_pct};">{flecha_pct} {cambio_pct:+.2f}%</span></p>

        <h3>📈 Aprobación estimada</h3>
        <p><strong>{ultima['prediccion_aprobacion']:.4f}</strong> 
        <span style="color:{color_aprob};">{flecha_aprob} {cambio_aprob:+.4f}</span></p>
    """

    if wordcloud_path:
        html_base += f"""<h3>☁️ Wordcloud del día</h3>
        <img src="cid:wordcloud" style="max-width:100%; border:1px solid #ddd; border-radius:8px;">"""

    html_base += "</div></body></html>"

    msg = MIMEMultipart("related")
    msg["Subject"] = f"📊 Resumen diario – Aprobación presidencial ({fecha_str})"
    msg["From"] = f"{nombre_remitente} <{remitente}>"
    msg["To"] = destinatario

    msg_alt = MIMEMultipart("alternative")
    msg_alt.attach(MIMEText(html_base, "html"))
    msg.attach(msg_alt)

    def embed(path, cid):
        with open(path, "rb") as f:
            img = MIMEImage(f.read())
            img.add_header("Content-ID", f"<{cid}>")
            img.add_header("Content-Disposition", "inline", filename=cid)
            msg.attach(img)

    if os.path.exists(LOGO_PATH):
        embed(LOGO_PATH, "logo")
    if wordcloud_path:
        embed(wordcloud_path, "wordcloud")

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(remitente, clave_app)
            server.send_message(msg)
        print(f"📬 Correo enviado profesionalmente a {destinatario}")
    except Exception as e:
        print(f"❌ Error al enviar correo: {e}")