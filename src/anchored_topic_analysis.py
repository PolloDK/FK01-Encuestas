from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from umap import UMAP

umap_model = UMAP(n_neighbors=15, n_components=5, metric="cosine", min_dist=0.1)


# === Modelo de embeddings ===
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# === Definir seed_topic_list ===
seed_topic_list = [
    # Seguridad ciudadana
    ["delincuencia", "asaltos", "robos", "hurto", "vandalismo", "carabineros", "víctimas", "denuncia", "crimen", "arresto", "bandas criminales", "testigo", "cárcel", "juicio", "seguridad", "patrullaje", "seguridad ciudadana", "secuestro", "encerrona", "portonazo", "narco", "fiscalía"],
    # Pensiones
    ["pensión", "jubilación", "AFP", "reforma de pensiones", "ahorro previsional", "PGU", "IPS", "edad jubilación", "sistema de reparto", "capitalización individual", "APV", "retiro de fondos"],
    # Salud
    ["salud", "hospital", "tratamiento", "medicamentos", "prevención", "vacuna", "enfermedad", "cesfam", "médicos", "enfermeros", "TENS", "terapia", "salud mental", "cirugía", "paciente", "Isapre", "Fonasa", "clínica", "listas de espera"],
    # Sueldos / Costo de vida
    ["sueldo", "salario", "nivel de ingresos", "bonos", "horas extra", "remuneración", "ingreso mensual", "contrato"],
    # Educación
    ["educación", "colegio", "profesores", "deuda histórica", "apoderados", "niños", "matrículas", "emblemáticos", "clase", "gratuidad", "CAE", "universidad", "evaluación", "Mineduc", "colegio de profesores", "simce", "PAES"],
    # Empleo / Trabajo
    ["empleo", "salario mínimo", "cesantía", "trabajo formal", "trabajo informal", "independientes", "pymes", "subcontratación", "Ministerio del trabajo", "emprendimiento", "empleo femenino", "tasa de desempleo", "seguro de cesantía", "inserción laboral", "sindicato", "ley de 40 horas"],
    # Medioambiente
    ["medioambiente", "desarrollo sostenible", "contaminación", "impacto ambiental", "evaluación ambiental", "huella de carbono", "energías renovables", "recursos naturales", "sustentabilidad", "cambio climático", "zonas de sacrificio", "incendios forestales"],
    # Corrupción
    ["corrupción", "soborno", "caso PENTA", "caso SQM", "Hermosilla", "ética", "transparencia", "anticorrupción", "delito cuello blanco", "ley de probidad"],
    # Vivienda
    ["vivienda", "déficit habitacional", "campamentos", "casa propia", "inmobiliaria", "crédito hipotecario", "arriendo", "obras", "MINVU", "permiso de edificación", "proyecto habitacional"],
    # Inmigración
    ["inmigración", "extranjeros", "paso fronterizo", "refugiados", "asilo político", "ley de migración", "políticas migratorias", "venezolanos", "colombianos", "haitianos", "control migratorio", "inmigración ilegal", "deportación", "xenofobia", "discriminación", "trabajo informal"],
    # Transporte público
    ["transporte", "metro", "RED", "micro", "locomoción", "hora punta", "tarjeta BIP", "tarifas", "recorridos", "congestión", "Transantiago"],
    # Derechos Humanos
    ["derechos humanos", "INDH", "defensoría de la niñez", "tortura", "dictadura", "derechos indígenas", "memoria", "verdad", "reparación", "comisión", "indulto", "represión"],
    # Relaciones Internacionales
    ["internacional", "palestina", "israel", "rafah", "ONU", "diplomacia", "conflicto", "solidaridad internacional", "guerra", "relaciones exteriores", "embajada", "asuntos internacionales"],
    # Narcotráfico
    ["narcotráfico", "carabineros", "PDI", "marihuana", "fiscalía", "tráfico de drogas", "carteles", "lavado de dinero", "red de distribución"],
    # Pobreza
    ["pobreza", "hogares vulnerables", "indigentes", "línea de la pobreza", "campamentos", "marginalidad", "subsidios", "Chile solidario", "asignación familiar"],
    # Desigualdad
    ["desigualdad", "concentración de la riqueza", "equidad", "inequidad", "movilidad social", "brecha salarial", "Gini", "ingresos", "redistribución", "protección social"],
    # Inflación
    ["inflación", "IPC", "inflación esperada", "alza de precios", "Banco Central", "política monetaria", "costo de vida", "meta de inflación", "Informe de Política Monetaria"],
    # Violencia
    ["violencia", "crimen organizado", "seguridad", "delitos", "violencia contra la mujer", "bullying", "agresión", "conflicto", "armas", "víctima", "PDI", "carabineros", "homicidio", "violación"],
    # Constitución
    ["constitución", "reforma constitucional", "plebiscito", "estallido social", "asamblea constituyente", "Servel", "sistema político", "carta fundamental"],
    # Otros (catch all)
    []
]
    # === Definir topic_id_to_label ===
topic_id_to_label = {
        0: "Seguridad ciudadana",
        1: "Pensiones",
        2: "Salud",
        3: "Sueldos / Costo de vida",
        4: "Educación",
        5: "Empleo / trabajo",
        6: "Medioambiente",
        7: "Corrupción",
        8: "Vivienda",
        9: "Inmigración",
        10: "Transporte",
        11: "Derechos Humanos",
        12: "Relaciones Internacionales",
        13: "Narcotráfico",
        14: "Pobreza",
        15: "Desigualdad",
        16: "Inflación",
        17: "Violencia",
        18: "Constitución",
        -1: "Otros"
    }

def train_anchored_bertopic(texts_topic, seed_topic_list):
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    topic_model = BERTopic(
        embedding_model=embedding_model,
        language="spanish",
        seed_topic_list=seed_topic_list,
        min_topic_size=10,
        verbose=True,
        umap_model=None
    )

    topics, probs = topic_model.fit_transform(texts_topic)
    df_topics = topic_model.get_document_info(texts_topic)

    return topic_model, df_topics

def assign_topic_labels(df_topics, df_all, topic_id_to_label):
    df_topics["candidato"] = df_all.loc[df_topics.index, "candidato"].values
    df_topics["topic_label"] = df_topics["Topic"].map(lambda x: topic_id_to_label.get(x, "Otros"))
    
    return df_topics

def plot_radar_chart_from_df_topics(df_topics, topic_id_to_label, normalize=False, exclude_otros=True, save_path=None):
    topic_order = [v for k, v in topic_id_to_label.items() if (not exclude_otros or v != "Otros")]

    df_pct = df_topics.groupby(["candidato", "topic_label"]).size().reset_index(name="count")
    df_pct["pct"] = df_pct.groupby("candidato")["count"].transform(lambda x: x / x.sum() * 100)

    df_radar = df_pct.pivot(index="topic_label", columns="candidato", values="pct").fillna(0)
    df_radar = df_radar.reindex(topic_order).fillna(0)

    df_radar_plot = df_radar.copy()
    if normalize:
        for candidato in df_radar_plot.columns:
            max_pct = df_radar_plot[candidato].max()
            if max_pct > 0:
                df_radar_plot[candidato] = df_radar_plot[candidato] / max_pct

    labels = df_radar_plot.index.tolist()
    num_labels = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_labels, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    for candidato in df_radar_plot.columns:
        values = df_radar_plot[candidato].tolist()
        values += values[:1]
        ax.plot(angles, values, label=candidato)
        ax.fill(angles, values, alpha=0.25)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    title = "Distribución de tópicos por candidato"
    if normalize:
        title += " (perfil normalizado)"
    else:
        title += " (% real de tweets)"
    ax.set_title(title, size=16)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()

    # Guardar imagen si se especifica
    if save_path:
        plt.savefig(save_path)
        print(f"✅ Radar chart guardado en {save_path}")

    plt.show()
    

if __name__ == "__main__":

    # === Preparar data ===
    df_all = pd.read_csv("data/df_all_tweets.csv")

    texts_topic = df_all["clean_text"].dropna().tolist()

    # === Entrenar modelo ===
    topic_model, df_topics = train_anchored_bertopic(texts_topic, seed_topic_list)

    # === Asignar topic labels ===
    df_topics = assign_topic_labels(df_topics, df_all, topic_id_to_label)

    # === Plot + guardar radar chart % real ===
    plot_radar_chart_from_df_topics(
        df_topics,
        topic_id_to_label,
        normalize=False,
        exclude_otros=True,
        save_path="radar_chart_real.png"
    )

    # === Plot + guardar radar chart normalizado ===
    plot_radar_chart_from_df_topics(
        df_topics,
        topic_id_to_label,
        normalize=True,
        exclude_otros=True,
        save_path="radar_chart_normalizado.png"
    )