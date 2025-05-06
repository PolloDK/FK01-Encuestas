import streamlit as st
from streamlit_navigation_bar import st_navbar
from pages import home, about
import os
# --- Ruta del logo ---
parent_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(parent_dir, "assets", "logo_fk.png")

# --- Configuración general ---
st.set_page_config( page_title="Aprobación Presidencial", page_icon=logo_path ,layout="wide", initial_sidebar_state="collapsed")

# --- Definición de páginas locales ---
pages = {
    "Inicio": home.show_home,
    "Proyecto": about.show_about,
}

# --- Definir TODAS las páginas para el navbar ---
pages_navbar = ["Inicio", "Proyecto", "Web", "LinkedIn"]

# --- Asociar URLs externas ---
urls = {
    "Web": "https://www.fkeconomics.com/",
    "LinkedIn": "https://www.linkedin.com/company/fkeconomics"
}

st.markdown("""
    <style>
    /* Botones generales */
    .stButton>button {
        background-color: #2697e1;
        color: white;
        border: none;
    }

    /* Etiquetas de multiselect */
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #2697e1 !important;
        color: white !important;
    }

    /* Input de fecha */
    .stDateInput input {
        border-color: #2697e1 !important;
    }

    /* Slider */
    .stSlider > div[data-baseweb="slider"] > div {
        background-color: #2697e1 !important;
    }

    /* Día seleccionado en el calendario */
    div[data-baseweb="datepicker"] button[aria-selected="true"] {
        background-color: #2697e1 !important;
        color: white !important;
    }

    /* Hover en días del calendario */
    div[data-baseweb="datepicker"] button:hover {
        background-color: #d1ecfa !important;
        color: black !important;
    }

    /* Fecha actual en el calendario */
    div[data-baseweb="datepicker"] button[aria-label*="Today"] {
        border: 1px solid #2697e1 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    /* Sidebar como overlay fijo */
    [data-testid="stSidebar"] {
        position: fixed !important;
        left: 0;
        top: 0;
        bottom: 0;
        z-index: 1001;
        width: 22rem !important;
        transition: all 0.3s ease-in-out;
        background-color: white;
        box-shadow: 0 0 10px rgba(0,0,0,0.2);
    }

    /* Oculta el margen del contenido cuando el sidebar está presente */
    [data-testid="stSidebar"] + div .block-container {
        padding-left: 1rem !important;
        transition: all 0.3s ease-in-out;
        margin-left: 0 !important;
    }

    /* Ajuste para pantallas pequeñas */
    @media (max-width: 768px) {
        [data-testid="stSidebar"] {
            width: 80% !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# --- Estilos del navbar ---
styles = {
    "nav": {
        "background-color": "#383838",
        "display": "flex",
        "justify-content": "flex-start",
        "align-items": "center",
        "gap": "1rem",
    },
    "img": {
        "padding-right": "14px",
    },
    "span": {
        "color": "white",
        "padding": "14px",
        "font-weight": "normal",
    },
    "active": {
        "background-color": "white",
        "color": "var(--text-color)",
        "font-weight": "normal",
        "padding": "14px",
    }
}

# --- Opciones ---
options = {
    "show_menu": False,
    "show_sidebar": True,
}

# --- Navbar ---
selected_page = st_navbar(
    pages=pages_navbar,
    styles=styles,
    options=options,
    urls=urls,  # 👈 añadimos urls externas
)

# --- Mostrar contenido solo si es interno ---
if selected_page in pages:
    pages[selected_page]()
else:
    pass  # No hacer nada para Página Web o LinkedIn, porque son links externos

# --- Sidebar ---
with st.sidebar:
    st.image(logo_path, width=80)
    st.markdown("## FK01 - Aprobación Presidencial")
    st.write("Este proyecto tiene como objetivo predecir la aprobación presidencial de Gabriel Boric a partir del análisis de sentimiento de los tweets que lo mencionan diariamente.")
    st.write("👉 Para más información, selecciona la pestaña **Proyecto** desde la barra de navegación.")
    st.write("</u>Última actualización</u>: 29 de abril de 2025", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("Desarrollado por [Cristián Rodríguez](https://github.com/PolloDK) Economista y Data Analyst de FK Economics")
