import streamlit as st
from streamlit_navigation_bar import st_navbar
from pages import home, about
import os

# --- Configuración general ---
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# --- Definición de páginas locales ---
pages = {
    "Home": home.show_home,
    "About": about.show_about,
}

# --- Ruta del logo ---
parent_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(parent_dir, "assets", "logo_fk.png")

# --- Definir TODAS las páginas para el navbar ---
pages_navbar = ["Home", "About", "Página web", "LinkedIn"]

# --- Asociar URLs externas ---
urls = {
    "Página web": "https://www.tu-pagina-web.com",  # 👈 cambia por tu sitio real
    "LinkedIn": "https://www.linkedin.com/in/tu-linkedin/"
}

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
    st.image("app/assets/logo_fk.png", use_container_width=True)
    st.markdown("## FK01 - Aprobación Presidencial")
    st.write("Proyecto de predicción de aprobación presidencial basado en ML.")
    st.write("Última actualización: 2025-04-11")
    st.markdown("---")
    st.markdown("Desarrollado por [Tu Nombre](https://github.com/PolloDK)")
