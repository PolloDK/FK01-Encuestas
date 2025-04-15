import streamlit as st
from streamlit_navigation_bar import st_navbar
from pages import home, about
import os
    
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# Páginas y funciones asociadas
pages = {
    "Home": home.show_home,
    "About": about.show_about
}

# Ruta del logo
parent_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(parent_dir, "assets", "logo_fk.png")

# Estilos del navbar
styles = {
    "nav": {
        "background-color": "#e5e5e5",  # gris claro
        "padding": "10px",
        "justify-content": "flex-end",
    },
    "img": {
        "padding-right": "10px",
    },
    "span": {
        "color": "black",
        "padding": "10px",
        "font-weight": "bold",
    },
    "active": {
        "background-color": "#d0d0d0",
        "color": "black",
    }
}

# Opciones del componente
options = {
    "show_menu": False,
    "show_sidebar": False,
}

# Renderizar barra de navegación
selected_page = st_navbar(
    list(pages.keys())
)

# Llamar a la página correspondiente
pages[selected_page]()

with st.sidebar:
    st.image("app/assets/logo_fk.png", use_container_width=True)
    st.markdown("## FK01 - Aprobación Presidencial")
    st.write("Proyecto de predicción de aprobación presidencial basado en ML.")
    st.write("Última actualización: 2025-04-11")
    st.markdown("---")
    st.markdown("Desarrollado por [Tu Nombre](https://github.com/PolloDK)")