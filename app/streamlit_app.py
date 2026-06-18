"""App de demo: predicción interactiva del precio de un portátil.

Ejecutar con:  streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.predict import load_model, predict_price

st.set_page_config(page_title="Predictor de precios de portátiles", page_icon="💻")
st.title("💻 Predictor de precio de portátiles")
st.caption("Modelo de regresión entrenado sobre especificaciones técnicas (RAM, CPU, GPU, etc.)")

try:
    model = load_model()
except FileNotFoundError:
    st.error("No se encontró el modelo entrenado. Ejecuta `python -m src.train_model` primero.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    brand = st.selectbox("Marca", ["Apple", "HP", "Acer", "Asus", "Dell", "Lenovo", "Chuwi", "MSI", "Microsoft", "Toshiba", "Huawei", "Xiaomi", "Vero", "Razer", "Mediacom", "Samsung", "Google", "Fujitsu", "LG"])
    laptop_type = st.selectbox("Tipo", ["Ultrabook", "Notebook", "Netbook", "Gaming", "2 in 1 Convertible", "Workstation"])
    cpu = st.selectbox("CPU", ["Intel Core i3", "Intel Core i5", "Intel Core i7", "Intel Celeron Dual", "Intel Pentium Quad", "Intel Core M", "AMD A9-Series", "AMD A6-Series", "AMD A12-Series"])
    ram_gb = st.select_slider("RAM (GB)", [2, 4, 6, 8, 12, 16, 32, 64], value=8)
    storage_gb = st.select_slider("Almacenamiento (GB)", [32, 64, 128, 256, 500, 512, 1024, 2048], value=256)
    storage_type = st.selectbox("Tipo de almacenamiento", ["HDD", "SSD"])
    gpu = st.selectbox("GPU (marca)", ["Intel", "Nvidia", "AMD"])

with col2:
    screen_size_in = st.selectbox("Tamaño de pantalla (in)", [11.6, 13.3, 14.0, 15.6, 17.3], index=3)
    resolution = st.selectbox("Resolución", ["1366x768", "1600x900", "1920x1080", "2560x1440", "3840x2160"])
    weight_kg = st.slider("Peso (kg)", 0.9, 4.5, 2.0)
    touchscreen = st.checkbox("Pantalla táctil")
    os_ = st.selectbox("Sistema operativo", ["Windows 10", "Windows 7", "macOS", "Mac OS X", "Linux", "Chrome OS", "Android", "No OS"])

specs = {
    "ram_gb": ram_gb, "storage_gb": storage_gb, "screen_size_in": screen_size_in,
    "weight_kg": weight_kg,
    "touchscreen": int(touchscreen), "brand": brand, "laptop_type": laptop_type, "cpu": cpu,
    "storage_type": storage_type, "gpu": gpu, "resolution": resolution, "os": os_,
}

if st.button("Estimar precio", type="primary"):
    price = predict_price(specs, model=model)
    st.success(f"💰 Precio estimado: **${price:,.2f}**")
