"""
App Streamlit - Previsión de Demanda con Incertidumbre (Conformal Prediction)
Módulo 2, Segundo Punto.

Lee los artefactos generados por Modulo2_Punto2_Solucion.ipynb (carpeta app_artifacts/)
y permite explorar de forma interactiva el pronóstico a 28 días y su banda de
incertidumbre del 90% para cada serie producto-tienda.

Ejecutar:
    conda run -n xgboost streamlit run app_streamlit.py
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------- #
# Configuración de la página
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Previsión de Demanda · Incertidumbre",
    page_icon="📦",
    layout="wide",
)

ART_DIR = Path(__file__).parent / "app_artifacts"
AZUL = "#4C72B0"
ROJO = "#C44E52"
VERDE = "#55A868"


# --------------------------------------------------------------------------- #
# Carga de artefactos (cacheada)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def cargar_artefactos():
    pred = pd.read_parquet(ART_DIR / "predicciones_intervalos.parquet")
    pred["date"] = pd.to_datetime(pred["date"])
    unc = pd.read_parquet(ART_DIR / "incertidumbre_por_serie.parquet")
    with open(ART_DIR / "meta.json") as f:
        meta = json.load(f)
    return pred, unc, meta


if not (ART_DIR / "predicciones_intervalos.parquet").exists():
    st.error(
        "No se encontraron los artefactos en `app_artifacts/`.\n\n"
        "Ejecuta primero el notebook `Modulo2_Punto2_Solucion.ipynb` "
        "(sección 8) para generarlos."
    )
    st.stop()

pred, unc, meta = cargar_artefactos()

# --------------------------------------------------------------------------- #
# Encabezado
# --------------------------------------------------------------------------- #
st.title("📦 Previsión de Demanda con Cuantificación de Incertidumbre")
st.caption(
    "Modelo LightGBM (objetivo Tweedie) + Split Conformal Prediction · "
    f"Ventana de test: {meta['fecha_min_test']} → {meta['fecha_max_test']} · "
    f"{int(meta['n_series'])} series producto-tienda"
)

# Métricas de calibración
c1, c2, c3, c4 = st.columns(4)
c1.metric("Cobertura objetivo", f"{meta['cobertura_objetivo']*100:.0f}%")
c2.metric("Cobertura real (normalizado)", f"{meta['cobertura_normalizado']*100:.1f}%")
c3.metric("Cobertura real (estándar)", f"{meta['cobertura_estandar']*100:.1f}%")
c4.metric("Nivel de confianza (1−α)", f"{(1-meta['alpha'])*100:.0f}%")

st.divider()

# --------------------------------------------------------------------------- #
# Barra lateral: filtros jerárquicos
# --------------------------------------------------------------------------- #
st.sidebar.header("Filtros")

cat = st.sidebar.selectbox("Categoría", ["(todas)"] + sorted(pred["cat_id"].unique()))
df_f = pred if cat == "(todas)" else pred[pred["cat_id"] == cat]

estado = st.sidebar.selectbox("Estado", ["(todos)"] + sorted(df_f["state_id"].unique()))
if estado != "(todos)":
    df_f = df_f[df_f["state_id"] == estado]

tienda = st.sidebar.selectbox("Tienda", ["(todas)"] + sorted(df_f["store_id"].unique()))
if tienda != "(todas)":
    df_f = df_f[df_f["store_id"] == tienda]

series_disponibles = sorted(df_f["id"].unique())
if not series_disponibles:
    st.warning("No hay series con esa combinación de filtros.")
    st.stop()

serie = st.sidebar.selectbox("Serie (producto-tienda)", series_disponibles)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Cómo leer la gráfica**\n\n"
    "- Línea azul: demanda real.\n"
    "- Línea roja: pronóstico del modelo.\n"
    "- Banda roja: intervalo del 90% (incertidumbre). "
    "Cuanto más ancha, más incierto el pronóstico."
)

# --------------------------------------------------------------------------- #
# Panel principal: pronóstico de la serie seleccionada
# --------------------------------------------------------------------------- #
s = df_f[df_f["id"] == serie].sort_values("date")

col_izq, col_der = st.columns([3, 1])

with col_izq:
    st.subheader(f"Pronóstico e incertidumbre · {serie}")
    fig = go.Figure()
    # Banda de incertidumbre (relleno entre hi y lo)
    fig.add_trace(go.Scatter(
        x=s["date"], y=s["hi"], mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=s["date"], y=s["lo"], mode="lines", fill="tonexty",
        fillcolor="rgba(196,78,82,0.20)", line=dict(width=0),
        name="Intervalo 90%",
    ))
    fig.add_trace(go.Scatter(
        x=s["date"], y=s["pred"], mode="lines",
        line=dict(color=ROJO, width=2.5), name="Predicción",
    ))
    fig.add_trace(go.Scatter(
        x=s["date"], y=s["real"], mode="lines+markers",
        line=dict(color=AZUL, width=2), marker=dict(size=6), name="Real",
    ))
    fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="Unidades/día", xaxis_title="Fecha",
        legend=dict(orientation="h", y=1.08),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

with col_der:
    st.subheader("Resumen de la serie")
    real = s["real"].to_numpy()
    predv = s["pred"].to_numpy()
    dentro = ((real >= s["lo"]) & (real <= s["hi"])).mean()
    mae = np.mean(np.abs(real - predv))
    ancho = np.mean(s["hi"] - s["lo"])
    st.metric("Venta media diaria (real)", f"{real.mean():.2f}")
    st.metric("MAE del pronóstico", f"{mae:.2f}")
    st.metric("Ancho medio del intervalo", f"{ancho:.2f}")
    st.metric("% real dentro del intervalo", f"{dentro*100:.0f}%")
    st.caption(
        "Categoría: **{}** · Depto: **{}** · Tienda: **{}**".format(
            s["cat_id"].iloc[0], s["dept_id"].iloc[0], s["store_id"].iloc[0]
        )
    )

st.divider()

# --------------------------------------------------------------------------- #
# Panel de incertidumbre: ranking de series
# --------------------------------------------------------------------------- #
st.subheader("¿Dónde tiene el modelo mayor incertidumbre?")
st.caption(
    "Series ordenadas por el ancho medio del intervalo del 90%. "
    "Las de arriba son las menos confiables: útiles para priorizar stock de seguridad."
)

col_a, col_b = st.columns([1, 1])

with col_a:
    top = unc.sort_values("width", ascending=False).head(15)
    fig2 = go.Figure(go.Bar(
        x=top["width"], y=top["id"], orientation="h",
        marker_color=ROJO,
    ))
    fig2.update_layout(
        height=430, margin=dict(l=10, r=10, t=30, b=10),
        title="Top 15 series más inciertas",
        xaxis_title="Ancho medio del intervalo",
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig2, use_container_width=True)

with col_b:
    por_cat = unc.groupby("cat", observed=True)["width"].mean().sort_values(ascending=False)
    fig3 = go.Figure(go.Bar(
        x=por_cat.index, y=por_cat.values, marker_color=VERDE,
    ))
    fig3.update_layout(
        height=430, margin=dict(l=10, r=10, t=30, b=10),
        title="Incertidumbre media por categoría",
        yaxis_title="Ancho medio del intervalo",
    )
    st.plotly_chart(fig3, use_container_width=True)

with st.expander("Ver tabla completa de incertidumbre por serie"):
    st.dataframe(
        unc.sort_values("width", ascending=False).round(2),
        use_container_width=True, height=300,
    )
