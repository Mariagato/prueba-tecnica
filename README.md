# Prueba Técnica · Módulos 1 y 2

Repositorio con la solución de los dos módulos de la prueba técnica.

## Estructura

```
.
├── Modulo1/                     # Optimización de asignación de órdenes
│   ├── Modulo1_Solucion.ipynb
│   └── Modulo1_Conclusiones.pdf
├── Modulo2/                     # Series de tiempo
│   ├── Modulo2_Punto1_Solucion.ipynb   # Proceso de extrusión (KPI, turnos, hipótesis)
│   ├── Modulo2_Punto2_Solucion.ipynb   # Previsión de demanda (M5) + Conformal Prediction
│   ├── app_streamlit.py                # App de visualización de pronósticos e incertidumbre
│   ├── app_artifacts/                  # Artefactos que consume la app (modelo, predicciones, intervalos)
│   └── Modulo2_Conclusiones.pdf
├── requirements.txt             # Dependencias de la app (para Streamlit Cloud)
└── .streamlit/config.toml       # Tema de la app
```

## App Streamlit — Previsión de demanda con incertidumbre

La app visualiza el pronóstico a 28 días por producto-tienda y su banda de
incertidumbre del 90% calculada con **Split Conformal Prediction**. Lee los
artefactos ya entrenados en `Modulo2/app_artifacts/`, por lo que **no reentrena**
ni necesita los CSV crudos.



> Los CSV crudos del dataset M5 (`Modulo2/Modulo_dos_segundo_punto/`) no se
> versionan por superar el límite de 100 MB de GitHub y no ser necesarios para
> la app. El notebook los usa solo para entrenar y regenerar los artefactos.
