import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json

from modelo_rayleigh import predecir_riesgo_defecto  

st.set_page_config(page_title="Predicción de Riesgo de Defectos", layout="centered")

st.title("Predicción de Riesgo de Defectos en Proyectos")
st.write("Completa el formulario para generar la predicción usando el modelo de Regresión + Rayleigh.")

# ---------------------------------------------------------
# FORMULARIO
# ---------------------------------------------------------
with st.form("form_prediccion"):
    tareas = st.number_input("Número estimado de tareas del nuevo proyecto:", min_value=1, value=30)
    semanas = st.number_input("Duración estimada del proyecto (semanas):", min_value=1, value=10)

    generar = st.form_submit_button("🔎 Generar Predicción")

# ---------------------------------------------------------
# PROCESAR PREDICCIÓN
# ---------------------------------------------------------
if generar:
    st.subheader("📊 Resultado del Modelo")

    # Ejecutar tu función principal
    prediccion_json_str = predecir_riesgo_defecto(tareas, semanas)

    # Mostrar JSON
    st.code(prediccion_json_str, language="json")

    # Convertir JSON a DataFrame para gráfica
    try:
        resultado = json.loads(prediccion_json_str)
        df_curva = pd.DataFrame(resultado.get('Curva_Riesgo_Rayleigh', []))

        if not df_curva.empty:
            st.subheader("📈 Curva de Riesgo (Distribución Rayleigh)")
            
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(df_curva["Semana"], df_curva["Defectos"], marker="o")
            ax.set_xlabel("Semana del Proyecto")
            ax.set_ylabel("Defectos Estimados")
            ax.set_title("Distribución de Riesgo Rayleigh")
            ax.grid(True)

            st.pyplot(fig)

    except Exception as e:
        st.error(f"No se pudo generar la gráfica: {e}")
