import streamlit as st
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt # Importación Simple
import seaborn as sns          # Importación Simple

# Importamos la función de predicción del modelo 
from modelo import hacer_prediccion 

st.set_page_config(page_title="Predicción de Riesgo de Defectos", layout="wide")

st.title("🛡️ Panel de Predicción de Riesgo de Defectos")

# ---------------------------------------------------------
# FORMULARIO (Estático y en la página principal)
# ---------------------------------------------------------
st.header("🛠️ Parámetros del Nuevo Proyecto")
st.markdown("Introduce las estimaciones clave para el nuevo proyecto.")

with st.form("form_prediccion", clear_on_submit=False):
    # Usamos columnas para un layout más limpio en el cuerpo principal
    col_form1, col_form2, col_form3 = st.columns(3)
    
    with col_form1:
        # Predictor 1 (Tamaño)
        tareas = st.number_input("1. Tareas Totales Estimadas:", min_value=1, value=35)
        
    with col_form2:
        # Predictor 2 (Calidad/Complejidad)
        automatizacion_input = st.number_input("2. Tareas de Automatización Estimadas:", min_value=0, value=10, 
                                               help="Número de tareas dedicadas a pruebas automatizadas o CI/CD.")
        
    with col_form3:
        # Input para la Curva Rayleigh (Tiempo)
        semanas = st.number_input("3. Duración del Proyecto (Semanas):", min_value=1, value=12,
                                  help="Define la duración del eje de tiempo de la curva de riesgo.")

    st.markdown("---")
    # Botón de submit del formulario
    generar = st.form_submit_button("🔎 Generar Predicción y Análisis", type="primary")

# Contenedores para el resultado
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

# ---------------------------------------------------------
# PROCESAR PREDICCIÓN
# ---------------------------------------------------------
if generar:
    
    # 1. Ejecutar el modelo
    try:
        prediccion_json_str = hacer_prediccion(tareas, automatizacion_input, semanas)
        
    except Exception as e:
        st.error(f"Error al ejecutar el modelo o conectarse a la DB: {e}")
        st.stop()
        
    # 2. Cargar y procesar JSON
    try:
        resultado = json.loads(prediccion_json_str)
        df_curva = pd.DataFrame(resultado.get('Curva_Riesgo_Rayleigh', []))
        total_defectos = resultado.get('Total_Defectos_Estimados', 0)
        metrics = resultado.get('Regresion_Metrics_Final', {})
        corr_data = resultado.get('Correlacion_Defectos', {})

    except Exception:
        st.error("Error al procesar el JSON de salida del modelo. Revise la consola del script `modelo_produccion.py`.")
        st.stop()

    
    st.markdown("## Resultados y Trazabilidad") 

    # ---------------------------------------------------------
    # COLUMNA 1 (Superior Izquierda): KPIs y Métricas
    # ---------------------------------------------------------
    with col1:
        st.markdown("### 2. Eficacia del Modelo Final")
        
        st.metric(
            label="Total de Defectos Estimados", 
            value=f"{total_defectos}", 
            help="Predicción final del Modelo de Regresión Múltiple."
        )
        
        st.info(f"""
            **Métricas de Regresión Múltiple (Final)**
            - **R² (Poder Explicativo):** `{metrics.get('R2')}` (El modelo explica el {metrics.get('R2') * 100:.1f}% de la varianza.)
            - **RMSE (Error Promedio):** `{metrics.get('RMSE')} defectos` (Precisión del modelo.)
            - **MAE (Error Absoluto):** `{metrics.get('MAE')} defectos`
        """)


    # ---------------------------------------------------------
    # COLUMNA 2 (Superior Derecha): Curva de Riesgo (Matplotlib)
    # ---------------------------------------------------------
    with col2:
        st.markdown("### 3. Curva de Riesgo Semanal (Rayleigh)")
        
        if not df_curva.empty:
            # Usando Matplotlib, que es más estable en entornos básicos
            fig, ax = plt.subplots(figsize=(10, 5))
            
            sns.lineplot(
                x='Semana', 
                y='Defectos', 
                data=df_curva, 
                marker='o', 
                color='#FF4B4B', 
                linewidth=3, 
                ax=ax
            )
            
            ax.set_title(f"Distribución de Riesgo de Defectos en {semanas} Semanas", fontsize=14)
            ax.set_xlabel("Semana del Proyecto", fontsize=12)
            ax.set_ylabel("Defectos Esperados", fontsize=12)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.set_xticks(df_curva['Semana'][::max(1, int(np.ceil(semanas/6)))]) 
            
            st.pyplot(fig)
            
            max_defectos = df_curva['Defectos'].max()
            semanas_pico = df_curva[df_curva['Defectos'] == max_defectos]['Semana'].tolist()
            
            st.success(f"**Pico de Riesgo:** Mayor número de defectos ({max_defectos}) esperado alrededor de las semanas {semanas_pico[0]} - {semanas_pico[-1]}.")
            
        else:
            st.warning("No hay datos de curva de riesgo para mostrar.")

    # ---------------------------------------------------------
    # COLUMNA 3 (Inferior Izquierda): Tabla de Datos de Riesgo
    # ---------------------------------------------------------
    with col3:
        st.markdown("### 4. Detalle Numérico del Riesgo Semanal")
        # Mostrar la tabla que viene del JSON
        df_curva.columns = ["Semana", "Defectos Esperados"]
        st.dataframe(df_curva, use_container_width=True)
        st.caption("Estos valores sumados dan el Total de Defectos Estimados.")
        
    # ---------------------------------------------------------
    # COLUMNA 4 (Inferior Derecha): Correlación
    # ---------------------------------------------------------
    with col4:
        st.markdown("### 5. Trazabilidad: Correlación con Defectos")
        
        corr_series = pd.Series(corr_data)
        df_corr = corr_series.reset_index()
        df_corr.columns = ['Variable', 'Correlación (r)']
        
        # Usamos Matplotlib/Seaborn para la gráfica de barras de correlación
        fig_corr, ax_corr = plt.subplots(figsize=(10, 5))
        
        # Eliminar Total_Defectos (correlación 1.0)
        corr_display = corr_series.drop('Total_Defectos').sort_values(ascending=False) 
        
        sns.barplot(x=corr_display.index, y=corr_display.values, palette="viridis", ax=ax_corr)
        ax_corr.set_title('Correlación de Variables con Total_Defectos', fontsize=14)
        ax_corr.set_xlabel('Variable Predictora', fontsize=12)
        ax_corr.set_ylabel('Coeficiente de Correlación (r)', fontsize=12)
        ax_corr.set_ylim(-1, 1)
        ax_corr.tick_params(axis='x', rotation=45)
        ax_corr.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        st.pyplot(fig_corr)
        st.caption("Gráfico que muestra la relación lineal de cada variable con el número total de defectos.")


# Pie de página descriptivo
st.markdown("---")
st.caption("Script de predicción ejecutado desde `modelo_produccion.py`.")