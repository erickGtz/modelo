import streamlit as st
import pandas as pd
import json
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt 
import seaborn as sns 

# Importamos la función de predicción del modelo (Asegúrate que el nombre del archivo sea correcto)
from modelo import hacer_prediccion 

st.set_page_config(page_title="Predicción de Defectos", layout="wide")

# =========================================================================
# 0. CONFIGURACIÓN DE ACCESO
# =========================================================================
USUARIOS = {
    "admin": {"password": "1234", "rol": "encargado"},
    "empleado": {"password": "1234", "rol": "empleado"}
}

# Inicialización de estado de sesión
if "log_ok" not in st.session_state:
    st.session_state["log_ok"] = False
    st.session_state["usuario_rol"] = None
    st.session_state["username"] = ""

# =========================================================================
# 1. FUNCIÓN DE LOGIN
# =========================================================================
def login_form():
    st.sidebar.title("Acceso al Panel")
    
    with st.sidebar.form("login_form"):
        username = st.text_input("Usuario")
        password = st.text_input("Contraseña", type="password")
        submitted = st.form_submit_button("Ingresar")

        if submitted:
            if username in USUARIOS and USUARIOS[username]["password"] == password:
                st.session_state["log_ok"] = True
                st.session_state["usuario_rol"] = USUARIOS[username]["rol"]
                st.session_state["username"] = username
                st.sidebar.success(f"Bienvenido, {username} ({st.session_state['usuario_rol']})")
                st.rerun() # CORRECCIÓN: Usamos st.rerun()
            else:
                st.sidebar.error("Usuario o contraseña incorrectos")

# =========================================================================
# MAIN APP FLOW
# =========================================================================

if st.session_state["log_ok"] == False:
    st.title("Panel de Predicción de Defectos")
    st.info("Solo los usuarios encargados tienen acceso al modelo de predicción.")
    login_form()
    
else:
    # Contenedor para el botón de cerrar sesión
    st.sidebar.title("Sesión Activa")
    st.sidebar.info(f"Usuario: {st.session_state['username']} | Rol: {st.session_state['usuario_rol'].upper()}")
    if st.sidebar.button("Cerrar Sesión"):
        st.session_state["log_ok"] = False
        st.session_state["usuario_rol"] = None
        st.session_state["username"] = ""
        st.rerun() # CORRECCIÓN: Usamos st.rerun()
        
    st.title("Panel de Predicción de Defectos")
    
    # -------------------------------------------------------------------------
    # RESTRICCIÓN DE FUNCIONALIDAD POR ROL
    # -------------------------------------------------------------------------
    
    if st.session_state["usuario_rol"] == "encargado":
        
        st.header("🛠️ Parámetros del Nuevo Proyecto")
        st.markdown("Introduce las estimaciones clave para el nuevo proyecto.")

        # El formulario ya no es un 'with st.form' completo para permitir que los inputs dinámicos funcionen sin generar un rerun
        # Mantenemos el bloque de validación de formulario para generar el evento 'generar'
        with st.form("form_prediccion_main", clear_on_submit=False):
            # Usamos columnas para un layout más limpio en el cuerpo principal
            col_form1, col_form2, col_form3 = st.columns(3)
            
            with col_form1:
                # Predictor 1 (Tamaño) - Clave para la validación de Max Value
                tareas = st.number_input("1. Tareas Totales Estimadas:", min_value=1, value=35, key="tareas_input")
                
            with col_form2:
                # Predictor 2 (Calidad/Complejidad)
                automatizacion_input = st.number_input(
                    "2. Tareas de Automatización Estimadas:", 
                    min_value=0, 
                    # Establecemos el máximo al valor actual de tareas
                    max_value=tareas, 
                    value=min(10, tareas), # Aseguramos que el valor inicial sea <= tareas
                    key="automatizacion_input",
                    help="Número de tareas dedicadas a pruebas automatizadas o CI/CD. No puede exceder las Tareas Totales."
                )
                
            with col_form3:
                # Input para la Curva Rayleigh (Tiempo)
                semanas = st.number_input("3. Duración del Proyecto (Semanas):", min_value=1, value=12,
                                          key="semanas_input",
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
                st.error(f"Error al ejecutar el modelo o conectarse a los CSV: {e}")
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
            # COLUMNA 2 (Superior Derecha): Curva de Riesgo (Plotly Express)
            # ---------------------------------------------------------
            with col2:
                st.markdown("### 3. Curva de Riesgo Semanal (Rayleigh)")
                
                if not df_curva.empty:
                    
                    # Usando Plotly Express (px) para la curva (Punto 3)
                    fig = px.line(
                        df_curva, 
                        x='Semana', 
                        y='Defectos', 
                        title=f"Distribución de Riesgo de Defectos en {semanas} Semanas",
                        markers=True,
                        color_discrete_sequence=['#FF4B4B']
                    )
                    
                    fig.update_layout(
                        xaxis_title="Semana del Proyecto", 
                        yaxis_title="Defectos Esperados",
                        hovermode="x unified"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
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
            # COLUMNA 4 (Inferior Derecha): Correlación (TABLA SIMPLE)
            # ---------------------------------------------------------
            with col4:
                st.markdown("### 5. Trazabilidad: Correlación con Defectos")
                
                corr_series = pd.Series(corr_data)
                df_corr = corr_series.reset_index()
                df_corr.columns = ['Variable', 'Correlación (r)']
                
                # Eliminar Total_Defectos (1.0) y formatear
                df_corr = df_corr[df_corr['Variable'] != 'Total_Defectos']
                
                # Formatear la tabla para la presentación (sin usar applymap)
                df_corr['Correlación (r)'] = df_corr['Correlación (r)'].map(lambda x: '{:.4f}'.format(x) if isinstance(x, (int, float)) else str(x))
                df_corr = df_corr.sort_values(by='Correlación (r)', ascending=False)
                
                # Mostrar como DataFrame simple (sin estilos ni errores)
                st.dataframe(df_corr.set_index('Variable'), use_container_width=True)
                st.caption("Valores más cercanos a 1.0 o -1.0 indican el predictor más fuerte.")

        st.markdown("---")
        
    else:
        st.warning("No tiene permisos para usar el modelo de predicción. Acceda con una cuenta de encargado (admin).")