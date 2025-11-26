import pandas as pd
import numpy as np
import mysql.connector
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt
import json
import random

# =========================================================================
# SECCIÓN 1: EXTRACCIÓN DE DATOS (FUNCIÓN ORIGINAL PARA OBTENER TOTALES)
# =========================================================================

def obtener_totales_proyectos():
    """
    Extrae los datos TOTALES de todos los proyectos, necesarios para la Regresión.
    """
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="13mysql22",
            database="db_soporte"
        )
        cursor = conn.cursor(dictionary=True)

        cursor.execute("""
            SELECT 
                p.idProyecto,
                p.nombre_proyecto,
                h.defectos_reportados
            FROM dim_proyecto p
            JOIN hecho_proyecto h ON h.idProyecto = p.idProyecto;
        """)
        proyectos = cursor.fetchall()
        registros = []

        for proyecto in proyectos:
            idp = proyecto["idProyecto"]
            
            # === Uso de Procedimientos Almacenados ===
            
            # Tamaño (contar_tareas)
            cursor2 = conn.cursor() 
            args_tareas = [idp, 0] 
            resultado_tareas = cursor2.callproc("contar_tareas", args_tareas)
            total_tareas = resultado_tareas[1]
            cursor2.close()

            # Tiempo (calcular_semanas_proyecto)
            cursor2 = conn.cursor()
            args_semanas = [idp, 0]
            resultado_semanas = cursor2.callproc("calcular_semanas_proyecto", args_semanas)
            semanas = resultado_semanas[1]
            cursor2.close()
            
            # =========================================

            # La predicción usará Total_Tareas y Tiempo_Semanas como variables predictoras (X)
            registros.append({
                "idProyecto": idp,
                "Tiempo_Semanas": semanas,
                "Total_Defectos": proyecto["defectos_reportados"], # Y (Target)
                "Total_Tareas": total_tareas, # X1
            })

        print(pd.DataFrame(registros).head())
        return pd.DataFrame(registros)

    except mysql.connector.Error as err:
        print(f"Error de MySQL al obtener datos totales: {err}")
        return pd.DataFrame()
        
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

# =========================================================================
# SECCIÓN 2: MODELO RAYLEIGH (SOLO DISTRIBUCIÓN)
# =========================================================================

def ModeloRayleighDistribucion(total_defects, duration_weeks, historic_b=0.005):
    """
    Distribuye el total de defectos predicho a lo largo de las semanas
    utilizando el PDF de Rayleigh.
    """
    total_defectos_estimados = int(np.round(total_defects))
    duracion_hist = int(duration_weeks)
    
    # 1. Parámetro 'b' fijo (para la forma de la curva).
    b_ajustado = historic_b 

    # 2. Distribución Temporal (Curva de riesgo)
    time_points = np.arange(1, duracion_hist + 1)
    
    # PDF del modelo ajustado: 2 * b * t * exp(-b * t^2)
    pdf_values = 2 * b_ajustado * time_points * np.exp(-b_ajustado * time_points**2)
    
    proportions = pdf_values / np.sum(pdf_values)
    defects_per_week_float = total_defectos_estimados * proportions
    defects_per_week = np.round(defects_per_week_float).astype(int)
    
    # Ajuste fino (para asegurar que el total entero sea exacto)
    diff = total_defectos_estimados - np.sum(defects_per_week)
    if diff != 0:
        decimal_parts = defects_per_week_float - defects_per_week
        indices_to_adjust = np.argsort(decimal_parts)[::-1][:abs(diff)]
        defects_per_week[indices_to_adjust] += np.sign(diff)
    
    df_distribucion = pd.DataFrame({
        'Semana': time_points, 
        'Defectos': defects_per_week
    })
    
    return {
        "Total_Defectos_Estimados": total_defectos_estimados,
        "Curva_Riesgo_Rayleigh": df_distribucion.to_dict('records'),
        "Parametro_Rayleigh_b": float(b_ajustado) 
    }

# =========================================================================
# SECCIÓN 3: API/SCRIPT ENDPOINT (MODELO HÍBRIDO SIMPLIFICADO)
# =========================================================================

def predecir_riesgo_defecto(total_tareas_nuevo, semanas_estimadas_nuevo):
    """
    Función API simplificada: predice el total de defectos basándose en Tareas/Semanas
    y distribuye el riesgo con Rayleigh.
    """
    print(f"Calculando predicción para proyecto nuevo (Tareas: {total_tareas_nuevo}, Semanas: {semanas_estimadas_nuevo})")
    
    # 1. Obtener todos los datos históricos (USA TODOS LOS PROYECTOS)
    df_totales = obtener_totales_proyectos()
    
    if df_totales.empty:
        return json.dumps({"Error": "No se encontraron datos históricos para entrenar el modelo."}, indent=4)
        
    # 2. ENTRENAMIENTO DEL MODELO DE REGRESIÓN (Predicción del Total de Defectos)
    
    X = df_totales[['Total_Tareas', 'Tiempo_Semanas']].values
    y = df_totales['Total_Defectos'].values
    
    modelo_regresion = LinearRegression()
    modelo_regresion.fit(X, y)
    
    # 3. PREDICCIÓN DEL TOTAL DE DEFECTOS PARA EL NUEVO PROYECTO
    new_project_data = np.array([[total_tareas_nuevo, semanas_estimadas_nuevo]]) 
    total_defects_pred = modelo_regresion.predict(new_project_data)[0]
    total_defects_pred = np.maximum(1, total_defects_pred) 
    
    print(f"-> Regresión Lineal predijo un total de {int(total_defects_pred)} defectos.")

    # 4. DISTRIBUCIÓN RAYLEIGH (Cumplimiento del Requisito)
    resultado_modelo = ModeloRayleighDistribucion(
        total_defects_pred, 
        semanas_estimadas_nuevo
    )
    
    # Aquí iría la validación del Punto 5 (Control de Accesos)
    return json.dumps(resultado_modelo, indent=4)


# =========================================================================
# SECCIÓN 4: EJECUCIÓN DEL FLUJO COMPLETO (PRUEBA FINAL)
# =========================================================================

if __name__ == "__main__":
    
    # No necesitamos el ID en la función, pero lo usamos para la etiqueta de la gráfica.
    NEW_PROJECT_TAG = "Nuevo_Proyecto_X" 
    
    # --- PARÁMETROS DEL NUEVO PROYECTO ---
    NUEVAS_TAREAS_ESTIMADAS = 35 
    NUEVAS_SEMANAS_ESTIMADAS = 12
    
    print("=" * 70)
    print(f"       DEMOSTRACIÓN CLAVE: PROYECTO NUEVO ({NEW_PROJECT_TAG})")
    print(f"       Parámetros: Tareas={NUEVAS_TAREAS_ESTIMADAS}, Semanas={NUEVAS_SEMANAS_ESTIMADAS}")
    print("=======================================================================")
    
    prediccion_json_str = predecir_riesgo_defecto(
        NUEVAS_TAREAS_ESTIMADAS,
        NUEVAS_SEMANAS_ESTIMADAS
    )
    
    print("\n" + "=" * 70)
    print("      SALIDA FINAL DE LA API/SCRIPT AL DASHBOARD (JSON)")
    print("      (Contiene la Predicción Total y la Curva de Riesgo)")
    print("=" * 70)
    print(prediccion_json_str)
    
    
    # ----------------------------------------------------
    # VISUALIZACIÓN DEL RIESGO
    # ----------------------------------------------------
    try:
        resultado = json.loads(prediccion_json_str)
        df_curva = pd.DataFrame(resultado.get('Curva_Riesgo_Rayleigh', []))
        total_defectos = resultado.get('Total_Defectos_Estimados', 0)

        if not df_curva.empty:
            plt.figure(figsize=(10, 6))
            # Gráfico que muestra la distribución de riesgo (tasa de aparición semanal)
            plt.plot(df_curva['Semana'], df_curva['Defectos'], color='red', marker='o', linestyle='-', linewidth=2, label='Curva de Riesgo (Rayleigh)')
            
            plt.title(f'Riesgo de Defectos Semanal (Proyecto: {NEW_PROJECT_TAG})\nTotal Estimado: {total_defectos} defectos', fontsize=14)
            plt.xlabel('Semana del Proyecto', fontsize=12)
            plt.ylabel('Número de Defectos Esperados (Riesgo)', fontsize=12)
            plt.xticks(df_curva['Semana'][::2]) 
            plt.grid(axis='y', linestyle='--', alpha=0.6)
            plt.legend()
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print(f"\nNo se pudo generar el gráfico para la demostración: {e}")