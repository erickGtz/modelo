import pandas as pd
import numpy as np
import mysql.connector
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns 
import json
import random

# =========================================================================
# SECCIÓN 1: EXTRACCIÓN DE DATOS
# =========================================================================

def obtener_totales_proyectos():
    """
    Extrae los datos históricos necesarios para entrenar el Modelo 
    Múltiple (Total_Tareas, Tareas_Automatizacion y Total_Defectos).
    """
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="db_soporte"
        )
        cursor = conn.cursor(dictionary=True)

        cursor.execute("""
            SELECT 
                p.idProyecto,
                h.defectos_reportados,
                h.tareas_automatizacion_total,
                h.costo_defecto 
            FROM hecho_proyecto h
            JOIN dim_proyecto p ON h.idProyecto = p.idProyecto;
        """)
        proyectos_hechos = cursor.fetchall()
        registros = []

        for proyecto in proyectos_hechos:
            idp = proyecto["idProyecto"]
            
            # Total Tareas
            cursor2 = conn.cursor() 
            args_tareas = [idp, 0] 
            resultado_tareas = cursor2.callproc("contar_tareas", args_tareas)
            total_tareas = resultado_tareas[1]
            cursor2.close()

            # Tiempo (calcular_semanas_proyecto) - Requerido para la Curva Rayleigh
            cursor2 = conn.cursor()
            args_semanas = [idp, 0]
            resultado_semanas = cursor2.callproc("calcular_semanas_proyecto", args_semanas)
            semanas = resultado_semanas[1]
            cursor2.close()

            registros.append({
                "idProyecto": idp,
                "Tiempo_Semanas": semanas,
                "Total_Defectos": proyecto["defectos_reportados"], 
                "Total_Tareas": total_tareas, 
                "Costo_Defectos": proyecto["costo_defecto"],
                "Tareas_Automatizacion": proyecto["tareas_automatizacion_total"],
                "Tareas_Reutilizadas": proyecto.get("tareas_reutilizadas_total", 0),
                "Horas_Reales": proyecto.get("horas_reales_total", 0)
            })

        df = pd.DataFrame(registros).fillna(0) 
        print(f"Total de registros cargados para el entrenamiento: {len(df)}")
        print(df.head())
        return df

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
    
    b_ajustado = historic_b 

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
# SECCIÓN 3: API/SCRIPT ENDPOINT - MODELO FINAL (REGRESIÓN MÚLTIPLE)
# =========================================================================

# NOTA: La función ahora recibe Tareas_Automatizacion directamente
def hacer_prediccion(total_tareas_nuevo, automatizacion_tareas_nuevo, semanas_estimadas_nuevo):
    
    print(f"Calculando predicción para proyecto nuevo (Tareas: {total_tareas_nuevo}, Automatización: {automatizacion_tareas_nuevo}, Semanas: {semanas_estimadas_nuevo})")
    
    df_totales = obtener_totales_proyectos()
    
    if df_totales.empty:
        return json.dumps({"Error": "No se encontraron datos históricos para entrenar el modelo."}, indent=4)

    # 1. ANÁLISIS DE CORRELACIÓN (Se mantiene la impresión para trazabilidad)
    variables_correlacion = [
        'Total_Defectos', 'Total_Tareas', 'Tiempo_Semanas', 
        'Costo_Defectos', 'Tareas_Automatizacion'
    ]
    corr_matrix = df_totales[variables_correlacion].corr()
    
    print("\n" + "=" * 70)
    print("      CORRELACIÓN DE VARIABLES CON EL OBJETIVO (Total_Defectos)")
    print("=" * 70)
    print(corr_matrix['Total_Defectos'].sort_values(ascending=False))

    print("\n" + "=" * 70)
    print("      MATRIZ DE CORRELACIÓN COMPLETA (Relación entre todas las variables)")
    print("=" * 70)
    print(corr_matrix)
    print("-" * 70)
    
    y = df_totales['Total_Defectos'].values
    
    # ----------------------------------------------------
    # MODELO 2: REGRESIÓN LINEAL MÚLTIPLE (FINAL)
    # ----------------------------------------------------
    
    # Predictoras: Total_Tareas y Tareas_Automatizacion
    X_FEATURES = ['Total_Tareas', 'Tareas_Automatizacion']
    X = df_totales[X_FEATURES].values
    
    modelo_regresion = LinearRegression()
    modelo_regresion.fit(X, y)
    
    # ----------------------------------------------------
    # 2. EVALUACIÓN Y PREDICCIÓN
    # ----------------------------------------------------
    
    # Evaluar el modelo final (en el set de entrenamiento)
    y_pred = modelo_regresion.predict(X)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    # Usamos el valor REAL que viene del input del Streamlit
    new_data_multiple = np.array([[total_tareas_nuevo, automatizacion_tareas_nuevo]])

    # Predicción final
    total_defects_pred = modelo_regresion.predict(new_data_multiple)[0]
    total_defects_pred = np.maximum(1, total_defects_pred) 
    
    print("\n" + "#" * 70)
    print("      EVALUACIÓN Y PREDICCIÓN DEL MODELO FINAL (MÚLTIPLE)")
    print("#" * 70)
    print(f"   FEATURES USADAS: {X_FEATURES}")
    print(f"   R² (Poder Explicativo): {r2:.4f}")
    print(f"   RMSE (Error promedio): {rmse:.2f} defectos")
    print(f"-> Predicción final: {int(total_defects_pred)} defectos.")
    print("#" * 70 + "\n")
    
    # 3. DISTRIBUCIÓN RAYLEIGH (Curva de Riesgo)
    resultado_modelo = ModeloRayleighDistribucion(
        total_defects_pred, 
        semanas_estimadas_nuevo # Semanas aún se usa aquí
    )
    
    # Añadir las métricas al JSON de salida
    resultado_modelo["Regresion_Metrics_Final"] = {
        "R2": round(r2, 4),
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "Coeficientes": modelo_regresion.coef_.tolist(),
    }
    
    resultado_modelo["Correlacion_Defectos"] = corr_matrix['Total_Defectos'].to_dict()
    
    return json.dumps(resultado_modelo, indent=4)


# =========================================================================
# SECCIÓN 4: EJECUCIÓN DEL FLUJO COMPLETO (PRUEBA FINAL)
# =========================================================================

if __name__ == "__main__":
    
    NEW_PROJECT_TAG = "Nuevo_Proyecto_X" 
    
    # --- PARÁMETROS DEL NUEVO PROYECTO ---
    NUEVAS_TAREAS_ESTIMADAS = 40
    NUEVAS_TAREAS_AUTOMATIZACION = 20 
    NUEVAS_SEMANAS_ESTIMADAS = 16
    
    print("=" * 70)
    print(f"       DEMOSTRACIÓN CLAVE: PROYECTO NUEVO ({NEW_PROJECT_TAG})")
    print(f"       Parámetros: Tareas={NUEVAS_TAREAS_ESTIMADAS}, Automatización={NUEVAS_TAREAS_AUTOMATIZACION}, Semanas={NUEVAS_SEMANAS_ESTIMADAS}")
    print("=======================================================================")
    
    prediccion_json_str = hacer_prediccion(
        NUEVAS_TAREAS_ESTIMADAS,
        NUEVAS_TAREAS_AUTOMATIZACION, # Pasar el nuevo valor
        NUEVAS_SEMANAS_ESTIMADAS
    )
    
    print("\n" + "=" * 70)
    print("      SALIDA FINAL DE LA API/SCRIPT AL DASHBOARD (JSON)")
    print("      (Contiene la Predicción Total y la Curva de Riesgo)")
    print("=" * 70)
    print(prediccion_json_str)
    
    
    # ----------------------------------------------------
    # VISUALIZACIÓN DEL RIESGO
    # ----------------------------------------------------
    try:
        resultado = json.loads(prediccion_json_str)
        df_curva = pd.DataFrame(resultado.get('Curva_Riesgo_Rayleigh', []))
        total_defectos = resultado.get('Total_Defectos_Estimados', 0)
        
        # --- GRÁFICA DE CORRELACIÓN ---
        corr_data = resultado.get('Correlacion_Defectos', {})
        corr_series = pd.Series(corr_data).drop('Total_Defectos').sort_values(ascending=False)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=corr_series.index, y=corr_series.values, palette="viridis")
        plt.title('Correlación de Variables con Total_Defectos', fontsize=14)
        plt.xlabel('Variable Predictora', fontsize=12)
        plt.ylabel('Coeficiente de Correlación (r)', fontsize=12)
        plt.ylim(-1, 1)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

        # Gráfico de distribución de riesgo (se mantiene)
        if not df_curva.empty:
            plt.figure(figsize=(10, 6))
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