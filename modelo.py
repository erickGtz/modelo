import pandas as pd
import numpy as np
import os 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import random
import datetime # Necesario para calcular diferencias de fechas

# Nombres de los archivos de datos que deben estar en el repositorio
CSV_PROYECTOS = 'dw_proyectos.csv'
CSV_HECHOS = 'dw_hechos_proyecto.csv'
CSV_TAREAS = 'dw_tareas.csv'
CSV_INCIDENTES = 'dw_incidentes.csv' # Incluido para evitar errores de referencia, aunque no se usa directamente en el modelo

# =========================================================================
# SECCIÓN 1: EXTRACCIÓN DE DATOS (AHORA USA UNIONES DE CSV)
# =========================================================================

def obtener_totales_proyectos():
    """
    Extrae los datos históricos de múltiples CSV, realiza las uniones necesarias
    y calcula las métricas Total_Tareas y Tiempo_Semanas usando Pandas.
    """
    try:
        # --- Carga de todos los CSV ---
        df_proyectos = pd.read_csv(CSV_PROYECTOS)
        df_hechos = pd.read_csv(CSV_HECHOS)
        df_tareas = pd.read_csv(CSV_TAREAS)
        
        # --- 1. Calcular Total_Tareas (Antes: PA contar_tareas) ---
        # Contamos el número de filas (tareas) por proyecto
        # NOTA: Usamos Proyecto_idProyecto de dw_tareas.csv para agrupar
        df_total_tareas = df_tareas.groupby('Proyecto_idProyecto').size().reset_index(name='Total_Tareas')
        df_total_tareas.rename(columns={'Proyecto_idProyecto': 'idProyecto'}, inplace=True)
        
        # --- 2. Calcular Tiempo_Semanas (Antes: PA calcular_semanas_proyecto) ---
        
        # Convertir columnas de fecha a datetime
        df_proyectos['fecha_inicio'] = pd.to_datetime(df_proyectos['fecha_inicio'])
        df_proyectos['fecha_fin_real'] = pd.to_datetime(df_proyectos['fecha_fin_real'])
        
        # Si fecha_fin_real es NULL, usamos la fecha actual (simulada)
        fecha_simulada_hoy = datetime.datetime.now() # Usamos la fecha actual de ejecución
        df_proyectos['fecha_fin_calculo'] = df_proyectos['fecha_fin_real'].fillna(fecha_simulada_hoy)
        
        # Calcular la diferencia en días y convertir a semanas
        df_proyectos['dias_duracion'] = (df_proyectos['fecha_fin_calculo'] - df_proyectos['fecha_inicio']).dt.days
        # Usamos ceil para redondear hacia arriba como en el PA
        df_proyectos['Tiempo_Semanas'] = (df_proyectos['dias_duracion'] / 7).apply(np.ceil).astype(int)
        
        # --- 3. Unión de Datos (Crear la tabla de entrenamiento) ---
        
        # Unir Hechos (Métricas) con Proyectos (Fechas/Dimensiones) usando 'idProyecto'
        # Usamos HOW='left' para mantener todos los proyectos de la tabla de Hechos y luego rellenar NaNs
        df_master = pd.merge(df_hechos, df_proyectos[['idProyecto', 'Tiempo_Semanas']], on='idProyecto', how='left')
        
        # Unir con el Total de Tareas calculado (ahora debería coincidir por 'idProyecto')
        df_master = pd.merge(df_master, df_total_tareas, on='idProyecto', how='left')
        
        # Renombrar columnas para la regresión
        df_master.rename(columns={
            'defectos_reportados': 'Total_Defectos',
            'tareas_automatizacion_total': 'Tareas_Automatizacion',
            'presupuesto': 'Costo_Defectos'
        }, inplace=True)
        
        # Limpieza final y selección de columnas para el modelo (manteniendo solo las usadas)
        df_final = df_master[[
            'idProyecto', 'Tiempo_Semanas', 'Total_Defectos', 'Total_Tareas',
            'Costo_Defectos', 'Tareas_Automatizacion'
        ]].fillna(0) # Sustituir NaNs resultantes por 0 para que la regresión no falle
        
        # Convertir tipos para la regresión
        df_final[['Total_Defectos', 'Total_Tareas', 'Tareas_Automatizacion', 'Tiempo_Semanas']] = \
            df_final[['Total_Defectos', 'Total_Tareas', 'Tareas_Automatizacion', 'Tiempo_Semanas']].astype(int)

        print(f"Total de registros cargados para el entrenamiento desde CSV: {len(df_final)}")
        print(df_final.head())
        
        return df_final

    except FileNotFoundError:
        print("Error: Asegúrate de que los archivos CSV (dw_proyectos.csv, dw_hechos_proyecto.csv, dw_tareas.csv) estén en la misma carpeta.")
        return pd.DataFrame()
        
    except Exception as err:
        print(f"Error al procesar los archivos CSV: {err}")
        return pd.DataFrame()

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

def hacer_prediccion(total_tareas_nuevo, automatizacion_tareas_nuevo, semanas_estimadas_nuevo):
    
    print(f"Calculando predicción para proyecto nuevo (Tareas: {total_tareas_nuevo}, Automatización: {automatizacion_tareas_nuevo}, Semanas: {semanas_estimadas_nuevo})")
    
    df_totales = obtener_totales_proyectos()
    
    if df_totales.empty:
        return json.dumps({"Error": "No se encontraron datos históricos para entrenar el modelo."}, indent=4)

    # 1. ANÁLISIS DE CORRELACIÓN
    variables_correlacion = [
        'Total_Defectos', 'Total_Tareas', 'Tiempo_Semanas', 
        'Costo_Defectos', 'Tareas_Automatizacion'
    ]
    
    # Solo calcular correlación si hay suficientes datos (más de 2 filas)
    if len(df_totales) > 2:
        corr_matrix = df_totales[variables_correlacion].corr()
    else:
        # Si no hay suficientes datos, la correlación no es significativa
        corr_matrix = pd.DataFrame(index=variables_correlacion, columns=variables_correlacion).fillna(np.nan)


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
    
    # Solo entrenar si hay más de 2 filas para evitar el sobreajuste 100%
    if len(df_totales) <= 2:
        # Asignar R2, RMSE y MAE como no aplicable o cero
        r2, rmse, mae = 0.0, 0.0, 0.0
        # Forzar una predicción simple si el modelo no puede entrenarse correctamente
        total_defects_pred = np.mean(df_totales['Total_Defectos']) if len(df_totales) > 0 else 1 
        print("ADVERTENCIA: Modelo entrenado con <= 2 registros. La predicción será la media simple.")
    else:
        modelo_regresion = LinearRegression()
        modelo_regresion.fit(X, y)
    
        # 2. EVALUACIÓN Y PREDICCIÓN
        y_pred = modelo_regresion.predict(X)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        # Usamos el valor REAL que viene del input del Streamlit
        new_data_multiple = np.array([[total_tareas_nuevo, automatizacion_tareas_nuevo]])
        total_defects_pred = modelo_regresion.predict(new_data_multiple)[0]

    total_defects_pred = np.maximum(1, total_defects_pred) # Asegura al menos 1 defecto
    
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
        semanas_estimadas_nuevo
    )
    
    # Añadir las métricas al JSON de salida
    resultado_modelo["Regresion_Metrics_Final"] = {
        "R2": round(r2, 4),
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "Coeficientes": modelo_regresion.coef_.tolist() if 'modelo_regresion' in locals() else [np.nan, np.nan],
    }
    
    resultado_modelo["Correlacion_Defectos"] = corr_matrix['Total_Defectos'].to_dict()
    
    return json.dumps(resultado_modelo, indent=4)


# =========================================================================
# SECCIÓN 4: EJECUCIÓN DEL FLUJO COMPLETO (PRUEBA FINAL)
# =========================================================================

if __name__ == "__main__":
    
    NEW_PROJECT_TAG = "Nuevo_Proyecto_X" 
    
    # --- PARÁMETROS DEL NUEVO PROYECTO ---
    NUEVAS_TAREAS_ESTIMADAS = 35 
    NUEVAS_TAREAS_AUTOMATIZACION = 10 
    NUEVAS_SEMANAS_ESTIMADAS = 12
    
    print("=" * 70)
    print(f"       DEMOSTRACIÓN CLAVE: PROYECTO NUEVO ({NEW_PROJECT_TAG})")
    print(f"       Parámetros: Tareas={NUEVAS_TAREAS_ESTIMADAS}, Automatización={NUEVAS_TAREAS_AUTOMATIZACION}, Semanas={NUEVAS_SEMANAS_ESTIMADAS}")
    print("=======================================================================")
    
    prediccion_json_str = hacer_prediccion(
        NUEVAS_TAREAS_ESTIMADAS,
        NUEVAS_TAREAS_AUTOMATIZACION,
        NUEVAS_SEMANAS_ESTIMADAS
    )
    
    print("\n" + "=" * 70)
    print("      SALIDA FINAL DE LA API/SCRIPT AL DASHBOARD (JSON)")
    print("      (Contiene la Predicción Total y la Curva de Riesgo)")
    print("=" * 70)
    print(prediccion_json_str)