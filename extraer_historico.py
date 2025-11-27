import pandas as pd
import mysql.connector

def obtener_dataset_proyectos():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="db_soporte"
    )
    cursor = conn.cursor(dictionary=True)

    # ======= lista de proyectos ==========
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

        # ===========================
        # Tamaño 
        # ===========================
        cursor2 = conn.cursor()       
        args = [idp, 0]               
        resultado = cursor2.callproc("contar_tareas", args)
        total_tareas = resultado[1]   
        cursor2.close()

        # ============================
        # Tiempo 
        # ============================
        cursor2 = conn.cursor()
        args = [idp, 0]
        resultado = cursor2.callproc("calcular_semanas_proyecto", args)
        semanas = resultado[1]
        cursor2.close()

        # =============================
        # Defectos 
        # =============================
        defectos = proyecto["defectos_reportados"]

        registros.append({
            "Proyecto": proyecto["nombre_proyecto"],
            "Tamaño (# Actividades)": total_tareas,
            "Tiempo (semanas)": semanas,
            "#Defectos totales": defectos
        })

    df = pd.DataFrame(registros)

    cursor.close()
    conn.close()
    return df


df = obtener_dataset_proyectos()
print(df)
