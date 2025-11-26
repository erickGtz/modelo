DELIMITER $$

CREATE PROCEDURE calcular_semanas_proyecto(IN p_idProyecto INT, OUT p_semanas INT)
BEGIN
    DECLARE fechaInicio DATE;
    DECLARE fechaFinReal DATE;

    -- Obtener fechas del proyecto
    SELECT fecha_inicio, fecha_fin_real
    INTO fechaInicio, fechaFinReal
    FROM dim_proyecto
    WHERE idProyecto = p_idProyecto;

    -- Si fecha_fin_real es NULL, usar fecha actual
    IF fechaFinReal IS NULL THEN
        SET fechaFinReal = CURDATE();
    END IF;

    -- Calcular semanas (ceil para redondear hacia arriba)
    SET p_semanas = CEIL(DATEDIFF(fechaFinReal, fechaInicio) / 7);
END$$

DELIMITER ;

-- LLAMAR así:
-- Declarar variable para el resultado
--SET @semanas = 0;

-- Llamar al procedimiento para proyecto con id = 3
--CALL calcular_semanas_proyecto(4, @semanas);

-- Ver resultado
--SELECT @semanas AS semanas;