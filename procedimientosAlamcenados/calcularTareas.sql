DELIMITER $$
CREATE PROCEDURE contar_tareas(
    IN p_idProyecto INT,
    OUT p_total INT
)
BEGIN
    SELECT COUNT(*) INTO p_total
    FROM dim_tarea
    WHERE idProyecto = p_idProyecto;
END$$
DELIMITER ;
