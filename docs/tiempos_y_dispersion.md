# Tiempos y dispersión entre folds

## Métricas reportadas

RC3 conserva una fila por clasificador y fold en
`output/metricas_por_fold.csv`. Además de accuracy, F1 macro y F1 ponderado,
cada fila contiene:

- `tiempo_entrenamiento_segundos`;
- `tiempo_inferencia_segundos`;
- `tiempo_total_segundos`.

La selección de columnas temporales también se exportará a
`output/tiempos_por_fold.csv` para facilitar su revisión.

`output/comparacion_consolidada.csv` reportará para cada combinación de
escenario, validación y clasificador:

- cantidad de folds;
- media y desviación estándar muestral de accuracy;
- media y desviación estándar muestral de F1 macro;
- media y desviación estándar muestral de F1 ponderado;
- media y desviación estándar muestral de entrenamiento, inferencia y tiempo
  total.

## Alcance del cronometraje

Se usa `time.perf_counter()`, un reloj monotónico apropiado para medir
duraciones cortas.

La medición excluye las operaciones comunes realizadas antes de comparar los
clasificadores: creación del fold, imputación compartida y cálculo de pesos.
Esto evita atribuir dos veces el mismo costo.

Para la red densa:

- entrenamiento comprende ajuste y transformación de train con el escalador,
  construcción/compilación de la red y `fit()`;
- inferencia comprende transformación de test con el escalador y `predict()`.

Para Random Forest:

- entrenamiento comprende construcción del estimador y `fit()`;
- inferencia comprende `predict()`.

## Interpretación

Los tiempos dependen del procesador, carga del sistema, versiones de librerías
y disponibilidad de aceleración. Permiten comparar los modelos dentro de una
misma ejecución, pero no deben presentarse como una garantía universal de
rendimiento.

Los valores temporales no intervienen en `fold_id`, hashes de datos ni otras
identidades deterministas. La corrida oficial de CP09 generará las duraciones
definitivas del entorno informado en `configuracion_ejecucion.json`.
