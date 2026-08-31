# Evidencia de folds compartidos

## Propósito

La comparación entre la red neuronal densa y Random Forest debe usar
exactamente las mismas observaciones de entrenamiento y prueba. En RC3 esta
condición deja de depender de una lectura informal del código y queda
materializada como evidencia verificable.

## Construcción

Para cada combinación de escenario (`a`, `aprima` o `b`) y validación se crea
una sola lista de cinco folds mediante `construir_folds()`:

- `aleatoria`: `StratifiedKFold`, con mezcla y semilla 42;
- `por_estacion`: `StratifiedGroupKFold`, con mezcla, semilla 42 y `s_id` como
  grupo.

La lista se materializa antes de recorrer los clasificadores. Dentro de cada
fold, la red densa y Random Forest reciben las mismas matrices ya imputadas.
El conjunto de prueba externo no se usa para ajustar el imputador.

Cada fold posee un `fold_id` SHA-256 calculado desde el protocolo, número de
fold, semilla, cantidad de folds e índices de train/test. Los arreglos de
índices se marcan como no modificables para evitar alteraciones accidentales
durante la ejecución.

## Artefacto generado

La ejecución completa generará `output/asignacion_folds.csv`. Cada registro
contendrá:

- escenario y tipo de validación;
- clasificador;
- número e identificador del fold;
- partición `train` o `test`;
- posición de la observación después de preparar los datos;
- número de fila original del CSV;
- estación y clase fenológica.

La misma asignación se registra una vez bajo `red_densa` y otra bajo
`random_forest`. Por tanto, un revisor puede eliminar la columna
`clasificador`, ordenar las filas y comprobar que ambas tablas sean idénticas.
El hash del CSV se incorporará a `configuracion_ejecucion.json`.

## Garantías verificadas sin entrenamiento

Las pruebas automáticas comprueban para A, A′ y B:

1. reproducibilidad con semilla 42;
2. cinco folds por protocolo;
3. ausencia de intersección entre train y test;
4. cobertura de todas las observaciones y aparición exactamente una vez en
   test;
5. ausencia de estaciones compartidas entre train y test en la validación
   agrupada;
6. igualdad completa de las asignaciones de la red y Random Forest;
7. igualdad de las matrices que el pipeline entrega a ambos entrenadores.

Las métricas por fold también incluyen `fold_id`, lo que permite relacionarlas
directamente con `asignacion_folds.csv`.

La evidencia puede regenerarse sin entrenar los clasificadores mediante:

```powershell
python ejecutar.py --solo-folds
```
