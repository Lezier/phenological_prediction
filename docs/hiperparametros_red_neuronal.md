# Estado de los hiperparámetros de la red neuronal

## Estado de trazabilidad

La procedencia y justificación original de los hiperparámetros de la red
neuronal no pudieron confirmarse porque al cierre de CP06 no se había recibido
respuesta del equipo. No corresponde atribuir los valores definitivamente a
una persona o herramienta ni construir una justificación retrospectiva.

## Configuración observada en la línea base

| Elemento | Valor observado | Procedencia confirmada |
|---|---:|---|
| Primera capa densa | 16 neuronas, ReLU | No confirmada |
| Primer dropout | 0,3 | No confirmada |
| Segunda capa densa | 8 neuronas, ReLU | No confirmada |
| Segundo dropout | 0,2 | No confirmada |
| Salida | `softmax` | Coherente con clasificación multiclase; origen exacto pendiente |
| Optimizador | Adam | No confirmado |
| Learning rate | 0,001 | No confirmado |
| Épocas máximas | 60 | No confirmado |
| Batch size | 16 | No confirmado |
| Early Stopping | `patience=8`, monitor `loss` | No confirmado |

## Información solicitada al equipo

Se requiere determinar:

1. Si los valores fueron solicitados expresamente, extraídos de un documento,
   generados por Claude u otro LLM, o modificados posteriormente.
2. Si existió tuning, comparación o prueba de sensibilidad.
3. Qué criterio se utilizó para capas, neuronas, dropout, optimizador, learning
   rate, épocas, batch size y patience.
4. Si el uso de Early Stopping sobre `loss` fue una decisión consciente de
   control de convergencia o una configuración provisional.

## Regla para RC3

La red neuronal se conserva como comparador experimental. Cualquier cambio
metodológico que pueda alterar sus resultados debe resolverse y documentarse
antes de la ejecución completa de RC3. El conjunto test externo no puede usarse
para seleccionar épocas ni ajustar hiperparámetros.

La ponderación de clases no se infiere desde test: RC3 calcula un diccionario
balanceado exclusivamente desde `y_train` de cada fold y entrega exactamente
ese mismo diccionario a la red y a Random Forest. Las frecuencias y los valores
aplicados quedan registrados en `pesos_clase_por_fold.csv`.

RC3 conserva el Early Stopping observado como control de convergencia sobre
`loss` de entrenamiento. No utiliza validación interna ni el test externo y no
se presenta como evidencia de control del overfitting. La decisión completa se
documenta en `early_stopping_red_neuronal.md`.
