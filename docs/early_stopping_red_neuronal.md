# Tratamiento conservador de Early Stopping

## Contexto de decisión

Al cerrar CP06 no se había recibido respuesta del equipo sobre la procedencia o
posible ajuste de los hiperparámetros de la red neuronal. RC3 no atribuye esos
valores a una persona o herramienta, no afirma que provengan de tuning y no
construye una justificación histórica retrospectiva.

Para conservar la comparabilidad con el baseline heredado se mantienen:

- máximo de 60 épocas;
- `EarlyStopping(monitor="loss")`;
- `patience=8`;
- `restore_best_weights=True`.

## Interpretación correcta

El callback observa solamente la pérdida de entrenamiento. Su función en RC3
es detener un ajuste que dejó de mejorar y restaurar los mejores pesos según
ese mismo criterio.

Esto implica que:

- no consulta el fold externo de test;
- no introduce fuga de información desde test;
- no requiere separar una validación interna;
- no debe describirse como una demostración de control del overfitting;
- puede reaccionar a fluctuaciones de la pérdida de entrenamiento producidas
  por el entrenamiento estocástico y dropout.

La ausencia de una validación interna es una decisión conservadora para no
cambiar el protocolo ni generar nuevos grados de libertad antes de la corrida
final. Una evaluación futura basada en `val_loss` requeriría definir y congelar
un subconjunto interno dentro de cada train externo.

## Evidencia de ejecución

Las pruebas comprueban que:

1. `entrenar_red()` no recibe `y_test`;
2. `fit()` no recibe `validation_data` ni `validation_split`;
3. el callback conserva monitor, paciencia y restauración declarados;
4. `x_test` solo se transforma y utiliza después de terminar `fit()`;
5. cada fila de métricas registra las épocas ejecutadas y si la detención fue
   anticipada.

## Declaración para el informe

> La red densa se conserva como baseline experimental heredado. Su
> configuración permanece fija para preservar la comparabilidad, pero no
> existe evidencia disponible de optimización sistemática ni confirmación
> completa de su procedencia. Early Stopping se utiliza únicamente como control
> de convergencia sobre la pérdida de entrenamiento y no consulta el fold
> externo de prueba.
