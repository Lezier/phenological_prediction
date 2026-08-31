# Auditoría de resultados RC3 — CP10

## Dictamen

**APROBADO con limitaciones declaradas.** La ejecución oficial RC3 es internamente
consistente y sustenta la elección de Random Forest A para el prototipo. La
aprobación no equivale a validación externa, calibración probabilística ni
demostración de desempeño universal fuera de las estaciones observadas.

## Alcance y fuentes

La auditoría se realizó sobre los artefactos oficiales generados en CP09:
`metricas_por_fold.csv`, `comparacion_consolidada.csv`,
`tiempos_por_fold.csv`, `asignacion_folds.csv`,
`pesos_clase_por_fold.csv`, las 12 matrices de confusión y
`configuracion_ejecucion.json`. También se contrastaron las métricas con la
línea base RC2 conservada en Git.

No se reentrenaron modelos ni se modificó el protocolo durante esta auditoría.

## Controles realizados

### Agregaciones y dispersión

- Se recalcularon media y desviación estándar muestral desde las 60 filas de
  `metricas_por_fold.csv`.
- Las 12 filas de `comparacion_consolidada.csv` coinciden con el recálculo. La
  máxima diferencia numérica fue `2,22 × 10⁻¹⁶`, atribuible a representación de
  punto flotante y menor que la tolerancia `1 × 10⁻¹²`.
- Los tiempos del CSV dedicado coinciden exactamente con los tiempos contenidos
  en las métricas por fold y todos son positivos.

### Folds compartidos

- Se auditaron 30 combinaciones de escenario, validación y fold.
- En cada fold, Random Forest y red densa comparten `fold_id`, número de filas y
  asignación exacta de train/test.
- En la validación agrupada no existe intersección de estaciones entre train y
  test.
- La evidencia contiene 48.100 asignaciones y concuerda con los tamaños de A
  (1.091 muestras) y A'/B (657 muestras).

### Ponderación de clases

- Las 300 filas de evidencia confirman que ambos clasificadores reciben el mismo
  peso para la misma clase y fold.
- Todos los pesos fueron calculados exclusivamente con train.
- La fórmula auditada es
  `n_train / (n_clases_presentes_train × frecuencia_clase_train)`; la máxima
  diferencia frente al recálculo fue `4,44 × 10⁻¹⁶`.

### Matrices de confusión

- Las 12 matrices tienen dimensión `5 × 5`, valores enteros no negativos y
  etiquetas de las cinco clases.
- Cada matriz suma 1.091 observaciones para A o 657 para A'/B, según corresponde.

### Comparación con RC2

Las medias y desviaciones de accuracy y F1 de RC3 son idénticas a las de RC2.
RC3 agrega evidencia de folds, pesos, tiempos y procedencia; no cambia las
métricas históricas de desempeño.

## Resultado que fundamenta el prototipo

La referencia principal es A con validación agrupada por estación, porque usa
las siete variables climáticas del prototipo y evalúa separación geográfica por
estación.

| Modelo | Accuracy | F1 macro | F1 weighted | Entrenamiento medio | Inferencia media |
|---|---:|---:|---:|---:|---:|
| Random Forest A | 0,834630 ± 0,137267 | 0,720435 ± 0,114232 | 0,838786 ± 0,129906 | 0,492875 s | 0,064380 s |
| Red densa A | 0,749204 ± 0,128915 | 0,641594 ± 0,139132 | 0,747154 ± 0,134848 | 6,001967 s | 0,109457 s |

Sobre exactamente los mismos folds, Random Forest supera a la red densa en
`0,085426` de accuracy, `0,078841` de F1 macro y `0,091632` de F1 weighted. La
red tarda aproximadamente `12,18×` más en entrenar y `1,70×` más en inferir.

En el subconjunto comparable, agregar NDVI no demuestra beneficio bajo
validación agrupada: B queda por debajo de A' en `0,018786` de accuracy,
`0,016948` de F1 macro y `0,020507` de F1 weighted. Por ello, B no justifica su
mayor requerimiento de datos para este prototipo.

A' obtiene una accuracy agrupada superior a A, pero cubre solo 657 muestras y
38 estaciones. No sustituye a A como prototipo: funciona como control para
comparar de manera justa el aporte de NDVI, mientras A conserva 1.091 muestras,
41 estaciones y mayor cobertura.

## Limitaciones y lectura crítica

- La validación aleatoria es optimista respecto de la agrupada. En Random
  Forest A, pasar a validación por estación reduce accuracy en `0,066359` y F1
  macro en `0,174911`.
- La dispersión agrupada es alta, por lo que el desempeño depende de las
  estaciones incluidas en test.
- En los folds agrupados 1, 2 y 4 de A no hay ejemplos de Floración en test.
  Esto refleja la distribución disponible y debe considerarse al interpretar
  F1 macro y las matrices; no se debe ocultar ni completar artificialmente.
- No hubo tuning sistemático de Random Forest. Su selección es relativa a la
  configuración y alternativas evaluadas, no una afirmación de optimalidad.
- `predict_proba` de Random Forest no fue calibrado ni se evaluó con métricas de
  calibración. Las probabilidades deben presentarse como puntajes del modelo,
  no como probabilidades clínicas u operacionales garantizadas.
- No existe validación externa en nuevas zonas de Chile. El prototipo no debe
  usarse para decisiones productivas reales sin validación geográfica y
  temporal adicional.

## Decisión

Se mantiene **Random Forest A** como modelo del prototipo por ofrecer el mejor
balance observado entre desempeño pareado frente a la red densa, cobertura de
datos, ausencia de dependencia de NDVI, velocidad y simplicidad operativa.

El siguiente paso es CP11: entrenar el artefacto final de inferencia con las
1.091 filas de A, conservando la configuración declarada y separando claramente
esa fase de entrenamiento final de la estimación de desempeño obtenida mediante
validación cruzada.
