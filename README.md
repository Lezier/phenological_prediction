# Predicción del estado fenológico de la vid

Release candidate académico que compara una red neuronal densa con Random
Forest para clasificar cinco macro-etapas fenológicas de *Vitis vinifera*.
Random Forest A, entrenado con siete variables climáticas, fue seleccionado
como modelo del prototipo mediante validación cruzada agrupada por estación.

> El sistema es una maqueta experimental entrenada con datos europeos. No está
> validado para uso operacional en Chile y no sustituye el criterio agronómico.

## Resultado de selección

La comparación limpia usa cinco folds, semilla 42 e imputación aprendida solo
desde entrenamiento dentro de cada fold. La prueba principal mantiene cada
`s_id` completamente en entrenamiento o prueba.

| Modelo | Accuracy por estación | F1 macro por estación |
|---|---:|---:|
| Random Forest A - clima | **83,46% +/- 13,73%** | **72,04% +/- 11,42%** |
| Red densa A - clima | 74,92% +/- 12,89% | 64,16% +/- 13,91% |
| Random Forest A' - control | 87,13% +/- 14,42% | 70,88% +/- 8,80% |
| Random Forest B - clima + NDVI | 85,25% +/- 14,98% | 69,19% +/- 9,06% |

Se seleccionó Random Forest A porque utiliza las 1.091 observaciones climáticas,
obtiene el mejor F1 macro entre los bosques evaluados y no condiciona su uso a
la disponibilidad de NDVI. A' es un control experimental y B permite evaluar el
aporte marginal del NDVI sobre las mismas 657 filas.

## Estructura

```text
phenological_prediction/
|-- data/                         CSV de entrada
|-- models/                       modelo final serializado
|-- output/                       evidencia de evaluación y metadatos
|-- tests/                        pruebas rápidas de integridad
|-- ejecutar.py                   comparación red vs. Random Forest
|-- entrenar_modelo_final.py      entrenamiento y empaquetado del modelo A
|-- demo.py                       inferencia interactiva o por argumentos
|-- DATA_PROVENANCE.md            procedencia y advertencias de redistribución
|-- requirements.txt              dependencias exactas del RC
|-- VERSION                       versión del proyecto
|-- CHANGELOG.md                  cambios del release candidate
`-- RELEASE_MANIFEST.json         identidad y hashes de artefactos del RC
```

## Requisitos

- Windows, Linux o macOS con Python 3.13.5.
- Los dos CSV presentes en `data/`.
- Dependencias de `requirements.txt`.

Crear e instalar el entorno:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

En Linux o macOS, reemplazar `.\.venv\Scripts\python.exe` por
`.venv/bin/python`.

## 1. Reproducir la comparación

La ejecución completa entrena 3 conjuntos de datos x 2 clasificadores x 2
estrategias de validación x 5 folds:

```powershell
.\.venv\Scripts\python.exe ejecutar.py
```

También puede ejecutarse un conjunto específico:

```powershell
.\.venv\Scripts\python.exe ejecutar.py --modelo a
.\.venv\Scripts\python.exe ejecutar.py --modelo aprima
.\.venv\Scripts\python.exe ejecutar.py --modelo b
```

Los resultados se escriben en `output/`. Las fuentes principales para el
informe son:

- `comparacion_consolidada.csv`: medias y desviaciones.
- `metricas_por_fold.csv`: detalle de los 60 experimentos.
- `configuracion_ejecucion.json`: versiones, parámetros, hashes y protocolo.
- `comparacion_metricas.png`: resumen visual.
- `matriz_*.csv`: matrices de confusión acumuladas.

## 2. Entrenar el modelo final

Después de validar la comparación, entrenar Random Forest A con las 1.091 filas
disponibles:

```powershell
.\.venv\Scripts\python.exe entrenar_modelo_final.py
```

El comando genera:

- `models/random_forest_a_final.joblib`
- `output/metadata_modelo_final.json`

Este entrenamiento no calcula una nueva métrica sobre las mismas filas. El
desempeño reportado siempre corresponde a la validación cruzada agrupada.

## 3. Ejecutar la demo

Modo interactivo:

```powershell
.\.venv\Scripts\python.exe demo.py
```

Modo reproducible con siete valores en el orden documentado:

```powershell
.\.venv\Scripts\python.exe demo.py --valores 18.5 24 12 15 18.2 62 210
```

La demo muestra la macro-etapa estimada, probabilidades no calibradas para las
cinco clases, advertencias por valores fuera del rango de entrenamiento y el
límite de uso. Esos valores sirven para describir la votación relativa del
bosque y no deben interpretarse como certeza estadística calibrada.

## 4. Ejecutar pruebas rápidas

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Las pruebas verifican integridad de datos, tamaño del conjunto A, carga del
artefacto, orden de variables y coherencia de probabilidades.

## Metodología y controles

- Validación principal: `StratifiedGroupKFold` por `s_id`, cinco folds.
- Comparación histórica: `StratifiedKFold` aleatoria, cinco folds.
- Imputación: mediana ajustada exclusivamente con entrenamiento en cada fold.
- Escalado de la red: ajustado exclusivamente con entrenamiento.
- Balance: pesos de clase en la red y `class_weight="balanced"` en Random Forest.
- Semilla: 42.
- Random Forest: 400 árboles, parámetros registrados en los metadatos.

## Limitaciones

- Los datos europeos no permiten afirmar aplicabilidad operacional en Chile.
- La validación por estación no deja temporadas completas fuera de entrenamiento.
- Floración tiene cobertura geográfica limitada y está ausente en algunos folds.
- La desviación entre folds evidencia heterogeneidad territorial.
- El NDVI disponible no mejoró consistentemente A' frente a B.
- El modelo clasifica una etapa alrededor de la fecha observada; no demuestra por
  sí solo una predicción prospectiva con anticipación definida.

## Estado del release

Versión: `0.1.0-rc.2`.

El RC incluye evidencia reproducible, modelo final, demo y controles mínimos.
La publicación pública de los CSV queda condicionada a revisar las reglas de
redistribución descritas en `DATA_PROVENANCE.md`. No se ha definido todavía una
licencia para el código.

`rc.2` corrige exclusivamente la representación de finales de línea de los CSV
entre clones Git. No modifica observaciones, métricas, parámetros ni el modelo.
`RELEASE_MANIFEST.json` vincula el candidato con la evidencia y el modelo
generados en `rc.1`, conservando su procedencia sin simular una nueva ejecución.
