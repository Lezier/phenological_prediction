# Phenological Prediction

Pipeline reproducible para clasificación de macro-etapas fenológicas de
*Vitis vinifera*. El repositorio conserva la comparación experimental entre
una red neuronal densa y Random Forest, y empaqueta **Random Forest A** como
modelo de inferencia del prototipo.

> Uso experimental: los datos son europeos y el modelo no ha sido validado
> geográfica ni temporalmente para Chile. No constituye una recomendación
> agronómica ni debe integrarse en decisiones productivas sin validación
> adicional.

## Estado

| Componente | Estado RC3 |
|---|---|
| Versión | `0.1.0-rc.3` |
| Comparación oficial | Ejecutada y auditada en CP09–CP10 |
| Modelo de inferencia | Random Forest A entrenado con 1.091 observaciones |
| Contrato del paquete | Esquema 2; siete variables y cinco clases |
| Suite actual | 41 pruebas, todas aprobadas después del congelamiento CP13 |
| Manifiesto general | RC3; controla datos, modelo y resultados oficiales |
| Commit RC3 | Pendiente de creación por el responsable del repositorio |

El estado `rc` identifica un pre-release. Los resultados, el modelo, la
metadata y el manifiesto corresponden a RC3. La identificación por commit se
completará cuando el responsable del repositorio registre este estado.

## Diseño del sistema

```text
data/*.csv
    │
    ├── ejecutar.py
    │     ├── folds.py                particiones compartidas
    │     ├── ponderacion.py          pesos calculados solo con train
    │     ├── medicion.py             tiempos por fold
    │     └── output/                 métricas, matrices y evidencias
    │
    ├── entrenar_modelo_final.py
    │     └── models/random_forest_a_final.joblib
    │         + output/metadata_modelo_final.json
    │
    └── demo.py                       contrato de inferencia
```

La configuración efectiva y su procedencia se centralizan en
[`configuracion.py`](configuracion.py). La evaluación y el entrenamiento final
son procesos separados:

- `ejecutar.py` estima desempeño mediante validación cruzada;
- `entrenar_modelo_final.py` ajusta el modelo seleccionado con todas las filas
  de A para generar un artefacto utilizable, sin producir una nueva estimación
  de desempeño.

## Escenarios experimentales

| Escenario | Variables | Muestras | Función |
|---|---:|---:|---|
| A | 7 climáticas | 1.091 | Candidato con máxima cobertura disponible |
| A′ | Las mismas 7 climáticas | 657 | Control sobre las filas con NDVI |
| B | 7 climáticas + NDVI | 657 | Evaluación del aporte satelital |

A′ y B contienen exactamente las mismas observaciones. Esto permite atribuir
sus diferencias a la incorporación de NDVI dentro del experimento, sin mezclar
el efecto con un cambio de muestra.

## Protocolo de evaluación

- Cinco folds con semilla 42.
- Protocolo principal: `StratifiedGroupKFold` agrupado por `s_id`.
- Control secundario: `StratifiedKFold` aleatorio.
- Los dos clasificadores reutilizan la misma asignación de cada fold.
- Imputadores, escaladores y ponderaciones se ajustan solo con train.
- La red densa conserva el baseline heredado y su configuración documentada.
- Random Forest usa 400 árboles, `class_weight="balanced"`,
  `random_state=42` y `n_jobs=-1`; no hubo tuning sistemático.

Resultados principales de A con validación agrupada por estación:

| Modelo | Accuracy | F1 macro | F1 weighted | Train medio | Inferencia media |
|---|---:|---:|---:|---:|---:|
| Random Forest | **0,834630 ± 0,137267** | **0,720435 ± 0,114232** | **0,838786 ± 0,129906** | 0,492875 s | 0,064380 s |
| Red densa | 0,749204 ± 0,128915 | 0,641594 ± 0,139132 | 0,747154 ± 0,134848 | 6,001967 s | 0,109457 s |

Random Forest A se selecciona por el balance entre desempeño pareado, cobertura,
velocidad, simplicidad y ausencia de dependencia de NDVI. En el subconjunto
comparable y bajo validación agrupada, B queda por debajo de A′ en accuracy y
F1 macro; este experimento no demuestra beneficio por agregar NDVI.

La interpretación completa se encuentra en
[`docs/auditoria_resultados_cp10.md`](docs/auditoria_resultados_cp10.md).

## Requisitos del entorno

- Python 3.13.x; artefacto final generado con Python 3.13.5.
- Dependencias fijadas en [`requirements.txt`](requirements.txt).
- Entorno verificado: NumPy 2.5.2, pandas 2.3.3, scikit-learn 1.6.1,
  joblib 1.5.3 y TensorFlow 2.21.0.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

macOS o Linux:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Los objetos joblib no garantizan compatibilidad binaria entre versiones
arbitrarias de scikit-learn y nunca deben cargarse desde fuentes no confiables.

## Contrato de inferencia

El artefacto [`models/random_forest_a_final.joblib`](models/random_forest_a_final.joblib)
empaqueta el pipeline, orden de variables, clases, rangos observados, hash de
datos, versión de proyecto y advertencias.

Orden posicional del CLI:

| Posición | Variable | Unidad |
|---:|---|---|
| 1 | `clima_temp_media` | °C |
| 2 | `clima_temp_max_media` | °C |
| 3 | `clima_temp_min_media` | °C |
| 4 | `clima_precip_acumulada` | mm/ventana |
| 5 | `clima_radiacion_media` | MJ/m²/día |
| 6 | `clima_humedad_media` | % |
| 7 | `clima_gdd_acumulado` | °C·día, base 10 °C |

Inferencia por CLI:

```powershell
python demo.py --valores 18.5 24 12 15 18.2 62 210
```

Inferencia programática con nombres —preferida sobre la interfaz posicional—:

```python
from demo import cargar_paquete, generar_prediccion

paquete = cargar_paquete()
resultado = generar_prediccion(
    paquete,
    {
        "clima_temp_media": 18.5,
        "clima_temp_max_media": 24.0,
        "clima_temp_min_media": 12.0,
        "clima_precip_acumulada": 15.0,
        "clima_radiacion_media": 18.2,
        "clima_humedad_media": 62.0,
        "clima_gdd_acumulado": 210.0,
    },
)
```

La respuesta contiene `macro_etapa`, un diccionario de cinco
`probabilidades`, `variables_fuera_rango`, `advertencia` y
`probabilidades_calibradas=False`. Los valores de `predict_proba` son puntajes
no calibrados y no representan una certeza operacional garantizada.

## Flujos ejecutables

### Suite automática

```powershell
python -m unittest discover -s tests -v
```

El resultado esperado para el conjunto congelado en CP13 es `OK`: 41 pruebas,
sin omisiones, fallos ni errores.

### Demo interactiva

```powershell
python demo.py
```

### Ensayo reducido no oficial

```powershell
python ejecutar.py --modelo a --ensayo-reducido
```

Genera un fold real aislado en `output/ensayo_reducido/`. Sus métricas no deben
incorporarse al informe ni sustituir la corrida oficial.

### Comparación completa

```powershell
python ejecutar.py
```

Ejecuta A, A′ y B; dos validaciones; cinco folds; y ambos clasificadores. El
comando sobrescribe los resultados oficiales de `output/`, por lo que debe
usarse deliberadamente y auditarse nuevamente.

Restricción por escenario:

```powershell
python ejecutar.py --modelo a
python ejecutar.py --modelo aprima
python ejecutar.py --modelo b
```

### Regeneración del modelo final

```powershell
python entrenar_modelo_final.py
```

Sobrescribe el `.joblib`, la metadata y su log. Después de ejecutarlo deben
recalcularse los hashes del manifiesto. No es necesario para consumir el
artefacto entregado.

## Artefactos

| Ruta | Contrato |
|---|---|
| `output/metricas_por_fold.csv` | 60 evaluaciones con métricas, tamaños, clases y tiempos |
| `output/comparacion_consolidada.csv` | 12 combinaciones con media y desviación muestral |
| `output/asignacion_folds.csv` | evidencia por fila de particiones compartidas |
| `output/pesos_clase_por_fold.csv` | frecuencias y pesos calculados desde train |
| `output/tiempos_por_fold.csv` | entrenamiento e inferencia por fold |
| `output/matriz_*.csv` | 12 matrices de confusión acumuladas |
| `output/configuracion_ejecucion.json` | entorno, parámetros y hashes de evidencias |
| `output/metadata_modelo_final.json` | contrato, entorno y hash del modelo final |
| `output/ejecucion_completa_rc3.log` | log de la comparación oficial |
| `output/entrenamiento_modelo_final_rc3.log` | log del entrenamiento final |

`RELEASE_MANIFEST.json` declara los hashes del modelo, los datos y los
resultados oficiales RC3. Excluye intencionalmente `output/ensayo_reducido/`,
porque ese directorio contiene una prueba acotada que no debe utilizarse en el
informe. El manifiesto tampoco se incluye a sí mismo, para evitar una
autorreferencia imposible.

## Estructura del repositorio

```text
phenological_prediction/
├── data/                       CSV derivados
├── docs/                       decisiones y auditorías técnicas
├── models/                     paquete de inferencia
├── output/                     evidencias y resultados
├── tests/                      pruebas unitarias y de integridad
├── configuracion.py            configuración central RC3
├── ejecutar.py                 evaluación cruzada
├── entrenar_modelo_final.py    entrenamiento para inferencia
├── demo.py                     interfaz de inferencia
├── DATA_LICENSE.md             condiciones de los datos
├── DATA_PROVENANCE.md          linaje y transformaciones
├── CODE_LICENSE.md             situación jurídica del código
├── RELEASE_MANIFEST.json       integridad de datos y artefactos oficiales RC3
├── requirements.txt
└── VERSION
```

## Limitaciones técnicas

- La caída entre validación aleatoria y agrupada evidencia sensibilidad
  territorial; en Random Forest A es 0,066359 de accuracy y 0,174911 de F1
  macro.
- Los folds agrupados 1, 2 y 4 de A no contienen Floración en test.
- La validación no reserva temporadas completas ni una región externa.
- No hubo búsqueda sistemática de hiperparámetros.
- La red densa se conserva como baseline heredado, no como arquitectura
  demostrada óptima.
- Las probabilidades no fueron calibradas ni evaluadas mediante Brier score,
  log-loss o curvas de calibración.
- Los CSV son snapshots procesados; este repositorio no reproduce por sí solo
  la descarga original desde PEP725, NASA POWER y Copernicus.

## Datos, código y documentación

Los CSV se distribuyen para uso académico no comercial bajo las condiciones
detalladas en [`DATA_LICENSE.md`](DATA_LICENSE.md) y el linaje de
[`DATA_PROVENANCE.md`](DATA_PROVENANCE.md). El campo `license_short` de todas
las observaciones PEP725 entregadas declara `CC BY-NC 4.0`.

El código no tiene una licencia abierta concedida; consulte
[`CODE_LICENSE.md`](CODE_LICENSE.md). La licencia de datos no se extiende al
código ni al modelo por inferencia automática.

La guía de recepción y ejecución destinada a la entrega se mantiene fuera del
ZIP del proyecto como **Anexo A — Guía de recepción y consumo del entregable
técnico**. Este README comienza con el repositorio ya disponible y se concentra
en su contrato de ingeniería.

Fuentes metodológicas y de uso:

- [PEP725 Dataset](https://pep725.eu/dataset/)
- [PEP725 Data Use Policy](https://pep725.eu/pep725_data_use_policy/)
- [NASA POWER Referencing Guide](https://power.larc.nasa.gov/docs/referencing/)
- [Copernicus Data Space Terms](https://dataspace.copernicus.eu/terms-and-conditions)

## Documentación técnica

- [`docs/decision_modelo.md`](docs/decision_modelo.md)
- [`docs/auditoria_resultados_cp10.md`](docs/auditoria_resultados_cp10.md)
- [`docs/modelo_final_cp11.md`](docs/modelo_final_cp11.md)
- [`docs/folds_compartidos.md`](docs/folds_compartidos.md)
- [`docs/ponderacion_clases.md`](docs/ponderacion_clases.md)
- [`docs/tiempos_y_dispersion.md`](docs/tiempos_y_dispersion.md)
- [`docs/hiperparametros_random_forest.md`](docs/hiperparametros_random_forest.md)
- [`docs/hiperparametros_red_neuronal.md`](docs/hiperparametros_red_neuronal.md)
- [`docs/early_stopping_red_neuronal.md`](docs/early_stopping_red_neuronal.md)

## Advertencias conocidas

- TensorFlow 2.21.0 no utiliza GPU en Windows nativo; la comparación oficial se
  ejecutó en CPU.
- TensorFlow puede informar retracing durante la validación cruzada. Es una
  advertencia de rendimiento y no invalidó la corrida completada.
- joblib 1.5.3 puede emitir un `DeprecationWarning` al cargar con NumPy 2.5.2.
  El artefacto fue cargado y probado correctamente con las versiones fijadas.
- Las advertencias no sustituyen el criterio de salida: cualquier `FAILED`,
  `ERROR`, excepción no controlada o hash distinto debe investigarse.
