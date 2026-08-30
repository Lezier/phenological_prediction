# Predicción de la macro-etapa fenológica de la vid

Proyecto académico en Python que clasifica cinco macro-etapas fenológicas de
*Vitis vinifera* a partir de variables climáticas. La solución compara una red
neuronal densa con Random Forest y utiliza **Random Forest A** como modelo del
prototipo.

> **Advertencia de uso:** es una maqueta experimental entrenada con datos
> europeos. No está validada para operación en Chile, no realiza una
> recomendación agronómica y no sustituye la evaluación de un especialista.

## Estado del entregable

| Campo | Estado |
|---|---|
| Versión | `0.1.0-rc.2` |
| Estado | Pre-release académico verificado desde un clon limpio |
| Commit verificado | `96b4e94687e8ff0aa7f904509ec0c2cdb4f0751d` |
| Modelo del prototipo | Random Forest A, solo clima |
| Entrenamiento final | 1.091 observaciones y 41 estaciones |
| Entradas de la demo | 7 variables climáticas |
| Salida | 1 macro-etapa y probabilidades no calibradas para 5 clases |
| Pruebas automáticas | 3/3 aprobadas en la verificación limpia |

El desarrollo funcional está completo: el repositorio contiene datos, código,
resultados, modelo entrenado, demo y pruebas. Continúa identificado como
**release candidate** (`rc`) porque todavía no ha sido promovido a la versión
final `0.1.0`. `rc.2` corrige portabilidad de finales de línea y no cambia los
datos, las métricas, los parámetros ni el modelo de `rc.1`.

## Qué puede hacer una persona con este proyecto

- Ejecutar una demostración con el modelo ya entrenado.
- Introducir siete valores climáticos y obtener una macro-etapa probable.
- Consultar la distribución de votos del bosque para las cinco clases.
- Ejecutar pruebas automáticas de integridad.
- Revisar las métricas y matrices ya generadas.
- Opcionalmente, reproducir la comparación completa entre modelos.
- Opcionalmente, volver a entrenar el artefacto final.

**Para probar el prototipo no es necesario entrenar nuevamente.** El modelo
`models/random_forest_a_final.joblib` ya está incluido.

## Inicio rápido para una persona con conocimientos básicos de Python

Este README comienza cuando el archivo de entrega ya fue extraído por completo.
La recepción y extracción del paquete se explican en la guía externa entregada
junto con el archivo ZIP. Desde este punto, todas las instrucciones se ejecutan
dentro de la carpeta extraída `phenological_prediction`.

### 1. Confirmar que está en la carpeta del proyecto

En Windows, abra la carpeta ya extraída `phenological_prediction`, haga clic en
la barra de direcciones del Explorador, escriba `powershell` y presione Enter.
La terminal debe quedar ubicada dentro de la carpeta donde se encuentra este
`README.md`.

Puede comprobarlo con:

```powershell
Get-Location
Get-ChildItem
```

Entre los archivos mostrados deberían aparecer `demo.py`, `requirements.txt`
y `VERSION`.

### 2. Comprobar la versión de Python

```powershell
python --version
```

El proyecto fue verificado con **Python 3.13.5**. Una versión `3.13.x` es la
opción recomendada. Si Windows indica que `python` no existe, instale Python
3.13 desde su distribuidor habitual, marque la opción para añadir Python al
`PATH`, cierre la terminal y ábrala nuevamente.

En algunos equipos Windows el comando disponible es `py`. Puede comprobarlo
con:

```powershell
py -3.13 --version
```

Si utiliza `py -3.13`, úselo también al crear el entorno del paso siguiente.

### 3. Crear un entorno virtual

Un **entorno virtual** es una carpeta privada que contiene una instalación de
Python y las librerías necesarias para este proyecto. Evita mezclar TensorFlow,
NumPy o scikit-learn con las versiones instaladas por otros programas.

Crear el entorno no modifica el Python general del computador:

```powershell
python -m venv .venv
```

Si en el paso anterior usó `py -3.13`, ejecute:

```powershell
py -3.13 -m venv .venv
```

El comando crea una carpeta llamada `.venv`. Solo debe crearse una vez.

### 4. Activar el entorno virtual

Activar significa indicarle a la terminal que, mientras esa ventana permanezca
abierta, debe usar el Python de `.venv`.

#### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

Cuando se activa correctamente, el inicio de la línea suele mostrar:

```text
(.venv) PS C:\ruta\phenological_prediction>
```

Si PowerShell bloquea el script de activación, habilítelo solamente para la
ventana actual y vuelva a intentarlo:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Esta autorización termina al cerrar la ventana y no cambia permanentemente la
política del computador.

#### Windows CMD

```bat
.venv\Scripts\activate.bat
```

#### macOS o Linux

```bash
source .venv/bin/activate
```

La activación se realiza cada vez que se abre una terminal nueva. No es
necesario volver a crear el entorno ni reinstalar las dependencias cada vez.

Para salir del entorno:

```text
deactivate
```

### 5. Instalar las dependencias

Con `(.venv)` visible al comienzo de la terminal:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

La descarga requiere conexión a internet y puede tardar varios minutos porque
incluye TensorFlow. Al finalizar, compruebe que no existan dependencias rotas:

```powershell
python -m pip check
```

La respuesta esperada es:

```text
No broken requirements found.
```

### 6. Ejecutar las pruebas automáticas

```powershell
python -m unittest discover -s tests -v
```

La ejecución correcta termina con:

```text
Ran 3 tests
OK
```

Una advertencia `DeprecationWarning` de NumPy o Joblib no equivale a una prueba
fallida. El resultado final debe ser `OK`.

### 7. Ejecutar la demo

Modo interactivo, solicitando los valores uno por uno:

```powershell
python demo.py
```

Modo reproducible con un ejemplo ya documentado:

```powershell
python demo.py --valores 18.5 24 12 15 18.2 62 210
```

La salida esperada para ese ejemplo comienza así:

```text
Resultado experimental
Macro-etapa estimada: Cosecha / post-cosecha
Probabilidades estimadas (no calibradas):
  - Cosecha / post-cosecha: 38.2%
  - Senescencia / caída de hojas: 35.5%
  - Floración: 19.8%
  - Envero / maduración: 4.2%
  - Brotación / desarrollo foliar: 2.2%
```

Las probabilidades representan la votación relativa de los árboles. No son una
medida de certeza estadística calibrada.

## Entradas de la demo

Los siete números se ingresan siempre en el siguiente orden. Resumen una
ventana de 30 días: los 29 días anteriores y el día del evento.

| Posición | Variable | Interpretación | Unidad usada por el pipeline | Ejemplo |
|---:|---|---|---|---:|
| 1 | `clima_temp_media` | Temperatura media de la ventana | °C | 18.5 |
| 2 | `clima_temp_max_media` | Promedio de las temperaturas máximas | °C | 24 |
| 3 | `clima_temp_min_media` | Promedio de las temperaturas mínimas | °C | 12 |
| 4 | `clima_precip_acumulada` | Precipitación acumulada | mm en la ventana | 15 |
| 5 | `clima_radiacion_media` | Radiación solar media diaria | MJ/m²/día | 18.2 |
| 6 | `clima_humedad_media` | Humedad relativa media | % | 62 |
| 7 | `clima_gdd_acumulado` | Grados-día acumulados con base 10 °C | °C·día | 210 |

La demo no convierte unidades. Los valores deben entregarse en las mismas
unidades utilizadas durante la preparación de los datos. Si una entrada queda
fuera del rango observado en entrenamiento, la aplicación mostrará una
advertencia, pero permitirá revisar el resultado experimental.

## Por qué se seleccionó Random Forest A

La hipótesis inicial utilizó una red neuronal densa. La retroalimentación
docente motivó compararla formalmente con una alternativa más sencilla. Ambos
clasificadores fueron evaluados con las mismas particiones dentro de cada
escenario y con validación agrupada por estación como protocolo principal.

| Modelo | Accuracy por estación | F1 macro por estación |
|---|---:|---:|
| Random Forest A - clima | **83,46% ± 13,73%** | **72,04% ± 11,42%** |
| Red densa A - clima | 74,92% ± 12,89% | 64,16% ± 13,91% |
| Random Forest A′ - control | 87,13% ± 14,42% | 70,88% ± 8,80% |
| Random Forest B - clima + NDVI | 85,25% ± 14,98% | 69,19% ± 9,06% |

Random Forest A no fue elegido únicamente por una métrica. Fue seleccionado
porque:

- superó a la red densa A en accuracy y F1 macro;
- utiliza 1.091 observaciones y 41 estaciones;
- obtuvo el mayor F1 macro entre los bosques evaluados;
- no depende de la disponibilidad de NDVI;
- es más simple de ejecutar, explicar y empaquetar para inferencia tabular.

A′ utiliza solamente las 657 filas con NDVI válido y funciona como control para
compararlo con B. Su mayor accuracy no implica automáticamente que deba
reemplazar A, porque cubre 434 observaciones y tres estaciones menos. En el
subconjunto comparable, añadir NDVI no mejoró los promedios frente a A′; esta
conclusión se limita al dataset y al diseño experimental utilizados.

## Reproducción técnica avanzada

Las siguientes operaciones no son necesarias para probar la demo.

### Reproducir la comparación completa

```powershell
python ejecutar.py
```

La ejecución entrena 3 escenarios × 2 clasificadores × 2 estrategias de
validación × 5 folds, es decir, 60 resultados de fold. Puede tardar y
sobrescribe archivos dentro de `output/`.

También puede ejecutar solamente un escenario:

```powershell
python ejecutar.py --modelo a
python ejecutar.py --modelo aprima
python ejecutar.py --modelo b
```

Durante esta ejecución TensorFlow puede mostrar advertencias sobre
`tf.function retracing`. Son advertencias de rendimiento y no un error si el
proceso continúa y genera los archivos de salida.

### Volver a entrenar el modelo final

```powershell
python entrenar_modelo_final.py
```

Este comando vuelve a ajustar Random Forest A con las 1.091 filas y reemplaza:

- `models/random_forest_a_final.joblib`;
- `output/metadata_modelo_final.json`.

No lo ejecute si solamente quiere demostrar el modelo entregado. El
entrenamiento final no calcula una métrica nueva sobre esas mismas filas; el
desempeño reportado procede de la validación cruzada agrupada.

## Archivos principales

```text
phenological_prediction/
|-- data/                         CSV usados por A, A′ y B
|-- models/                       modelo final Random Forest A
|-- output/                       métricas, matrices y metadatos verificados
|-- tests/                        pruebas de integridad del release candidate
|-- ejecutar.py                   comparación red vs. Random Forest
|-- entrenar_modelo_final.py      entrenamiento y empaquetado del modelo A
|-- demo.py                       inferencia interactiva o por argumentos
|-- DATA_PROVENANCE.md            procedencia y condiciones de los datos
|-- requirements.txt              versiones exactas de dependencias
|-- VERSION                       versión del proyecto
|-- CHANGELOG.md                  cambios por release candidate
`-- RELEASE_MANIFEST.json         identidad y hashes de artefactos
```

Las principales evidencias del informe son:

- `output/comparacion_consolidada.csv`: medias y desviaciones.
- `output/metricas_por_fold.csv`: detalle de las 60 evaluaciones.
- `output/configuracion_ejecucion.json`: versiones, parámetros y protocolo.
- `output/comparacion_metricas.png`: resumen visual.
- `output/matriz_*.csv`: matrices de confusión acumuladas.
- `output/metadata_modelo_final.json`: identidad y rangos del modelo final.

## Solución de problemas frecuentes

### `python` no se reconoce como comando

Pruebe `py -3.13 --version`. Si tampoco funciona, instale Python 3.13, active la
opción para añadirlo al `PATH` y abra una terminal nueva.

### La línea no muestra `(.venv)`

El entorno no está activado. Vuelva a ejecutar el comando correspondiente a su
sistema operativo. La activación se pierde al cerrar la terminal.

### PowerShell bloquea `Activate.ps1`

Use solamente para la ventana actual:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Error al importar NumPy, SciPy, scikit-learn o TensorFlow

Compruebe que `(.venv)` esté visible y reinstale las versiones del proyecto:

```powershell
python -m pip install --force-reinstall -r requirements.txt
python -m pip check
```

No instale paquetes individualmente con versiones diferentes, porque puede
crear una combinación binaria incompatible.

### No se encuentra un CSV o el archivo Joblib

Asegúrese de ejecutar los comandos desde la carpeta raíz del proyecto y de que
existan `data/` y `models/random_forest_a_final.joblib`.

### Aparecen advertencias, pero no `FAILED` ni una excepción final

Las advertencias de TensorFlow, NumPy o Joblib no necesariamente significan que
la ejecución falló. En las pruebas, revise que el resumen final sea `OK`. En la
demo, confirme que aparezca una macro-etapa y las cinco probabilidades.

## Metodología y controles

- Validación principal: `StratifiedGroupKFold` por `s_id`, cinco folds.
- Validación aleatoria: control secundario con `StratifiedKFold`, cinco folds.
- Imputación: mediana ajustada exclusivamente con entrenamiento en cada fold.
- Escalado: aplicado solo a la red y ajustado con entrenamiento.
- Balance: pesos de clase en la red y `class_weight="balanced"` en el bosque.
- Semilla: 42.
- Random Forest: 400 árboles.
- Integridad: hashes de datos, resultados y modelo en el manifiesto.

## Limitaciones

- Los datos europeos no demuestran aplicabilidad operacional en Chile.
- La validación por estación no reserva temporadas completas.
- Floración tiene cobertura geográfica limitada en algunos folds.
- La desviación entre folds evidencia heterogeneidad territorial.
- El NDVI disponible no mejoró los promedios de A′ frente a B.
- El modelo clasifica alrededor de una fecha observada; no demuestra una
  anticipación prospectiva con horizonte definido.
- Las probabilidades del bosque no están calibradas.

## Procedencia y distribución

Los dos CSV integran observaciones de PEP725, variables climáticas de NASA POWER
e información Sentinel-2 para la línea con NDVI. Consulte
`DATA_PROVENANCE.md` antes de copiar o publicar los datos.

La entrega académica se realiza en un contexto privado. Este repositorio no
concede derechos adicionales sobre las fuentes originales y no tiene todavía
una licencia pública para el código.

## Trazabilidad del pre-release

- Versión: `0.1.0-rc.2`.
- Commit técnico verificado: `96b4e94687e8ff0aa7f904509ec0c2cdb4f0751d`.
- Python: 3.13.5.
- TensorFlow: 2.21.0.
- scikit-learn: 1.6.1.
- Parámetros y hashes: `RELEASE_MANIFEST.json` y archivos de `output/`.

El release candidate contiene la evidencia reproducible, el modelo final y la
demo. La liberación `0.1.0` requiere una decisión posterior y no debe inferirse
solamente de que la ejecución local haya sido satisfactoria.
