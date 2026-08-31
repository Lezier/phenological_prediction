# Procedencia y linaje de los datos

## Resumen

Los CSV del repositorio no son archivos originales de una única fuente. Son
snapshots derivados que combinan:

1. observaciones fenológicas de *Vitis vinifera* proporcionadas por PEP725;
2. ventanas climáticas derivadas de NASA POWER;
3. atributos NDVI derivados de imágenes Copernicus Sentinel-2.

El proyecto reproduce la evaluación desde estos snapshots. No incluye el flujo
completo de descarga y enriquecimiento desde los tres servicios externos.

## Identidad de los archivos

| Archivo | Filas | Columnas | Periodo | SHA-256 |
|---|---:|---:|---|---|
| `data/base_fenologia_clima.csv` | 1.130 | 19 | 1952–2025 | `0397C7A0B61B76388C22A1CDD1F13BCB2B7E10069C7BBB2935F0ADCC2E5CF6B7` |
| `data/base_fenologia_clima_satelite.csv` | 1.130 | 24 | 1952–2025 | `0C307E8CAAEEE04A87EB572AA675C4BB7C97EF1C1BC5C0D36C7018FB7129ADCA` |

Ambos archivos contienen 41 estaciones y preservan `license_short=CC BY-NC
4.0` en todas las filas.

## Fuente fenológica

- Proyecto: Pan European Phenology Project, PEP725.
- Export: `PEP725_FranciscoLopezBrombley_20260817`.
- Fecha de recuperación informada por el archivo de origen: 2026-08-17.
- Redes proveedoras indicadas en el export: provider 14, COST725 Spain;
  provider 1401, SMC-Meteocat; provider 1402, AEMET.
- Especie de trabajo: *Vitis vinifera*.
- Identificadores preservados: `s_id`, `year`, `day`, `date` y `phase_id`.
- Condición por fila: `CC BY-NC 4.0`.

PEP725 agrega observaciones de redes asociadas y las armoniza en un formato
común; no realiza por sí mismo todas las observaciones. La fuente debe citarse
según [`DATA_LICENSE.md`](DATA_LICENSE.md).

## Enriquecimiento climático

Las variables climáticas resumen una ventana de 30 días asociada a cada evento:

- temperatura media;
- promedio de máximas y mínimas;
- precipitación acumulada;
- radiación solar media;
- humedad relativa media;
- grados-día acumulados con base 10 °C;
- cantidad de días disponibles en la ventana.

El origen declarado es NASA POWER. El snapshot no conserva endpoint, versión
del servicio ni fecha exacta de cada consulta. Esta omisión impide reconstruir
bit a bit la fase de enriquecimiento externo únicamente desde este repositorio;
no impide reproducir el entrenamiento a partir de los CSV congelados.

## Enriquecimiento satelital

`base_fenologia_clima_satelite.csv` agrega:

- `ndvi`;
- fecha de la imagen;
- diferencia de días respecto del evento;
- nubosidad de la escena;
- número de imágenes candidatas.

El origen declarado es Copernicus Sentinel-2. Existen 657 observaciones con
NDVI válido; esas filas definen B y el control comparable A′. El snapshot no
incluye los identificadores completos de producto ni el historial del proceso
de selección de imágenes, por lo que la extracción satelital original tampoco
es reproducible bit a bit desde este repositorio.

## Transformaciones para modelado

```text
Export PEP725: 1.130 observaciones
        │
        ├── + ventana NASA POWER ──> base_fenologia_clima.csv
        │                                │
        │                                └── filtro A: 1.091 filas
        │
        └── + NASA POWER + Sentinel-2 ─> base_fenologia_clima_satelite.csv
                                             │
                                             └── NDVI válido: 657 filas
                                                   ├── A′: 7 variables clima
                                                   └── B: clima + NDVI
```

Para A se eliminan filas sin objetivo, estación o variables climáticas
obligatorias. La radiación ausente se conserva y se imputa por mediana dentro
de cada train. A′ y B se construyen sobre las mismas 657 filas. La variable
`phase_id` se agrupa en cinco `macro_etapa`; los modelos no reciben identificador
de estación, coordenadas, fecha, año, día, licencia ni fase como predictores.

La preparación efectiva está implementada en `ejecutar.py` y el contrato de
variables en `configuracion.py`. Folds, imputación, escalado y pesos se ajustan
sin usar el test del fold.

## Fuentes que no originan estos CSV

El dataset “A Multisource Grapevine Phenology Dataset for Smart Farming and AI
Modeling” de Zenodo fue consultado como antecedente académico, pero no es la
fuente de los dos snapshots entregados. Sus registros 2016–2022, parcelas y
variables no deben confundirse con el export PEP725 1952–2025 utilizado aquí.

## Reproducibilidad y límites

Reproducible dentro del repositorio:

- integridad de los dos CSV mediante SHA-256;
- definición de A, A′ y B;
- folds compartidos y ponderaciones;
- comparación completa y agregación de resultados;
- entrenamiento del artefacto final desde A.

No reproducible sin fuentes externas adicionales:

- solicitud original del export PEP725;
- consultas NASA POWER;
- búsqueda, filtrado y cálculo original de NDVI;
- reconstrucción exacta de los CSV desde datos crudos.

Estas limitaciones deben conservarse al describir el notebook como
“reproducible”: reproduce el pipeline analítico desde los CSV publicados, no la
adquisición primaria de las tres fuentes.

## Referencias

- [PEP725 Dataset](https://pep725.eu/dataset/)
- [Templ et al. (2018)](https://doi.org/10.1007/s00484-018-1512-8)
- [NASA POWER](https://power.larc.nasa.gov/)
- [NASA POWER Referencing Guide](https://power.larc.nasa.gov/docs/referencing/)
- [Copernicus Sentinel-2](https://dataspace.copernicus.eu/data-collections/copernicus-sentinel-missions/sentinel-2)
- [Copernicus Data Space Terms](https://dataspace.copernicus.eu/terms-and-conditions)
