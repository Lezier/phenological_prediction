# Condiciones de uso y atribución de los datos

## Alcance

Este documento aplica a los dos snapshots derivados incluidos en `data/`:

| Archivo | SHA-256 |
|---|---|
| `base_fenologia_clima.csv` | `0397C7A0B61B76388C22A1CDD1F13BCB2B7E10069C7BBB2935F0ADCC2E5CF6B7` |
| `base_fenologia_clima_satelite.csv` | `0C307E8CAAEEE04A87EB572AA675C4BB7C97EF1C1BC5C0D36C7018FB7129ADCA` |

No aplica al código. La situación del software se documenta por separado en
[`CODE_LICENSE.md`](CODE_LICENSE.md).

## Observaciones fenológicas PEP725

La fuente fenológica es el export PEP725
`PEP725_FranciscoLopezBrombley_20260817`, recuperado el 17 de agosto de 2026.
Su README de origen explica que PEP725 admite varios modelos de licencia y que
la licencia aplicable debe evaluarse por observación.

En los dos CSV entregados:

- las 1.130 filas conservan el campo `license_short`;
- el único valor presente es `CC BY-NC 4.0`;
- no se mezclan observaciones PEP725 con otra licencia.

Por ello, las observaciones fenológicas incorporadas pueden compartirse y
adaptarse con atribución, indicación de cambios y **sin uso comercial**, según
[Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/).
También se deben respetar la
[PEP725 Data Use Policy](https://pep725.eu/pep725_data_use_policy/) y las
obligaciones de atribución informadas con el export.

La página actual de PEP725 ofrece además un dataset de ejemplo bajo CC BY 4.0.
Ese ejemplo público no es la fuente de estos CSV y su licencia no reemplaza el
valor `CC BY-NC 4.0` del export utilizado.

### Atribución PEP725 requerida

Incluir el siguiente reconocimiento:

> Data were provided by the members of the PEP725 project.

Citar:

> Templ, B., Koch, E., Bolmgren, K., Ungersböck, M., Paul, A., Scheifinger,
> H., et al. (2018). Pan European Phenological database (PEP725): a single
> point of access for European data. *International Journal of
> Biometeorology*, 62, 1109–1113.
> https://doi.org/10.1007/s00484-018-1512-8. Dataset accessed 2026-08-17 at
> https://pep725.eu.

La política recibida solicita informar a PEP725 (`pep725@geosphere.at`) los
datos bibliográficos de publicaciones que utilicen el dataset.

## Variables climáticas NASA POWER

Las columnas `clima_*` contienen variables derivadas de NASA POWER. La guía
oficial solicita reconocer al NASA Langley Research Center POWER Project,
financiado por la NASA Earth Science Division, e identificar servicio, versión
y fecha de acceso cuando estén disponibles.

El snapshot entregado no conserva esos tres últimos datos con precisión; no se
inventan en este documento. Esta ausencia es una limitación de procedencia que
debe corregirse si se vuelve a descargar la fuente.

Reconocimiento utilizado:

> Climate data were obtained from the NASA Langley Research Center Prediction
> Of Worldwide Energy Resources (POWER) Project, funded through the NASA Earth
> Science Division.

Referencia oficial:
[NASA POWER Referencing Guide](https://power.larc.nasa.gov/docs/referencing/).
NASA POWER solicita además notificación cuando sus datos se redistribuyen a
otros investigadores.

## NDVI derivado de Copernicus Sentinel-2

Las columnas `ndvi`, `ndvi_fecha_imagen`, `ndvi_dif_dias`,
`ndvi_nubosidad_escena` y `ndvi_n_imagenes_candidatas` contienen información
derivada de Sentinel-2.

El acceso y uso de los datos Sentinel se ofrece de forma libre, completa y
abierta bajo el aviso legal aplicable a Copernicus Sentinel Data and Service
Information. Debe reconocerse a Copernicus/Unión Europea y no debe inferirse
respaldo institucional del proyecto.

Reconocimiento utilizado:

> Contains modified Copernicus Sentinel-2 data processed for this academic
> project.

Referencia oficial:
[Copernicus Data Space Terms and Conditions](https://dataspace.copernicus.eu/terms-and-conditions).

## Regla para los CSV combinados

Los archivos combinan material con condiciones distintas. La incorporación de
variables NASA POWER o Sentinel-2 no elimina las restricciones de las
observaciones PEP725. En consecuencia, la redistribución de estos snapshots se
limita al uso no comercial y debe conservar todas las atribuciones anteriores.

No se autoriza eliminar `license_short`, presentar los datos como propios,
usar el nombre de las instituciones como respaldo del modelo ni eliminar la
indicación de que los CSV son derivados.

## Ausencia de garantía

Los datos se suministran para evaluación académica, sin garantía de exactitud,
idoneidad para un propósito productivo ni representatividad de Chile. Quien los
reutilice debe revisar las condiciones vigentes de cada fuente y conservar la
trazabilidad del export específico.
