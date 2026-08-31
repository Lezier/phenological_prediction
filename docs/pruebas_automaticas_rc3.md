# Pruebas automáticas de RC3

## Ejecución

Desde el entorno virtual activo:

```powershell
python -m unittest discover -s tests -v
```

## Cobertura anterior a la corrida oficial

La suite cubre:

- integridad, cantidad y preparación de los datos A, A′ y B;
- configuración efectiva y construcción de ambos modelos;
- folds compartidos, cobertura, reproducibilidad y aislamiento de estaciones;
- ponderación de clases calculada solo desde train;
- separación de tiempos y dispersión entre folds;
- tratamiento conservador de Early Stopping sin uso del test externo;
- contrato del artefacto Joblib heredado;
- siete variables en orden y cinco clases únicas;
- inferencia y suma de probabilidades igual a uno;
- advertencia explícita de probabilidades no calibradas;
- entradas con cantidad incorrecta, valores no finitos, texto, campos faltantes
  o campos adicionales;
- advertencia, sin bloqueo, cuando una entrada queda fuera del rango observado;
- esquema y relaciones de los CSV deterministas de folds y pesos.

## Activación progresiva durante la construcción

Cuatro pruebas se omiten deliberadamente mientras sus artefactos aún no
existen en versión RC3:

- tres pruebas de métricas, tiempos, consolidado y hashes se activan cuando
  `output/configuracion_ejecucion.json` declara la misma versión que `VERSION`;
- la prueba del manifiesto se activa cuando `RELEASE_MANIFEST.json` declara la
  misma versión que `VERSION`.

Durante la construcción se esperaban los siguientes estados:

- antes de CP09 se espera `OK (skipped=4)`;
- después de CP09 se esperan solo las omisiones asociadas al manifiesto;
- después de CP13 se espera la suite completa sin omisiones.

Una prueba omitida por estas condiciones no equivale a una prueba aprobada. Es
un control preparado para fallar automáticamente si el artefacto se declara
RC3 pero no cumple el contrato esperado.

## Estado congelado en CP13

`RELEASE_MANIFEST.json` ya declara RC3. La prueba de manifiesto está activa y
comprueba todos los datos, el modelo y los resultados oficiales enumerados.
El resultado exigido es 41 pruebas aprobadas, sin omisiones, fallos ni errores.
`output/ensayo_reducido/` se excluye del manifiesto porque no es evidencia
oficial ni debe alimentar el informe.

## Estado del modelo usado en CP07

En CP07, las pruebas de inferencia cargaban el `.joblib` heredado únicamente
para verificar compatibilidad, entradas y salidas; todavía no era el artefacto
final. CP11 reentrenó el modelo, generó sus metadatos RC3 y repitió estas mismas
pruebas en un proceso nuevo.
