# Changelog

## 0.1.0-rc.2 - 2026-08-29

- Fija finales de línea LF para los CSV mediante `.gitattributes`.
- Marca modelos Joblib e imágenes PNG como archivos binarios.
- Amplía las pruebas de integridad a ambos CSV y verifica ausencia de CRLF.
- Declara hashes `sha256-lf` para texto y `sha256-raw` para binarios, evitando
  falsos negativos entre clones con configuraciones Git distintas.
- Conserva sin cambios los datos, métricas, parámetros y modelo de `0.1.0-rc.1`.

## 0.1.0-rc.1 - 2026-08-29

- Incorpora comparación reproducible entre red densa y Random Forest.
- Evalúa A, A' y B mediante validación aleatoria y agrupada por estación.
- Ajusta la imputación exclusivamente con entrenamiento dentro de cada fold.
- Selecciona Random Forest A como candidato para el prototipo.
- Añade entrenamiento final, artefacto serializado, demo y pruebas mínimas.
- Registra versiones, parámetros, fechas y hashes SHA-256 de datos y modelo.
