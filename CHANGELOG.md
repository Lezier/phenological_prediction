# Changelog

## 0.1.0-rc.3 - 2026-08-31

- Inicia la migración reanudable desde `0.1.0-rc.2` mediante checkpoints
  verificables.
- Confirma Random Forest A como modelo de inferencia del prototipo y conserva
  la red neuronal como comparador experimental.
- Documenta la procedencia asistida por ChatGPT de la configuración de Random
  Forest, la ausencia de tuning sistemático y el uso de valores predeterminados
  de scikit-learn para los parámetros no especificados.
- Deja la procedencia y justificación de los hiperparámetros de la red neuronal
  pendientes de la respuesta solicitada al equipo.
- Materializa folds compartidos con identificadores reproducibles y evidencia
  por fila, estación, clase, partición y clasificador.
- Calcula una sola ponderación balanceada desde train por fold, la reutiliza en
  ambos clasificadores y exporta frecuencias y pesos auditables.
- Separa tiempos de entrenamiento e inferencia por fold y agrega media y
  desviación estándar de todas las métricas y duraciones al consolidado.
- Conserva Early Stopping como control de convergencia del baseline heredado,
  prueba que el test externo no participa en `fit()` y registra las épocas
  ejecutadas por fold.
- Amplía las pruebas automáticas de datos, evidencias, outputs e inferencia;
  las verificaciones de CP09 y CP13 se activan por versión cuando sus artefactos
  oficiales están disponibles.
- Añade un modo de ensayo reducido aislado, marcado como no oficial, que
  ejecuta un fold real y genera un manifiesto propio sin sobrescribir los
  resultados de la comparación completa.
- Completa y audita la ejecución oficial RC3: los agregados coinciden con las
  métricas por fold, los folds y pesos son compartidos, y los tiempos y matrices
  son estructuralmente válidos. La auditoría mantiene Random Forest A para el
  prototipo y declara las limitaciones de dispersión, cobertura de clases,
  calibración y generalización externa.
- Entrena y empaqueta el Random Forest A final con las 1.091 filas: el paquete
  de esquema 2 declara versión RC3, siete variables ordenadas, cinco clases,
  rangos, hash de datos y probabilidades no calibradas. La metadata obtiene su
  evidencia del consolidado auditado y registra versiones de serialización.
- Reestructura el README como referencia de ingeniería RC3 y separa el recorrido
  operativo en el Anexo A externo al ZIP.
- Documenta el linaje verificable de PEP725, NASA POWER y Sentinel-2, las
  condiciones `CC BY-NC 4.0` del export fenológico utilizado y las limitaciones
  para reproducir la adquisición externa desde los snapshots.
- Separa las condiciones de datos y código: los CSV permiten uso académico no
  comercial con atribución; el repositorio privado no concede por sí mismo una
  licencia abierta de software.
- Verifica instalación, pruebas, demo y consulta de resultados en una copia
  aislada con un entorno Python 3.13 nuevo.
- Congela el conjunto Python RC3 mediante un manifiesto SHA-256 de los datos,
  el modelo y todos los resultados oficiales; excluye expresamente el ensayo
  reducido y activa la última prueba de integridad pendiente.

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
