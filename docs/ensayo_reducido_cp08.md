# Ensayo reducido de CP08

## Objetivo

El ensayo reducido comprueba con entrenamiento real que el pipeline integrado
puede completar un fold y generar todos los tipos de artefacto antes de iniciar
la corrida oficial. No reemplaza la validación cruzada de cinco folds y sus
resultados no deben utilizarse en el informe ni en la presentación.

## Alcance congelado

- escenario: A, clima completo;
- validación: agrupada por estación;
- fold: 1;
- clasificadores: red densa y Random Forest;
- semilla: 42;
- salida aislada: `output/ensayo_reducido/`.

Comando:

```powershell
python ejecutar.py --modelo a --ensayo-reducido
```

## Artefactos

El directorio contiene:

- métricas y tiempos del fold;
- resumen no oficial;
- asignación de filas;
- pesos de clase;
- dos matrices de confusión;
- gráfico comparativo;
- configuración del ensayo;
- advertencia `NO_USAR_EN_INFORME.md`;
- manifiesto con SHA-256 de los archivos anteriores.

El manifiesto no se incluye a sí mismo para evitar una autorreferencia de hash.

## Resultado operativo del 31 de agosto de 2026

La ejecución terminó con código cero. Ambos clasificadores usaron el mismo
`fold_id`, 923 observaciones de train y 168 de test. La red completó las 60
épocas, por lo que Early Stopping no la detuvo en este fold.

Las métricas y tiempos observados se conservan únicamente como diagnóstico de
funcionamiento. No se reproducen aquí como resultados científicos porque
provienen de un solo fold. La desviación estándar queda sin valor: no puede
estimarse dispersión con una sola observación.

TensorFlow emitió mensajes sobre ejecución CPU, una API deprecada y un atributo
interno desconocido que fue ignorado. El proceso continuó, produjo todos los
artefactos y superó las verificaciones; los mensajes quedan como observaciones
no bloqueantes del entorno local.

## Criterio de aprobación

- ambos modelos entrenan y predicen;
- folds, pesos, tiempos, métricas, matrices, JSON y PNG se generan;
- todos los hashes del manifiesto coinciden;
- las salidas oficiales ubicadas directamente en `output/` permanecen sin
  cambios;
- la suite posterior finaliza sin fallos.
