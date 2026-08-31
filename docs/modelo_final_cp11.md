# Modelo final de inferencia RC3 — CP11

## Propósito

El archivo `models/random_forest_a_final.joblib` es el artefacto de inferencia
del prototipo. Se entrenó con las 1.091 observaciones y 41 estaciones del
escenario A después de seleccionar el enfoque mediante validación cruzada.

Este entrenamiento con todos los datos no estima desempeño. Las métricas
reportadas proceden exclusivamente de la validación agrupada por estación
registrada en `output/comparacion_consolidada.csv`.

## Pipeline empaquetado

1. `SimpleImputer(strategy="median")` ajustado con las 1.091 filas.
2. `RandomForestClassifier` con 400 árboles, pesos balanceados, semilla 42 y
   paralelización `n_jobs=-1`.

El paquete usa esquema 2 y contiene:

- versión del proyecto `0.1.0-rc.3`;
- pipeline ajustado;
- orden de las siete variables climáticas;
- cinco clases y sus rangos observados;
- hash del CSV de entrenamiento;
- declaración explícita de probabilidades no calibradas;
- advertencia de uso experimental y falta de validación para Chile.

## Integridad y entorno

| Artefacto | SHA-256 |
|---|---|
| `data/base_fenologia_clima.csv` | `0397C7A0B61B76388C22A1CDD1F13BCB2B7E10069C7BBB2935F0ADCC2E5CF6B7` |
| `output/comparacion_consolidada.csv` | `0AB8AA5FCE99070CE17B0DC9C562D48DD7AC524A517B804D34DEB69020145C37` |
| `models/random_forest_a_final.joblib` | `64CEDBFD3F73A74F1BBF651BAF193E2F7C9A998EDB566C4A3EE03EF30DAC04CD` |
| `output/metadata_modelo_final.json` | `49EFDEF7154B77BD33190B2507D9C4447A5184BDE25D45FD7ACB0FF17B3D3F8D` |
| `output/entrenamiento_modelo_final_rc3.log` | `25290FC427E7473AB14A6C27057EE3F5156FC3AF7ACD8D10BE4A9A9BEDFC4C3E` |

Entorno de serialización y carga verificado:

- Python 3.13.5;
- NumPy 2.5.2;
- pandas 2.3.3;
- scikit-learn 1.6.1;
- joblib 1.5.3.

Los consumidores deben recrear las versiones de `requirements.txt`. Los
artefactos joblib no se consideran un formato portable entre versiones
arbitrarias de scikit-learn y solo deben cargarse desde una fuente confiable.

## Verificación

- Carga del `.joblib` en un proceso nuevo: aprobada.
- Inferencia con siete valores: aprobada; entrega etapa y cinco puntajes que
  suman uno.
- Entrada con cantidad incorrecta, valores no finitos, texto o claves
  incorrectas: rechazada.
- Orden de variables, clases, rangos, versión, hashes y estado de calibración:
  verificados por pruebas automáticas.
- Suite posterior: 41 pruebas, 40 aprobadas y una omisión condicional del
  manifiesto RC2 pendiente de CP13; cero fallos y cero errores.

La advertencia `DeprecationWarning` observada al cargar joblib con NumPy 2.5.2
no impidió la deserialización ni la inferencia. Queda mitigada mediante versiones
fijadas y será reevaluada si se actualiza alguna dependencia.
