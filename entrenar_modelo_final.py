"""Entrena y empaqueta Random Forest A para la demostración del proyecto.

La selección del modelo se hizo previamente mediante validación agrupada por
estación en ``ejecutar.py``. Este script no estima desempeño: ajusta el modelo
seleccionado con todas las observaciones disponibles para permitir inferencia.
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

RAIZ = Path(__file__).resolve().parent
RUTA_DATOS = RAIZ / "data" / "base_fenologia_clima.csv"
RUTA_MODELO = RAIZ / "models" / "random_forest_a_final.joblib"
RUTA_METADATA = RAIZ / "output" / "metadata_modelo_final.json"
VERSION = (RAIZ / "VERSION").read_text(encoding="utf-8").strip()

VARIABLES = [
    "clima_temp_media",
    "clima_temp_max_media",
    "clima_temp_min_media",
    "clima_precip_acumulada",
    "clima_radiacion_media",
    "clima_humedad_media",
    "clima_gdd_acumulado",
]
OBJETIVO = "macro_etapa"
GRUPO = "s_id"
SEMILLA = 42
HASH_DATOS_ESPERADO = "0397C7A0B61B76388C22A1CDD1F13BCB2B7E10069C7BBB2935F0ADCC2E5CF6B7"
PARAMETROS_RANDOM_FOREST = {
    "n_estimators": 400,
    "class_weight": "balanced",
    "random_state": SEMILLA,
    "n_jobs": -1,
}
ADVERTENCIA = (
    "Uso experimental con datos europeos; no validado para operación en Chile "
    "y no sustituye evaluación agronómica."
)


def sha256_archivo(ruta: Path) -> str:
    resumen = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            resumen.update(bloque)
    return resumen.hexdigest().upper()


def cargar_modelo_a() -> pd.DataFrame:
    if not RUTA_DATOS.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {RUTA_DATOS}")

    hash_datos = sha256_archivo(RUTA_DATOS)
    if hash_datos != HASH_DATOS_ESPERADO:
        raise ValueError(
            "El CSV no coincide con la fuente validada. "
            f"Esperado: {HASH_DATOS_ESPERADO}; recibido: {hash_datos}."
        )

    datos = pd.read_csv(RUTA_DATOS)
    requeridas = VARIABLES + [OBJETIVO, GRUPO]
    faltantes = sorted(set(requeridas) - set(datos.columns))
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {faltantes}")

    # Replica la definición evaluada de A: excluye filas sin clima, pero conserva
    # radiación ausente para que el Pipeline la impute durante el ajuste.
    obligatorias = [
        columna for columna in VARIABLES if columna != "clima_radiacion_media"
    ] + [OBJETIVO, GRUPO]
    datos = datos.dropna(subset=obligatorias).reset_index(drop=True)

    if len(datos) != 1091:
        raise ValueError(f"Se esperaban 1091 muestras de Modelo A; se obtuvieron {len(datos)}.")
    if datos[OBJETIVO].nunique() != 5:
        raise ValueError("El objetivo debe contener exactamente cinco macro-etapas.")
    return datos


def main() -> None:
    datos = cargar_modelo_a()
    x = datos[VARIABLES]
    y = datos[OBJETIVO].astype(str)

    modelo = Pipeline(
        [
            ("imputador", SimpleImputer(strategy="median")),
            (
                "clasificador",
                RandomForestClassifier(**PARAMETROS_RANDOM_FOREST),
            ),
        ]
    )
    modelo.fit(x, y)

    clases = list(modelo.named_steps["clasificador"].classes_)
    rangos = {
        variable: {
            "min": float(datos[variable].min()),
            "max": float(datos[variable].max()),
        }
        for variable in VARIABLES
    }
    metadata = {
        "version_proyecto": VERSION,
        "fecha_entrenamiento": datetime.now().astimezone().isoformat(timespec="seconds"),
        "modelo": "Random Forest A - clima",
        "proposito": "Artefacto de inferencia para la demostración académica.",
        "muestras_entrenamiento": int(len(datos)),
        "estaciones": int(datos[GRUPO].nunique()),
        "variables": VARIABLES,
        "clases": clases,
        "imputacion": "Mediana ajustada con las 1091 filas del entrenamiento final.",
        "parametros_random_forest": PARAMETROS_RANDOM_FOREST,
        "semilla": SEMILLA,
        "version_python": platform.python_version(),
        "version_scikit_learn": sklearn.__version__,
        "archivo_datos": RUTA_DATOS.name,
        "sha256_datos": sha256_archivo(RUTA_DATOS),
        "rangos_entrenamiento": rangos,
        "evidencia_desempeno": {
            "protocolo": "StratifiedGroupKFold por s_id, 5 folds",
            "accuracy_promedio": 0.8346299027206735,
            "accuracy_desviacion": 0.13726734207631164,
            "f1_macro_promedio": 0.7204350871824743,
            "f1_macro_desviacion": 0.11423188916818808,
            "fuente": "output/comparacion_consolidada.csv",
        },
        "advertencia": ADVERTENCIA,
    }
    paquete = {
        "version_esquema": 1,
        "pipeline": modelo,
        "variables": VARIABLES,
        "clases": clases,
        "rangos_entrenamiento": rangos,
        "advertencia": ADVERTENCIA,
    }

    RUTA_MODELO.parent.mkdir(parents=True, exist_ok=True)
    RUTA_METADATA.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(paquete, RUTA_MODELO, compress=3)
    metadata["archivo_modelo"] = str(RUTA_MODELO.relative_to(RAIZ))
    metadata["sha256_modelo"] = sha256_archivo(RUTA_MODELO)
    RUTA_METADATA.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Modelo final generado correctamente.")
    print(f"Muestras: {len(datos)} | Estaciones: {datos[GRUPO].nunique()}")
    print(f"Modelo: {RUTA_MODELO}")
    print(f"Metadatos: {RUTA_METADATA}")
    print(f"SHA-256: {metadata['sha256_modelo']}")


if __name__ == "__main__":
    main()
