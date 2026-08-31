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
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from configuracion import (
    ADVERTENCIA_USO,
    GRUPO,
    HASH_DATOS_MODELO_A,
    OBJETIVO,
    PARAMETROS_RANDOM_FOREST,
    PROCEDENCIA_HIPERPARAMETROS,
    SEMILLA,
    VARIABLES_CLIMA,
    VERSION_PROYECTO,
    parametros_random_forest_efectivos,
)

RAIZ = Path(__file__).resolve().parent
RUTA_DATOS = RAIZ / "data" / "base_fenologia_clima.csv"
RUTA_MODELO = RAIZ / "models" / "random_forest_a_final.joblib"
RUTA_METADATA = RAIZ / "output" / "metadata_modelo_final.json"
RUTA_RESULTADOS = RAIZ / "output" / "comparacion_consolidada.csv"
VERSION = VERSION_PROYECTO
VARIABLES = VARIABLES_CLIMA
HASH_DATOS_ESPERADO = HASH_DATOS_MODELO_A
ADVERTENCIA = ADVERTENCIA_USO


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


def cargar_evidencia_desempeno() -> dict:
    """Obtiene la evidencia auditada sin volver a estimar desempeño."""

    if not RUTA_RESULTADOS.exists():
        raise FileNotFoundError(
            f"No existe la comparación oficial requerida: {RUTA_RESULTADOS}"
        )
    resultados = pd.read_csv(RUTA_RESULTADOS)
    seleccion = resultados.loc[
        (resultados["modelo_datos"] == "a")
        & (resultados["validacion"] == "por_estacion")
        & (resultados["clasificador"] == "random_forest")
    ]
    if len(seleccion) != 1:
        raise ValueError(
            "La comparación oficial debe contener exactamente una fila para "
            "Random Forest A con validación por estación."
        )
    fila = seleccion.iloc[0]
    return {
        "protocolo": "StratifiedGroupKFold por s_id, 5 folds",
        "n_folds": int(fila["n_folds"]),
        "accuracy_promedio": float(fila["accuracy_promedio"]),
        "accuracy_desviacion": float(fila["accuracy_desviacion"]),
        "f1_macro_promedio": float(fila["f1_macro_promedio"]),
        "f1_macro_desviacion": float(fila["f1_macro_desviacion"]),
        "f1_weighted_promedio": float(fila["f1_weighted_promedio"]),
        "f1_weighted_desviacion": float(fila["f1_weighted_desviacion"]),
        "fuente": RUTA_RESULTADOS.relative_to(RAIZ).as_posix(),
        "sha256_fuente": sha256_archivo(RUTA_RESULTADOS),
        "nota": (
            "Estimación obtenida por validación cruzada; no corresponde al "
            "entrenamiento final con todas las filas."
        ),
    }


def main() -> None:
    datos = cargar_modelo_a()
    evidencia_desempeno = cargar_evidencia_desempeno()
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
        "version_esquema_paquete": 2,
        "fecha_entrenamiento": datetime.now().astimezone().isoformat(timespec="seconds"),
        "modelo": "Random Forest A - clima",
        "proposito": "Artefacto de inferencia para la demostración académica.",
        "muestras_entrenamiento": int(len(datos)),
        "estaciones": int(datos[GRUPO].nunique()),
        "variables": VARIABLES,
        "clases": clases,
        "imputacion": "Mediana ajustada con las 1091 filas del entrenamiento final.",
        "parametros_random_forest_explicitos": PARAMETROS_RANDOM_FOREST,
        "parametros_random_forest_efectivos": parametros_random_forest_efectivos(),
        "procedencia_hiperparametros": PROCEDENCIA_HIPERPARAMETROS["random_forest"],
        "semilla": SEMILLA,
        "version_python": platform.python_version(),
        "version_numpy": np.__version__,
        "version_pandas": pd.__version__,
        "version_scikit_learn": sklearn.__version__,
        "version_joblib": joblib.__version__,
        "archivo_datos": RUTA_DATOS.name,
        "sha256_datos": sha256_archivo(RUTA_DATOS),
        "rangos_entrenamiento": rangos,
        "evidencia_desempeno": evidencia_desempeno,
        "probabilidades_calibradas": False,
        "nota_probabilidades": (
            "predict_proba entrega puntajes no calibrados; no deben "
            "interpretarse como probabilidades garantizadas."
        ),
        "advertencia": ADVERTENCIA,
    }
    paquete = {
        "version_esquema": 2,
        "version_proyecto": VERSION,
        "pipeline": modelo,
        "variables": VARIABLES,
        "clases": clases,
        "rangos_entrenamiento": rangos,
        "sha256_datos": sha256_archivo(RUTA_DATOS),
        "probabilidades_calibradas": False,
        "advertencia": ADVERTENCIA,
    }

    RUTA_MODELO.parent.mkdir(parents=True, exist_ok=True)
    RUTA_METADATA.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(paquete, RUTA_MODELO, compress=3)
    metadata["archivo_modelo"] = RUTA_MODELO.relative_to(RAIZ).as_posix()
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
