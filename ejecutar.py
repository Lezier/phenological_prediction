"""Validacion robusta y baseline para responder al feedback del Avance 1.

Compara una red densa con Random Forest usando las mismas filas y variables
de los modelos A, A' y B. Cada combinacion se evalua con particion aleatoria
y con estaciones completas fuera del entrenamiento.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DATOS_LOCALES = Path(__file__).resolve().parent / "data"
SALIDA = Path(__file__).resolve().parent / "output"
CONFIG_MATPLOTLIB = Path(__file__).resolve().parent / ".matplotlib"
CONFIG_MATPLOTLIB.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CONFIG_MATPLOTLIB))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

CLIMA = [
    "clima_temp_media",
    "clima_temp_max_media",
    "clima_temp_min_media",
    "clima_precip_acumulada",
    "clima_radiacion_media",
    "clima_humedad_media",
    "clima_gdd_acumulado",
]
CONFIG = {
    "a": ("modelo_a_clima", "base_fenologia_clima.csv", CLIMA, "Modelo A - clima"),
    "aprima": (
        "modelo_aprima_control",
        "base_fenologia_clima_satelite.csv",
        CLIMA,
        "Modelo A' - control clima",
    ),
    "b": (
        "modelo_b_clima_ndvi",
        "base_fenologia_clima_satelite.csv",
        CLIMA + ["ndvi"],
        "Modelo B - clima + NDVI",
    ),
}
SEMILLA = 42
FOLDS = 5
VERSION_PROYECTO = "0.1.0-rc.1"
PARAMETROS_RANDOM_FOREST = {
    "n_estimators": 400,
    "class_weight": "balanced",
    "random_state": SEMILLA,
    "n_jobs": -1,
}


def sha256_archivo(ruta: Path) -> str:
    resumen = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            resumen.update(bloque)
    return resumen.hexdigest().upper()


def fijar_semilla() -> None:
    os.environ["PYTHONHASHSEED"] = str(SEMILLA)
    random.seed(SEMILLA)
    np.random.seed(SEMILLA)
    tf.keras.utils.set_random_seed(SEMILLA)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def cargar_datos(nombre: str) -> tuple[pd.DataFrame, list[str], str]:
    carpeta, archivo, variables, titulo = CONFIG[nombre]
    ruta_local = DATOS_LOCALES / archivo
    ruta_datos = ruta_local if ruta_local.exists() else RAIZ / carpeta / "data" / archivo
    datos = pd.read_csv(ruta_datos)

    if nombre in {"aprima", "b"}:
        datos = datos[datos["ndvi"].notna()].copy()

    # En A se conservan las filas con radiacion ausente para imputarlas dentro
    # de cada fold usando exclusivamente las observaciones de entrenamiento.
    if nombre == "a":
        requeridas = [
            columna for columna in variables if columna != "clima_radiacion_media"
        ] + ["macro_etapa", "s_id"]
    else:
        requeridas = variables + ["macro_etapa", "s_id"]
    datos = datos.dropna(subset=requeridas).reset_index(drop=True)
    return datos, variables, titulo


def crear_red(n_variables: int, n_clases: int) -> tf.keras.Model:
    red = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(n_variables,)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(n_clases, activation="softmax"),
        ]
    )
    red.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return red


def particiones(x: np.ndarray, y: np.ndarray, grupos: np.ndarray, tipo: str):
    if tipo == "aleatoria":
        divisor = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEMILLA)
        return divisor.split(x, y)
    divisor = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEMILLA)
    return divisor.split(x, y, groups=grupos)


def entrenar_red(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, n_clases: int) -> np.ndarray:
    escalador = StandardScaler()
    x_train = escalador.fit_transform(x_train)
    x_test = escalador.transform(x_test)
    pesos = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    red = crear_red(x_train.shape[1], n_clases)
    parada = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=8, restore_best_weights=True)
    red.fit(
        x_train,
        y_train,
        epochs=60,
        batch_size=16,
        class_weight=dict(zip(np.unique(y_train), pesos)),
        callbacks=[parada],
        verbose=0,
    )
    prediccion = np.argmax(red.predict(x_test, verbose=0), axis=1)
    tf.keras.backend.clear_session()
    return prediccion


def entrenar_random_forest(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    bosque = RandomForestClassifier(**PARAMETROS_RANDOM_FOREST)
    bosque.fit(x_train, y_train)
    return bosque.predict(x_test)


def evaluar(nombre: str, validacion: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    datos, variables, titulo = cargar_datos(nombre)
    codificador = LabelEncoder()
    y = codificador.fit_transform(datos["macro_etapa"])
    x = datos[variables].to_numpy()
    grupos = datos["s_id"].to_numpy()
    acumuladas = {
        "red_densa": np.zeros((len(codificador.classes_), len(codificador.classes_)), dtype=int),
        "random_forest": np.zeros((len(codificador.classes_), len(codificador.classes_)), dtype=int),
    }
    filas = []

    for fold, (idx_train, idx_test) in enumerate(particiones(x, y, grupos, validacion), start=1):
        # El imputador aprende las medianas solo desde entrenamiento. Aplicar
        # luego esas mismas medianas a test evita fuga de informacion.
        imputador = SimpleImputer(strategy="median")
        x_train = imputador.fit_transform(x[idx_train])
        x_test = imputador.transform(x[idx_test])

        if np.isnan(x_train).any() or np.isnan(x_test).any():
            raise ValueError(f"Quedaron valores NaN despues de imputar el fold {fold}.")
        if validacion == "por_estacion":
            estaciones_train = set(grupos[idx_train])
            estaciones_test = set(grupos[idx_test])
            if not estaciones_train.isdisjoint(estaciones_test):
                raise ValueError(f"Hay estaciones compartidas en el fold {fold}.")

        etiquetas = list(range(len(codificador.classes_)))
        presentes_train = set(np.unique(y[idx_train]))
        presentes_test = set(np.unique(y[idx_test]))
        for clasificador, entrenar in {
            "red_densa": lambda: entrenar_red(
                x_train, y[idx_train], x_test, len(codificador.classes_)
            ),
            "random_forest": lambda: entrenar_random_forest(
                x_train, y[idx_train], x_test
            ),
        }.items():
            pred = entrenar()
            acumuladas[clasificador] += confusion_matrix(
                y[idx_test], pred, labels=range(len(codificador.classes_))
            )
            filas.append(
                {
                    "modelo_datos": nombre,
                    "validacion": validacion,
                    "clasificador": clasificador,
                    "fold": fold,
                    "n_train": len(idx_train),
                    "n_test": len(idx_test),
                    "estaciones_train": len(np.unique(grupos[idx_train])),
                    "estaciones_test": len(np.unique(grupos[idx_test])),
                    "clases_train": len(presentes_train),
                    "clases_test": len(presentes_test),
                    "clases_ausentes_train": "|".join(
                        codificador.classes_[i] for i in etiquetas if i not in presentes_train
                    ),
                    "clases_ausentes_test": "|".join(
                        codificador.classes_[i] for i in etiquetas if i not in presentes_test
                    ),
                    "accuracy": accuracy_score(y[idx_test], pred),
                    "f1_macro": f1_score(
                        y[idx_test], pred, labels=etiquetas, average="macro", zero_division=0
                    ),
                    "f1_weighted": f1_score(
                        y[idx_test], pred, labels=etiquetas, average="weighted", zero_division=0
                    ),
                }
            )

    matrices = []
    for clasificador, matriz in acumuladas.items():
        tabla = pd.DataFrame(matriz, index=codificador.classes_, columns=codificador.classes_)
        tabla.index.name = "clase_real"
        tabla.to_csv(SALIDA / f"matriz_{nombre}_{validacion}_{clasificador}.csv")
        matrices.append(tabla.assign(clasificador=clasificador, clase_real=tabla.index))

    metadata = {
        "modelo_datos": nombre,
        "titulo": titulo,
        "validacion": validacion,
        "grupo": "s_id" if validacion == "por_estacion" else None,
        "muestras": int(len(datos)),
        "estaciones": int(datos["s_id"].nunique()),
        "variables": variables,
        "clases": list(codificador.classes_),
        "imputacion": "Mediana ajustada exclusivamente con train dentro de cada fold.",
    }
    return pd.DataFrame(filas), pd.concat(matrices, ignore_index=True), metadata


def crear_grafico(resumen: pd.DataFrame) -> None:
    etiquetas = [
        f"{fila.modelo_datos}\n{fila.validacion}\n{fila.clasificador}"
        for fila in resumen.itertuples()
    ]
    posiciones = np.arange(len(resumen))
    figura, eje = plt.subplots(figsize=(max(12, len(resumen) * 1.15), 6))
    eje.bar(posiciones - 0.18, resumen["accuracy_promedio"], 0.36, label="Accuracy")
    eje.bar(posiciones + 0.18, resumen["f1_macro_promedio"], 0.36, label="F1 macro")
    eje.set(
        title="Comparacion de modelos y estrategias de validacion",
        ylabel="Metrica promedio",
        ylim=(0, 1),
        xticks=posiciones,
        xticklabels=etiquetas,
    )
    plt.setp(eje.get_xticklabels(), rotation=45, ha="right")
    eje.grid(axis="y", alpha=0.3)
    eje.legend()
    figura.tight_layout()
    figura.savefig(SALIDA / "comparacion_metricas.png", dpi=160)
    plt.close(figura)


def ejecutar(modelos: list[str]) -> None:
    SALIDA.mkdir(exist_ok=True)
    fijar_semilla()
    resultados, metadatos = [], []
    for nombre in modelos:
        for validacion in ("aleatoria", "por_estacion"):
            print(f"Ejecutando {nombre}: {validacion}")
            filas, _, metadata = evaluar(nombre, validacion)
            resultados.append(filas)
            metadatos.append(metadata)

    detalle = pd.concat(resultados, ignore_index=True)
    detalle.to_csv(SALIDA / "metricas_por_fold.csv", index=False)
    resumen = (
        detalle.groupby(["modelo_datos", "validacion", "clasificador"], as_index=False)
        .agg(
            accuracy_promedio=("accuracy", "mean"),
            accuracy_desviacion=("accuracy", "std"),
            f1_macro_promedio=("f1_macro", "mean"),
            f1_macro_desviacion=("f1_macro", "std"),
            f1_weighted_promedio=("f1_weighted", "mean"),
        )
    )
    resumen.to_csv(SALIDA / "comparacion_consolidada.csv", index=False)
    crear_grafico(resumen)
    archivos_datos = [
        DATOS_LOCALES / "base_fenologia_clima.csv",
        DATOS_LOCALES / "base_fenologia_clima_satelite.csv",
    ]
    configuracion = {
        "version_proyecto": VERSION_PROYECTO,
        "fecha_ejecucion": datetime.now().astimezone().isoformat(timespec="seconds"),
        "semilla": SEMILLA,
        "folds": FOLDS,
        "version_python": platform.python_version(),
        "version_scikit_learn": sklearn.__version__,
        "version_tensorflow": tf.__version__,
        "parametros_random_forest": PARAMETROS_RANDOM_FOREST,
        "archivos_datos": [
            {
                "archivo": ruta.name,
                "sha256": sha256_archivo(ruta),
            }
            for ruta in archivos_datos
        ],
        "modelos": metadatos,
        "nota": "La validacion por estacion usa StratifiedGroupKFold con s_id como grupo.",
    }
    (SALIDA / "configuracion_ejecucion.json").write_text(
        json.dumps(configuracion, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nResultado consolidado:")
    print(resumen.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline y validacion agrupada por estacion.")
    parser.add_argument("--modelo", choices=[*CONFIG, "todos"], default="todos")
    argumento = parser.parse_args()
    ejecutar(list(CONFIG) if argumento.modelo == "todos" else [argumento.modelo])
