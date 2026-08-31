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
from time import perf_counter

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
from sklearn.preprocessing import LabelEncoder, StandardScaler

from configuracion import (
    FOLDS,
    GRUPO,
    OBJETIVO,
    PARAMETROS_RANDOM_FOREST,
    PARAMETROS_RED_NEURONAL,
    PROCEDENCIA_HIPERPARAMETROS,
    SEMILLA,
    TRATAMIENTO_EARLY_STOPPING,
    VARIABLES_CLIMA,
    VARIABLE_NDVI,
    VERSION_PROYECTO,
    parametros_random_forest_efectivos,
)
from folds import (
    COLUMNA_FILA_FUENTE,
    FoldCompartido,
    construir_folds,
    crear_asignacion_folds,
)
from medicion import ResultadoPrediccion
from ponderacion import calcular_pesos_fold, crear_evidencia_pesos

CLIMA = VARIABLES_CLIMA
CLASIFICADORES = ("red_densa", "random_forest")
COLUMNAS_TIEMPOS = [
    "modelo_datos",
    "validacion",
    "clasificador",
    "fold",
    "fold_id",
    "n_train",
    "n_test",
    "tiempo_entrenamiento_segundos",
    "tiempo_inferencia_segundos",
    "tiempo_total_segundos",
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
        CLIMA + [VARIABLE_NDVI],
        "Modelo B - clima + NDVI",
    ),
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
    datos[COLUMNA_FILA_FUENTE] = datos.index

    if nombre in {"aprima", "b"}:
        datos = datos[datos[VARIABLE_NDVI].notna()].copy()

    # En A se conservan las filas con radiacion ausente para imputarlas dentro
    # de cada fold usando exclusivamente las observaciones de entrenamiento.
    if nombre == "a":
        requeridas = [
            columna for columna in variables if columna != "clima_radiacion_media"
        ] + [OBJETIVO, GRUPO]
    else:
        requeridas = variables + [OBJETIVO, GRUPO]
    datos = datos.dropna(subset=requeridas).reset_index(drop=True)
    return datos, variables, titulo


def crear_red(n_variables: int, n_clases: int) -> tf.keras.Model:
    capas: list[tf.keras.layers.Layer] = [tf.keras.layers.Input(shape=(n_variables,))]
    for configuracion_capa in PARAMETROS_RED_NEURONAL["capas_ocultas"]:
        capas.append(
            tf.keras.layers.Dense(
                configuracion_capa["unidades"],
                activation=configuracion_capa["activacion"],
            )
        )
        capas.append(tf.keras.layers.Dropout(configuracion_capa["dropout"]))
    capas.append(
        tf.keras.layers.Dense(
            n_clases, activation=PARAMETROS_RED_NEURONAL["activacion_salida"]
        )
    )
    red = tf.keras.Sequential(capas)
    red.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=PARAMETROS_RED_NEURONAL["learning_rate"]
        ),
        loss=PARAMETROS_RED_NEURONAL["funcion_perdida"],
        metrics=PARAMETROS_RED_NEURONAL["metricas"],
    )
    return red


def entrenar_red(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    n_clases: int,
    pesos_clase: dict[int, float],
) -> ResultadoPrediccion:
    inicio_entrenamiento = perf_counter()
    escalador = StandardScaler()
    x_train = escalador.fit_transform(x_train)
    red = crear_red(x_train.shape[1], n_clases)
    parada = tf.keras.callbacks.EarlyStopping(
        **PARAMETROS_RED_NEURONAL["early_stopping"]
    )
    historial = red.fit(
        x_train,
        y_train,
        epochs=PARAMETROS_RED_NEURONAL["epocas_maximas"],
        batch_size=PARAMETROS_RED_NEURONAL["batch_size"],
        class_weight=pesos_clase,
        callbacks=[parada],
        verbose=0,
    )
    epocas_ejecutadas = len(historial.history.get("loss", []))
    if epocas_ejecutadas < 1:
        raise RuntimeError("Keras no informó épocas ejecutadas en history['loss'].")
    tiempo_entrenamiento = perf_counter() - inicio_entrenamiento

    inicio_inferencia = perf_counter()
    x_test = escalador.transform(x_test)
    prediccion = np.argmax(red.predict(x_test, verbose=0), axis=1)
    tiempo_inferencia = perf_counter() - inicio_inferencia
    tf.keras.backend.clear_session()
    return ResultadoPrediccion(
        prediccion,
        tiempo_entrenamiento,
        tiempo_inferencia,
        epocas_ejecutadas=epocas_ejecutadas,
        early_stopping_detencion=(
            epocas_ejecutadas < PARAMETROS_RED_NEURONAL["epocas_maximas"]
        ),
    )


def entrenar_random_forest(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    pesos_clase: dict[int, float],
) -> ResultadoPrediccion:
    inicio_entrenamiento = perf_counter()
    parametros = {**PARAMETROS_RANDOM_FOREST, "class_weight": pesos_clase}
    bosque = RandomForestClassifier(**parametros)
    bosque.fit(x_train, y_train)
    tiempo_entrenamiento = perf_counter() - inicio_entrenamiento

    inicio_inferencia = perf_counter()
    prediccion = bosque.predict(x_test)
    tiempo_inferencia = perf_counter() - inicio_inferencia
    return ResultadoPrediccion(
        prediccion,
        tiempo_entrenamiento,
        tiempo_inferencia,
    )


def preparar_folds(
    nombre: str,
    validacion: str,
    datos: pd.DataFrame,
    y: np.ndarray,
    grupos: np.ndarray,
) -> tuple[list[FoldCompartido], pd.DataFrame]:
    """Materializa una vez los folds y la evidencia para ambos modelos."""

    folds = construir_folds(y, grupos, validacion)
    asignacion = crear_asignacion_folds(
        nombre,
        validacion,
        folds,
        datos[COLUMNA_FILA_FUENTE].to_numpy(),
        grupos,
        datos[OBJETIVO].to_numpy(),
        CLASIFICADORES,
    )
    return folds, asignacion


def generar_asignacion_folds(modelos: list[str]) -> pd.DataFrame:
    """Genera la evidencia de particiones sin entrenar clasificadores."""

    asignaciones = []
    for nombre in modelos:
        datos, _, _ = cargar_datos(nombre)
        y = LabelEncoder().fit_transform(datos[OBJETIVO])
        grupos = datos[GRUPO].to_numpy()
        for validacion in ("aleatoria", "por_estacion"):
            _, asignacion = preparar_folds(
                nombre, validacion, datos, y, grupos
            )
            asignaciones.append(asignacion)
    return pd.concat(asignaciones, ignore_index=True)


def exportar_asignacion_folds(modelos: list[str]) -> Path:
    """Escribe el CSV determinista de folds sin ejecutar entrenamiento."""

    SALIDA.mkdir(exist_ok=True)
    ruta = SALIDA / "asignacion_folds.csv"
    generar_asignacion_folds(modelos).to_csv(ruta, index=False)
    return ruta


def generar_evidencia_pesos(modelos: list[str]) -> pd.DataFrame:
    """Genera frecuencias y pesos por fold sin entrenar clasificadores."""

    evidencias = []
    for nombre in modelos:
        datos, _, _ = cargar_datos(nombre)
        codificador = LabelEncoder()
        y = codificador.fit_transform(datos[OBJETIVO])
        grupos = datos[GRUPO].to_numpy()
        for validacion in ("aleatoria", "por_estacion"):
            folds, _ = preparar_folds(nombre, validacion, datos, y, grupos)
            for fold in folds:
                evidencias.append(
                    crear_evidencia_pesos(
                        nombre,
                        validacion,
                        fold.numero,
                        fold.fold_id,
                        y[fold.idx_train],
                        codificador.classes_,
                        CLASIFICADORES,
                    )
                )
    return pd.concat(evidencias, ignore_index=True)


def exportar_evidencia_pesos(modelos: list[str]) -> Path:
    """Escribe el CSV determinista de pesos sin ejecutar entrenamiento."""

    SALIDA.mkdir(exist_ok=True)
    ruta = SALIDA / "pesos_clase_por_fold.csv"
    generar_evidencia_pesos(modelos).to_csv(ruta, index=False)
    return ruta


def evaluar(
    nombre: str,
    validacion: str,
    *,
    fold_numero: int | None = None,
    directorio_salida: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    salida_evaluacion = SALIDA if directorio_salida is None else directorio_salida
    salida_evaluacion.mkdir(parents=True, exist_ok=True)
    datos, variables, titulo = cargar_datos(nombre)
    codificador = LabelEncoder()
    y = codificador.fit_transform(datos[OBJETIVO])
    x = datos[variables].to_numpy()
    grupos = datos[GRUPO].to_numpy()
    folds, asignacion = preparar_folds(nombre, validacion, datos, y, grupos)
    if fold_numero is not None:
        folds = [fold for fold in folds if fold.numero == fold_numero]
        if not folds:
            raise ValueError(f"El fold debe estar entre 1 y {FOLDS}.")
        asignacion = asignacion[asignacion["fold"] == fold_numero].reset_index(
            drop=True
        )
    acumuladas = {
        "red_densa": np.zeros((len(codificador.classes_), len(codificador.classes_)), dtype=int),
        "random_forest": np.zeros((len(codificador.classes_), len(codificador.classes_)), dtype=int),
    }
    filas = []
    evidencias_pesos = []

    for fold_compartido in folds:
        fold = fold_compartido.numero
        idx_train = fold_compartido.idx_train
        idx_test = fold_compartido.idx_test
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
        pesos_clase = calcular_pesos_fold(y, idx_train)
        evidencias_pesos.append(
            crear_evidencia_pesos(
                nombre,
                validacion,
                fold,
                fold_compartido.fold_id,
                y[idx_train],
                codificador.classes_,
                CLASIFICADORES,
            )
        )
        for clasificador, entrenar in {
            "red_densa": lambda: entrenar_red(
                x_train,
                y[idx_train],
                x_test,
                len(codificador.classes_),
                pesos_clase,
            ),
            "random_forest": lambda: entrenar_random_forest(
                x_train, y[idx_train], x_test, pesos_clase
            ),
        }.items():
            resultado = entrenar()
            pred = resultado.prediccion
            acumuladas[clasificador] += confusion_matrix(
                y[idx_test], pred, labels=range(len(codificador.classes_))
            )
            filas.append(
                {
                    "modelo_datos": nombre,
                    "validacion": validacion,
                    "clasificador": clasificador,
                    "fold": fold,
                    "fold_id": fold_compartido.fold_id,
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
                    "tiempo_entrenamiento_segundos": (
                        resultado.tiempo_entrenamiento_segundos
                    ),
                    "tiempo_inferencia_segundos": (
                        resultado.tiempo_inferencia_segundos
                    ),
                    "tiempo_total_segundos": resultado.tiempo_total_segundos,
                    "epocas_ejecutadas": resultado.epocas_ejecutadas,
                    "early_stopping_detencion": (
                        resultado.early_stopping_detencion
                    ),
                }
            )

    matrices = []
    for clasificador, matriz in acumuladas.items():
        tabla = pd.DataFrame(matriz, index=codificador.classes_, columns=codificador.classes_)
        tabla.index.name = "clase_real"
        tabla.to_csv(
            salida_evaluacion
            / f"matriz_{nombre}_{validacion}_{clasificador}.csv"
        )
        matrices.append(tabla.assign(clasificador=clasificador, clase_real=tabla.index))

    metadata = {
        "modelo_datos": nombre,
        "titulo": titulo,
        "validacion": validacion,
        "grupo": GRUPO if validacion == "por_estacion" else None,
        "muestras": int(len(datos)),
        "estaciones": int(datos[GRUPO].nunique()),
        "variables": variables,
        "clases": list(codificador.classes_),
        "imputacion": "Mediana ajustada exclusivamente con train dentro de cada fold.",
        "folds_compartidos": True,
        "clasificadores": list(CLASIFICADORES),
        "tratamiento_early_stopping": TRATAMIENTO_EARLY_STOPPING,
        "folds_ejecutados": [fold.numero for fold in folds],
    }
    return (
        pd.DataFrame(filas),
        pd.concat(matrices, ignore_index=True),
        metadata,
        asignacion,
        pd.concat(evidencias_pesos, ignore_index=True),
    )


def crear_grafico(
    resumen: pd.DataFrame, directorio_salida: Path | None = None
) -> None:
    salida_grafico = SALIDA if directorio_salida is None else directorio_salida
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
    figura.savefig(salida_grafico / "comparacion_metricas.png", dpi=160)
    plt.close(figura)


def crear_resumen(detalle: pd.DataFrame) -> pd.DataFrame:
    """Consolida media y dispersión entre folds para métricas y tiempos."""

    return (
        detalle.groupby(
            ["modelo_datos", "validacion", "clasificador"], as_index=False
        )
        .agg(
            n_folds=("fold", "nunique"),
            accuracy_promedio=("accuracy", "mean"),
            accuracy_desviacion=("accuracy", "std"),
            f1_macro_promedio=("f1_macro", "mean"),
            f1_macro_desviacion=("f1_macro", "std"),
            f1_weighted_promedio=("f1_weighted", "mean"),
            f1_weighted_desviacion=("f1_weighted", "std"),
            tiempo_entrenamiento_promedio_segundos=(
                "tiempo_entrenamiento_segundos",
                "mean",
            ),
            tiempo_entrenamiento_desviacion_segundos=(
                "tiempo_entrenamiento_segundos",
                "std",
            ),
            tiempo_inferencia_promedio_segundos=(
                "tiempo_inferencia_segundos",
                "mean",
            ),
            tiempo_inferencia_desviacion_segundos=(
                "tiempo_inferencia_segundos",
                "std",
            ),
            tiempo_total_promedio_segundos=("tiempo_total_segundos", "mean"),
            tiempo_total_desviacion_segundos=("tiempo_total_segundos", "std"),
        )
    )


def ejecutar_ensayo_reducido(
    nombre: str = "a",
    validacion: str = "por_estacion",
    fold_numero: int = 1,
    directorio_salida: Path | None = None,
) -> Path:
    """Ejecuta ambos modelos en un fold y separa sus resultados oficiales."""

    salida_ensayo = (
        SALIDA / "ensayo_reducido"
        if directorio_salida is None
        else directorio_salida
    )
    salida_ensayo.mkdir(parents=True, exist_ok=True)
    fijar_semilla()
    detalle, _, metadata, asignacion, pesos = evaluar(
        nombre,
        validacion,
        fold_numero=fold_numero,
        directorio_salida=salida_ensayo,
    )

    rutas_csv = {
        "metricas": salida_ensayo / "metricas_por_fold.csv",
        "tiempos": salida_ensayo / "tiempos_por_fold.csv",
        "resumen": salida_ensayo / "comparacion_consolidada.csv",
        "folds": salida_ensayo / "asignacion_folds.csv",
        "pesos": salida_ensayo / "pesos_clase_por_fold.csv",
    }
    detalle.to_csv(rutas_csv["metricas"], index=False)
    detalle[COLUMNAS_TIEMPOS].to_csv(rutas_csv["tiempos"], index=False)
    resumen = crear_resumen(detalle)
    resumen.to_csv(rutas_csv["resumen"], index=False)
    asignacion.to_csv(rutas_csv["folds"], index=False)
    pesos.to_csv(rutas_csv["pesos"], index=False)
    crear_grafico(resumen, salida_ensayo)

    advertencia = salida_ensayo / "NO_USAR_EN_INFORME.md"
    advertencia.write_text(
        "# Ensayo reducido de CP08\n\n"
        "Estos resultados ejecutan un solo fold para verificar el pipeline. "
        "No son resultados oficiales de RC3 y no deben copiarse al informe, "
        "presentación ni comparación final.\n",
        encoding="utf-8",
    )
    configuracion = {
        "version_proyecto": VERSION_PROYECTO,
        "tipo_ejecucion": "ensayo_reducido_cp08_no_oficial",
        "usar_en_informe": False,
        "fecha_ejecucion": datetime.now().astimezone().isoformat(timespec="seconds"),
        "modelo_datos": nombre,
        "validacion": validacion,
        "fold": fold_numero,
        "semilla": SEMILLA,
        "version_python": platform.python_version(),
        "version_scikit_learn": sklearn.__version__,
        "version_tensorflow": tf.__version__,
        "metadata": metadata,
        "parametros_random_forest": PARAMETROS_RANDOM_FOREST,
        "parametros_red_neuronal": PARAMETROS_RED_NEURONAL,
        "tratamiento_early_stopping": TRATAMIENTO_EARLY_STOPPING,
    }
    ruta_configuracion = salida_ensayo / "configuracion_ensayo.json"
    ruta_configuracion.write_text(
        json.dumps(configuracion, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    artefactos = [
        *rutas_csv.values(),
        salida_ensayo / "comparacion_metricas.png",
        salida_ensayo / f"matriz_{nombre}_{validacion}_red_densa.csv",
        salida_ensayo / f"matriz_{nombre}_{validacion}_random_forest.csv",
        advertencia,
        ruta_configuracion,
    ]
    manifiesto = {
        "tipo_ejecucion": "ensayo_reducido_cp08_no_oficial",
        "usar_en_informe": False,
        "artefactos": [
            {"archivo": ruta.name, "sha256": sha256_archivo(ruta)}
            for ruta in artefactos
        ],
    }
    (salida_ensayo / "manifest_ensayo.json").write_text(
        json.dumps(manifiesto, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nResultado del ensayo reducido (NO OFICIAL):")
    print(resumen.to_string(index=False))
    return salida_ensayo


def ejecutar(modelos: list[str]) -> None:
    SALIDA.mkdir(exist_ok=True)
    fijar_semilla()
    resultados, metadatos, asignaciones, ponderaciones = [], [], [], []
    for nombre in modelos:
        for validacion in ("aleatoria", "por_estacion"):
            print(f"Ejecutando {nombre}: {validacion}")
            filas, _, metadata, asignacion, pesos = evaluar(nombre, validacion)
            resultados.append(filas)
            metadatos.append(metadata)
            asignaciones.append(asignacion)
            ponderaciones.append(pesos)

    detalle = pd.concat(resultados, ignore_index=True)
    detalle.to_csv(SALIDA / "metricas_por_fold.csv", index=False)
    ruta_tiempos = SALIDA / "tiempos_por_fold.csv"
    detalle[COLUMNAS_TIEMPOS].to_csv(ruta_tiempos, index=False)
    ruta_asignacion = SALIDA / "asignacion_folds.csv"
    pd.concat(asignaciones, ignore_index=True).to_csv(ruta_asignacion, index=False)
    ruta_pesos = SALIDA / "pesos_clase_por_fold.csv"
    pd.concat(ponderaciones, ignore_index=True).to_csv(ruta_pesos, index=False)
    resumen = crear_resumen(detalle)
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
        "parametros_random_forest_explicitos": PARAMETROS_RANDOM_FOREST,
        "parametros_random_forest_efectivos": parametros_random_forest_efectivos(),
        "nota_class_weight_random_forest": (
            "balanced declara la estrategia base; en cada fold se reemplaza "
            "por el diccionario explicito registrado en "
            "pesos_clase_por_fold.csv."
        ),
        "parametros_red_neuronal": PARAMETROS_RED_NEURONAL,
        "tratamiento_early_stopping": TRATAMIENTO_EARLY_STOPPING,
        "procedencia_hiperparametros": PROCEDENCIA_HIPERPARAMETROS,
        "evidencia_folds": {
            "archivo": ruta_asignacion.name,
            "sha256": sha256_archivo(ruta_asignacion),
            "clasificadores": list(CLASIFICADORES),
            "descripcion": (
                "Cada fold se materializa una vez y se reutiliza para ambos "
                "clasificadores; el CSV repite la misma asignacion para "
                "hacer comprobable su igualdad."
            ),
        },
        "evidencia_ponderacion_clases": {
            "archivo": ruta_pesos.name,
            "sha256": sha256_archivo(ruta_pesos),
            "estrategia": "balanced_explicito",
            "calculado_solo_con_train": True,
            "clasificadores": list(CLASIFICADORES),
        },
        "medicion_tiempos": {
            "archivo": ruta_tiempos.name,
            "reloj": "time.perf_counter",
            "unidad": "segundos",
            "incluido_en_fold_id": False,
            "nota": (
                "Los tiempos dependen del entorno de ejecucion y no forman "
                "parte de los identificadores deterministas de folds."
            ),
        },
        "archivos_datos": [
            {
                "archivo": ruta.name,
                "sha256": sha256_archivo(ruta),
            }
            for ruta in archivos_datos
        ],
        "modelos": metadatos,
        "nota": (
            "La validacion por estacion usa StratifiedGroupKFold con "
            f"{GRUPO} como grupo."
        ),
    }
    (SALIDA / "configuracion_ejecucion.json").write_text(
        json.dumps(configuracion, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nResultado consolidado:")
    print(resumen.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline y validacion agrupada por estacion.")
    parser.add_argument("--modelo", choices=[*CONFIG, "todos"], default="todos")
    grupo_modo = parser.add_mutually_exclusive_group()
    grupo_modo.add_argument(
        "--solo-folds",
        action="store_true",
        help="Exporta asignacion_folds.csv sin entrenar los clasificadores.",
    )
    grupo_modo.add_argument(
        "--solo-pesos",
        action="store_true",
        help="Exporta pesos_clase_por_fold.csv sin entrenar clasificadores.",
    )
    grupo_modo.add_argument(
        "--ensayo-reducido",
        action="store_true",
        help="Ejecuta A, validacion por estacion, fold 1, como prueba no oficial.",
    )
    argumento = parser.parse_args()
    modelos_seleccionados = (
        list(CONFIG) if argumento.modelo == "todos" else [argumento.modelo]
    )
    if argumento.solo_folds:
        ruta_generada = exportar_asignacion_folds(modelos_seleccionados)
        print(f"Evidencia de folds generada en: {ruta_generada}")
    elif argumento.solo_pesos:
        ruta_generada = exportar_evidencia_pesos(modelos_seleccionados)
        print(f"Evidencia de pesos generada en: {ruta_generada}")
    elif argumento.ensayo_reducido:
        if modelos_seleccionados != ["a"]:
            parser.error("--ensayo-reducido requiere --modelo a")
        ruta_generada = ejecutar_ensayo_reducido()
        print(f"Ensayo reducido generado en: {ruta_generada}")
    else:
        ejecutar(modelos_seleccionados)
