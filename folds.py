"""Construcción y evidencia auditable de folds compartidos."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from configuracion import FOLDS, SEMILLA

VALIDACIONES = ("aleatoria", "por_estacion")
COLUMNA_FILA_FUENTE = "_fila_fuente_csv"


@dataclass(frozen=True)
class FoldCompartido:
    """Índices inmutables por contrato y su identificación reproducible."""

    numero: int
    fold_id: str
    idx_train: np.ndarray
    idx_test: np.ndarray


def _fold_id(
    validacion: str,
    numero: int,
    idx_train: np.ndarray,
    idx_test: np.ndarray,
    semilla: int,
    n_splits: int,
) -> str:
    encabezado = json.dumps(
        {
            "validacion": validacion,
            "fold": numero,
            "semilla": semilla,
            "n_splits": n_splits,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    resumen = hashlib.sha256(encabezado)
    resumen.update(np.asarray(idx_train, dtype="<i8").tobytes())
    resumen.update(b"|")
    resumen.update(np.asarray(idx_test, dtype="<i8").tobytes())
    return resumen.hexdigest().upper()


def construir_folds(
    y: Sequence[object] | np.ndarray,
    grupos: Sequence[object] | np.ndarray,
    validacion: str,
    *,
    semilla: int = SEMILLA,
    n_splits: int = FOLDS,
) -> list[FoldCompartido]:
    """Construye una sola lista materializada para todos los clasificadores."""

    if validacion not in VALIDACIONES:
        raise ValueError(f"Validación desconocida: {validacion!r}.")
    y_array = np.asarray(y)
    grupos_array = np.asarray(grupos)
    if y_array.ndim != 1 or grupos_array.ndim != 1:
        raise ValueError("y y grupos deben ser arreglos unidimensionales.")
    if len(y_array) != len(grupos_array):
        raise ValueError("y y grupos deben tener la misma cantidad de filas.")

    indices = np.arange(len(y_array), dtype=np.int64)
    if validacion == "aleatoria":
        divisor = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=semilla
        )
        particiones = divisor.split(indices, y_array)
    else:
        divisor = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=semilla
        )
        particiones = divisor.split(indices, y_array, groups=grupos_array)

    folds = []
    for numero, (idx_train, idx_test) in enumerate(particiones, start=1):
        train = np.asarray(idx_train, dtype=np.int64)
        test = np.asarray(idx_test, dtype=np.int64)
        train.setflags(write=False)
        test.setflags(write=False)
        folds.append(
            FoldCompartido(
                numero=numero,
                fold_id=_fold_id(
                    validacion, numero, train, test, semilla, n_splits
                ),
                idx_train=train,
                idx_test=test,
            )
        )
    return folds


def crear_asignacion_folds(
    modelo_datos: str,
    validacion: str,
    folds: Sequence[FoldCompartido],
    filas_fuente: Sequence[int] | np.ndarray,
    grupos: Sequence[object] | np.ndarray,
    clases: Sequence[object] | np.ndarray,
    clasificadores: Sequence[str],
) -> pd.DataFrame:
    """Expande la misma asignación para hacer auditable su reutilización."""

    filas_fuente_array = np.asarray(filas_fuente)
    grupos_array = np.asarray(grupos)
    clases_array = np.asarray(clases)
    cantidad = len(clases_array)
    if not (len(filas_fuente_array) == len(grupos_array) == cantidad):
        raise ValueError("Filas, grupos y clases deben tener igual longitud.")

    registros: list[dict[str, object]] = []
    for fold in folds:
        for particion, indices in (
            ("train", fold.idx_train),
            ("test", fold.idx_test),
        ):
            for clasificador in clasificadores:
                for indice in indices:
                    registros.append(
                        {
                            "modelo_datos": modelo_datos,
                            "validacion": validacion,
                            "clasificador": clasificador,
                            "fold": fold.numero,
                            "fold_id": fold.fold_id,
                            "particion": particion,
                            "fila_preparada": int(indice),
                            "fila_fuente_csv": int(filas_fuente_array[indice]),
                            "estacion": grupos_array[indice],
                            "clase": clases_array[indice],
                        }
                    )
    return pd.DataFrame.from_records(registros)
