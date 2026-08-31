"""Cálculo y evidencia de pesos de clase sin acceso al conjunto de prueba."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

FORMULA_PESO_BALANCEADO = (
    "n_train / (n_clases_presentes_train * frecuencia_clase_train)"
)


def calcular_pesos_clase(y_train: Sequence[int] | np.ndarray) -> dict[int, float]:
    """Calcula pesos balanceados utilizando exclusivamente etiquetas de train."""

    y_train_array = np.asarray(y_train)
    if y_train_array.ndim != 1:
        raise ValueError("y_train debe ser un arreglo unidimensional.")
    if len(y_train_array) == 0:
        raise ValueError("y_train no puede estar vacío.")
    clases = np.unique(y_train_array)
    pesos = compute_class_weight("balanced", classes=clases, y=y_train_array)
    return {
        int(clase): float(peso)
        for clase, peso in zip(clases.tolist(), pesos.tolist())
    }


def calcular_pesos_fold(
    y: Sequence[int] | np.ndarray,
    idx_train: Sequence[int] | np.ndarray,
) -> dict[int, float]:
    """Selecciona train antes de calcular; las demás etiquetas no intervienen."""

    y_array = np.asarray(y)
    indices = np.asarray(idx_train, dtype=np.int64)
    return calcular_pesos_clase(y_array[indices])


def crear_evidencia_pesos(
    modelo_datos: str,
    validacion: str,
    fold: int,
    fold_id: str,
    y_train: Sequence[int] | np.ndarray,
    nombres_clases: Sequence[object],
    clasificadores: Sequence[str],
) -> pd.DataFrame:
    """Registra frecuencias y el mismo peso aplicado por ambos modelos."""

    y_train_array = np.asarray(y_train)
    pesos = calcular_pesos_clase(y_train_array)
    clases_presentes, frecuencias = np.unique(y_train_array, return_counts=True)
    frecuencias_por_clase = {
        int(clase): int(frecuencia)
        for clase, frecuencia in zip(
            clases_presentes.tolist(), frecuencias.tolist()
        )
    }
    registros: list[dict[str, object]] = []
    for clasificador in clasificadores:
        for clase_codificada, clase in enumerate(nombres_clases):
            frecuencia = frecuencias_por_clase.get(clase_codificada, 0)
            registros.append(
                {
                    "modelo_datos": modelo_datos,
                    "validacion": validacion,
                    "clasificador": clasificador,
                    "fold": fold,
                    "fold_id": fold_id,
                    "n_train": int(len(y_train_array)),
                    "n_clases_presentes_train": int(len(clases_presentes)),
                    "clase_codificada": clase_codificada,
                    "clase": clase,
                    "frecuencia_train": frecuencia,
                    "peso_clase": pesos.get(clase_codificada, np.nan),
                    "peso_aplicado": clase_codificada in pesos,
                    "estrategia": "balanced_explicito",
                    "formula": FORMULA_PESO_BALANCEADO,
                    "calculado_solo_con_train": True,
                }
            )
    return pd.DataFrame.from_records(registros)
