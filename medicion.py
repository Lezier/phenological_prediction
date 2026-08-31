"""Tipos comunes para registrar tiempos de entrenamiento e inferencia."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ResultadoPrediccion:
    """Predicción y duraciones medidas con un reloj monotónico."""

    prediccion: np.ndarray
    tiempo_entrenamiento_segundos: float
    tiempo_inferencia_segundos: float
    epocas_ejecutadas: int | None = None
    early_stopping_detencion: bool | None = None

    def __post_init__(self) -> None:
        prediccion = np.asarray(self.prediccion)
        if prediccion.ndim != 1:
            raise ValueError("La predicción debe ser un arreglo unidimensional.")
        tiempos = (
            self.tiempo_entrenamiento_segundos,
            self.tiempo_inferencia_segundos,
        )
        if not all(np.isfinite(tiempo) and tiempo >= 0 for tiempo in tiempos):
            raise ValueError("Los tiempos deben ser finitos y no negativos.")
        if self.epocas_ejecutadas is not None and self.epocas_ejecutadas < 1:
            raise ValueError("Las épocas ejecutadas deben ser positivas.")
        if self.epocas_ejecutadas is None and self.early_stopping_detencion is not None:
            raise ValueError(
                "No se puede informar detención anticipada sin épocas ejecutadas."
            )
        object.__setattr__(self, "prediccion", prediccion)

    @property
    def tiempo_total_segundos(self) -> float:
        return self.tiempo_entrenamiento_segundos + self.tiempo_inferencia_segundos
