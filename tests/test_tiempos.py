"""Pruebas de tiempos y dispersión sin entrenamiento completo."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

import ejecutar
from configuracion import FOLDS
from medicion import ResultadoPrediccion


class TiemposYDispersionTest(unittest.TestCase):
    def test_resultado_rechaza_tiempos_invalidos(self) -> None:
        with self.assertRaises(ValueError):
            ResultadoPrediccion(np.array([0]), -0.1, 0.1)
        with self.assertRaises(ValueError):
            ResultadoPrediccion(np.array([0]), 0.1, np.nan)
        with self.assertRaises(ValueError):
            ResultadoPrediccion(np.array([[0]]), 0.1, 0.1)

    def test_random_forest_separa_entrenamiento_e_inferencia(self) -> None:
        modelo = Mock()
        modelo.predict.return_value = np.array([1])
        with (
            patch.object(ejecutar, "RandomForestClassifier", return_value=modelo),
            patch.object(
                ejecutar,
                "perf_counter",
                side_effect=[10.0, 10.4, 20.0, 20.1],
            ),
        ):
            resultado = ejecutar.entrenar_random_forest(
                np.array([[0.0], [1.0]]),
                np.array([0, 1]),
                np.array([[0.5]]),
                {0: 1.0, 1: 1.0},
            )

        self.assertAlmostEqual(resultado.tiempo_entrenamiento_segundos, 0.4)
        self.assertAlmostEqual(resultado.tiempo_inferencia_segundos, 0.1)
        self.assertAlmostEqual(resultado.tiempo_total_segundos, 0.5)
        modelo.fit.assert_called_once()
        modelo.predict.assert_called_once()

    def test_metricas_y_resumen_incluyen_media_y_desviacion(self) -> None:
        contador = {"red": 0, "bosque": 0}

        def red_simulada(_x_train, _y_train, x_test, _n_clases, _pesos):
            contador["red"] += 1
            fold = contador["red"]
            return ResultadoPrediccion(
                np.zeros(len(x_test), dtype=int), float(fold), fold / 10
            )

        def bosque_simulado(_x_train, _y_train, x_test, _pesos):
            contador["bosque"] += 1
            fold = contador["bosque"]
            return ResultadoPrediccion(
                np.zeros(len(x_test), dtype=int), fold / 2, fold / 20
            )

        with TemporaryDirectory() as temporal:
            with (
                patch.object(ejecutar, "SALIDA", Path(temporal)),
                patch.object(ejecutar, "entrenar_red", side_effect=red_simulada),
                patch.object(
                    ejecutar,
                    "entrenar_random_forest",
                    side_effect=bosque_simulado,
                ),
            ):
                metricas, _, _, _, _ = ejecutar.evaluar("a", "por_estacion")

        for columna in ejecutar.COLUMNAS_TIEMPOS[-3:]:
            self.assertIn(columna, metricas.columns)
            self.assertTrue(pd.api.types.is_numeric_dtype(metricas[columna]))
            self.assertTrue((metricas[columna] >= 0).all())
        np.testing.assert_allclose(
            metricas["tiempo_total_segundos"],
            metricas["tiempo_entrenamiento_segundos"]
            + metricas["tiempo_inferencia_segundos"],
        )

        resumen = ejecutar.crear_resumen(metricas)
        self.assertEqual(set(resumen["n_folds"]), {FOLDS})
        columnas_dispersion = [
            "accuracy_desviacion",
            "f1_macro_desviacion",
            "f1_weighted_desviacion",
            "tiempo_entrenamiento_desviacion_segundos",
            "tiempo_inferencia_desviacion_segundos",
            "tiempo_total_desviacion_segundos",
        ]
        for columna in columnas_dispersion:
            self.assertIn(columna, resumen.columns)
        fila_red = resumen[resumen["clasificador"] == "red_densa"].iloc[0]
        self.assertAlmostEqual(
            fila_red["tiempo_entrenamiento_promedio_segundos"], 3.0
        )
        self.assertAlmostEqual(
            fila_red["tiempo_entrenamiento_desviacion_segundos"],
            np.std([1, 2, 3, 4, 5], ddof=1),
        )


if __name__ == "__main__":
    unittest.main()
