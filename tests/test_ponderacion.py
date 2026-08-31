"""Pruebas de ponderación de clases calculada solo desde train."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

import ejecutar
from configuracion import FOLDS
from medicion import ResultadoPrediccion
from ponderacion import (
    FORMULA_PESO_BALANCEADO,
    calcular_pesos_clase,
    calcular_pesos_fold,
    crear_evidencia_pesos,
)


class PonderacionClasesTest(unittest.TestCase):
    def test_formula_balanceada(self) -> None:
        pesos = calcular_pesos_clase(np.array([0, 0, 0, 1]))
        self.assertAlmostEqual(pesos[0], 4 / (2 * 3))
        self.assertAlmostEqual(pesos[1], 4 / (2 * 1))

    def test_modificar_y_test_no_cambia_los_pesos(self) -> None:
        y_original = np.array([0, 0, 0, 1, 1, 2, 2, 2])
        idx_train = np.array([0, 1, 2, 3, 4])
        y_test_modificado = y_original.copy()
        y_test_modificado[[5, 6, 7]] = [0, 1, 1]

        self.assertEqual(
            calcular_pesos_fold(y_original, idx_train),
            calcular_pesos_fold(y_test_modificado, idx_train),
        )

    def test_evidencia_incluye_clases_ausentes_y_es_compartida(self) -> None:
        evidencia = crear_evidencia_pesos(
            "a",
            "por_estacion",
            1,
            "FOLD-DE-PRUEBA",
            np.array([0, 0, 1]),
            ["clase_a", "clase_b", "clase_c"],
            ejecutar.CLASIFICADORES,
        )
        columnas_comparables = [
            columna
            for columna in evidencia.columns
            if columna != "clasificador"
        ]
        red = (
            evidencia[evidencia["clasificador"] == "red_densa"]
            [columnas_comparables]
            .reset_index(drop=True)
        )
        bosque = (
            evidencia[evidencia["clasificador"] == "random_forest"]
            [columnas_comparables]
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(red, bosque)

        ausente = red[red["clase_codificada"] == 2].iloc[0]
        self.assertEqual(ausente["frecuencia_train"], 0)
        self.assertTrue(pd.isna(ausente["peso_clase"]))
        self.assertFalse(ausente["peso_aplicado"])
        self.assertEqual(ausente["formula"], FORMULA_PESO_BALANCEADO)
        self.assertTrue(ausente["calculado_solo_con_train"])

    def test_pipeline_entrega_el_mismo_diccionario_a_ambos_modelos(self) -> None:
        pesos_red = []
        pesos_bosque = []

        def red_simulada(x_train, _y_train, x_test, _n_clases, pesos_clase):
            pesos_red.append(pesos_clase.copy())
            return ResultadoPrediccion(
                np.zeros(len(x_test), dtype=int), 1.0, 0.1
            )

        def bosque_simulado(x_train, _y_train, x_test, pesos_clase):
            pesos_bosque.append(pesos_clase.copy())
            return ResultadoPrediccion(
                np.zeros(len(x_test), dtype=int), 0.5, 0.05
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
                _, _, _, _, evidencia = ejecutar.evaluar("a", "por_estacion")

        self.assertEqual(len(pesos_red), FOLDS)
        self.assertEqual(pesos_red, pesos_bosque)
        self.assertEqual(len(evidencia), FOLDS * 5 * 2)
        self.assertTrue(evidencia["calculado_solo_con_train"].all())

    def test_generacion_completa_sin_entrenamiento(self) -> None:
        evidencia = ejecutar.generar_evidencia_pesos(["a", "aprima", "b"])
        self.assertEqual(len(evidencia), 3 * 2 * FOLDS * 5 * 2)
        self.assertEqual(
            set(evidencia["clasificador"]), set(ejecutar.CLASIFICADORES)
        )
        self.assertTrue(evidencia["calculado_solo_con_train"].all())
        self.assertTrue((evidencia["frecuencia_train"] >= 0).all())

    def test_random_forest_recibe_pesos_explicitos(self) -> None:
        pesos = {0: 0.75, 1: 1.5}
        modelo_simulado = unittest.mock.Mock()
        modelo_simulado.predict.return_value = np.array([0])
        with patch.object(
            ejecutar, "RandomForestClassifier", return_value=modelo_simulado
        ) as constructor:
            ejecutar.entrenar_random_forest(
                np.array([[0.0], [1.0]]),
                np.array([0, 1]),
                np.array([[0.5]]),
                pesos,
            )
        self.assertEqual(constructor.call_args.kwargs["class_weight"], pesos)


if __name__ == "__main__":
    unittest.main()
