"""Pruebas acotadas de la configuración central de RC3."""

from __future__ import annotations

import unittest

from configuracion import (
    FOLDS,
    PARAMETROS_RANDOM_FOREST,
    PARAMETROS_RED_NEURONAL,
    PROCEDENCIA_HIPERPARAMETROS,
    SEMILLA,
    VARIABLES_CLIMA,
    VERSION_PROYECTO,
    parametros_random_forest_efectivos,
)


class ConfiguracionRC3Test(unittest.TestCase):
    def test_configuracion_general(self) -> None:
        self.assertEqual(VERSION_PROYECTO, "0.1.0-rc.3")
        self.assertEqual(SEMILLA, 42)
        self.assertEqual(FOLDS, 5)
        self.assertEqual(len(VARIABLES_CLIMA), 7)
        self.assertEqual(len(set(VARIABLES_CLIMA)), 7)

    def test_random_forest_explicito_y_efectivo(self) -> None:
        self.assertEqual(
            PARAMETROS_RANDOM_FOREST,
            {
                "n_estimators": 400,
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": -1,
            },
        )
        efectivos = parametros_random_forest_efectivos()
        for nombre, valor in PARAMETROS_RANDOM_FOREST.items():
            self.assertEqual(efectivos[nombre], valor)
        self.assertEqual(efectivos["criterion"], "gini")
        self.assertIsNone(efectivos["max_depth"])
        self.assertEqual(efectivos["min_samples_split"], 2)
        self.assertEqual(efectivos["min_samples_leaf"], 1)
        self.assertEqual(efectivos["max_features"], "sqrt")
        self.assertTrue(efectivos["bootstrap"])
        self.assertFalse(
            PROCEDENCIA_HIPERPARAMETROS["random_forest"]["tuning_sistematico"]
        )

    def test_red_neuronal_observada(self) -> None:
        self.assertEqual(
            PARAMETROS_RED_NEURONAL["capas_ocultas"],
            [
                {"unidades": 16, "activacion": "relu", "dropout": 0.3},
                {"unidades": 8, "activacion": "relu", "dropout": 0.2},
            ],
        )
        self.assertEqual(PARAMETROS_RED_NEURONAL["learning_rate"], 0.001)
        self.assertEqual(PARAMETROS_RED_NEURONAL["epocas_maximas"], 60)
        self.assertEqual(PARAMETROS_RED_NEURONAL["batch_size"], 16)
        self.assertEqual(
            PARAMETROS_RED_NEURONAL["early_stopping"],
            {"monitor": "loss", "patience": 8, "restore_best_weights": True},
        )
        self.assertIsNone(
            PROCEDENCIA_HIPERPARAMETROS["red_neuronal"]["tuning_sistematico"]
        )

    def test_red_neuronal_se_construye_desde_la_configuracion(self) -> None:
        from ejecutar import crear_red

        red = crear_red(n_variables=7, n_clases=5)
        self.assertEqual(
            [capa.__class__.__name__ for capa in red.layers],
            ["Dense", "Dropout", "Dense", "Dropout", "Dense"],
        )
        self.assertEqual([red.layers[0].units, red.layers[2].units], [16, 8])
        self.assertEqual([red.layers[1].rate, red.layers[3].rate], [0.3, 0.2])
        self.assertEqual(red.layers[-1].units, 5)


if __name__ == "__main__":
    unittest.main()
