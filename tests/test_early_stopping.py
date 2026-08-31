"""Pruebas de la posición conservadora sobre Early Stopping."""

from __future__ import annotations

import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import ejecutar
from configuracion import (
    PARAMETROS_RED_NEURONAL,
    PROCEDENCIA_HIPERPARAMETROS,
    TRATAMIENTO_EARLY_STOPPING,
)


class EarlyStoppingConservadorTest(unittest.TestCase):
    def test_decision_conservadora_esta_declarada(self) -> None:
        self.assertEqual(
            PARAMETROS_RED_NEURONAL["early_stopping"],
            {"monitor": "loss", "patience": 8, "restore_best_weights": True},
        )
        self.assertFalse(TRATAMIENTO_EARLY_STOPPING["usa_validacion_interna"])
        self.assertFalse(TRATAMIENTO_EARLY_STOPPING["usa_fold_externo_test"])
        self.assertFalse(
            TRATAMIENTO_EARLY_STOPPING[
                "interpretable_como_control_de_overfitting"
            ]
        )
        procedencia = PROCEDENCIA_HIPERPARAMETROS["red_neuronal"]
        self.assertFalse(procedencia["respuesta_equipo_recibida"])
        self.assertIsNone(procedencia["tuning_sistematico"])

    def test_entrenamiento_no_recibe_y_test(self) -> None:
        parametros = inspect.signature(ejecutar.entrenar_red).parameters
        self.assertNotIn("y_test", parametros)

    def test_fit_no_usa_test_ni_validacion_interna(self) -> None:
        red = Mock()
        red.fit.return_value = SimpleNamespace(history={"loss": [1.0] * 12})
        red.predict.return_value = np.array([[0.2, 0.8], [0.7, 0.3]])
        x_train = np.array([[0.0], [1.0], [2.0], [3.0]])
        y_train = np.array([0, 0, 1, 1])
        x_test = np.array([[100.0], [200.0]])

        with (
            patch.object(ejecutar, "crear_red", return_value=red),
            patch.object(
                ejecutar,
                "perf_counter",
                side_effect=[10.0, 10.5, 20.0, 20.1],
            ),
            patch.object(ejecutar.tf.keras.backend, "clear_session"),
        ):
            resultado = ejecutar.entrenar_red(
                x_train,
                y_train,
                x_test,
                n_clases=2,
                pesos_clase={0: 1.0, 1: 1.0},
            )

        argumentos_fit = red.fit.call_args.args
        opciones_fit = red.fit.call_args.kwargs
        self.assertEqual(len(argumentos_fit[0]), len(x_train))
        np.testing.assert_array_equal(argumentos_fit[1], y_train)
        self.assertNotIn("validation_data", opciones_fit)
        self.assertNotIn("validation_split", opciones_fit)
        self.assertNotIn("y_test", opciones_fit)

        callback = opciones_fit["callbacks"][0]
        self.assertEqual(callback.monitor, "loss")
        self.assertEqual(callback.patience, 8)
        self.assertTrue(callback.restore_best_weights)
        self.assertEqual(len(red.predict.call_args.args[0]), len(x_test))
        self.assertEqual(resultado.epocas_ejecutadas, 12)
        self.assertTrue(resultado.early_stopping_detencion)


if __name__ == "__main__":
    unittest.main()
