"""Pruebas de construcción y reutilización de folds sin entrenar modelos."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

import ejecutar
from configuracion import FOLDS, GRUPO, OBJETIVO, SEMILLA
from ejecutar import CLASIFICADORES, cargar_datos
from folds import COLUMNA_FILA_FUENTE, construir_folds, crear_asignacion_folds
from medicion import ResultadoPrediccion


class FoldsCompartidosTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.escenarios = {}
        for nombre in ("a", "aprima", "b"):
            datos, _, _ = cargar_datos(nombre)
            cls.escenarios[nombre] = datos

    def _folds_y_asignacion(self, nombre: str, validacion: str):
        datos = self.escenarios[nombre]
        y = datos[OBJETIVO].to_numpy()
        grupos = datos[GRUPO].to_numpy()
        folds = construir_folds(y, grupos, validacion)
        asignacion = crear_asignacion_folds(
            nombre,
            validacion,
            folds,
            datos[COLUMNA_FILA_FUENTE].to_numpy(),
            grupos,
            y,
            CLASIFICADORES,
        )
        return folds, asignacion

    def test_reproducibilidad_con_semilla_42(self) -> None:
        datos = self.escenarios["a"]
        y = datos[OBJETIVO].to_numpy()
        grupos = datos[GRUPO].to_numpy()
        for validacion in ("aleatoria", "por_estacion"):
            with self.subTest(validacion=validacion):
                primera = construir_folds(y, grupos, validacion)
                segunda = construir_folds(
                    y, grupos, validacion, semilla=SEMILLA, n_splits=FOLDS
                )
                self.assertEqual(
                    [fold.fold_id for fold in primera],
                    [fold.fold_id for fold in segunda],
                )
                for fold_1, fold_2 in zip(primera, segunda):
                    np.testing.assert_array_equal(fold_1.idx_train, fold_2.idx_train)
                    np.testing.assert_array_equal(fold_1.idx_test, fold_2.idx_test)
                    self.assertFalse(fold_1.idx_train.flags.writeable)
                    self.assertFalse(fold_1.idx_test.flags.writeable)

    def test_misma_asignacion_para_ambos_clasificadores(self) -> None:
        columnas = [
            "modelo_datos",
            "validacion",
            "fold",
            "fold_id",
            "particion",
            "fila_preparada",
            "fila_fuente_csv",
            "estacion",
            "clase",
        ]
        orden = ["fold", "particion", "fila_preparada"]
        for nombre in self.escenarios:
            for validacion in ("aleatoria", "por_estacion"):
                with self.subTest(nombre=nombre, validacion=validacion):
                    _, asignacion = self._folds_y_asignacion(nombre, validacion)
                    red = (
                        asignacion[asignacion["clasificador"] == "red_densa"]
                        [columnas]
                        .sort_values(orden)
                        .reset_index(drop=True)
                    )
                    bosque = (
                        asignacion[asignacion["clasificador"] == "random_forest"]
                        [columnas]
                        .sort_values(orden)
                        .reset_index(drop=True)
                    )
                    pd.testing.assert_frame_equal(red, bosque)

    def test_cobertura_completa_sin_solapamiento(self) -> None:
        for nombre, datos in self.escenarios.items():
            esperadas = set(range(len(datos)))
            for validacion in ("aleatoria", "por_estacion"):
                with self.subTest(nombre=nombre, validacion=validacion):
                    folds, _ = self._folds_y_asignacion(nombre, validacion)
                    self.assertEqual(len(folds), FOLDS)
                    pruebas_acumuladas = []
                    for fold in folds:
                        train = set(fold.idx_train.tolist())
                        test = set(fold.idx_test.tolist())
                        self.assertTrue(train.isdisjoint(test))
                        self.assertEqual(train | test, esperadas)
                        pruebas_acumuladas.extend(fold.idx_test.tolist())
                    self.assertEqual(sorted(pruebas_acumuladas), sorted(esperadas))

    def test_estaciones_aisladas_en_validacion_agrupada(self) -> None:
        for nombre, datos in self.escenarios.items():
            grupos = datos[GRUPO].to_numpy()
            folds = construir_folds(
                datos[OBJETIVO].to_numpy(), grupos, "por_estacion"
            )
            for fold in folds:
                with self.subTest(nombre=nombre, fold=fold.numero):
                    estaciones_train = set(grupos[fold.idx_train])
                    estaciones_test = set(grupos[fold.idx_test])
                    self.assertTrue(estaciones_train.isdisjoint(estaciones_test))

    def test_evaluacion_entrega_las_mismas_matrices_a_ambos_modelos(self) -> None:
        llamadas_red = []
        llamadas_bosque = []

        def red_simulada(x_train, y_train, x_test, _n_clases, _pesos_clase):
            llamadas_red.append((x_train.copy(), y_train.copy(), x_test.copy()))
            return ResultadoPrediccion(
                np.zeros(len(x_test), dtype=int), 1.0, 0.1
            )

        def bosque_simulado(x_train, y_train, x_test, _pesos_clase):
            llamadas_bosque.append((x_train.copy(), y_train.copy(), x_test.copy()))
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
                metricas, _, metadata, asignacion, _ = ejecutar.evaluar(
                    "a", "por_estacion"
                )

        self.assertEqual(len(llamadas_red), FOLDS)
        self.assertEqual(len(llamadas_bosque), FOLDS)
        for llamada_red, llamada_bosque in zip(llamadas_red, llamadas_bosque):
            for matriz_red, matriz_bosque in zip(llamada_red, llamada_bosque):
                np.testing.assert_array_equal(matriz_red, matriz_bosque)

        ids_por_clasificador = metricas.pivot(
            index="fold", columns="clasificador", values="fold_id"
        )
        self.assertTrue(
            (
                ids_por_clasificador["red_densa"]
                == ids_por_clasificador["random_forest"]
            ).all()
        )
        self.assertTrue(metadata["folds_compartidos"])
        self.assertEqual(set(asignacion["clasificador"]), set(CLASIFICADORES))


if __name__ == "__main__":
    unittest.main()
