"""Pruebas ampliadas de datos preparados para A, A' y B."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from configuracion import GRUPO, OBJETIVO, VARIABLES_CLIMA, VARIABLE_NDVI
from ejecutar import cargar_datos
from folds import COLUMNA_FILA_FUENTE


class DatosRC3Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.escenarios = {
            nombre: cargar_datos(nombre) for nombre in ("a", "aprima", "b")
        }

    def test_cantidad_y_cobertura_de_escenarios(self) -> None:
        datos_a, variables_a, _ = self.escenarios["a"]
        datos_aprima, variables_aprima, _ = self.escenarios["aprima"]
        datos_b, variables_b, _ = self.escenarios["b"]
        self.assertEqual(len(datos_a), 1091)
        self.assertEqual(len(datos_aprima), 657)
        self.assertEqual(len(datos_b), 657)
        self.assertEqual(variables_a, VARIABLES_CLIMA)
        self.assertEqual(variables_aprima, VARIABLES_CLIMA)
        self.assertEqual(variables_b, VARIABLES_CLIMA + [VARIABLE_NDVI])
        self.assertEqual(datos_a[OBJETIVO].nunique(), 5)
        self.assertEqual(datos_aprima[OBJETIVO].nunique(), 5)
        self.assertEqual(datos_b[OBJETIVO].nunique(), 5)

    def test_aprima_y_b_utilizan_las_mismas_filas(self) -> None:
        datos_aprima = self.escenarios["aprima"][0]
        datos_b = self.escenarios["b"][0]
        pd.testing.assert_series_equal(
            datos_aprima[COLUMNA_FILA_FUENTE],
            datos_b[COLUMNA_FILA_FUENTE],
        )

    def test_campos_criticos_sin_nulos_ni_infinitos(self) -> None:
        for nombre, (datos, variables, _) in self.escenarios.items():
            with self.subTest(nombre=nombre):
                self.assertFalse(datos[[OBJETIVO, GRUPO]].isna().any().any())
                columnas_sin_nulos = variables
                if nombre == "a":
                    columnas_sin_nulos = [
                        variable
                        for variable in variables
                        if variable != "clima_radiacion_media"
                    ]
                self.assertFalse(datos[columnas_sin_nulos].isna().any().any())
                valores = datos[variables].to_numpy(dtype=float)
                self.assertFalse(np.isinf(valores).any())
                self.assertTrue(datos[COLUMNA_FILA_FUENTE].is_unique)

    def test_modelo_a_conserva_casos_para_imputacion_interna(self) -> None:
        datos_a = self.escenarios["a"][0]
        self.assertGreater(datos_a["clima_radiacion_media"].isna().sum(), 0)


if __name__ == "__main__":
    unittest.main()
