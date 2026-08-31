"""Pruebas del contrato de inferencia con el artefacto final RC3."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

import numpy as np

from configuracion import VARIABLES_CLIMA
from demo import cargar_paquete, generar_prediccion, predecir, validar_paquete


class InferenciaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.paquete = cargar_paquete()
        cls.valores_validos = [
            (
                cls.paquete["rangos_entrenamiento"][variable]["min"]
                + cls.paquete["rangos_entrenamiento"][variable]["max"]
            )
            / 2
            for variable in VARIABLES_CLIMA
        ]

    def test_contrato_variables_clases_y_probabilidades(self) -> None:
        self.assertEqual(self.paquete["version_esquema"], 2)
        self.assertEqual(self.paquete["version_proyecto"], "0.1.0-rc.3")
        self.assertEqual(self.paquete["variables"], VARIABLES_CLIMA)
        self.assertEqual(len(self.paquete["clases"]), 5)
        self.assertEqual(len(set(self.paquete["clases"])), 5)
        resultado = generar_prediccion(self.paquete, self.valores_validos)
        self.assertIn(resultado["macro_etapa"], self.paquete["clases"])
        self.assertEqual(set(resultado["probabilidades"]), set(self.paquete["clases"]))
        self.assertTrue(
            np.isclose(sum(resultado["probabilidades"].values()), 1.0)
        )
        self.assertFalse(resultado["probabilidades_calibradas"])

    def test_diccionario_se_ordena_por_nombre_de_variable(self) -> None:
        entrada = dict(reversed(list(zip(VARIABLES_CLIMA, self.valores_validos))))
        desde_lista = generar_prediccion(self.paquete, self.valores_validos)
        desde_diccionario = generar_prediccion(self.paquete, entrada)
        self.assertEqual(desde_lista, desde_diccionario)

    def test_rechaza_entradas_invalidas(self) -> None:
        casos = [
            self.valores_validos[:-1],
            [*self.valores_validos[:-1], np.nan],
            [*self.valores_validos[:-1], np.inf],
            [*self.valores_validos[:-1], "texto"],
            {variable: 0.0 for variable in VARIABLES_CLIMA[:-1]},
            {**{variable: 0.0 for variable in VARIABLES_CLIMA}, "extra": 1.0},
        ]
        for valores in casos:
            with self.subTest(valores=valores):
                with self.assertRaises(ValueError):
                    generar_prediccion(self.paquete, valores)

    def test_fuera_de_rango_advierte_sin_bloquear(self) -> None:
        valores = self.valores_validos.copy()
        variable = VARIABLES_CLIMA[0]
        valores[0] = self.paquete["rangos_entrenamiento"][variable]["min"] - 1
        resultado = generar_prediccion(self.paquete, valores)
        self.assertEqual(resultado["variables_fuera_rango"], [variable])

    def test_salida_humana_advierte_probabilidades_no_calibradas(self) -> None:
        salida = io.StringIO()
        with redirect_stdout(salida):
            resultado = predecir(self.paquete, self.valores_validos)
        self.assertIn("Probabilidades estimadas (no calibradas)", salida.getvalue())
        self.assertFalse(resultado["probabilidades_calibradas"])

    def test_rechaza_paquetes_inconsistentes(self) -> None:
        variables_invertidas = {**self.paquete, "variables": VARIABLES_CLIMA[::-1]}
        with self.assertRaises(ValueError):
            validar_paquete(variables_invertidas)

        clases_duplicadas = {
            **self.paquete,
            "clases": [self.paquete["clases"][0]] * 5,
        }
        with self.assertRaises(ValueError):
            validar_paquete(clases_duplicadas)

        rangos_incompletos = {
            **self.paquete,
            "rangos_entrenamiento": {
                variable: self.paquete["rangos_entrenamiento"][variable]
                for variable in VARIABLES_CLIMA[:-1]
            },
        }
        with self.assertRaises(ValueError):
            validar_paquete(rangos_incompletos)


if __name__ == "__main__":
    unittest.main()
