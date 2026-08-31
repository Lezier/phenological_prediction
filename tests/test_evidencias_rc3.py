"""Pruebas de los CSV deterministas ya generados por CP03 y CP04."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from folds import VALIDACIONES
from ponderacion import FORMULA_PESO_BALANCEADO

RAIZ = Path(__file__).resolve().parents[1]
RUTA_FOLDS = RAIZ / "output" / "asignacion_folds.csv"
RUTA_PESOS = RAIZ / "output" / "pesos_clase_por_fold.csv"


class EvidenciasRC3Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.folds = pd.read_csv(RUTA_FOLDS)
        cls.pesos = pd.read_csv(RUTA_PESOS)

    def test_esquema_y_claves_de_folds(self) -> None:
        requeridas = {
            "modelo_datos",
            "validacion",
            "clasificador",
            "fold",
            "fold_id",
            "particion",
            "fila_preparada",
            "fila_fuente_csv",
            "estacion",
            "clase",
        }
        self.assertTrue(requeridas.issubset(self.folds.columns))
        self.assertEqual(len(self.folds), 48100)
        self.assertEqual(set(self.folds["validacion"]), set(VALIDACIONES))
        self.assertEqual(set(self.folds["particion"]), {"train", "test"})
        self.assertTrue(
            self.folds["fold_id"].map(
                lambda valor: bool(re.fullmatch(r"[0-9A-F]{64}", valor))
            ).all()
        )

    def test_esquema_y_formula_de_pesos(self) -> None:
        requeridas = {
            "modelo_datos",
            "validacion",
            "clasificador",
            "fold",
            "fold_id",
            "n_train",
            "n_clases_presentes_train",
            "clase_codificada",
            "clase",
            "frecuencia_train",
            "peso_clase",
            "peso_aplicado",
            "formula",
            "calculado_solo_con_train",
        }
        self.assertTrue(requeridas.issubset(self.pesos.columns))
        self.assertEqual(len(self.pesos), 300)
        self.assertTrue(self.pesos["calculado_solo_con_train"].all())
        self.assertEqual(set(self.pesos["formula"]), {FORMULA_PESO_BALANCEADO})
        aplicados = self.pesos[self.pesos["peso_aplicado"]]
        esperados = aplicados["n_train"] / (
            aplicados["n_clases_presentes_train"]
            * aplicados["frecuencia_train"]
        )
        np.testing.assert_allclose(aplicados["peso_clase"], esperados)

    def test_pesos_referencian_folds_existentes(self) -> None:
        claves_folds = set(
            self.folds[["modelo_datos", "validacion", "fold", "fold_id"]]
            .itertuples(index=False, name=None)
        )
        claves_pesos = set(
            self.pesos[["modelo_datos", "validacion", "fold", "fold_id"]]
            .itertuples(index=False, name=None)
        )
        self.assertTrue(claves_pesos.issubset(claves_folds))


if __name__ == "__main__":
    unittest.main()
