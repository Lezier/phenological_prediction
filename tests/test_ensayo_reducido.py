"""Prueba estructural del ensayo reducido sin entrenamiento costoso."""

from __future__ import annotations

import hashlib
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

import ejecutar
from medicion import ResultadoPrediccion


def sha256_archivo(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest().upper()


class EnsayoReducidoTest(unittest.TestCase):
    def test_genera_todos_los_tipos_de_salida_y_los_marca_no_oficiales(self) -> None:
        def red_simulada(_x_train, _y_train, x_test, _n_clases, _pesos):
            return ResultadoPrediccion(
                np.zeros(len(x_test), dtype=int),
                1.2,
                0.2,
                epocas_ejecutadas=10,
                early_stopping_detencion=True,
            )

        def bosque_simulado(_x_train, _y_train, x_test, _pesos):
            return ResultadoPrediccion(
                np.zeros(len(x_test), dtype=int), 0.4, 0.05
            )

        with TemporaryDirectory() as temporal:
            salida = Path(temporal) / "ensayo"
            with (
                patch.object(ejecutar, "entrenar_red", side_effect=red_simulada),
                patch.object(
                    ejecutar,
                    "entrenar_random_forest",
                    side_effect=bosque_simulado,
                ),
            ):
                with redirect_stdout(io.StringIO()):
                    ejecutar.ejecutar_ensayo_reducido(
                        directorio_salida=salida
                    )

            esperados = {
                "metricas_por_fold.csv",
                "tiempos_por_fold.csv",
                "comparacion_consolidada.csv",
                "asignacion_folds.csv",
                "pesos_clase_por_fold.csv",
                "matriz_a_por_estacion_red_densa.csv",
                "matriz_a_por_estacion_random_forest.csv",
                "comparacion_metricas.png",
                "configuracion_ensayo.json",
                "manifest_ensayo.json",
                "NO_USAR_EN_INFORME.md",
            }
            self.assertTrue(esperados.issubset({ruta.name for ruta in salida.iterdir()}))

            metricas = pd.read_csv(salida / "metricas_por_fold.csv")
            self.assertEqual(len(metricas), 2)
            self.assertEqual(set(metricas["fold"]), {1})
            self.assertEqual(
                set(metricas["clasificador"]), set(ejecutar.CLASIFICADORES)
            )

            configuracion = json.loads(
                (salida / "configuracion_ensayo.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                configuracion["tipo_ejecucion"],
                "ensayo_reducido_cp08_no_oficial",
            )
            self.assertFalse(configuracion["usar_en_informe"])

            manifiesto = json.loads(
                (salida / "manifest_ensayo.json").read_text(encoding="utf-8")
            )
            self.assertFalse(manifiesto["usar_en_informe"])
            for artefacto in manifiesto["artefactos"]:
                ruta = salida / artefacto["archivo"]
                self.assertEqual(sha256_archivo(ruta), artefacto["sha256"])


if __name__ == "__main__":
    unittest.main()
