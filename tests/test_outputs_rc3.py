"""Pruebas que se activan automáticamente después de la corrida oficial RC3."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "output"
RUTA_CONFIGURACION = SALIDA / "configuracion_ejecucion.json"


def outputs_oficiales_rc3_disponibles() -> bool:
    version = (RAIZ / "VERSION").read_text(encoding="utf-8").strip()
    configuracion = json.loads(RUTA_CONFIGURACION.read_text(encoding="utf-8"))
    return configuracion.get("version_proyecto") == version


def sha256_archivo(ruta: Path) -> str:
    resumen = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            resumen.update(bloque)
    return resumen.hexdigest().upper()


@unittest.skipUnless(
    outputs_oficiales_rc3_disponibles(),
    "Los outputs oficiales RC3 se generaran en CP09.",
)
class OutputsOficialesRC3Test(unittest.TestCase):
    def test_metricas_y_tiempos(self) -> None:
        metricas = pd.read_csv(SALIDA / "metricas_por_fold.csv")
        tiempos = pd.read_csv(SALIDA / "tiempos_por_fold.csv")
        requeridas = {
            "fold_id",
            "accuracy",
            "f1_macro",
            "f1_weighted",
            "tiempo_entrenamiento_segundos",
            "tiempo_inferencia_segundos",
            "tiempo_total_segundos",
            "epocas_ejecutadas",
            "early_stopping_detencion",
        }
        self.assertTrue(requeridas.issubset(metricas.columns))
        self.assertEqual(len(metricas), 3 * 2 * 2 * 5)
        self.assertEqual(len(tiempos), len(metricas))
        for columna in (
            "tiempo_entrenamiento_segundos",
            "tiempo_inferencia_segundos",
            "tiempo_total_segundos",
        ):
            self.assertTrue((metricas[columna] >= 0).all())

    def test_consolidado_con_media_y_dispersion(self) -> None:
        resumen = pd.read_csv(SALIDA / "comparacion_consolidada.csv")
        requeridas = {
            "n_folds",
            "accuracy_promedio",
            "accuracy_desviacion",
            "f1_macro_promedio",
            "f1_macro_desviacion",
            "f1_weighted_promedio",
            "f1_weighted_desviacion",
            "tiempo_entrenamiento_promedio_segundos",
            "tiempo_entrenamiento_desviacion_segundos",
            "tiempo_inferencia_promedio_segundos",
            "tiempo_inferencia_desviacion_segundos",
        }
        self.assertTrue(requeridas.issubset(resumen.columns))
        self.assertEqual(len(resumen), 3 * 2 * 2)
        self.assertTrue((resumen["n_folds"] == 5).all())

    def test_hashes_de_evidencias_en_configuracion(self) -> None:
        configuracion = json.loads(RUTA_CONFIGURACION.read_text(encoding="utf-8"))
        for campo in ("evidencia_folds", "evidencia_ponderacion_clases"):
            evidencia = configuracion[campo]
            ruta = SALIDA / evidencia["archivo"]
            self.assertTrue(ruta.exists())
            self.assertEqual(sha256_archivo(ruta), evidencia["sha256"])


if __name__ == "__main__":
    unittest.main()
