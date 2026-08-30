"""Pruebas mínimas de integridad del release candidate."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RUTA_DATOS = RAIZ / "data" / "base_fenologia_clima.csv"
RUTA_MODELO = RAIZ / "models" / "random_forest_a_final.joblib"
RUTA_METADATA = RAIZ / "output" / "metadata_modelo_final.json"
HASH_DATOS = "0397C7A0B61B76388C22A1CDD1F13BCB2B7E10069C7BBB2935F0ADCC2E5CF6B7"
VARIABLES = [
    "clima_temp_media",
    "clima_temp_max_media",
    "clima_temp_min_media",
    "clima_precip_acumulada",
    "clima_radiacion_media",
    "clima_humedad_media",
    "clima_gdd_acumulado",
]


def sha256_archivo(ruta: Path) -> str:
    resumen = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            resumen.update(bloque)
    return resumen.hexdigest().upper()


class ReleaseCandidateTest(unittest.TestCase):
    def test_datos_modelo_a(self) -> None:
        self.assertTrue(RUTA_DATOS.exists())
        self.assertEqual(sha256_archivo(RUTA_DATOS), HASH_DATOS)
        datos = pd.read_csv(RUTA_DATOS)
        obligatorias = [
            columna for columna in VARIABLES if columna != "clima_radiacion_media"
        ] + ["macro_etapa", "s_id"]
        preparados = datos.dropna(subset=obligatorias)
        self.assertEqual(len(preparados), 1091)
        self.assertEqual(preparados["macro_etapa"].nunique(), 5)

    def test_modelo_y_metadata(self) -> None:
        self.assertTrue(RUTA_MODELO.exists())
        self.assertTrue(RUTA_METADATA.exists())
        paquete = joblib.load(RUTA_MODELO)
        metadata = json.loads(RUTA_METADATA.read_text(encoding="utf-8"))
        self.assertEqual(paquete["variables"], VARIABLES)
        self.assertEqual(metadata["muestras_entrenamiento"], 1091)
        self.assertEqual(metadata["sha256_datos"], HASH_DATOS)
        self.assertEqual(metadata["sha256_modelo"], sha256_archivo(RUTA_MODELO))

        datos = pd.read_csv(RUTA_DATOS).dropna(subset=VARIABLES).head(1)
        probabilidades = paquete["pipeline"].predict_proba(datos[VARIABLES])[0]
        self.assertEqual(len(probabilidades), 5)
        self.assertTrue(np.isclose(probabilidades.sum(), 1.0))


if __name__ == "__main__":
    unittest.main()
