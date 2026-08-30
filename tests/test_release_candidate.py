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
RUTA_DATOS_SATELITE = RAIZ / "data" / "base_fenologia_clima_satelite.csv"
RUTA_MODELO = RAIZ / "models" / "random_forest_a_final.joblib"
RUTA_METADATA = RAIZ / "output" / "metadata_modelo_final.json"
RUTA_MANIFIESTO = RAIZ / "RELEASE_MANIFEST.json"
HASH_DATOS = "0397C7A0B61B76388C22A1CDD1F13BCB2B7E10069C7BBB2935F0ADCC2E5CF6B7"
HASH_DATOS_SATELITE = "0C307E8CAAEEE04A87EB572AA675C4BB7C97EF1C1BC5C0D36C7018FB7129ADCA"
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


def sha256_texto_lf(ruta: Path) -> str:
    """Calcula SHA-256 tras normalizar finales de línea de texto a LF."""
    contenido = ruta.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(contenido).hexdigest().upper()


class ReleaseCandidateTest(unittest.TestCase):
    def test_datos_modelo_a(self) -> None:
        self.assertTrue(RUTA_DATOS.exists())
        self.assertTrue(RUTA_DATOS_SATELITE.exists())
        self.assertNotIn(b"\r\n", RUTA_DATOS.read_bytes())
        self.assertNotIn(b"\r\n", RUTA_DATOS_SATELITE.read_bytes())
        self.assertEqual(sha256_archivo(RUTA_DATOS), HASH_DATOS)
        self.assertEqual(sha256_archivo(RUTA_DATOS_SATELITE), HASH_DATOS_SATELITE)
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

    def test_manifiesto_release(self) -> None:
        version = (RAIZ / "VERSION").read_text(encoding="utf-8").strip()
        manifiesto = json.loads(RUTA_MANIFIESTO.read_text(encoding="utf-8"))
        self.assertEqual(manifiesto["version"], version)
        artefactos = manifiesto["artefactos_reutilizados"] + manifiesto["datos"]
        for artefacto in artefactos:
            ruta = RAIZ / artefacto["ruta"]
            self.assertTrue(ruta.exists())
            modo_hash = artefacto["modo_hash"]
            self.assertIn(modo_hash, {"sha256-raw", "sha256-lf"})
            hash_observado = (
                sha256_archivo(ruta)
                if modo_hash == "sha256-raw"
                else sha256_texto_lf(ruta)
            )
            self.assertEqual(hash_observado, artefacto["sha256"])


if __name__ == "__main__":
    unittest.main()
