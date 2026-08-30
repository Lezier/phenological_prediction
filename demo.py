"""Demo de inferencia para el modelo final Random Forest A."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
RUTA_MODELO = RAIZ / "models" / "random_forest_a_final.joblib"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def cargar_paquete() -> dict:
    if not RUTA_MODELO.exists():
        raise FileNotFoundError(
            "No existe el modelo final. Ejecute primero: python entrenar_modelo_final.py"
        )
    paquete = joblib.load(RUTA_MODELO)
    requeridas = {
        "version_esquema",
        "pipeline",
        "variables",
        "clases",
        "rangos_entrenamiento",
        "advertencia",
    }
    faltantes = requeridas - set(paquete)
    if faltantes:
        raise ValueError(f"El artefacto no contiene campos requeridos: {sorted(faltantes)}")
    return paquete


def solicitar_valores(paquete: dict) -> list[float]:
    valores = []
    print("Ingrese las siete variables climáticas del caso a clasificar.")
    for variable in paquete["variables"]:
        rango = paquete["rangos_entrenamiento"][variable]
        while True:
            texto = input(
                f"{variable} (rango observado {rango['min']:.2f} a {rango['max']:.2f}): "
            ).strip()
            try:
                valor = float(texto)
                if not np.isfinite(valor):
                    raise ValueError
                valores.append(valor)
                break
            except ValueError:
                print("Ingrese un número finito; no use texto ni deje el campo vacío.")
    return valores


def predecir(paquete: dict, valores: list[float]) -> None:
    if len(valores) != len(paquete["variables"]):
        raise ValueError(f"Se requieren exactamente {len(paquete['variables'])} valores.")
    if not np.isfinite(np.asarray(valores, dtype=float)).all():
        raise ValueError("Todos los valores deben ser números finitos.")

    entrada = pd.DataFrame([valores], columns=paquete["variables"])
    modelo = paquete["pipeline"]
    clase = str(modelo.predict(entrada)[0])
    probabilidades = modelo.predict_proba(entrada)[0]
    clases = list(modelo.named_steps["clasificador"].classes_)

    fuera_rango = []
    for variable, valor in zip(paquete["variables"], valores):
        rango = paquete["rangos_entrenamiento"][variable]
        if valor < rango["min"] or valor > rango["max"]:
            fuera_rango.append(variable)

    print("\nResultado experimental")
    print(f"Macro-etapa estimada: {clase}")
    print("Probabilidades estimadas (no calibradas):")
    for etiqueta, probabilidad in sorted(
        zip(clases, probabilidades), key=lambda elemento: elemento[1], reverse=True
    ):
        print(f"  - {etiqueta}: {probabilidad:.1%}")
    if fuera_rango:
        print(
            "Advertencia de extrapolación: fuera del rango observado: "
            + ", ".join(fuera_rango)
        )
    print("Advertencia de uso: " + paquete["advertencia"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo del clasificador fenológico.")
    parser.add_argument(
        "--valores",
        nargs=7,
        type=float,
        metavar=("T_MEDIA", "T_MAX", "T_MIN", "PRECIP", "RADIACION", "HUMEDAD", "GDD"),
        help="Siete variables climáticas en el orden documentado.",
    )
    argumentos = parser.parse_args()
    paquete = cargar_paquete()
    valores = argumentos.valores or solicitar_valores(paquete)
    predecir(paquete, list(valores))


if __name__ == "__main__":
    main()
