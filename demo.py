"""Demo de inferencia para el modelo final Random Forest A."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from configuracion import VARIABLES_CLIMA

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
    validar_paquete(paquete)
    return paquete


def validar_paquete(paquete: dict) -> None:
    """Valida contrato, variables, clases y rangos del artefacto cargado."""

    requeridas = {
        "version_esquema",
        "version_proyecto",
        "pipeline",
        "variables",
        "clases",
        "rangos_entrenamiento",
        "sha256_datos",
        "probabilidades_calibradas",
        "advertencia",
    }
    faltantes = requeridas - set(paquete)
    if faltantes:
        raise ValueError(f"El artefacto no contiene campos requeridos: {sorted(faltantes)}")
    if paquete["version_esquema"] != 2:
        raise ValueError("La versión del esquema del artefacto no es compatible.")
    if paquete["variables"] != VARIABLES_CLIMA:
        raise ValueError(
            "El orden de variables del artefacto no coincide con la configuración RC3."
        )
    if paquete["probabilidades_calibradas"] is not False:
        raise ValueError("El artefacto debe declarar sus probabilidades como no calibradas.")
    clases = list(paquete["clases"])
    if len(clases) != 5 or len(set(clases)) != 5:
        raise ValueError("El artefacto debe declarar exactamente cinco clases únicas.")
    modelo = paquete["pipeline"]
    try:
        clases_modelo = list(modelo.named_steps["clasificador"].classes_)
    except (AttributeError, KeyError) as error:
        raise ValueError("El pipeline no expone el clasificador esperado.") from error
    if clases_modelo != clases:
        raise ValueError("Las clases declaradas no coinciden con las del clasificador.")

    rangos = paquete["rangos_entrenamiento"]
    if set(rangos) != set(VARIABLES_CLIMA):
        raise ValueError("Los rangos no cubren exactamente las siete variables.")
    for variable in VARIABLES_CLIMA:
        try:
            minimo = float(rangos[variable]["min"])
            maximo = float(rangos[variable]["max"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Rango inválido para {variable}.") from error
        if not np.isfinite([minimo, maximo]).all() or minimo > maximo:
            raise ValueError(f"Rango inválido para {variable}.")


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


def normalizar_valores(
    paquete: dict,
    valores: Sequence[object] | Mapping[str, object],
) -> list[float]:
    """Ordena entradas nominales o valida una secuencia posicional."""

    variables = paquete["variables"]
    if isinstance(valores, Mapping):
        faltantes = set(variables) - set(valores)
        adicionales = set(valores) - set(variables)
        if faltantes or adicionales:
            raise ValueError(
                "Variables de entrada incorrectas. "
                f"Faltantes: {sorted(faltantes)}; adicionales: {sorted(adicionales)}."
            )
        valores_ordenados = [valores[variable] for variable in variables]
    else:
        valores_ordenados = list(valores)

    if len(valores_ordenados) != len(variables):
        raise ValueError(f"Se requieren exactamente {len(paquete['variables'])} valores.")
    try:
        arreglo = np.asarray(valores_ordenados, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Todos los valores deben ser números finitos.") from error
    if arreglo.ndim != 1 or not np.isfinite(arreglo).all():
        raise ValueError("Todos los valores deben ser números finitos.")
    return arreglo.tolist()


def generar_prediccion(
    paquete: dict,
    valores: Sequence[object] | Mapping[str, object],
) -> dict[str, Any]:
    """Ejecuta una inferencia y devuelve un resultado estructurado."""

    validar_paquete(paquete)
    valores_normalizados = normalizar_valores(paquete, valores)

    entrada = pd.DataFrame([valores_normalizados], columns=paquete["variables"])
    modelo = paquete["pipeline"]
    clase = str(modelo.predict(entrada)[0])
    probabilidades = np.asarray(modelo.predict_proba(entrada)[0], dtype=float)
    clases = list(paquete["clases"])
    if (
        len(probabilidades) != len(clases)
        or not np.isfinite(probabilidades).all()
        or (probabilidades < 0).any()
        or not np.isclose(probabilidades.sum(), 1.0)
    ):
        raise ValueError("El modelo devolvió probabilidades inválidas.")
    if clase not in clases:
        raise ValueError("El modelo devolvió una clase no declarada.")

    fuera_rango = []
    for variable, valor in zip(paquete["variables"], valores_normalizados):
        rango = paquete["rangos_entrenamiento"][variable]
        if valor < rango["min"] or valor > rango["max"]:
            fuera_rango.append(variable)

    return {
        "macro_etapa": clase,
        "probabilidades": {
            etiqueta: float(probabilidad)
            for etiqueta, probabilidad in zip(clases, probabilidades)
        },
        "variables_fuera_rango": fuera_rango,
        "advertencia": paquete["advertencia"],
        "probabilidades_calibradas": paquete["probabilidades_calibradas"],
    }


def predecir(
    paquete: dict,
    valores: Sequence[object] | Mapping[str, object],
) -> dict[str, Any]:
    resultado = generar_prediccion(paquete, valores)

    print("\nResultado experimental")
    print(f"Macro-etapa estimada: {resultado['macro_etapa']}")
    print("Probabilidades estimadas (no calibradas):")
    for etiqueta, probabilidad in sorted(
        resultado["probabilidades"].items(),
        key=lambda elemento: elemento[1],
        reverse=True,
    ):
        print(f"  - {etiqueta}: {probabilidad:.1%}")
    if resultado["variables_fuera_rango"]:
        print(
            "Advertencia de extrapolación: fuera del rango observado: "
            + ", ".join(resultado["variables_fuera_rango"])
        )
    print("Advertencia de uso: " + resultado["advertencia"])
    return resultado


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
