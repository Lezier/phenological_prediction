"""Configuración auditable y compartida del experimento RC3."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sklearn.ensemble import RandomForestClassifier

RAIZ_PROYECTO = Path(__file__).resolve().parent
VERSION_PROYECTO = (RAIZ_PROYECTO / "VERSION").read_text(encoding="utf-8").strip()

VARIABLES_CLIMA = [
    "clima_temp_media",
    "clima_temp_max_media",
    "clima_temp_min_media",
    "clima_precip_acumulada",
    "clima_radiacion_media",
    "clima_humedad_media",
    "clima_gdd_acumulado",
]
VARIABLE_NDVI = "ndvi"
OBJETIVO = "macro_etapa"
GRUPO = "s_id"
SEMILLA = 42
FOLDS = 5

HASH_DATOS_MODELO_A = (
    "0397C7A0B61B76388C22A1CDD1F13BCB2B7E10069C7BBB2935F0ADCC2E5CF6B7"
)

PARAMETROS_RANDOM_FOREST = {
    "n_estimators": 400,
    "class_weight": "balanced",
    "random_state": SEMILLA,
    "n_jobs": -1,
}

PARAMETROS_RED_NEURONAL = {
    "capas_ocultas": [
        {"unidades": 16, "activacion": "relu", "dropout": 0.3},
        {"unidades": 8, "activacion": "relu", "dropout": 0.2},
    ],
    "activacion_salida": "softmax",
    "optimizador": "Adam",
    "learning_rate": 0.001,
    "funcion_perdida": "sparse_categorical_crossentropy",
    "metricas": ["accuracy"],
    "epocas_maximas": 60,
    "batch_size": 16,
    "early_stopping": {
        "monitor": "loss",
        "patience": 8,
        "restore_best_weights": True,
    },
    "ponderacion_clases": "balanced_calculado_solo_con_train",
}

TRATAMIENTO_EARLY_STOPPING = {
    "rol": "control_de_convergencia_sobre_loss_de_train",
    "usa_validacion_interna": False,
    "usa_fold_externo_test": False,
    "interpretable_como_control_de_overfitting": False,
    "decision_rc3": "conservar_baseline_heredado_sin_cambiar_resultados",
}

PROCEDENCIA_HIPERPARAMETROS = {
    "random_forest": {
        "estado": "configuracion_heuristica_no_optimizada",
        "procedencia": "incorporada_por_chatgpt_en_clean_comparison",
        "seleccion_personal_por_francisco_lopez": False,
        "tuning_sistematico": False,
        "nota": (
            "Los parametros no especificados conservan los valores "
            "predeterminados de la version instalada de scikit-learn."
        ),
    },
    "red_neuronal": {
        "estado": "sin_respuesta_del_equipo_al_cierre_cp06",
        "procedencia": "no_completamente_confirmada",
        "respuesta_equipo_recibida": False,
        "tuning_sistematico": None,
        "nota": (
            "La configuracion se conserva como baseline heredado. No atribuir "
            "los valores ni afirmar que fueron optimizados."
        ),
    },
}

ADVERTENCIA_USO = (
    "Uso experimental con datos europeos; no validado para operación en Chile "
    "y no sustituye evaluación agronómica. Las probabilidades mostradas no "
    "han sido calibradas."
)


def parametros_random_forest_efectivos() -> dict[str, Any]:
    """Devuelve los parámetros efectivos de la versión instalada."""

    return RandomForestClassifier(**PARAMETROS_RANDOM_FOREST).get_params(deep=False)
