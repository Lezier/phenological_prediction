# Trazabilidad de hiperparámetros de Random Forest A

## Conclusión

La configuración de Random Forest fue incorporada durante el desarrollo de la
comparación limpia asistido por ChatGPT y posteriormente heredada por este
proyecto. Francisco López no seleccionó personalmente estos valores ni solicitó
un proceso de tuning.

El hecho de que el primer commit del repositorio esté firmado por Francisco
López registra quién incorporó los archivos a Git, pero no demuestra autoría
personal sobre cada decisión técnica contenida en ellos.

## Parámetros establecidos explícitamente

```python
{
    "n_estimators": 400,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}
```

| Parámetro | Procedencia | Interpretación |
|---|---|---|
| `n_estimators=400` | Heurística incorporada por ChatGPT | Cantidad amplia de árboles orientada a estabilizar el ensamble; no se demostró que sea óptima. |
| `class_weight="balanced"` | Incorporado por ChatGPT | Compensa automáticamente la frecuencia de las clases presentes en entrenamiento. |
| `random_state=42` | Convención incorporada por ChatGPT | Permite repetir la construcción pseudoaleatoria del bosque. |
| `n_jobs=-1` | Decisión operativa incorporada por ChatGPT | Utiliza los procesadores disponibles para reducir tiempo de ejecución. |

## Parámetros no especificados

Los parámetros restantes son resueltos mediante los valores predeterminados de
la versión instalada de scikit-learn. Entre los más relevantes se encuentran:

| Parámetro | Valor predeterminado utilizado |
|---|---|
| `criterion` | `"gini"` |
| `max_depth` | `None` |
| `min_samples_split` | `2` |
| `min_samples_leaf` | `1` |
| `max_features` | `"sqrt"` |
| `bootstrap` | `True` |

RC3 deberá registrar la configuración efectiva obtenida directamente desde el
clasificador, junto con la versión de scikit-learn, para evitar depender de una
descripción implícita.

En la comparación RC3, `balanced` se materializa una vez por fold como un
diccionario de pesos calculado exclusivamente desde `y_train`. El mismo
diccionario se entrega a la red neuronal y a Random Forest y se registra en
`pesos_clase_por_fold.csv`; esto hace auditable la estrategia sin modificar su
fórmula.

## Alcance de la evaluación

Se evaluó una configuración fija mediante validación cruzada aleatoria y
validación agrupada por estación. Esto permite estimar su comportamiento, pero
no equivale a una búsqueda de hiperparámetros.

Por lo tanto:

- el modelo es reproducible;
- no se realizó `GridSearchCV`, `RandomizedSearchCV` ni un procedimiento
  equivalente;
- no se puede afirmar que 400 árboles sea el valor óptimo;
- el modelo no debe describirse como optimizado;
- un tuning futuro constituiría un experimento adicional y requeriría un
  protocolo que separe selección y evaluación.

## Probabilidades

El prototipo puede utilizar `predict_proba()` para mostrar la distribución de
votos normalizada del bosque. Estas probabilidades no han sido sometidas a un
procedimiento de calibración y deben presentarse con esa advertencia.
