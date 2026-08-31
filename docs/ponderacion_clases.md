# Ponderación de clases por fold

## Decisión metodológica

RC3 utiliza una única estrategia balanceada para la red neuronal y Random
Forest. Los pesos se calculan después de construir cada fold y exclusivamente
con las etiquetas de entrenamiento:

```text
peso(clase) = n_train / (n_clases_presentes_train * frecuencia_clase_train)
```

Las etiquetas del conjunto de prueba no se entregan a la función de cálculo.
Una prueba adicional modifica artificialmente `y_test` y confirma que el
resultado permanece idéntico.

## Aplicación a los modelos

El cálculo produce un solo diccionario `{clase_codificada: peso}` por fold.
Ese mismo objeto lógico se entrega a:

- `class_weight` de `Model.fit()` para la red densa;
- `class_weight` de `RandomForestClassifier` para Random Forest.

La configuración base de Random Forest conserva `class_weight="balanced"`
como declaración de estrategia. Durante la comparación, RC3 materializa esa
estrategia como un diccionario explícito para que sus valores puedan auditarse
y reutilizarse de forma idéntica en ambos clasificadores.

Si una clase global no aparece en el train de un fold, su frecuencia se
registra como cero, su peso queda vacío y `peso_aplicado` queda en falso. No se
inventa un peso para una clase que el modelo no puede observar durante el
entrenamiento.

## Evidencia generada

La ejecución genera `output/pesos_clase_por_fold.csv` con:

- escenario, validación, clasificador, fold y `fold_id`;
- tamaño de train y cantidad de clases presentes;
- clase codificada y nombre de clase;
- frecuencia observada solamente en train;
- peso aplicado, estrategia y fórmula;
- indicador `calculado_solo_con_train`.

La tabla repite los mismos valores bajo `red_densa` y `random_forest` para que
la igualdad sea comprobable directamente.

Puede regenerarse sin entrenar modelos mediante:

```powershell
python ejecutar.py --solo-pesos
```

La corrida oficial de RC3 volverá a generar el archivo y registrará su SHA-256
en `configuracion_ejecucion.json`.
