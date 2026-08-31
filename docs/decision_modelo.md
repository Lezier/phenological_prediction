# Decisión del modelo del prototipo

## Estado

**Decisión confirmada:** Random Forest A será el modelo de inferencia del
prototipo. La red neuronal se conservará como comparador experimental y su
evidencia no se eliminará del proyecto, del notebook ni del informe final.

## Significado de Random Forest A

La letra **A** identifica el escenario de datos que utiliza siete variables
climáticas:

1. Temperatura media.
2. Temperatura máxima media.
3. Temperatura mínima media.
4. Precipitación acumulada.
5. Radiación media.
6. Humedad media.
7. Grados-día acumulados.

La denominación no significa que sea una variante optimizada de Random Forest.

## Fundamento de la selección

La selección se fundamenta en el balance observado entre:

- desempeño predictivo;
- cobertura de datos;
- estabilidad bajo validación agrupada por estación;
- simplicidad de entrenamiento e inferencia;
- facilidad para construir una demostración con siete entradas climáticas.

Las cifras definitivas de RC3 se incorporarán después de la ejecución
completa y de la auditoría de resultados. Hasta entonces, los CSV y el modelo
serializado presentes corresponden a la base técnica de RC2.

## Alcance de la red neuronal

La red neuronal continúa siendo necesaria para demostrar que el modelo elegido
fue comparado con una alternativa de mayor complejidad mediante el mismo
protocolo y los mismos folds. No se presenta como un segundo modelo operativo.

## Limitaciones

- Random Forest A no fue sometido a tuning sistemático.
- Sus probabilidades no han sido calibradas.
- La evaluación utiliza datos europeos y no constituye validación operacional
  para Chile.
- El prototipo no sustituye el criterio agronómico ni una validación de campo.

## Documentos relacionados

- `hiperparametros_random_forest.md`
- `hiperparametros_red_neuronal.md`
- `../DATA_PROVENANCE.md`

