# Informe Teórico-Práctico 02 — Machine Learning

## Descripción

Este proyecto corresponde al Informe Teórico-Práctico 02 de Machine Learning, centrado en el análisis del dataset `dataset_desercion_estudiantes.csv`. El objetivo principal es predecir la deserción estudiantil (Target: Desertó, donde 0 = No, 1 = Sí).

El informe se divide en cuatro partes principales:

1. **Análisis NO supervisado**: Implementación de algoritmos de clustering como KMeans, Fuzzy C-Means, Subtractive Clustering, DBSCAN y otros de la familia de clustering.
2. **Re-evaluación de etiquetas**: Revisión de aproximadamente el 30% de las etiquetas que pueden estar incorrectas.
3. **Modelos supervisados con etiquetas re-evaluadas**: Aplicación de Árbol de Decisión, Regresión Logística y Regresión Lineal.
4. **Comparación**: Evaluación comparativa entre el dataset original y el re-etiquetado.

## Requisitos

- Python 3.x
- Librerías necesarias:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn
  - pathlib

Instala las dependencias con:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Uso

1. Asegúrate de que el archivo `dataset_desercion_estudiantes.csv` esté en el mismo directorio que el script.
2. Ejecuta el script principal:

```bash
python informe_desercion.py
```

El script generará gráficos, modelos y resultados en la carpeta `output_desercion/output/`.

## Estructura del Proyecto

- `informe_desercion.py`: Script principal que ejecuta el análisis completo.
- `dataset_desercion_estudiantes.csv`: Dataset de deserción estudiantil.
- `output_desercion/output/`: Carpeta donde se guardan los resultados, incluyendo `comparacion_modelos.csv`.

## Resultados

Los resultados incluyen métricas de evaluación de modelos, gráficos de clustering, matrices de confusión y una comparación final entre modelos en `comparacion_modelos.csv`.

## Notas

- El script utiliza una semilla fija (SEED = 42) para reproducibilidad.
- Las advertencias están suprimidas para una salida más limpia.
- Los gráficos se generan en modo no interactivo (Agg backend) para compatibilidad con entornos sin GUI.