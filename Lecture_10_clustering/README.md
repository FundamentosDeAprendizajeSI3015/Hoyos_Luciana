# Clustering Avanzado - FIRE UdeA

Este repositorio contiene proyectos de clustering aplicados a datasets sintéticos y realistas del conjunto FIRE UdeA.

## Archivos principales

- `lecture_10.py`
  - Implementa un análisis de clustering con KMeans y DBSCAN.
  - Incluye preprocesamiento, reducción de dimensionalidad con PCA, selección de `k` con el método del codo y silhouette, visualización 2D/3D y evaluación de ARI si existe etiqueta real.

- `lecture_10_substractive.py`
  - Contiene implementaciones de `SubtractiveClustering` y `FuzzyCMeans` desde cero.
  - Compara resultados con KMeans y DBSCAN sobre un dataset sintético.

- `lecture_10_realista.py`
  - Ejecuta clustering sobre un dataset más realista.
  - Incluye KMeans, DBSCAN, Subtractive Clustering y Fuzzy C-Means.
  - Guarda resultados en `clusters_resultado.csv`.

- `dataset_sintetico_FIRE_UdeA.csv`
  - Dataset sintético original con etiquetas de clase.

- `dataset_sintetico_FIRE_UdeA_realista.csv`
  - Dataset de mayor complejidad y realismo con variables adicionales.

## Requisitos

- Python 3.8+ recomendado
- Bibliotecas:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scikit-learn`

## Instalación

Puedes instalar las dependencias con `pip`:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Uso

Ejecuta los scripts desde la carpeta del proyecto:

```bash
python lecture_10.py
python lecture_10_substractive.py
python lecture_10_realista.py
```

`lecture_10_realista.py` generará un archivo de salida:

- `clusters_resultado.csv`

## Nota

Los scripts asumen que los archivos CSV están en la misma carpeta del proyecto.
