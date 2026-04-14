# 📘 Fundamentos de Aprendizaje Automático – SI3015

Repositorio de trabajos prácticos – Luciana Hoyos

Este repositorio contiene el desarrollo progresivo del ciclo de vida de Machine Learning a lo largo de varias semanas, incluyendo:

* Definición del problema
* Análisis exploratorio de datos (EDA)
* Limpieza y preprocesamiento
* Ingeniería de características
* Partición de datos
* Exportación para modelado
* Informe 1 Teórico Práctico

---

# 📅 Semana 2 – Ciclo de Vida ML con Iris

📂 Archivo: `Hoyos_Luciana_iris_analysis_interactive.py`

## 🎯 Objetivo

Implementar el ciclo completo de Machine Learning utilizando el dataset clásico **Iris**.

## 🔎 Problema

Clasificación supervisada multiclase para predecir la especie de flor:

* Setosa
* Versicolor
* Virginica

## 🧠 Etapas implementadas

### 1️⃣ Definición del problema

Clasificación multiclase con variable objetivo `species`.

### 2️⃣ Recolección de datos

Se usa el dataset Iris desde `sklearn.datasets`.

### 3️⃣ Procesamiento

* Validación de valores nulos
* Normalización con `StandardScaler`
* División Train/Test (75% / 25%) con estratificación

### 4️⃣ Entrenamiento

Modelo:

* **SVM (Support Vector Machine)** con kernel RBF
* Implementado mediante `Pipeline`

### 5️⃣ Evaluación

Métricas:

* Accuracy
* Precision
* Recall
* F1-score
* Matriz de confusión
* Classification report

📌 Resultado: Se implementa correctamente un pipeline profesional de ML desde cero.

---

# 📅 Semana 3 – Laboratorio FinTech Sintético (EDA + Preprocesamiento)

📂 Archivo: `lect_03_hoyos_luciana_lab_fintech.py`

## 🎯 Objetivo

Realizar un análisis exploratorio completo y preparar datos financieros sintéticos para modelado futuro.

Dataset 100% sintético con fines académicos.

## 🧠 Etapas implementadas

### 0️⃣ Carga y validación del diccionario

* Validación del JSON de metadatos

### 1️⃣ Carga del CSV

* Parsing de fechas
* Ordenamiento temporal

### 2️⃣ EDA básico

* Info del dataset
* Análisis de nulos

### 2.5️⃣ EDA visual interactivo

Se generan archivos HTML con:

* Scatter Matrix
* Coordenadas paralelas
* Scatter 3D
* UMAP 2D
* UMAP 3D

Todos exportados en:

```
data_output_finanzas_sintetico/
```

### 3️⃣ Limpieza

* Imputación:

  * Numéricas → mediana
  * Categóricas → `"__MISSING__"`

### 4️⃣ Ingeniería de características

* Retornos porcentuales
* Log-retornos de precio
* Agrupación por empresa y fecha

### 5️⃣ Preparación para ML

* Eliminación de IDs y fecha
* One-hot encoding
* Escalado
* Split temporal (evita fuga de datos)

### 6️⃣ Exportación

Se generan:

* `fintech_train.parquet`
* `fintech_test.parquet`
* `processed_schema.json`
* `features_columns.txt`

📌 Resultado: Pipeline robusto de preprocesamiento financiero listo para modelado.

---

# 📅 Semana 4 – Identificar patrones en la deserción estudiantil

📂 Archivo: `lecture4_EDA.py`

## 📊 Lo que contiene:

### 1️⃣ Medidas de Tendencia Central

* Media, mediana, moda de variables numéricas
* Modas de variables categóricas
* Proporciones

### 2️⃣ Cuartiles e IQR

* Q1, Q2 (mediana), Q3
* Rango Intercuartílico
* Límites para outliers

### 3️⃣ Percentiles

* P10, P25, P50, P75, P90

### 4️⃣ Correlaciones

* Matriz de correlación completa
* Pearson vs Spearman
* Heatmap visual

### 5️⃣ Tablas Pivote

* Promedio académico por Beca/Deserción
* Materias perdidas por Beca/Deserción
* Conteos cruzados

### 6️⃣ Visualizaciones (6 gráficos PNG)
 
* ✅ Histogramas de distribución
* ✅ Boxplots por deserción
* ✅ Scatter plot (Promedio vs Materias)
* ✅ Barras de proporciones
* ✅ Gráfico stacked
* ✅ Heatmap de correlación

### 7️⃣ Resumen por clase

* Estadísticas descriptivas separadas por Desertó
* Comparación de medias

### 8️⃣ Identificación de Outliers

* Detección por método IQR
* Porcentaje de outliers

### 9️⃣ Insights Finales

* Conclusiones automáticas del análisis

### 📁 Estructura de salida:
```
eda_output/
├── 00_descripcion_basica.csv
├── 01_tendencia_central_numericas.csv
├── 01_moda_categoricas.json
├── 01_proporciones_categoricas.json
├── 02_iqr_results.json
├── 03_percentiles.json
├── 04_correlation_stats.json
├── 04_heatmap_correlacion.png
├── 05_pivot_*.csv (3 archivos)
├── 06_histogramas_distribuciones.png
├── 06_boxplots_por_desercion.png
├── 06_scatter_promedio_materias.png
├── 06_barras_proporciones.png
├── 06_stacked_desercion_beca.png
├── 07_resumen_estadistico_por_clase.csv
├── 07_comparacion_medias_por_clase.csv
└── 08_outliers_info.json
```
---

# 📅 Semana 5 – Informe 1 del Proyecto de Aprendizaje

📂 Carpeta: informe_teorico_practico_ML_LucianaHoyosPerez

El informe consolida todo el trabajo realizado en las semanas anteriores y formaliza el desarrollo del proyecto bajo estándares académicos.

---

# 📅 Semana 5 – Modelado Predictivo con Dataset de Películas

📂 Carpeta: Lecture_05_movies_exercise

## 🎯 Objetivo

Implementar técnicas de aprendizaje supervisado para predecir ingresos de películas y clasificar si pertenecen a altos ingresos.

## 🧠 Etapas implementadas

### 1️⃣ Predicción continua

* Estimación de ingresos (`Gross`) con Regresión Lineal regularizada (Ridge y Lasso).

### 2️⃣ Clasificación binaria

* Determinación de altos ingresos con Regresión Logística.

### 3️⃣ Preprocesamiento

* Limpieza de datos: extracción de año, conversión de duración, eliminación de separadores y símbolos monetarios.

📂 Archivos principales: `lecture05_movies_exercise_luciana_hoyos.py`, `movies.csv`, `output/`

---

# 📅 Semana 6 – Clasificación de Deserción Estudiantil

📂 Carpeta: Lecture_06_model_exercise

## 🎯 Objetivo

Predecir si un estudiante presenta riesgo de deserción utilizando modelos basados en árboles.

## 🧠 Etapas implementadas

### 1️⃣ Modelos comparados

* Random Forest
* Gradient Boosting

### 2️⃣ Preprocesamiento

* Eliminación de duplicados, imputación de nulos, codificación OneHot, escalado StandardScaler.

📂 Archivos principales: `lecture_06_model_exercise_luciana_hoyos.py`, `dataset_desercion_estudiantes.csv`

---

# 📅 Semana 9 – Clustering Básico

📂 Carpeta: Lecture_09_clustering

## 🎯 Objetivo

Implementar análisis de clustering básico con KMeans y DBSCAN en datasets sintéticos.

## 🧠 Etapas implementadas

### 1️⃣ Preprocesamiento

* Separación de variables, escalado, PCA.

### 2️⃣ Clustering

* KMeans con selección de k por método del codo y silhouette.
* DBSCAN.

### 3️⃣ Evaluación

* Métricas de silhouette, comparación con etiquetas reales si disponibles.

📂 Archivos principales: `lecture_09.py`, `lecture_09_realista.py`, `dataset_sintetico_FIRE_UdeA.csv`, `dataset_sintetico_FIRE_UdeA_realista.csv`

---

# 📅 Semana 10 – Clustering Avanzado

📂 Carpeta: Lecture_10_clustering

## 🎯 Objetivo

Aplicar técnicas avanzadas de clustering incluyendo implementaciones desde cero.

## 🧠 Etapas implementadas

### 1️⃣ Algoritmos

* KMeans, DBSCAN, Subtractive Clustering, Fuzzy C-Means.

### 2️⃣ Comparación

* Resultados sobre datasets sintéticos y realistas.

### 3️⃣ Evaluación

* ARI, visualizaciones 2D/3D, guardado de resultados.

📂 Archivos principales: `lecture_10.py`, `lecture_10_substractive.py`, `lecture_10_realista.py`, datasets sintéticos.

---

# 📅 Semana 11 – Informe 2 Teórico-Práctico: Análisis de Deserción Estudiantil

📂 Carpeta: informe_2_teorico_practico

## 🎯 Objetivo

Predecir la deserción estudiantil utilizando técnicas de aprendizaje no supervisado y supervisado, incluyendo clustering y modelos de clasificación.

## 🧠 Etapas implementadas

### 1️⃣ Análisis NO supervisado

* Algoritmos de clustering: KMeans, Fuzzy C-Means, Subtractive Clustering, DBSCAN.

### 2️⃣ Re-evaluación de etiquetas

* Revisión del 30% de etiquetas potencialmente incorrectas.

### 3️⃣ Modelos supervisados

* Árbol de Decisión, Regresión Logística, Regresión Lineal con etiquetas re-evaluadas.

### 4️⃣ Comparación

* Evaluación comparativa entre dataset original y re-etiquetado.

📂 Archivos principales: `informe_desercion.py`, `dataset_desercion_estudiantes.csv`, `output_desercion/output/comparacion_modelos.csv`

---
