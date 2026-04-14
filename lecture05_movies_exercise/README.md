```markdown
# 🎬 Modelado Predictivo con Dataset de Películas

## 🧩 Introducción

En este proyecto se implementan técnicas de **aprendizaje supervisado** para analizar un conjunto de datos de películas. Se abordan dos problemas principales:

- 📈 **Predicción continua**: estimar los ingresos (`Gross`) mediante modelos de regresión.
- 🏷 **Clasificación binaria**: determinar si una película pertenece al grupo de altos ingresos.

Para ello, se utilizan modelos de **Regresión Lineal regularizada (Ridge y Lasso)** y **Regresión Logística**, junto con técnicas de optimización y preprocesamiento.

---

## 📁 Fuente de datos

El dataset utilizado es:

```

movies.csv

```

Incluye variables relevantes como:

- `YEAR`: año de lanzamiento  
- `RunTime`: duración de la película  
- `RATING`: calificación promedio  
- `VOTES`: número de votos recibidos  
- `Gross`: ingresos generados  

---

## 🔧 Preparación de los datos

Antes de modelar, se realizó un proceso de limpieza para asegurar consistencia:

- Se extrajo el año en formato numérico desde `YEAR`
- Se convirtió `RunTime` a minutos (valor numérico)
- Se eliminaron separadores (comas) en `VOTES`
- Se limpiaron símbolos monetarios en `Gross`
- Se transformaron todas las variables a tipo numérico
- Se eliminaron registros con valores faltantes

---

## 📈 Modelos de Regresión

### 🎯 Propósito

Predecir el valor de ingresos (`Gross`) a partir de variables explicativas.

### 🔍 Variables utilizadas

- `RunTime`
- `RATING`
- `VOTES`
- `YEAR`

### 🧠 Algoritmos empleados

Se trabajó con dos variantes de regresión lineal regularizada:

- **Ridge Regression** (penalización L2)
- **Lasso Regression** (penalización L1)

Ambos modelos se implementaron mediante un `Pipeline` que incluye:

1. Generación de características polinómicas  
2. Escalamiento de variables  
3. Modelo de regresión  

---

### ⚙️ Ajuste de hiperparámetros

Se utilizó:

```

RandomizedSearchCV

```

Para encontrar configuraciones óptimas de:

- Grado del polinomio (1 a 3)
- Nivel de regularización (`alpha`)

---

### 📊 Evaluación del modelo

Las métricas utilizadas fueron:

- **R²**: mide la proporción de varianza explicada  
- **MAE**: error promedio absoluto  

---

### 🖼 Resultados visuales

Las siguientes gráficas se generan automáticamente:

- `reg_lineal_train_test.png` → distribución de datos  
- `ridge_prediction.png` → predicciones del modelo Ridge  
- `lasso_prediction.png` → predicciones del modelo Lasso  

---

## 📊 Clasificación con Regresión Logística

### 🎯 Objetivo

Determinar si una película tiene ingresos altos o bajos.

Se definió una variable binaria:

```

High_Gross = 1 si el ingreso supera la mediana
High_Gross = 0 en caso contrario

```

---

### 🧠 Modelo implementado

Se utilizó un pipeline compuesto por:

1. Transformación polinómica  
2. Normalización de datos  
3. Modelo de regresión logística  

---

### ⚙️ Optimización

También se aplicó `RandomizedSearchCV` para ajustar:

- Grado de las características polinómicas  
- Parámetro de regularización (`C`)  

---

### 📈 Métricas de desempeño

- **Accuracy**: proporción de predicciones correctas  
- **F1-score**: balance entre precisión y recall  
- **Matriz de confusión**: análisis detallado de clasificación  

---

### 🖼 Salida gráfica

- `confusion_matrix.png`

---

## 🗂 Organización del proyecto

```

.
│
├── movies.csv
├── lecture05_movies_exercise_santiago_manco.py
├── output/
│   ├── reg_lineal_train_test.png
│   ├── ridge_prediction.png
│   ├── lasso_prediction.png
│   └── confusion_matrix.png

````

---

## ⚙️ Instalación de dependencias

Ejecutar:

```bash
pip install numpy pandas matplotlib scikit-learn scipy
````

---

## ▶️ Ejecución

Para correr el proyecto:

```bash
python lecture05_movies_exercise_luciana_hoyos.py
```

Los resultados se guardarán automáticamente en la carpeta `output/`.

---

## 🧠 Análisis final

* Los modelos **Ridge y Lasso** presentan comportamientos similares, lo que sugiere consistencia en las variables utilizadas.
* La predicción de ingresos es más compleja en valores extremos, lo que indica posibles outliers o alta variabilidad.
* La **regresión logística** logra un desempeño sólido en la clasificación.
* La incorporación de **regularización** resulta clave para evitar sobreajuste y mejorar la generalización.

```

---
