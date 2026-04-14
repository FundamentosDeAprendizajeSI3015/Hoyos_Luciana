
```markdown
# 🎓 Clasificación de Deserción Estudiantil con Machine Learning

## 📌 Descripción del Proyecto

Este proyecto implementa modelos de **aprendizaje supervisado** para analizar la deserción estudiantil a partir de un dataset académico.

El objetivo principal es **predecir si un estudiante está preparado laboralmente o presenta riesgo de deserción**, utilizando modelos de clasificación basados en árboles.

Se comparan dos enfoques:

- 🌳 **Random Forest**
- ⚡ **Gradient Boosting**

---

## 📂 Dataset

Archivo utilizado:

```

dataset_desercion_estudiantes (1).csv

```

El dataset contiene información académica, demográfica y de desempeño de estudiantes, junto con una variable objetivo que indica si el estudiante presenta deserción.

---

## 🧹 Preprocesamiento de Datos

Antes del modelado, se realizaron las siguientes transformaciones:

- Eliminación de registros duplicados
- Relleno de valores nulos en variables numéricas usando la mediana
- Separación entre variables numéricas y categóricas
- Codificación de variables categóricas con **OneHotEncoder**
- Escalamiento de variables numéricas con **StandardScaler**

---

## 🎯 Variable Objetivo

La variable objetivo (`desercion`) originalmente contiene valores categóricos:

```

"No", "Sí"

```

Se transformó a formato numérico para facilitar el modelado:

```

No → 0
Sí → 1

```

Esto permite utilizar correctamente métricas de clasificación como precisión, recall y F1-score.

---

## 🔀 División del Dataset

Los datos se dividieron de manera estratificada en:

- **60% Entrenamiento**
- **20% Validación**
- **20% Prueba**

Esto asegura una distribución balanceada de clases en cada subconjunto.

---

## ⚙️ Pipeline de Procesamiento

Se utilizó un `ColumnTransformer` para aplicar transformaciones diferenciadas:

- 🔢 Variables numéricas → `StandardScaler`
- 🏷 Variables categóricas → `OneHotEncoder`

Este preprocesamiento se integra directamente dentro de los modelos mediante `Pipeline`.

---

## 🧠 Modelos Implementados

### 🌳 Random Forest

- Conjunto de árboles de decisión
- Reduce el overfitting mediante promediado
- Configuración:
  - `n_estimators = 150`

---

### ⚡ Gradient Boosting

- Construcción secuencial de árboles
- Optimiza errores de modelos anteriores
- Configuración:
  - `n_estimators = 150`

---

## 📊 Evaluación de Modelos

Se evaluaron los modelos en tres conjuntos:

- Entrenamiento
- Validación
- Prueba

### Métricas utilizadas:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **AUC (Área bajo la curva ROC)**

---

## 📈 Visualizaciones

El proyecto incluye múltiples gráficos para análisis:

### 🌳 Interpretabilidad
- Visualización de un árbol representativo de cada modelo

### 🔍 Evaluación
- Matriz de confusión (numérica y gráfica)
- Curva ROC

Estas visualizaciones permiten entender tanto el rendimiento como el comportamiento interno de los modelos.

---

## 🗂 Estructura del Proyecto

```

.
│
├── dataset_desercion_estudiantes.csv
├── modelo_desercion.py
├── README.md

````

---

## ⚙️ Requisitos

Instalar dependencias con:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
````

---

## ▶️ Ejecución

Para ejecutar el proyecto:

```bash
python lecture_06_model_exercise_luciana_hoyos.py
```

---

## 🧠 Conclusiones

* Ambos modelos logran una buena capacidad de clasificación.
* **Random Forest** es más robusto frente a ruido.
* **Gradient Boosting** puede capturar patrones más complejos.
* La transformación de la variable objetivo fue clave para evitar errores en métricas.
* El uso de pipelines permite un flujo limpio y reproducible.

---

## 🚀 Posibles Mejoras

* Ajuste de hiperparámetros (GridSearch / RandomSearch)
* Análisis de importancia de variables
* Manejo de desbalance de clases (SMOTE, class_weight)
* Exportación de resultados a reportes automáticos

---

## 👩‍💻 Autora

**Luciana Hoyos**

```

---
