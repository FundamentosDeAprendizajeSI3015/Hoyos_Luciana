# ==========================================================
# PROYECTO: MODELOS DE REGRESIÓN Y CLASIFICACIÓN
# DATASET: PELÍCULAS (MOVIES)
# ==========================================================
# Objetivo:
# 1. Predecir los ingresos (Gross) usando regresión
# 2. Clasificar si una película es de alto ingreso o no
# ==========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import reciprocal
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.metrics import mean_absolute_error, f1_score, confusion_matrix, ConfusionMatrixDisplay


# ==========================================================
# CONFIGURACIÓN GENERAL DEL ENTORNO
# ==========================================================

SEED = 42  # Semilla para garantizar reproducibilidad
np.random.seed(SEED)

# Configuración estética de gráficas
plt.rc('font', family='serif', size=12)

# Carpeta donde se almacenarán los resultados
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# FUNCIONES AUXILIARES
# ==========================================================

def limpiar_dataset(df):
    """
    Limpia y transforma las variables del dataset.

    Se convierten columnas que originalmente son texto a formato numérico,
    eliminando símbolos y caracteres innecesarios.
    """

    # YEAR: extrae el primer año válido (ej: "1994–1998" → 1994)
    df["YEAR"] = pd.to_numeric(
        df["YEAR"].astype(str).str.extract(r'(\d{4})')[0],
        errors='coerce'
    )

    # RunTime: extrae duración en minutos
    df["RunTime"] = pd.to_numeric(
        df["RunTime"].astype(str).str.extract(r'(\d+)')[0],
        errors='coerce'
    )

    # VOTES: elimina comas para permitir conversión numérica
    df["VOTES"] = pd.to_numeric(
        df["VOTES"].astype(str).str.replace(",", ""),
        errors='coerce'
    )

    # Gross: elimina símbolos monetarios (ej: $, ,)
    df["Gross"] = pd.to_numeric(
        df["Gross"].astype(str).str.replace(r"[^\d.]", "", regex=True),
        errors='coerce'
    )

    # RATING: asegurar formato numérico
    df["RATING"] = pd.to_numeric(df["RATING"], errors='coerce')

    # Eliminación de registros incompletos
    df = df.dropna()

    return df


def graficar_train_test(X_train, X_test, y_train, y_test):
    """
    Visualiza la separación entre datos de entrenamiento y prueba.
    Permite verificar distribución y posible sesgo.
    """
    plt.figure()
    plt.scatter(X_train["RATING"], y_train, label="Train", c='c')
    plt.scatter(X_test["RATING"], y_test, label="Test", c='m')
    plt.xlabel("Rating")
    plt.ylabel("Gross")
    plt.title("Distribución de datos (Train vs Test)")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/train_test_split.png")
    plt.close()


def construir_pipeline_regresion(modelo):
    """
    Construye un pipeline para modelos de regresión que incluye:
    1. Expansión polinómica (capturar relaciones no lineales)
    2. Estandarización (mejora estabilidad numérica)
    3. Modelo (Ridge o Lasso)
    """
    return Pipeline([
        ("poly", PolynomialFeatures(include_bias=False)),
        ("scaler", StandardScaler()),
        ("model", modelo)
    ])


def construir_pipeline_clasificacion():
    """
    Pipeline para clasificación con regresión logística.
    Sigue el mismo preprocesamiento que regresión.
    """
    return Pipeline([
        ("poly", PolynomialFeatures(include_bias=False)),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=10000))
    ])


def plot_predicciones(y_real, y_pred, nombre):
    """
    Genera gráfico de valores reales vs predichos.
    Permite evaluar visualmente el desempeño del modelo.
    """
    plt.figure()
    plt.scatter(y_real, y_pred)
    plt.xlabel("Valor real")
    plt.ylabel("Predicción")
    plt.title(nombre)
    plt.savefig(f"{OUTPUT_DIR}/{nombre}.png")
    plt.close()


# ==========================================================
# CARGA Y PREPROCESAMIENTO DE DATOS
# ==========================================================

# Lectura del dataset
data = pd.read_csv("movies.csv")
print("Columnas disponibles:", data.columns)

# Limpieza de datos
data = limpiar_dataset(data)


# ==========================================================
# REGRESIÓN LINEAL
# Objetivo: predecir ingresos (Gross)
# ==========================================================

print("\n===== REGRESIÓN LINEAL =====\n")

# Variables predictoras
variables = ["RunTime", "RATING", "VOTES", "YEAR"]
X = data[variables]

# Variable objetivo
y = data["Gross"]

# División en entrenamiento y prueba
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

# Visualización de la división
graficar_train_test(X_tr, X_te, y_tr, y_te)


# -------- Definición de modelos --------

ridge_pipe = construir_pipeline_regresion(Ridge())
lasso_pipe = construir_pipeline_regresion(Lasso(max_iter=10000))

# Espacio de búsqueda de hiperparámetros
parametros = {
    "poly__degree": range(1, 4),  # complejidad del modelo
    "model__alpha": reciprocal(1e-4, 1e2)  # regularización
}

# Búsqueda aleatoria de hiperparámetros
ridge_search = RandomizedSearchCV(
    ridge_pipe, parametros, n_iter=40, cv=4, random_state=SEED
)

lasso_search = RandomizedSearchCV(
    lasso_pipe, parametros, n_iter=40, cv=4, random_state=SEED
)

# Entrenamiento
ridge_search.fit(X_tr, y_tr)
lasso_search.fit(X_tr, y_tr)


# -------- Evaluación --------

print("Ridge mejores parámetros:", ridge_search.best_params_)
print("Lasso mejores parámetros:", lasso_search.best_params_)

print("\nEvaluación Ridge")
print("R2:", ridge_search.score(X_te, y_te))
print("MAE:", mean_absolute_error(y_te, ridge_search.predict(X_te)))

print("\nEvaluación Lasso")
print("R2:", lasso_search.score(X_te, y_te))
print("MAE:", mean_absolute_error(y_te, lasso_search.predict(X_te)))


# Visualización de predicciones
plot_predicciones(y_te, ridge_search.predict(X_te), "ridge_results")
plot_predicciones(y_te, lasso_search.predict(X_te), "lasso_results")


# ==========================================================
# REGRESIÓN LOGÍSTICA
# Objetivo: clasificar películas de alto ingreso
# ==========================================================

print("\n===== REGRESIÓN LOGÍSTICA =====\n")

# Definición de umbral (mediana)
umbral = data["Gross"].median()

# Variable binaria: 1 = alto ingreso, 0 = bajo ingreso
data["High_Gross"] = (data["Gross"] > umbral).astype(int)

X_c = data[variables]
y_c = data["High_Gross"]

# División de datos
X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
    X_c, y_c, test_size=0.2, random_state=SEED
)

# Pipeline de clasificación
logistic_pipe = construir_pipeline_clasificacion()

# Hiperparámetros
parametros_lr = {
    "poly__degree": range(1, 4),
    "clf__C": reciprocal(1e-4, 1e2)
}

# Búsqueda de mejores parámetros
logistic_search = RandomizedSearchCV(
    logistic_pipe, parametros_lr, n_iter=40, cv=4, random_state=SEED
)

# Entrenamiento
logistic_search.fit(X_tr_c, y_tr_c)


# -------- Evaluación --------

print("Mejores parámetros:", logistic_search.best_params_)
print("Accuracy:", logistic_search.score(X_te_c, y_te_c))
print("F1-score:", f1_score(y_te_c, logistic_search.predict(X_te_c)))


# -------- Matriz de confusión --------

cm = confusion_matrix(y_te_c, logistic_search.predict(X_te_c))
disp = ConfusionMatrixDisplay(cm)
disp.plot()

plt.title("Matriz de Confusión")
plt.savefig(f"{OUTPUT_DIR}/conf_matrix.png")
plt.close()


# ==========================================================
# FINAL
# ==========================================================

print("\n✔ Ejecución completada correctamente")
print(f"📁 Resultados disponibles en: {OUTPUT_DIR}/")