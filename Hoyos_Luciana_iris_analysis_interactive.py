# iris_analysis_interactive.py
# -*- coding: utf-8 -*-
"""
Pipeline completo de análisis del dataset Iris.
Incluye:
- Análisis exploratorio (EDA)
- Visualizaciones estáticas e interactivas
- Ingeniería de características
- Entrenamiento y comparación de modelos
- Evaluación final y exportación de resultados

Los gráficos interactivos se exportan a HTML.
"""

from __future__ import annotations
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    StratifiedKFold, GridSearchCV
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from joblib import dump

# Visualización interactiva
import plotly.express as px
import plotly.graph_objects as go
from plotly.io import write_html

# -----------------------------
# Configuración general
# -----------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# -----------------------------
# 1) Carga y preparación del dataset
# -----------------------------
iris = datasets.load_iris()

# Variables predictoras
X = pd.DataFrame(iris.data, columns=iris.feature_names)

# Variable objetivo
y = pd.Series(iris.target, name="target")

class_names = iris.target_names

# Limpieza de nombres de columnas
X.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in X.columns]

# Dataset completo para análisis exploratorio
species_map = {i: name for i, name in enumerate(class_names)}
df = pd.concat([X, y.map(species_map).rename("species")], axis=1)

# Crear carpeta de salida si no existe
os.makedirs("outputs", exist_ok=True)

# -----------------------------
# 2) Análisis exploratorio (EDA)
# -----------------------------
print("\n=== Dimensiones del dataset ===", df.shape)
print("\n=== Primeras filas ===\n", df.head())
print("\n=== Estadísticas descriptivas ===\n", df.describe(include="all"))
print("\n=== Distribución de clases ===\n", df["species"].value_counts())

# Heatmap de correlaciones
corr = df.drop(columns=["species"]).corr()
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap="viridis", fmt=".2f")
plt.title("Matriz de correlación - Iris")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png", dpi=150)
plt.close()

# Pairplot clásico
sns.pairplot(df, hue="species", corner=True)
plt.suptitle("Pairplot Iris", y=1.02)
plt.savefig("outputs/pairplot.png", dpi=150)
plt.close()

# Codificación numérica de la especie
le = LabelEncoder()
df["species_num"] = le.fit_transform(df["species"])

# -----------------------------
# 3) Visualizaciones interactivas
# -----------------------------

# a) Scatter matrix interactivo
fig_scatter_matrix = px.scatter_matrix(
    df,
    dimensions=X.columns,
    color="species",
    title="Iris - Scatter Matrix (Interactivo)",
    height=800
)
write_html(fig_scatter_matrix, "outputs/interactive_scatter_matrix.html", include_plotlyjs="cdn")

# b) Coordenadas paralelas (features normalizadas)
X_norm = (X - X.min()) / (X.max() - X.min())
X_norm["species_num"] = df["species_num"]

fig_parallel = px.parallel_coordinates(
    X_norm,
    dimensions=X.columns,
    color="species_num",
    color_continuous_scale=px.colors.diverging.Tealrose,
    title="Iris - Coordenadas paralelas (Interactivo)"
)
write_html(fig_parallel, "outputs/interactive_parallel_coordinates.html", include_plotlyjs="cdn")

# c) Dispersión 3D
fig_3d = px.scatter_3d(
    df,
    x="petal_length",
    y="petal_width",
    z="sepal_length",
    color="species",
    title="Iris - Dispersión 3D (Interactivo)",
    height=600
)
write_html(fig_3d, "outputs/interactive_scatter_3d.html", include_plotlyjs="cdn")

# d) PCA en 2D
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(StandardScaler().fit_transform(X))

fig_pca = px.scatter(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    color=df["species"],
    labels={"x": "PC1", "y": "PC2"},
    title="Iris - PCA 2D (Interactivo)"
)
write_html(fig_pca, "outputs/interactive_pca.html", include_plotlyjs="cdn")

# e) t-SNE (puede tardar un poco)
tsne = TSNE(
    n_components=2,
    random_state=RANDOM_STATE,
    init="pca",
    learning_rate="auto"
)
X_tsne = tsne.fit_transform(StandardScaler().fit_transform(X))

fig_tsne = px.scatter(
    x=X_tsne[:, 0],
    y=X_tsne[:, 1],
    color=df["species"],
    labels={"x": "t-SNE 1", "y": "t-SNE 2"},
    title="Iris - t-SNE (Interactivo)"
)
write_html(fig_tsne, "outputs/interactive_tsne.html", include_plotlyjs="cdn")

# -----------------------------
# 4) Ingeniería de características
# -----------------------------
X_feat = X.copy()
X_feat["sepal_ratio"] = X["sepal_length"] / X["sepal_width"]
X_feat["petal_ratio"] = X["petal_length"] / X["petal_width"]
X_feat["sepal_area"] = X["sepal_length"] * X["sepal_width"]
X_feat["petal_area"] = X["petal_length"] * X["petal_width"]

# -----------------------------
# 5) Entrenamiento y validación
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_feat,
    y,
    test_size=0.25,
    stratify=y,
    random_state=RANDOM_STATE
)

# Modelos a evaluar
models = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier())
    ]),
    "DecisionTree": Pipeline([
        ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))
    ]),
    "RandomForest": Pipeline([
        ("clf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE))
    ]),
    "SVM_RBF": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))
    ])
}

# Validación cruzada estratificada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

print("\n=== Validación cruzada ===")
cv_summary = []

for name, pipe in models.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
    cv_summary.append({
        "modelo": name,
        "accuracy_mean": scores.mean(),
        "accuracy_std": scores.std()
    })
    print(f"{name:>15}: {scores.mean():.3f} ± {scores.std():.3f}")

cv_df = pd.DataFrame(cv_summary)

# Tabla interactiva de resultados CV
fig_cv = go.Figure(data=[go.Table(
    header=dict(values=list(cv_df.columns), fill_color='lightgrey', align='left'),
    cells=dict(values=[cv_df[c] for c in cv_df.columns], align='left')
)])
fig_cv.update_layout(title_text="Resultados de validación cruzada")
write_html(fig_cv, "outputs/interactive_cv_results.html", include_plotlyjs="cdn")

# -----------------------------
# 6) Búsqueda de hiperparámetros
# -----------------------------
param_grid_svm = {
    "clf__C": [0.1, 1, 10, 100],
    "clf__gamma": ["scale", 0.1, 0.01, 0.001]
}

svm_grid = GridSearchCV(
    models["SVM_RBF"],
    param_grid_svm,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)
svm_grid.fit(X_train, y_train)

param_grid_rf = {
    "clf__n_estimators": [100, 200, 400],
    "clf__max_depth": [None, 3, 5, 7]
}

rf_grid = GridSearchCV(
    models["RandomForest"],
    param_grid_rf,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)
rf_grid.fit(X_train, y_train)

# Selección del mejor modelo
best_estimator = svm_grid if svm_grid.best_score_ >= rf_grid.best_score_ else rf_grid
best_model = best_estimator.best_estimator_
best_name = "SVM_RBF" if best_estimator is svm_grid else "RandomForest"

print(f"\nModelo seleccionado: {best_name}")

# -----------------------------
# 7) Evaluación final
# -----------------------------
y_pred = best_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average="weighted"
)
cm = confusion_matrix(y_test, y_pred)

print("\n=== Resultados en test ===")
print(f"Accuracy: {acc:.3f}")
print(f"Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")
print("\nReporte de clasificación:\n",
      classification_report(y_test, y_pred, target_names=class_names))

# Matriz de confusión interactiva
fig_cm = px.imshow(
    cm,
    text_auto=True,
    color_continuous_scale="Blues",
    labels=dict(x="Predicción", y="Real", color="Cuenta"),
    x=class_names,
    y=class_names,
    title=f"Matriz de confusión - {best_name}"
)
write_html(fig_cm, "outputs/interactive_confusion_matrix.html", include_plotlyjs="cdn")

# -----------------------------
# 8) Guardado de resultados
# -----------------------------
dump(best_model, "outputs/iris_best_model.joblib")

with open("outputs/summary.txt", "w", encoding="utf-8") as f:
    f.write("Resultados del análisis Iris\n")
    f.write(f"Modelo: {best_name}\n")
    f.write(f"Accuracy test: {acc:.3f}\n")
    f.write(f"Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}\n")

print("\nTodo listo 🚀 Revisa la carpeta 'outputs/'")
