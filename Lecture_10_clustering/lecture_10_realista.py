# ==========================================================
# PROYECTO: CLUSTERING (KMEANS + DBSCAN)
# Dataset: FIRE UdeA (versión realista)
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.impute import SimpleImputer

from mpl_toolkits.mplot3d import Axes3D


# ==========================================================
# 🔹 CONFIGURACIÓN INICIAL
# ==========================================================

SEED = 42
plt.rc('font', family='serif', size=12)


# ==========================================================
# 🔹 1. CARGA Y EXPLORACIÓN DEL DATASET
# ==========================================================

archivo = "dataset_sintetico_FIRE_UdeA_realista.csv"
data = pd.read_csv(archivo)

print("Vista previa:")
print(data.head())

print("\nDimensiones:", data.shape)
print("\nColumnas:", list(data.columns))
print("\nTipos de datos:")
print(data.dtypes)

print("\nValores faltantes:")
print(data.isna().sum())


# ==========================================================
# 🔹 2. SEPARACIÓN DE FEATURES Y LABEL (si existe)
# ==========================================================

if "label" in data.columns:
    y_real = data["label"]
    X = data.drop(columns=["label"])
else:
    y_real = None
    X = data.copy()

print("\nVariables usadas:")
print(list(X.columns))


# ==========================================================
# 🔹 3. IDENTIFICACIÓN DE VARIABLES
# ==========================================================

cols_num = X.select_dtypes(include=np.number).columns
cols_cat = X.select_dtypes(include=["object", "category"]).columns

print("\nNuméricas:", list(cols_num))
print("Categóricas:", list(cols_cat))


# ==========================================================
# 🔹 4. PREPROCESAMIENTO
# ==========================================================
# Se imputan valores faltantes y se transforman variables

pipe_num = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

pipe_cat = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocesador = ColumnTransformer([
    ("num", pipe_num, cols_num),
    ("cat", pipe_cat, cols_cat)
])

# Aplicar transformación
X_transf = preprocesador.fit_transform(X)

print("\nShape después del preprocesamiento:", X_transf.shape)
print("Valores NaN restantes:", np.isnan(X_transf).sum())


# ==========================================================
# 🔹 5. REDUCCIÓN DE DIMENSIONALIDAD (PCA)
# ==========================================================

# ---------- PCA 2D ----------
pca_2 = PCA(n_components=2, random_state=SEED)
X_2d = pca_2.fit_transform(X_transf)

plt.figure(figsize=(8,5))
plt.scatter(X_2d[:, 0], X_2d[:, 1])
plt.title("Proyección PCA (2D)")
plt.xlabel("CP1")
plt.ylabel("CP2")
plt.show()

# ---------- PCA 3D ----------
pca_3 = PCA(n_components=3, random_state=SEED)
X_3d = pca_3.fit_transform(X_transf)

fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], s=30)
ax.set_title("Proyección PCA (3D)")
plt.show()


# ==========================================================
# 🔹 6. SELECCIÓN DE K (CODO + SILHOUETTE)
# ==========================================================

valores_k = range(2, 11)
inercia = []
sil_scores = []

for k in valores_k:
    modelo_temp = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    etiquetas = modelo_temp.fit_predict(X_transf)

    inercia.append(modelo_temp.inertia_)
    sil_scores.append(silhouette_score(X_transf, etiquetas))

# Gráfica codo
plt.figure(figsize=(8,5))
plt.plot(valores_k, inercia, marker='o')
plt.title("Método del codo")
plt.xlabel("k")
plt.ylabel("Inercia")
plt.show()

# Gráfica silhouette
plt.figure(figsize=(8,5))
plt.plot(valores_k, sil_scores, marker='o')
plt.title("Silhouette Score")
plt.xlabel("k")
plt.ylabel("Score")
plt.show()


# ==========================================================
# 🔹 7. MODELO FINAL KMEANS
# ==========================================================

k_final = 2
print(f"\nNúmero de clusters seleccionado: {k_final}")

kmeans = KMeans(n_clusters=k_final, random_state=SEED, n_init=10)
labels_kmeans = kmeans.fit_predict(X_transf)

print("\nResultados KMeans:")
print("Inercia:", kmeans.inertia_)
print("Silhouette:", silhouette_score(X_transf, labels_kmeans))

# Visualización 2D
plt.figure(figsize=(8,5))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_kmeans)
plt.title("Clusters KMeans (2D)")
plt.show()

# Visualización 3D
fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=labels_kmeans)
plt.title("Clusters KMeans (3D)")
plt.show()

# Centroides
centros = kmeans.cluster_centers_
centros_3d = pca_3.transform(centros)

fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=labels_kmeans, s=30)
ax.scatter(centros_3d[:, 0], centros_3d[:, 1], centros_3d[:, 2], marker='X', s=200)
plt.title("KMeans con centroides")
plt.show()


# ==========================================================
# 🔹 8. MODELO DBSCAN
# ==========================================================

dbscan = DBSCAN(eps=1.2, min_samples=5)
labels_db = dbscan.fit_predict(X_transf)

print("\nResultados DBSCAN:")
print("Clusters encontrados:", np.unique(labels_db))

if len(set(labels_db) - {-1}) > 1:
    mask = labels_db != -1
    print("Silhouette:", silhouette_score(X_transf[mask], labels_db[mask]))
else:
    print("No hay suficientes clusters para silhouette")

# Visualización
plt.figure(figsize=(8,5))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_db)
plt.title("DBSCAN (2D)")
plt.show()


# ==========================================================
# 🔹 9. COMPARACIÓN CON LABEL REAL
# ==========================================================

if y_real is not None:
    print("\nComparación con etiquetas reales:")
    print("ARI:", adjusted_rand_score(y_real, labels_kmeans))


# ==========================================================
# 🔹 10. ANÁLISIS DE ERRORES POR UNIDAD
# ==========================================================

if y_real is not None and "unidad" in data.columns:

    y_true = np.array(y_real)
    y_pred = np.array(labels_kmeans)

    # Ajuste por posible inversión de clusters
    if len(np.unique(y_pred)) == 2:
        y_pred_inv = 1 - y_pred

        if (y_true != y_pred_inv).sum() < (y_true != y_pred).sum():
            y_pred = y_pred_inv
            print("\nSe invirtieron los clusters para mejor correspondencia.")

    errores = (y_true != y_pred).astype(int)

    df_error = data.copy()
    df_error["error"] = errores

    resumen = (
        df_error.groupby("unidad")
        .agg(total=("error", "count"),
             errores=("error", "sum"),
             tasa=("error", "mean"))
        .sort_values(by="errores", ascending=False)
    )

    print("\nErrores por unidad:")
    print(resumen)

    resumen.to_csv("resumen_errores.csv")
    print("\nArchivo guardado: resumen_errores.csv")


# ==========================================================
# 🔹 11. EXPORTAR RESULTADOS
# ==========================================================

output = data.copy()
output["kmeans"] = labels_kmeans
output["dbscan"] = labels_db

print("\nPreview resultados:")
print(output.head())