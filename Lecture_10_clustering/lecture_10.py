# ==========================================================
# PROYECTO: CLUSTERING - KMEANS vs DBSCAN
# Dataset sintético FIRE UdeA
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

# Soporte para gráficos 3D
from mpl_toolkits.mplot3d import Axes3D

# ==========================================================
# CONFIGURACIÓN GENERAL
# ==========================================================

SEED = 42
plt.rc('font', family='serif', size=12)

# ==========================================================
# 1. CARGA DEL DATASET
# ==========================================================

ruta_archivo = "dataset_sintetico_FIRE_UdeA.csv"
df = pd.read_csv(ruta_archivo)

print("Vista inicial del dataset:")
print(df.head())

print("\nTamaño del dataset:", df.shape)
print("\nColumnas disponibles:", list(df.columns))

# ==========================================================
# 2. SEPARACIÓN DE VARIABLES
# ==========================================================
# En clustering NO se usa la etiqueta para entrenar,
# pero se puede conservar para evaluación posterior

if "label" in df.columns:
    etiquetas_reales = df["label"]
    datos = df.drop(columns=["label"])
else:
    etiquetas_reales = None
    datos = df.copy()

print("\nVariables utilizadas:")
print(list(datos.columns))

# ==========================================================
# 3. PREPROCESAMIENTO
# ==========================================================
# Se normalizan todas las variables para evitar sesgos por escala

columnas_numericas = datos.columns.tolist()

pipeline_numerico = Pipeline([
    ("scaler", StandardScaler())
])

preprocesador = ColumnTransformer([
    ("num", pipeline_numerico, columnas_numericas)
])

# Transformación de los datos
datos_escalados = preprocesador.fit_transform(datos)

# ==========================================================
# 4. REDUCCIÓN DE DIMENSIONALIDAD (PCA)
# ==========================================================
# Se usa PCA para visualizar en 2D y 3D

pca_2d = PCA(n_components=2, random_state=SEED)
datos_2d = pca_2d.fit_transform(datos_escalados)

plt.figure(figsize=(8,5))
plt.scatter(datos_2d[:, 0], datos_2d[:, 1], s=30)
plt.title("Proyección 2D con PCA")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.show()

# ---------- PCA en 3D ----------

pca_3d = PCA(n_components=3, random_state=SEED)
datos_3d = pca_3d.fit_transform(datos_escalados)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(datos_3d[:, 0], datos_3d[:, 1], datos_3d[:, 2], s=30)
ax.set_title("Proyección 3D con PCA")
ax.set_xlabel("CP1")
ax.set_ylabel("CP2")
ax.set_zlabel("CP3")
ax.view_init(elev=25, azim=45)

plt.show()

# ==========================================================
# 5. SELECCIÓN DE K (MÉTODO DEL CODO + SILHOUETTE)
# ==========================================================

inercia = []
sil_scores = []
rango_k = range(2, 11)

for k in rango_k:
    
    modelo_temp = Pipeline([
        ("prep", preprocesador),
        ("kmeans", KMeans(n_clusters=k, random_state=SEED, n_init=10))
    ])
    
    modelo_temp.fit(datos)
    
    etiquetas = modelo_temp["kmeans"].labels_
    
    inercia.append(modelo_temp["kmeans"].inertia_)
    sil_scores.append(silhouette_score(preprocesador.transform(datos), etiquetas))

# Gráfica del codo
plt.figure(figsize=(8,5))
plt.plot(rango_k, inercia, marker='o')
plt.title("Método del codo")
plt.xlabel("Número de clusters")
plt.ylabel("Inercia")
plt.show()

# Gráfica silhouette
plt.figure(figsize=(8,5))
plt.plot(rango_k, sil_scores, marker='o')
plt.title("Silhouette Score")
plt.xlabel("Número de clusters")
plt.ylabel("Score")
plt.show()

# ==========================================================
# 6. MODELO KMEANS FINAL
# ==========================================================

k_seleccionado = 2

modelo_kmeans = Pipeline([
    ("prep", preprocesador),
    ("kmeans", KMeans(n_clusters=k_seleccionado, random_state=SEED, n_init=10))
])

modelo_kmeans.fit(datos)

labels_kmeans = modelo_kmeans["kmeans"].labels_

print(f"\nKMeans con k = {k_seleccionado}")
print("Inercia:", modelo_kmeans["kmeans"].inertia_)
print("Silhouette:", silhouette_score(preprocesador.transform(datos), labels_kmeans))

# ---------- Visualización 2D ----------
plt.figure(figsize=(8,5))
plt.scatter(datos_2d[:, 0], datos_2d[:, 1], c=labels_kmeans, s=35)
plt.title("Clusters KMeans (2D)")
plt.show()

# ---------- Visualización 3D ----------
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(datos_3d[:, 0], datos_3d[:, 1], datos_3d[:, 2], c=labels_kmeans, s=35)
ax.set_title("Clusters KMeans (3D)")
plt.show()

# ---------- Centroides ----------
centros = modelo_kmeans["kmeans"].cluster_centers_
centros_3d = pca_3d.transform(centros)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(datos_3d[:, 0], datos_3d[:, 1], datos_3d[:, 2], c=labels_kmeans, s=30)
ax.scatter(centros_3d[:, 0], centros_3d[:, 1], centros_3d[:, 2], marker='X', s=200)

ax.set_title("KMeans con centroides")
plt.show()

# ==========================================================
# 7. MODELO DBSCAN
# ==========================================================

modelo_dbscan = DBSCAN(eps=0.8, min_samples=10)
labels_db = modelo_dbscan.fit_predict(datos_escalados)

print("\nDBSCAN")
print("Clusters encontrados:", np.unique(labels_db))
print("Distribución:", np.unique(labels_db, return_counts=True))

# Silhouette (sin ruido)
if len(set(labels_db) - {-1}) > 1:
    mask = labels_db != -1
    sil = silhouette_score(datos_escalados[mask], labels_db[mask])
    print("Silhouette (sin ruido):", sil)
else:
    print("No se puede calcular silhouette (muy pocos clusters)")

# ---------- Visualización ----------
plt.figure(figsize=(8,5))
plt.scatter(datos_2d[:, 0], datos_2d[:, 1], c=labels_db, s=35)
plt.title("DBSCAN (2D)")
plt.show()

# ==========================================================
# 8. COMPARACIÓN CON ETIQUETAS REALES (OPCIONAL)
# ==========================================================

if etiquetas_reales is not None:
    
    plt.figure(figsize=(8,5))
    plt.scatter(datos_2d[:, 0], datos_2d[:, 1], c=etiquetas_reales, s=35)
    plt.title("Etiquetas reales")
    plt.show()
    
    ari = adjusted_rand_score(etiquetas_reales, labels_kmeans)
    print(f"\nARI (KMeans vs Real): {ari:.4f}")