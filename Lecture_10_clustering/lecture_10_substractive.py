# ==========================================================
# PROYECTO: CLUSTERING AVANZADO
# Métodos: KMeans, DBSCAN, Subtractive Clustering y Fuzzy C-Means
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
# 🔹 IMPLEMENTACIÓN DE MODELOS PERSONALIZADOS
# ==========================================================

class SubtractiveClustering:
    """
    Método de clustering basado en densidad.
    Identifica centros potenciales evaluando la densidad local de los puntos.
    """

    def __init__(self, ra=0.5, rb=0.75, eps_upper=0.5, eps_lower=0.15):
        self.ra = ra
        self.rb = rb
        self.eps_upper = eps_upper
        self.eps_lower = eps_lower
        self.centers_ = None
        self.n_clusters_ = 0

    def _normalizar(self, X):
        """Escala los datos entre 0 y 1"""
        self.x_min = X.min(axis=0)
        self.x_range = X.max(axis=0) - self.x_min
        self.x_range[self.x_range == 0] = 1
        return (X - self.x_min) / self.x_range

    def fit(self, X):
        X_norm = self._normalizar(X)

        # Cálculo inicial de densidad
        densidad = np.zeros(len(X_norm))
        for punto in X_norm:
            dist = np.sum(((X_norm - punto) / (self.ra / 2))**2, axis=1)
            densidad += np.exp(-dist)

        centros = []
        densidad_original = densidad.copy()

        while True:
            idx = np.argmax(densidad)
            valor = densidad[idx]

            if valor == 0:
                break

            candidato = X_norm[idx]

            if not centros:
                centros.append(candidato)
            else:
                dist_min = min(np.linalg.norm(candidato - c) for c in centros)
                if (dist_min / self.ra) < 1:
                    densidad[idx] = 0
                    continue
                centros.append(candidato)

            # Reducción de densidad alrededor del nuevo centro
            dist = np.sum(((X_norm - candidato) / (self.rb / 2))**2, axis=1)
            densidad -= valor * np.exp(-dist)
            densidad = np.clip(densidad, 0, None)

        if centros:
            self.centers_ = np.array(centros) * self.x_range + self.x_min
            self.n_clusters_ = len(self.centers_)
        else:
            self.centers_ = np.mean(X, axis=0, keepdims=True)
            self.n_clusters_ = 1

        return self

    def predict(self, X):
        distancias = np.array([np.linalg.norm(X - c, axis=1) for c in self.centers_])
        return np.argmin(distancias, axis=0)


class FuzzyCMeans:
    """
    Clustering difuso: cada punto pertenece parcialmente a varios clusters.
    """

    def __init__(self, n_clusters=3, m=2, max_iter=300, tol=1e-6, random_state=42):
        self.k = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _init_U(self, n):
        rng = np.random.default_rng(self.random_state)
        U = rng.random((n, self.k))
        return U / U.sum(axis=1, keepdims=True)

    def fit(self, X, init_centers=None):
        n = len(X)

        if init_centers is not None:
            centers = init_centers
            U = self._update_U(X, centers)
        else:
            U = self._init_U(n)
            centers = self._update_centers(X, U)

        for i in range(self.max_iter):
            U_old = U.copy()
            centers = self._update_centers(X, U)
            U = self._update_U(X, centers)

            if np.max(np.abs(U - U_old)) < self.tol:
                break

        self.centers_ = centers
        self.U_ = U
        self.n_iter_ = i + 1
        return self

    def _update_centers(self, X, U):
        um = U ** self.m
        return (um.T @ X) / um.sum(axis=0)[:, None]

    def _update_U(self, X, centers):
        dist = np.array([np.linalg.norm(X - c, axis=1) for c in centers]).T
        dist = np.fmax(dist, 1e-10)

        exp = 2 / (self.m - 1)
        U = np.zeros_like(dist)

        for i in range(len(centers)):
            ratio = dist[:, i:i+1] / dist
            U[:, i] = 1 / np.sum(ratio**exp, axis=1)

        return U

    def predict(self, X):
        U = self._update_U(X, self.centers_)
        return np.argmax(U, axis=1)


# ==========================================================
# 🔹 FUNCIÓN PARA MÉTRICA SEGURA
# ==========================================================

def calcular_silhouette(X, labels, nombre):
    if len(np.unique(labels)) < 2:
        print(f"{nombre}: no se puede calcular silhouette.")
        return None
    return silhouette_score(X, labels)


# ==========================================================
# 🔹 CONFIGURACIÓN Y CARGA
# ==========================================================

SEED = 42
plt.rc('font', family='serif', size=12)

data = pd.read_csv("dataset_sintetico_FIRE_UdeA_realista.csv")

print("Información básica:")
print(data.info())

# ==========================================================
# 🔹 PREPARACIÓN DE DATOS
# ==========================================================

if "label" in data.columns:
    y_real = data["label"]
    X = data.drop(columns=["label"])
else:
    y_real = None
    X = data.copy()

num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(include="object").columns

# Pipeline de limpieza
preprocesador = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ]), num_cols),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), cat_cols)
])

X_proc = preprocesador.fit_transform(X)

# ==========================================================
# 🔹 REDUCCIÓN DE DIMENSIONALIDAD
# ==========================================================

pca2 = PCA(n_components=2, random_state=SEED)
X_2d = pca2.fit_transform(X_proc)

pca3 = PCA(n_components=3, random_state=SEED)
X_3d = pca3.fit_transform(X_proc)

# ==========================================================
# 🔹 KMEANS
# ==========================================================

k = 2
modelo_k = KMeans(n_clusters=k, random_state=SEED, n_init=10)
labels_k = modelo_k.fit_predict(X_proc)

print("\nKMeans:", calcular_silhouette(X_proc, labels_k, "KMeans"))

# ==========================================================
# 🔹 DBSCAN
# ==========================================================

modelo_db = DBSCAN(eps=1.2, min_samples=5)
labels_db = modelo_db.fit_predict(X_proc)

print("DBSCAN:", calcular_silhouette(X_proc, labels_db, "DBSCAN"))

# ==========================================================
# 🔹 SUBTRACTIVE CLUSTERING
# ==========================================================

modelo_sub = SubtractiveClustering()
modelo_sub.fit(X_proc)

labels_sub = modelo_sub.predict(X_proc)
print("Subtractive:", calcular_silhouette(X_proc, labels_sub, "Subtractive"))

# ==========================================================
# 🔹 FUZZY C-MEANS
# ==========================================================

k_fcm = modelo_sub.n_clusters_ if modelo_sub.n_clusters_ >= 2 else k

modelo_fcm = FuzzyCMeans(n_clusters=k_fcm)
modelo_fcm.fit(X_proc)

labels_fcm = modelo_fcm.predict(X_proc)
print("Fuzzy C-Means:", calcular_silhouette(X_proc, labels_fcm, "FCM"))

# ==========================================================
# 🔹 COMPARACIÓN CON ETIQUETAS REALES
# ==========================================================

if y_real is not None:
    print("\nComparación con etiquetas reales:")
    print("ARI KMeans:", adjusted_rand_score(y_real, labels_k))
    print("ARI Subtractive:", adjusted_rand_score(y_real, labels_sub))
    print("ARI FCM:", adjusted_rand_score(y_real, labels_fcm))

# ==========================================================
# 🔹 GUARDAR RESULTADOS
# ==========================================================

resultado = data.copy()
resultado["kmeans"] = labels_k
resultado["dbscan"] = labels_db
resultado["subtractive"] = labels_sub
resultado["fcm"] = labels_fcm

resultado.to_csv("clusters_resultado.csv", index=False)

print("\nArchivo guardado: clusters_resultado.csv")