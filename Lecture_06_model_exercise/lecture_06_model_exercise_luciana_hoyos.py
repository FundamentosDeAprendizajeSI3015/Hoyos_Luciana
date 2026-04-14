# ==========================================================
# PROYECTO: CLASIFICACIÓN - DESERCIÓN ESTUDIANTIL
# Modelos: Random Forest vs Gradient Boosting
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import *

# ==========================================================
# CONFIGURACIÓN
# ==========================================================

SEED = 42
plt.rc('font', family='serif', size=12)

# ==========================================================
# CARGA DE DATOS
# ==========================================================

print("\n📥 Leyendo dataset...")
df = pd.read_csv("dataset_desercion_estudiantes (1).csv")

print("\n🔍 Vista general:")
print(df.head())

print("\n📊 Info:")
print(df.info())

# ==========================================================
# DEFINICIÓN DEL TARGET
# ==========================================================

TARGET = "desercion" if "desercion" in df.columns else df.columns[-1]
print(f"\n🎯 Variable objetivo: {TARGET}")

# ==========================================================
# LIMPIEZA
# ==========================================================

print("\n🧹 Limpieza...")

df = df.drop_duplicates()

# Rellenar nulos numéricos
df.fillna(df.median(numeric_only=True), inplace=True)

# ==========================================================
# FEATURES Y TARGET
# ==========================================================

X = df.drop(columns=TARGET)
y = df[TARGET]

# 🔥 SOLUCIÓN 1: Convertir a binario (0 y 1)
y = y.astype(str).str.lower().map({
    "no": 0,
    "sí": 1,
    "si": 1
})

# Verificación rápida
print("\nValores únicos del target después de transformación:")
print(y.value_counts())

# ==========================================================
# TIPOS DE VARIABLES
# ==========================================================

num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(include="object").columns

print("\nColumnas numéricas:", list(num_cols))
print("Columnas categóricas:", list(cat_cols))

# ==========================================================
# SPLIT 60 / 20 / 20
# ==========================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=SEED
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED
)

# ==========================================================
# PREPROCESAMIENTO
# ==========================================================

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# ==========================================================
# MODELOS
# ==========================================================

rf = Pipeline([
    ("prep", preprocess),
    ("model", RandomForestClassifier(n_estimators=150, random_state=SEED))
])

gb = Pipeline([
    ("prep", preprocess),
    ("model", GradientBoostingClassifier(n_estimators=150, random_state=SEED))
])

# ==========================================================
# ENTRENAMIENTO
# ==========================================================

print("\n🚀 Entrenando modelos...")

rf.fit(X_train, y_train)
gb.fit(X_train, y_train)

# ==========================================================
# VISUALIZACIÓN DE ÁRBOLES
# ==========================================================

def graficar_arbol(modelo, titulo):
    clf = modelo.named_steps["model"]
    features = modelo.named_steps["prep"].get_feature_names_out()

    if hasattr(clf, "estimators_"):
        if isinstance(clf.estimators_[0], np.ndarray):
            tree = clf.estimators_[0][0]
        else:
            tree = clf.estimators_[0]

        plt.figure(figsize=(18,8))
        plot_tree(
            tree,
            feature_names=features,
            filled=True,
            max_depth=3
        )
        plt.title(titulo)
        plt.show()

graficar_arbol(rf, "Árbol - Random Forest")
graficar_arbol(gb, "Árbol - Gradient Boosting")

# ==========================================================
# FUNCIÓN DE EVALUACIÓN
# ==========================================================

def evaluar(modelo, X, y, nombre):

    pred = modelo.predict(X)
    prob = modelo.predict_proba(X)[:,1]

    print(f"\n===== {nombre} =====")
    print("Accuracy :", accuracy_score(y, pred))
    print("Precision:", precision_score(y, pred))
    print("Recall   :", recall_score(y, pred))
    print("F1 Score :", f1_score(y, pred))
    print("AUC      :", roc_auc_score(y, prob))

    print("\nMatriz:")
    print(confusion_matrix(y, pred))

    # Heatmap
    plt.figure()
    sns.heatmap(confusion_matrix(y, pred), annot=True, fmt='d')
    plt.title(f"Confusion Matrix - {nombre}")
    plt.show()

    # ROC
    fpr, tpr, _ = roc_curve(y, prob)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0,1], [0,1])
    plt.title(f"ROC - {nombre}")
    plt.show()

# ==========================================================
# EVALUACIÓN
# ==========================================================

print("\n📊 Evaluación TRAIN")
evaluar(rf, X_train, y_train, "RF Train")
evaluar(gb, X_train, y_train, "GB Train")

print("\n📊 Evaluación VALIDATION")
evaluar(rf, X_val, y_val, "RF Validation")
evaluar(gb, X_val, y_val, "GB Validation")

print("\n📊 Evaluación TEST")
evaluar(rf, X_test, y_test, "RF Test")
evaluar(gb, X_test, y_test, "GB Test")