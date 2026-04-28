# ==========================================================
# ANÁLISIS FINANCIERO SIN MACHINE LEARNING
# Clasificación basada en reglas (EDA)
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# CARGA DE DATOS
# ==========================================================

data_A = pd.read_csv("dataset_sintetico_FIRE_UdeA_realista.csv")
data_B = pd.read_csv("dataset_sintetico_FIRE_UdeA.csv")

# ==========================================================
# FUNCIÓN DE ANÁLISIS EXPLORATORIO
# ==========================================================

def analizar_dataset(data, nombre):

    print(f"\n{'='*50}")
    print(f"ANÁLISIS - {nombre}")
    print(f"{'='*50}")

    print("\nPrimeras filas:")
    print(data.head())

    print("\nInformación general:")
    print(data.info())

    print("\nValores nulos:")
    print(data.isnull().sum())

    print("\nEstadísticas:")
    print(data.describe())

    print("\nCorrelación:")
    plt.figure(figsize=(10,8))
    sns.heatmap(data.corr(numeric_only=True), cmap="coolwarm", annot=False)
    plt.title(f"Heatmap - {nombre}")
    plt.show()


# Ejecutar análisis
analizar_dataset(data_A, "Dataset A")
analizar_dataset(data_B, "Dataset B")

# ==========================================================
# LIMPIEZA BÁSICA
# ==========================================================

def limpiar_datos(data):

    data = data.copy()

    # Eliminar duplicados
    data.drop_duplicates(inplace=True)

    # Rellenar nulos numéricos con mediana
    for col in data.select_dtypes(include=np.number).columns:
        data[col] = data[col].fillna(data[col].median())

    # Rellenar categóricos
    for col in data.select_dtypes(include="object").columns:
        data[col] = data[col].fillna("Unknown")

    return data

data_A = limpiar_datos(data_A)
data_B = limpiar_datos(data_B)

# ==========================================================
# REGLAS DE CLASIFICACIÓN FINANCIERA
# ==========================================================

def clasificar_finanzas(row, data):

    score = 0

    # Liquidez baja
    if "liquidez" in row and row["liquidez"] < data["liquidez"].quantile(0.25):
        score += 1

    # Flujo de caja bajo
    if "cfo" in row and row["cfo"] < data["cfo"].quantile(0.25):
        score += 1

    # Alto endeudamiento
    if "endeudamiento" in row and row["endeudamiento"] > data["endeudamiento"].quantile(0.75):
        score += 1

    # Gastos altos
    if "gastos_personal" in row and row["gastos_personal"] > data["gastos_personal"].quantile(0.75):
        score += 1

    # Baja tendencia de ingresos
    if "tendencia_ingresos" in row and row["tendencia_ingresos"] < data["tendencia_ingresos"].quantile(0.25):
        score += 1

    # Clasificación final
    if score >= 2:
        return "Crítico"
    else:
        return "Estable"

# ==========================================================
# APLICAR CLASIFICACIÓN
# ==========================================================

def aplicar_clasificacion(data, nombre):

    data = data.copy()

    data["estado_financiero"] = data.apply(
        lambda row: clasificar_finanzas(row, data), axis=1
    )

    print(f"\nResultados - {nombre}")
    print(data["estado_financiero"].value_counts())

    return data

data_A = aplicar_clasificacion(data_A, "Dataset A")
data_B = aplicar_clasificacion(data_B, "Dataset B")

# ==========================================================
# IDENTIFICAR UNIDADES CRÍTICAS
# ==========================================================
def obtener_criticos(data):
    return data[data["estado_financiero"] == "Crítico"]

#def obtener_criticos(data):

    #columnas = [col for col in ["unidad", "anio"] if col in data.columns]

    #return data[data["estado_financiero"] == "Crítico"][columnas]

criticos_A = obtener_criticos(data_A)
criticos_B = obtener_criticos(data_B)

print("\nUnidades críticas - Dataset A:")
print(criticos_A)

print("\nUnidades críticas - Dataset B:")
print(criticos_B)

# ==========================================================
# COMPARACIÓN FINAL
# ==========================================================

print("\nCOMPARACIÓN FINAL")

print("\nDataset A:")
print(data_A["estado_financiero"].value_counts(normalize=True))

print("\nDataset B:")
print(data_B["estado_financiero"].value_counts(normalize=True))

# ==========================================================
# VISUALIZACIÓN FINAL
# ==========================================================

def graficar_resultados(data, nombre):

    plt.figure()
    data["estado_financiero"].value_counts().plot(kind="bar")
    plt.title(f"Estado Financiero - {nombre}")
    plt.xticks(rotation=0)
    plt.show()

graficar_resultados(data_A, "Dataset A")
graficar_resultados(data_B, "Dataset B")