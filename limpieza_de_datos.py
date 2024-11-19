import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/danny/OneDrive/Escritorio/airline_haro/train_airlines_delay_challenge.csv')
#df = pd.read_csv('/Users/danny/OneDrive/Documents/GitHub/airline_haro/train_airlines_delay_challenge.csv')
#df.shape

# Resumen de valores nulos antes de la limpieza
print("Valores nulos antes de la limpieza:")
print(df.isnull().sum())

# Eliminar filas con valores nulos en caso de necesitarlo
    #df_sin_nulos = df.dropna()

# Resumen de valores nulos después de la limpieza
    #print("Valores nulos después de la limpieza:")
    #print(df_sin_nulos.isnull().sum())
    
# Número de filas duplicadas antes de la limpieza
duplicados = df.duplicated().sum()
print(f"Número de filas duplicadas antes de la limpieza: {duplicados}")

# Eliminar filas duplicadas en caso de necesitarlo
df_sin_duplicados = df.drop_duplicates()

# Número de filas duplicadas después de la limpieza
duplicados_despues = df_sin_duplicados.duplicated().sum()
print(f"Número de filas duplicadas después de la limpieza: {duplicados_despues}")

# Revisar los tipos de datos de cada columna
print("Tipos de datos antes de la limpieza:")
print(df.dtypes)

# Seleccionar columnas numéricas para análisis de outliers
columnas_numericas = ['DEP_TIME', 'DEP_DELAY', 'TAXI_OUT', 'WHEELS_OFF', 'ARR_DELAY']
# Definir una función para eliminar outliers basados en el IQR
def eliminar_outliers(df, columna):
    # Calcular el primer y tercer cuartil (Q1 y Q3)
    Q1 = df[columna].quantile(0.25)
    Q3 = df[columna].quantile(0.75)
    IQR = Q3 - Q1  # Rango intercuartílico

    # Definir límites inferior y superior
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    # Filtrar los datos que están dentro de los límites
    df_sin_outliers = df[(df[columna] >= limite_inferior) & (df[columna] <= limite_superior)]
    return df_sin_outliers

# Aplicar la función a cada columna numérica
for columna in columnas_numericas:
    df = eliminar_outliers(df, columna)

# Mostrar dataset después de eliminar outliers
print("Datos después de eliminar outliers:")
print(df.describe())

def boxplot_columnas(columnas):

    plt.figure(figsize=(15, 10))
    for i, col in enumerate(columnas):
        plt.subplot(1, len(columnas), i + 1)  # Crear subplots en una sola fila
        sns.boxplot(y=df[col], color='skyblue')
        plt.title(f"Boxplot de {col}")
        plt.xlabel(col)
    plt.tight_layout()
    plt.show()

columnas_seleccionadas = ['DEP_TIME', 'DEP_DELAY', 'TAXI_OUT', 'WHEELS_OFF', 'ARR_DELAY']
boxplot_columnas(columnas_seleccionadas)