import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/danny/OneDrive/Escritorio/airline_haro/train_airlines_delay_challenge.csv')
#df = pd.read_csv('/Users/danny/OneDrive/Documents/GitHub/airline_haro/train_airlines_delay_challenge.csv')

#print(df.head())

def estadisticas_descriptivas():
    # Resumen de estadísticas descriptivas
    print("\nEstadísticas descriptivas de las columnas numéricas:")
    print(df.describe())

    print("\nEstadísticas descriptivas de las columnas categóricas:")
    print(df.describe(include=['object']))

    # Revisar los tipos de datos de cada columna
    print("Tipos de datos antes de la limpieza:")
    print(df.dtypes)
    
def dist_variables_numericas():
    # Distribución de las variables numéricas
    columnas_numericas = df.select_dtypes(include=['float64', 'int64']).columns
    df[columnas_numericas].hist(bins=15, figsize=(15, 10))
    plt.suptitle("Distribución de las variables numéricas")
    plt.show()
    
    # Matriz de correlación para variables numéricas
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[columnas_numericas].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Matriz de correlación entre variables numéricas")
    plt.show()
        
def vis_variables_categoricas():
    # Visualización de variables categóricas
    columnas_categoricas = df.select_dtypes(include=['object']).columns
    for columna in columnas_categoricas:
        plt.figure(figsize=(10, 5))
        sns.countplot(data=df, x=columna)
        plt.title(f"Distribución de la variable categórica: {columna}")
        plt.xticks(rotation=45)
        plt.show()
    
def boxplot_columnas(columnas):

    plt.figure(figsize=(15, 10))
    for i, col in enumerate(columnas):
        plt.subplot(1, len(columnas), i + 1)  # Crear subplots en una sola fila
        sns.boxplot(y=df[col], color='skyblue')
        plt.title(f"Boxplot de {col}")
        plt.xlabel(col)
    plt.tight_layout()
    plt.show()

        
#Main

#estadisticas_descriptivas()
#dist_variables_numericas()
#vis_variables_categoricas()

columnas_seleccionadas = ['DEP_TIME', 'DEP_DELAY', 'TAXI_OUT', 'WHEELS_OFF', 'ARR_DELAY']
boxplot_columnas(columnas_seleccionadas)