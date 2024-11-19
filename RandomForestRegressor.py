from limpieza_de_datos import eliminar_outliers

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold

df = pd.read_csv('/Users/danny/OneDrive/Escritorio/airline_haro/train_airlines_delay_challenge.csv')
#df = pd.read_csv('/Users/danny/OneDrive/Documents/GitHub/airline_haro/train_airlines_delay_challenge.csv')

# Seleccionar columnas numéricas para análisis de outliers
columnas_numericas = ['DEP_TIME', 'DEP_DELAY', 'TAXI_OUT', 'WHEELS_OFF', 'ARR_DELAY']
for columna in columnas_numericas:
    df = eliminar_outliers(df, columna)
    
# Parámetro para ajustar el tamaño de la muestra (proporción de los datos)
sample_size = 0.10  #Define el porcentaje de datos que deseas usar.

# Seleccionar solo las características numéricas y obtener una muestra
df_airlines_numeric = df.select_dtypes(include=['number']).sample(frac=sample_size, random_state=42)
#Selecciona aleatoriamente la fracción de datos especificada para reducir el tiempo de procesamiento.

# Separar las variables predictoras y la variable objetivo
X = df_airlines_numeric.drop(columns=['ARR_DELAY'])  # Elimina 'ARR_DELAY' de las características predictoras
y = df_airlines_numeric['ARR_DELAY']  # Variable objetivo

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo Random Forest
modelo_rf = RandomForestRegressor(random_state=42)
modelo_rf.fit(X_train, y_train)

# Obtener la importancia de las características
importancia = modelo_rf.feature_importances_
caracteristicas = X.columns

# Crear un DataFrame para mostrar la importancia
importancia_df = pd.DataFrame({'Característica': caracteristicas, 'Importancia': importancia})
importancia_df = importancia_df.sort_values(by='Importancia', ascending=False)

# Visualizar la importancia de las características
plt.figure(figsize=(10, 8))
plt.barh(importancia_df['Característica'], importancia_df['Importancia'], color='skyblue')
plt.xlabel("Importancia")
plt.ylabel("Características")
plt.title("Importancia de las Características Numéricas usando Random Forest")
plt.gca().invert_yaxis()
plt.show()

print("Características numéricas ordenadas por importancia:")
print(importancia_df)