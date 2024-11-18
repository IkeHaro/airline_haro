import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/danny/OneDrive/Escritorio/airline_haro/train_airlines_delay_challenge.csv')
#print(df.head())

# 2. Resumen de estadísticas descriptivas
print("\nEstadísticas descriptivas de las columnas numéricas:")
print(df.describe())

print("\nEstadísticas descriptivas de las columnas categóricas:")
print(df.describe(include=['object']))

# 3. Revisión de valores nulos
print("\nValores nulos por columna:")
print(df.isnull().sum())

# 4. Distribución de las variables numéricas
columnas_numericas = df.select_dtypes(include=['float64', 'int64']).columns
df[columnas_numericas].hist(bins=15, figsize=(15, 10))
plt.suptitle("Distribución de las variables numéricas")
plt.show()

# 5. Visualización de variables categóricas
columnas_categoricas = df.select_dtypes(include=['object']).columns
for columna in columnas_categoricas:
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x=columna)
    plt.title(f"Distribución de la variable categórica: {columna}")
    plt.xticks(rotation=45)
    plt.show()
    
# 6. Matriz de correlación para variables numéricas
plt.figure(figsize=(10, 8))
sns.heatmap(df[columnas_numericas].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriz de correlación entre variables numéricas")
plt.show()

# 7. Boxplots para detectar outliers visualmente
for columna in columnas_numericas:
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x=columna)
    plt.title(f"Boxplot de la variable: {columna}")
    plt.show()
    
