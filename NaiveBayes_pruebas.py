from extr_trans_carac import df_airlines
from sklearn.utils import resample
import pandas as pd

print("Pereme tantito, ando chambeando")

# Discretizar ARR_DELAY en categorías
def categorizar_retraso(delay):
    if delay >= 15:
        return 'Delay'
    elif delay <= -15:
        return 'Early'
    else:
        return 'OnTime'
    
df_airlines['ARR_DELAY_CATEGORY'] = df_airlines['ARR_DELAY'].apply(categorizar_retraso)

# Separar las clases
df_majority = df_airlines[df_airlines['ARR_DELAY_CATEGORY'] == 'OnTime']
df_minority_early = df_airlines[df_airlines['ARR_DELAY_CATEGORY'] == 'Early']
df_minority_delay = df_airlines[df_airlines['ARR_DELAY_CATEGORY'] == 'Delay']

# Reducir la clase mayoritaria al tamaño de la clase minoritaria más pequeña
minority_size = min(len(df_minority_early), len(df_minority_delay))
df_majority_downsampled = resample(df_majority,replace=False,n_samples=minority_size,random_state=42)

# Combinar las clases balanceadas
df_balanced = pd.concat([df_majority_downsampled, df_minority_early, df_minority_delay])

import matplotlib.pyplot as plt
def dist_variables_numericas():
    # Distribución ARR_DELAY_CATEGORY
    columnas_numericas = df_balanced.select_dtypes(include=['float64', 'int64']).columns
    df_balanced[columnas_numericas].hist(bins=15, figsize=(15, 10))
    plt.suptitle("Distribución de las variables numéricas")
    plt.show()    
columnas_seleccionadas = ['ARR_DELAY_CATEGORY']
dist_variables_numericas()

# Selección de muestra
sample_size = 0.30  
df_sample = df_balanced.sample(frac=sample_size, random_state=42)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Escalar los datos (solo las columnas numéricas)
X_scaled = StandardScaler().fit_transform(df_sample.drop(columns=['ARR_DELAY', 'ARR_DELAY_CATEGORY']))

# Aplicar PCA para reducir la dimensionalidad
pca = PCA(n_components=100, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Definir X (características) e y (variable objetivo)
X = X_pca
y = df_sample['ARR_DELAY_CATEGORY']

# Paso 3: Dividir los datos en entrenamiento y prueba
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Paso 4: Entrenamiento del modelo Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Entrenar el modelo
modelo_nb = GaussianNB()
modelo_nb.fit(X_train, y_train)

# Predicción y evaluación
y_pred = modelo_nb.predict(X_test)

# Reporte de resultados
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

print("\nExactitud del modelo:")
print(accuracy_score(y_test, y_pred))