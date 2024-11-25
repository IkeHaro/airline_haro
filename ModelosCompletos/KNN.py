import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample

df = pd.read_csv('/Users/danny/OneDrive/Escritorio/airline_haro/train_airlines_delay_challenge.csv')

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

# Crear intervalos de tiempo para horas de salida y llegada
def categorizar_hora(hora):
    if 5 <= hora < 12:
        return 'mañana'
    elif 12 <= hora < 18:
        return 'tarde'
    elif 18 <= hora < 24:
        return 'noche'
    else:
        return 'madrugada'

df['DEP_TIME_CATEGORY'] = ((df['CRS_DEP_TIME'] // 100) % 24).apply(categorizar_hora)
df['ARR_TIME_CATEGORY'] = ((df['CRS_ARR_TIME'] // 100) % 24).apply(categorizar_hora)

# Extraer día de la semana y mes de la fecha
df['DAY_OF_WEEK'] = pd.to_datetime(df['FL_DATE']).dt.dayofweek
df['MONTH'] = pd.to_datetime(df['FL_DATE']).dt.month

# Calcular la velocidad media planeada
df['AVG_SPEED_PLANNED'] = df['DISTANCE'] / (df['CRS_ELAPSED_TIME'] / 60)

# Crear variable binaria para indicar si el retraso fue significativo
df['LATE_ARRIVAL'] = (df['ARR_DELAY'] > 15).astype(int)

# Convertir variables categóricas en dummies
df_airlines = pd.get_dummies(df, columns=['OP_CARRIER', 'ORIGIN', 'DEST', 'DEP_TIME_CATEGORY', 'ARR_TIME_CATEGORY'], drop_first=True)

# Eliminar columnas irrelevantes o ya procesadas
df_airlines = df_airlines.drop(columns=['FL_DATE', 'WHEELS_OFF', 'DEP_TIME', 'OP_CARRIER_FL_NUM'])

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

# Reducir las clases mayoritarias al tamaño de la clase minoritaria
minority_size = min(len(df_minority_early), len(df_minority_delay))
df_majority_downsampled = resample(df_majority,replace=False,n_samples=minority_size,random_state=42)
df_early_downsampled = resample(df_minority_early,replace=False,n_samples=minority_size,random_state=42)


# Combinar las clases balanceadas
df_balanced = pd.concat([df_majority_downsampled, df_early_downsampled, df_minority_delay])

# Verificar conteos de clases tras submuestreo
class_counts = df_balanced['ARR_DELAY_CATEGORY'].value_counts()
print("Distribución de clases tras submuestreo:")
print(class_counts)

# Visualizar la distribución de las clases
plt.figure(figsize=(8, 6))
sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')
plt.title('Distribución de clases tras submuestreo')
plt.xlabel('Categoría de ARR_DELAY')
plt.ylabel('Número de instancias')
plt.show()

# Selección de muestra
sample_size = 0.80  
df_sample = df_balanced.sample(frac=sample_size, random_state=42)

# Dividir en características (X) y objetivo (y)
X = df_sample.drop(columns=['ARR_DELAY_CATEGORY'])
y = df_sample['ARR_DELAY_CATEGORY']

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Validación cruzada para determinar el mejor número de vecinos
k_range = range(1, 21)  # Probamos valores de k entre 1 y 20
cv_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

# Visualizar los resultados de la validación cruzada
plt.figure(figsize=(10, 6))
plt.plot(k_range, cv_scores, marker='o', linestyle='-', color='b')
plt.title('Validación Cruzada - Determinación del Mejor k')
plt.xlabel('Número de Vecinos (k)')
plt.ylabel('Accuracy promedio')
plt.xticks(k_range)
plt.grid()
plt.show()

# Encontrar el mejor valor de k
best_k = k_range[cv_scores.index(max(cv_scores))]
print(f"El mejor número de vecinos (k) es: {best_k}")

# Entrenar el modelo con el mejor número de vecinos
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train_scaled, y_train)

# Realizar predicciones
y_pred = knn_best.predict(X_test_scaled)

# Evaluar el modelo
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred, labels=knn_best.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn_best.classes_)
disp.plot(cmap='Blues', values_format='d')
plt.title("Matriz de Confusión para KNN (k óptimo)")
plt.show()